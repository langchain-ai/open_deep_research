# Multi-Turn Research: Options and Trade-offs

The graph is closer to multi-turn than it looks out of the box:

- `AgentState` extends `MessagesState`, which uses LangGraph's `add_messages` reducer — messages accumulate automatically across invocations if you use a checkpointer
- Both `write_research_brief` and `final_report_generation` pass `get_buffer_string(state["messages"])` into their prompts — so a follow-up message in state automatically becomes context
- `clarify_with_user` already reads the full message history and is told not to re-ask questions it already asked

The only gap: after `final_report_generation → END`, state is discarded unless you persist it.

---

## Option 1: Script-level message accumulation

Manually carry messages forward between invocations. No changes to the graph.

**File to change:** `run_query.py`

```python
messages = [HumanMessage(content=QUERY)]

while True:
    result = await deep_researcher.ainvoke(
        {"messages": messages},
        config={"configurable": {"allow_clarification": False}},
    )
    print(result["final_report"])

    # Carry the full conversation forward
    messages = result["messages"]   # includes AI's report as the last message

    follow_up = input("\nFollow-up question (or 'quit'): ").strip()
    if follow_up.lower() == "quit":
        break
    messages.append(HumanMessage(content=follow_up))
```

**Trade-off:** Simple — no graph changes. But the entire research pipeline re-runs from scratch each turn. The supervisor has to re-derive everything from the conversation history.

---

## Option 2: LangGraph checkpointing (recommended)

### How it works (conceptually)

LangGraph has a built-in concept of a "thread" — a named conversation that persists across multiple graph runs. When you attach a checkpointer to the graph, LangGraph saves the entire state (all messages, the research brief, the notes, everything) to storage after every node execution. The next time you invoke the graph with the same thread ID, it loads that saved state, appends your new message to it, and runs the graph again from `START`.

So from the agent's perspective, turn 2 looks exactly like turn 1 — except the `messages` list already contains the full history including the previous report. The `write_research_brief` node sees all of that and writes a new brief that incorporates the follow-up context. The `final_report_generation` node does the same.

**What happens between turns:** The graph fully completes and exits to `END` after each turn. Nothing is running between turns — the state is just sitting in storage (in-memory, or a SQLite file) waiting to be resumed.

**When to use this:** When your follow-up is a new research question that happens to be informed by the previous answer. For example: first query is "explain transformer architecture", follow-up is "how does that compare to Mamba?" — the agent benefits from knowing what it already told you, but it still needs to go out and do fresh research on Mamba. The full pipeline re-runs, which is what you want. The previous report being in the message history means the new report won't repeat itself unnecessarily.

### Implementation

LangGraph's built-in mechanism for stateful, resumable graphs. With a checkpointer, re-invoking the same `thread_id` merges new messages into saved state and re-runs from `START` — so `write_research_brief` sees the full conversation including the prior report.

### Change 1 — `src/open_deep_research/deep_researcher.py`

At the bottom where `compile()` is called (line ~719):

```python
from langgraph.checkpoint.memory import MemorySaver

deep_researcher = deep_researcher_builder.compile(checkpointer=MemorySaver())
```

For persistence across process restarts, swap `MemorySaver` for `AsyncSqliteSaver`:

```python
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

async with AsyncSqliteSaver.from_conn_string("checkpoints.db") as checkpointer:
    deep_researcher = deep_researcher_builder.compile(checkpointer=checkpointer)
```

### Change 2 — `run_query.py`

Add a `thread_id` and loop. On follow-up turns, only pass the new message — LangGraph merges it with saved state automatically.

```python
import uuid

thread_id = str(uuid.uuid4())   # same thread = same conversation

messages = [HumanMessage(content=QUERY)]
while True:
    result = await deep_researcher.ainvoke(
        {"messages": messages},
        config={
            "configurable": {
                "thread_id": thread_id,
                "allow_clarification": False,
            }
        },
    )
    print(result["final_report"])

    follow_up = input("\nFollow-up (or 'quit'): ").strip()
    if follow_up.lower() == "quit":
        break
    # Only pass the new message — LangGraph merges it with saved state
    messages = [HumanMessage(content=follow_up)]
```

**Trade-off:** Cleanest solution. State is persisted per thread. The previous `final_report` and all prior messages are in state when the follow-up runs. However, the graph still re-runs research from scratch — it does not skip sub-topics it already covered.

---

## Option 3: Graph loop with interrupt (most invasive)

### How it works (conceptually)

Instead of the graph ending after each report, it loops back to the beginning and pauses — using LangGraph's `interrupt()` mechanism — waiting for user input before proceeding. The graph never truly exits between turns; it is suspended mid-execution at a specific node, holding all its accumulated internal state in memory.

When the user sends a follow-up, the graph resumes from exactly where it paused. Critically, `SupervisorState` — which holds `supervisor_messages`, `notes`, and everything the supervisor accumulated — is still live. A new supervisor run can be told "here is what we already researched, here is the follow-up question, decide what additional research is needed." It can choose to skip sub-topics it already covered and only research the delta.

**The key difference from Option 2:** In Option 2, the supervisor starts from scratch every turn and re-derives its research plan from the conversation history. In Option 3, the supervisor can explicitly see its own prior internal state and make smarter decisions about what NOT to research again.

**When to use this:** When your follow-up questions are genuinely incremental and tightly coupled to the previous research — not new standalone questions. For example: first query is "compare transformers and Mamba", follow-up is "go deeper on Mamba's memory efficiency specifically." Here you do not want to re-research transformers at all. Option 3 lets the supervisor recognise that and only dispatch researchers for the new angle. Option 2 would re-run everything.

The tradeoff is complexity: you have to redesign the graph's loop structure and write prompts that teach the supervisor how to reason about what it has already done versus what is new.

### Implementation

Turn the graph into a true conversational loop by routing back after report generation and using `interrupt()` to pause for user input.

### Change 1 — `src/open_deep_research/deep_researcher.py`

Use `interrupt` in `clarify_with_user` and loop the final edge back:

```python
from langgraph.types import interrupt

async def clarify_with_user(state, config):
    ...
    if response.need_clarification:
        user_reply = interrupt(response.question)   # pauses graph, returns control to caller
        return Command(
            goto="write_research_brief",
            update={"messages": [
                AIMessage(content=response.question),
                HumanMessage(content=user_reply)
            ]}
        )
    ...

# Change the final edge from:
deep_researcher_builder.add_edge("final_report_generation", END)
# To:
deep_researcher_builder.add_edge("final_report_generation", "clarify_with_user")
```

### Change 2 — `run_query.py`

Use `astream` with `Command(resume=...)` to handle interrupts:

```python
from langgraph.types import Command

config = {"configurable": {"thread_id": thread_id, "allow_clarification": True}}

# First run
async for chunk in deep_researcher.astream(
    {"messages": [HumanMessage(content=QUERY)]},
    config=config,
    stream_mode="updates",
):
    if "__interrupt__" in chunk:
        question = chunk["__interrupt__"][0].value
        user_reply = input(f"Agent: {question}\nYou: ")
        # Resume with user's answer
        async for chunk in deep_researcher.astream(
            Command(resume=user_reply),
            config=config,
            stream_mode="updates",
        ):
            ...
```

**Trade-off:** True stateful loop — the graph never fully ends, so accumulated research notes and `supervisor_messages` are preserved across turns. Requires a checkpointer (graph must be stateful to support `interrupt`). Also needs prompt changes to distinguish "first report" from "follow-up report" contexts. Significant restructure overall.

---

## Summary

| Option | Files changed | Research re-runs? | State persists? | Complexity |
|---|---|---|---|---|
| 1. Script accumulation | `run_query.py` only | Yes, fully | No | Low |
| 2. Checkpointing | `deep_researcher.py` + `run_query.py` | Yes, from START | Yes, per thread | Low |
| 3. Graph loop + interrupt | 3 files, significant restructure | No (can skip) | Yes, in-memory | High |

**Start with Option 2.** It's two small changes, works with LangGraph's design, and the message history naturally gives the follow-up full context of the prior report. Option 3 is only worth it if you need the agent to remember which sub-topics it already researched and explicitly avoid re-doing them.
