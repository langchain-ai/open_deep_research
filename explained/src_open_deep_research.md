# src/open_deep_research/ — Core Agent Implementation

This is the only folder that matters for running the agent. All five files are active; none are optional.

---

## File Map

```
src/open_deep_research/
├── deep_researcher.py   ← entry point, all graph logic
├── state.py             ← data structures passed between nodes
├── configuration.py     ← all configurable knobs
├── prompts.py           ← all LLM prompt strings
└── utils.py             ← tools, search, token handling, helpers
```

---

## `deep_researcher.py` — The Graph

**What it does:** Defines and compiles all three LangGraph subgraphs. Contains every node function. This is the file you read when you want to understand what the agent actually does step by step.

**Key pieces:**

| Symbol | Type | Role |
|---|---|---|
| `clarify_with_user` | node | Asks a clarifying question if query is ambiguous; routes to `write_research_brief` or `END` |
| `write_research_brief` | node | Transforms the raw user messages into a structured research brief |
| `supervisor` | node | LLM that plans research strategy using `ConductResearch`, `ResearchComplete`, `think_tool` |
| `supervisor_tools` | node | Executes supervisor tool calls; spawns researcher subgraphs in parallel via `asyncio.gather()` |
| `researcher` | node | Individual researcher LLM that calls search tools in a loop |
| `researcher_tools` | node | Executes search tool calls; decides whether to loop or move to compression |
| `compress_research` | node | Distills a researcher's full message history into a clean, cited summary |
| `final_report_generation` | node | Synthesizes all compressed summaries into the final markdown report |
| `supervisor_subgraph` | compiled graph | `supervisor → supervisor_tools` loop |
| `researcher_subgraph` | compiled graph | `researcher → researcher_tools → compress_research` |
| `deep_researcher` | compiled graph | Main graph: `clarify → brief → supervisor_subgraph → final_report` |

**How nodes are wired:**

```
START
  └─► clarify_with_user ──► write_research_brief ──► research_supervisor ──► final_report_generation ──► END
                        └──► END (if clarification needed)

research_supervisor (supervisor subgraph):
  START ──► supervisor ──► supervisor_tools ──► supervisor (loop)
                                           └──► END (ResearchComplete / iteration limit)

supervisor_tools spawns N researcher subgraphs in parallel:
  START ──► researcher ──► researcher_tools ──► researcher (loop)
                                           └──► compress_research ──► END
```

**Where to look when debugging:**
- Agent loops too many times → check `max_researcher_iterations` and `max_react_tool_calls` guard conditions in `supervisor_tools` and `researcher_tools`
- Report is empty → check `get_notes_from_tool_calls()` in `supervisor_tools` and the `notes` field passed to `final_report_generation`
- Token limit errors → handled in `compress_research` (trims messages) and `final_report_generation` (truncates findings)

**Imports from:** `state.py` (all state types), `configuration.py` (`Configuration`), `prompts.py` (all prompt strings), `utils.py` (`get_all_tools`, `think_tool`, token helpers)

---

## `state.py` — Data Structures

**What it does:** Defines every TypedDict and Pydantic model used as graph state. No logic here — purely data shape definitions.

**State types:**

| Class | Used by | Key fields |
|---|---|---|
| `AgentInputState` | Main graph input | `messages` only |
| `AgentState` | Main graph full state | `messages`, `supervisor_messages`, `research_brief`, `notes`, `raw_notes`, `final_report` |
| `SupervisorState` | Supervisor subgraph | `supervisor_messages`, `research_brief`, `notes`, `raw_notes`, `research_iterations` |
| `ResearcherState` | Researcher subgraph | `researcher_messages`, `tool_call_iterations`, `research_topic`, `compressed_research`, `raw_notes` |
| `ResearcherOutputState` | Researcher subgraph output | `compressed_research`, `raw_notes` |

**Structured outputs (used as tool schemas):**

| Class | Used as |
|---|---|
| `ConductResearch` | Tool the supervisor calls to spawn a researcher |
| `ResearchComplete` | Tool called to signal research is done |
| `ClarifyWithUser` | Structured output of `clarify_with_user` node |
| `ResearchQuestion` | Structured output of `write_research_brief` node |
| `Summary` | Structured output of the Tavily webpage summarizer |

**The `override_reducer`:** Lists in LangGraph state normally append. `override_reducer` lets a node replace the list entirely by passing `{"type": "override", "value": [...]}`. Used for `supervisor_messages`, `notes`, and `raw_notes` so nodes can reset these fields rather than accumulate them.

**Imports from:** Nothing in this repo — only standard library and LangGraph/Pydantic.

---

## `configuration.py` — Settings

**What it does:** Single `Configuration` Pydantic model that holds every tunable parameter. Loaded at the start of each node via `Configuration.from_runnable_config(config)`.

**Key fields:**

| Field | Default | Effect |
|---|---|---|
| `allow_clarification` | `True` | If `False`, skips `clarify_with_user` entirely |
| `search_api` | `"tavily"` | Which search backend researchers use (`tavily`, `openai`, `anthropic`, `none`) |
| `max_concurrent_research_units` | `5` | Max parallel researcher subgraphs per supervisor iteration |
| `max_researcher_iterations` | `6` | Max supervisor loop iterations before forced exit |
| `max_react_tool_calls` | `10` | Max search calls per individual researcher before forced compression |
| `research_model` | `"openai:gpt-4.1"` | Model for supervisor + researcher nodes |
| `summarization_model` | `"openai:gpt-4.1-mini"` | Model for summarizing raw Tavily webpage content |
| `compression_model` | `"openai:gpt-4.1"` | Model for `compress_research` node |
| `final_report_model` | `"openai:gpt-4.1"` | Model for `final_report_generation` node |
| `mcp_config` | `None` | Optional MCP server connection for custom tools |

**How configuration flows in:** LangGraph passes a `RunnableConfig` dict to every node. `Configuration.from_runnable_config()` reads from both `config["configurable"]` and environment variables, with configurable values taking precedence.

**Imports from:** Nothing in this repo.

---

## `prompts.py` — All LLM Prompts

**What it does:** Stores every prompt string used by the agent as module-level constants. No logic. Changing behaviour without touching graph logic means changing these strings.

**Prompt map:**

| Constant | Used in node | Purpose |
|---|---|---|
| `clarify_with_user_instructions` | `clarify_with_user` | Asks model whether the query needs clarification |
| `transform_messages_into_research_topic_prompt` | `write_research_brief` | Converts conversation history into a focused research brief |
| `lead_researcher_prompt` | `supervisor` | System prompt for the supervisor; instructs it how to delegate and when to stop |
| `research_system_prompt` | `researcher` | System prompt for each researcher; instructs it how to search and when to stop |
| `compress_research_system_prompt` | `compress_research` | Instructs compression model to preserve all info while cleaning up format |
| `compress_research_simple_human_message` | `compress_research` | The human turn appended to trigger compression |
| `final_report_generation_prompt` | `final_report_generation` | Instructs writer model to produce a structured markdown report from findings |
| `summarize_webpage_prompt` | `utils.summarize_webpage` | Used by Tavily tool to summarize raw webpage content before returning to researcher |

**Imports from:** Nothing in this repo.

---

## `utils.py` — Tools and Helpers

**What it does:** Everything that is not graph logic or prompts. Defines the Tavily search tool, the `think_tool`, MCP loading, and a collection of helpers for token limits, API keys, and message manipulation.

**Key functions grouped by role:**

### Search

| Function | What it does |
|---|---|
| `tavily_search` | LangChain `@tool` — runs multiple queries in parallel, deduplicates by URL, summarizes each page, returns formatted string |
| `tavily_search_async` | Raw async Tavily API calls; called by `tavily_search` |
| `summarize_webpage` | Calls `summarization_model` on raw page content; falls back to original content on timeout |
| `get_search_tool` | Returns the right search tool list based on `SearchAPI` enum value |
| `get_all_tools` | Assembles the full tool list: `[ResearchComplete, think_tool, <search tool>, <mcp tools>]` |

### Reflection

| Function | What it does |
|---|---|
| `think_tool` | LangChain `@tool` — a no-op that records a reflection string; used by both supervisor and researcher for structured thinking |

### MCP

| Function | What it does |
|---|---|
| `load_mcp_tools` | Connects to configured MCP server, filters to specified tools, wraps with auth error handling |
| `wrap_mcp_authenticate_tool` | Wraps an MCP tool to convert `-32003` auth errors into readable `ToolException` messages |
| `fetch_tokens` / `get_tokens` / `set_tokens` | OAuth token exchange and storage for authenticated MCP servers |

### Token limits

| Function | What it does |
|---|---|
| `is_token_limit_exceeded` | Inspects an exception to determine if it's a context-length error (handles OpenAI, Anthropic, Gemini) |
| `get_model_token_limit` | Looks up max context length from `MODEL_TOKEN_LIMITS` dict |
| `remove_up_to_last_ai_message` | Trims message list by removing up to the last AI message; used as retry strategy in `compress_research` |

### Config / API keys

| Function | What it does |
|---|---|
| `get_api_key_for_model` | Returns the right env var key based on model prefix (`openai:`, `anthropic:`, etc.) |
| `get_tavily_api_key` | Returns Tavily key from env or config |
| `get_config_value` | Unwraps enum values and handles None safely |
| `get_today_str` | Returns formatted date string injected into all prompts |

**Imports from:** `configuration.py` (`Configuration`, `SearchAPI`), `prompts.py` (`summarize_webpage_prompt`), `state.py` (`ResearchComplete`, `Summary`)

---

## How the Files Interlink

```
deep_researcher.py
    ├── imports state.py        (AgentState, SupervisorState, ResearcherState, ConductResearch, ...)
    ├── imports configuration.py (Configuration)
    ├── imports prompts.py      (all prompt strings)
    └── imports utils.py        (get_all_tools, think_tool, token helpers, API key helpers)

utils.py
    ├── imports configuration.py (Configuration, SearchAPI)
    ├── imports prompts.py       (summarize_webpage_prompt)
    └── imports state.py         (ResearchComplete, Summary)

prompts.py     → no internal imports
state.py       → no internal imports
configuration.py → no internal imports
```

The dependency graph is a clean DAG: `deep_researcher.py` sits at the top and imports everything; `prompts.py`, `state.py`, and `configuration.py` are leaves with no internal imports; `utils.py` sits in the middle importing config, prompts, and state.
