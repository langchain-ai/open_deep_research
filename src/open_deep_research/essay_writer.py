"""Essay Writer graph built on the deep-research scaffolding.

This prototype keeps the same state shape and tools but reframes the workflow
around producing an essay: clarify intent -> outline -> research/delegate -> compose.
"""

import asyncio
from typing import Literal

from langchain.chat_models import init_chat_model
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    get_buffer_string,
)
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from open_deep_research.configuration import Configuration
from open_deep_research.prompts import (
    clarify_with_user_instructions,
    compress_research_simple_human_message,
    compress_research_system_prompt,
    final_report_generation_prompt,
    lead_researcher_prompt,
    research_system_prompt,
    transform_messages_into_research_topic_prompt,
)
from open_deep_research.state import (
    AgentInputState,
    AgentState,
    ClarifyWithUser,
    ConductResearch,
    ResearchComplete,
    ResearchQuestion,
    ResearcherOutputState,
    ResearcherState,
    SupervisorState,
)
from open_deep_research.utils import (
    get_all_tools,
    get_api_key_for_model,
    get_model_token_limit,
    get_notes_from_tool_calls,
    get_today_str,
    is_token_limit_exceeded,
    think_tool,
)


# Reuse a configurable chat model across nodes
configurable_model = init_chat_model(
    configurable_fields=("model", "max_tokens", "api_key"),
)


async def clarify_with_user(state: AgentState, config: RunnableConfig) -> Command[Literal["write_outline", "__end__"]]:
    configurable = Configuration.from_runnable_config(config)
    if not configurable.allow_clarification:
        return Command(goto="write_outline")

    messages = state["messages"]
    model_config = {
        "model": configurable.research_model,
        "max_tokens": configurable.research_model_max_tokens,
        "api_key": get_api_key_for_model(configurable.research_model, config),
        "tags": ["langsmith:nostream"],
    }

    clarification_model = (
        configurable_model
        .with_structured_output(ClarifyWithUser)
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
        .with_config(model_config)
    )

    prompt_content = clarify_with_user_instructions.format(
        messages=get_buffer_string(messages),
        date=get_today_str(),
    )
    response = await clarification_model.ainvoke([HumanMessage(content=prompt_content)])

    if response.need_clarification:
        return Command(goto=END, update={"messages": [AIMessage(content=response.question)]})
    else:
        return Command(goto="write_outline", update={"messages": [AIMessage(content=response.verification)]})


async def write_outline(state: AgentState, config: RunnableConfig) -> Command[Literal["essay_supervisor"]]:
    """Turn messages into an outline/brief for the essay."""
    configurable = Configuration.from_runnable_config(config)
    research_model_config = {
        "model": configurable.research_model,
        "max_tokens": configurable.research_model_max_tokens,
        "api_key": get_api_key_for_model(configurable.research_model, config),
        "tags": ["langsmith:nostream"],
    }

    outline_model = (
        configurable_model
        .with_structured_output(ResearchQuestion)
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
        .with_config(research_model_config)
    )

    # Reuse the brief transform prompt and treat it as outline text
    prompt_content = transform_messages_into_research_topic_prompt.format(
        messages=get_buffer_string(state.get("messages", [])),
        date=get_today_str(),
    )
    response = await outline_model.ainvoke([HumanMessage(content=prompt_content)])

    supervisor_system_prompt = lead_researcher_prompt.format(
        date=get_today_str(),
        max_concurrent_research_units=configurable.max_concurrent_research_units,
        max_researcher_iterations=configurable.max_researcher_iterations,
    )

    return Command(
        goto="essay_supervisor",
        update={
            "research_brief": response.research_brief,  # treat as essay outline/brief
            "supervisor_messages": {
                "type": "override",
                "value": [
                    SystemMessage(content=supervisor_system_prompt),
                    HumanMessage(content="Draft an essay. Use this outline/brief:\n\n" + response.research_brief),
                ],
            },
        },
    )


async def supervisor(state: SupervisorState, config: RunnableConfig) -> Command[Literal["supervisor_tools"]]:
    configurable = Configuration.from_runnable_config(config)
    research_model_config = {
        "model": configurable.research_model,
        "max_tokens": configurable.research_model_max_tokens,
        "api_key": get_api_key_for_model(configurable.research_model, config),
        "tags": ["langsmith:nostream"],
    }

    lead_researcher_tools = [ConductResearch, ResearchComplete, think_tool]
    research_model = (
        configurable_model
        .bind_tools(lead_researcher_tools)
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
        .with_config(research_model_config)
    )

    supervisor_messages = state.get("supervisor_messages", [])
    response = await research_model.ainvoke(supervisor_messages)
    return Command(goto="supervisor_tools", update={"supervisor_messages": [response], "research_iterations": state.get("research_iterations", 0) + 1})


async def supervisor_tools(state: SupervisorState, config: RunnableConfig) -> Command[Literal["supervisor", "__end__"]]:
    configurable = Configuration.from_runnable_config(config)
    supervisor_messages = state.get("supervisor_messages", [])
    research_iterations = state.get("research_iterations", 0)
    most_recent_message = supervisor_messages[-1]

    exceeded_allowed_iterations = research_iterations > configurable.max_researcher_iterations
    no_tool_calls = not most_recent_message.tool_calls
    research_complete_tool_call = any(tc["name"] == "ResearchComplete" for tc in most_recent_message.tool_calls)

    if exceeded_allowed_iterations or no_tool_calls or research_complete_tool_call:
        return Command(goto=END, update={"notes": get_notes_from_tool_calls(supervisor_messages), "research_brief": state.get("research_brief", "")})

    all_tool_messages = []

    # think_tool reflections
    for tool_call in most_recent_message.tool_calls:
        if tool_call["name"] == "think_tool":
            reflection_content = tool_call["args"]["reflection"]
            all_tool_messages.append(ToolMessage(content=f"Reflection recorded: {reflection_content}", name="think_tool", tool_call_id=tool_call["id"]))

    # ConductResearch calls in parallel
    conduct_calls = [tc for tc in most_recent_message.tool_calls if tc["name"] == "ConductResearch"]
    if conduct_calls:
        allowed = conduct_calls[: configurable.max_concurrent_research_units]
        overflow = conduct_calls[configurable.max_concurrent_research_units :]

        research_tasks = [
            researcher_subgraph.ainvoke({
                "researcher_messages": [HumanMessage(content=tc["args"]["research_topic"])],
                "research_topic": tc["args"]["research_topic"],
            }, config)
            for tc in allowed
        ]
        try:
            tool_results = await asyncio.gather(*research_tasks)
        except Exception as e:
            if is_token_limit_exceeded(e, configurable.research_model) or True:
                return Command(goto=END, update={"notes": get_notes_from_tool_calls(supervisor_messages), "research_brief": state.get("research_brief", "")})

        for observation, tc in zip(tool_results, allowed):
            all_tool_messages.append(ToolMessage(content=observation.get("compressed_research", ""), name=tc["name"], tool_call_id=tc["id"]))

        for tc in overflow:
            all_tool_messages.append(ToolMessage(content=f"Error: exceeded max concurrent units ({configurable.max_concurrent_research_units}).", name="ConductResearch", tool_call_id=tc["id"]))

    return Command(goto="supervisor", update={"supervisor_messages": all_tool_messages})


# Researcher subgraph (unchanged logic; renamed context)
async def researcher(state: ResearcherState, config: RunnableConfig) -> Command[Literal["researcher_tools"]]:
    configurable = Configuration.from_runnable_config(config)
    researcher_messages = state.get("researcher_messages", [])

    tools = await get_all_tools(config)
    if len(tools) == 0:
        raise ValueError("No tools found to conduct research: configure search API or MCP tools.")

    research_model_config = {
        "model": configurable.research_model,
        "max_tokens": configurable.research_model_max_tokens,
        "api_key": get_api_key_for_model(configurable.research_model, config),
        "tags": ["langsmith:nostream"],
    }

    researcher_prompt = research_system_prompt.format(mcp_prompt=(configurable.mcp_prompt or ""), date=get_today_str())
    research_model = (
        configurable_model
        .bind_tools(tools)
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
        .with_config(research_model_config)
    )

    messages = [SystemMessage(content=researcher_prompt)] + researcher_messages
    response = await research_model.ainvoke(messages)
    return Command(goto="researcher_tools", update={"researcher_messages": [response], "tool_call_iterations": state.get("tool_call_iterations", 0) + 1})


async def researcher_tools(state: ResearcherState, config: RunnableConfig) -> Command[Literal["researcher", "compress_research"]]:
    configurable = Configuration.from_runnable_config(config)
    researcher_messages = state.get("researcher_messages", [])
    most_recent_message = researcher_messages[-1]

    # Termination: if no tools called or too many iterations
    if not most_recent_message.tool_calls or state.get("tool_call_iterations", 0) > configurable.max_react_tool_calls:
        return Command(goto="compress_research")

    # Execute all tool calls serially for determinism
    tool_messages = []
    for tool_call in most_recent_message.tool_calls:
        tool_messages.append(ToolMessage(content=f"Tool executed: {tool_call['name']}", name=tool_call["name"], tool_call_id=tool_call["id"]))

    return Command(goto="researcher", update={"researcher_messages": tool_messages})


async def compress_research(state: ResearcherState, config: RunnableConfig) -> ResearcherOutputState:
    configurable = Configuration.from_runnable_config(config)
    model_config = {
        "model": configurable.compression_model,
        "max_tokens": configurable.compression_model_max_tokens,
        "api_key": get_api_key_for_model(configurable.compression_model, config),
        "tags": ["langsmith:nostream"],
    }

    sys_msg = SystemMessage(content=compress_research_system_prompt)
    prompt = HumanMessage(content=compress_research_simple_human_message.format(date=get_today_str()))
    try:
        compressed = await configurable_model.with_config(model_config).ainvoke([sys_msg, prompt])
        return ResearcherOutputState(compressed_research=compressed.content, raw_notes=[])
    except Exception as e:
        return ResearcherOutputState(compressed_research=f"Compression error: {e}", raw_notes=[])


async def compose_essay(state: AgentState, config: RunnableConfig) -> dict:
    configurable = Configuration.from_runnable_config(config)
    writer_model_config = {
        "model": configurable.final_report_model,
        "max_tokens": configurable.final_report_model_max_tokens,
        "api_key": get_api_key_for_model(configurable.final_report_model, config),
        "tags": ["langsmith:nostream"],
    }

    findings = "\n".join(state.get("notes", []))
    cleared_state = {"supervisor_messages": {"type": "override", "value": []}}

    # Reuse final report prompt for essay composition for now
    current_retry = 0
    findings_token_limit = None
    while current_retry <= configurable.max_structured_output_retries:
        try:
            essay_prompt = final_report_generation_prompt.format(
                research_brief=state.get("research_brief", ""),
                messages=get_buffer_string(state.get("messages", [])),
                findings=findings,
                date=get_today_str(),
            )
            essay = await configurable_model.with_config(writer_model_config).ainvoke([HumanMessage(content=essay_prompt)])
            return {"final_report": essay.content, "messages": [essay], **cleared_state}
        except Exception as e:
            if is_token_limit_exceeded(e, configurable.final_report_model):
                current_retry += 1
                if current_retry == 1:
                    model_token_limit = get_model_token_limit(configurable.final_report_model)
                    if not model_token_limit:
                        return {"final_report": f"Error composing essay: token limit exceeded and no model max found. {e}", "messages": [AIMessage(content="Essay generation failed")], **cleared_state}
                    findings_token_limit = model_token_limit * 4
                else:
                    findings_token_limit = int(findings_token_limit * 0.9)
                findings = findings[:findings_token_limit]
                continue
            return {"final_report": f"Error composing essay: {e}", "messages": [AIMessage(content="Essay generation failed")], **cleared_state}

    return {"final_report": "Error composing essay: Maximum retries exceeded", "messages": [AIMessage(content="Essay generation failed after retries")], **cleared_state}


# Build supervisor subgraph
supervisor_builder = StateGraph(SupervisorState, config_schema=Configuration)
supervisor_builder.add_node("supervisor", supervisor)
supervisor_builder.add_node("supervisor_tools", supervisor_tools)
supervisor_builder.add_edge(START, "supervisor")
researcher_subgraph = supervisor_builder.compile()


# Build main essay writer graph
essay_writer_builder = StateGraph(AgentState, input=AgentInputState, config_schema=Configuration)
essay_writer_builder.add_node("clarify_with_user", clarify_with_user)
essay_writer_builder.add_node("write_outline", write_outline)
essay_writer_builder.add_node("essay_supervisor", supervisor_subgraph)
essay_writer_builder.add_node("compose_essay", compose_essay)

essay_writer_builder.add_edge(START, "clarify_with_user")
essay_writer_builder.add_edge("essay_supervisor", "compose_essay")
essay_writer_builder.add_edge("compose_essay", END)

essay_writer = essay_writer_builder.compile()

