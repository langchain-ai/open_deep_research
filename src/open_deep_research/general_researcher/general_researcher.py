from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage, get_buffer_string
from langchain_core.runnables import RunnableConfig
from langgraph.graph import START, END, StateGraph
from langgraph.types import Command
import asyncio
from typing import Literal
from open_deep_research.general_researcher.configuration import (
    WorkflowConfiguration, 
    SearchAPI, 
)
from open_deep_research.general_researcher.state import (
    ResearchUnitState,
    GeneralResearcherState,
    GeneralResearcherStateInput,
    GeneralResearcherStateOutput,
    Outline,
    ReflectionResult,
    ClarifyWithUser
)
from open_deep_research.utils import (
    get_config_value,
)
from open_deep_research.general_researcher.prompts import (
    response_structure_instructions, 
    initial_upfront_model_provider_web_search_system_prompt,
    follow_up_upfront_model_provider_web_search_system_prompt,
    upfront_model_provider_reflection_system_prompt,
    gap_context_prompt,
    final_report_generation_prompt,
    clarify_with_user_instructions
)
from open_deep_research.general_researcher.utils import (
    get_search_tool,
    load_mcp_tools,
    extract_notes_from_research_messages,
    get_conversation_history,
)

# Initialize a configurable model that we will use throughout the agent
configurable_model = init_chat_model(
    max_tokens=10000,
    configurable_fields=("model", "max_tokens"),
)

def initial_router(state: GeneralResearcherState, config: RunnableConfig):
    configurable = WorkflowConfiguration.from_runnable_config(config)
    if configurable.clarify_with_user and not len(state.get("messages", [])) >= 3:
        return "clarify_with_user"
    else:
        return "research"


async def clarify_with_user(state: GeneralResearcherState, config: RunnableConfig):
    messages = state["messages"]
    configurable = WorkflowConfiguration.from_runnable_config(config)
    model_config = {
        "model": configurable.final_report_model
    }
    model = configurable_model.with_config(model_config).with_structured_output(ClarifyWithUser).with_retry(stop_after_attempt=configurable.max_structured_output_retries)
    response = await model.ainvoke([HumanMessage(content=clarify_with_user_instructions.format(messages=get_buffer_string(messages)))])
    return {
        "messages": [AIMessage(content=response.question)]
    }
    
    
async def research(state: ResearchUnitState, config: RunnableConfig) -> Command[Literal["reflection", "__end__"]]:
    configurable = WorkflowConfiguration.from_runnable_config(config)
    research_model_config = {
        "model": configurable.research_model,
        "max_tokens": configurable.research_model_max_tokens,
        "tags": ["langsmith:nostream"]
    }
    search_api = SearchAPI(get_config_value(configurable.search_api))
    research_messages = state.get("research_messages", [])
    num_initial_messages = len(research_messages)
    if len(research_messages) == 0:
        research_messages = [HumanMessage(content=get_conversation_history(state.get("messages", [])))]
    research_iterations = state.get("research_iterations", 1)
    tools = []
    tools.extend(await get_search_tool(search_api)) # TODO: UNDO
    existing_tool_names = {tool.name if hasattr(tool, "name") else tool.get("name", "web_search") for tool in tools}
    mcp_tools = await load_mcp_tools(config, existing_tool_names)
    tools.extend(mcp_tools)
    tools_by_name = {tool.name if hasattr(tool, "name") else tool.get("name", "web_search"):tool for tool in tools}
    research_model = configurable_model.with_config(research_model_config).bind_tools(tools).with_retry(stop_after_attempt=configurable.max_structured_output_retries)
    tool_calling_iterations = 0
    system_prompt = initial_upfront_model_provider_web_search_system_prompt.format(
        mcp_prompt=configurable.mcp_prompt or ""
    ) if research_iterations == 1 else follow_up_upfront_model_provider_web_search_system_prompt.format(
        mcp_prompt=configurable.mcp_prompt or ""
    )
    # ReAct Tool Calling Loop
    while tool_calling_iterations < 5: # TODO: Replace with configurable
        response = await research_model.ainvoke([SystemMessage(content=system_prompt), *research_messages])
        research_messages.append(response)
        if len(response.tool_calls) == 0:
            # If no MCP or non-native web search tools were called, then this research iteration is complete.
            break
        tool_calls = response.tool_calls
        web_search_called = False
        for tool_call in tool_calls:
            tool = tools_by_name[tool_call["name"]]
            metadata = getattr(tool, "metadata", {}) or {}
            if metadata.get("type") == "search":
                web_search_called = True
            try:
                observation = await tool.ainvoke(tool_call["args"], config)
            except NotImplementedError:
                observation = tool.invoke(tool_call["args"], config)
            research_messages.append(ToolMessage(
                content=observation,
                name=tool_call["name"],
                tool_call_id=tool_call["id"]
            ))
        # Once we perform a web search, we should exit the loop for feedback.
        if web_search_called:
            break
        tool_calling_iterations += 1
    # We need to assemble clean notes from our research_messages
    new_notes, collected_sources = extract_notes_from_research_messages(research_messages[num_initial_messages:], research_iterations, search_api)
    return Command(
        goto="reflection",
        update={
            "notes": [new_notes],
            "collected_sources": collected_sources,
            "research_iterations": research_iterations + 1,
        }
    )


async def reflection(state: ResearchUnitState, config: RunnableConfig) -> Command[Literal["research", "__end__"]]:
    messages = state["messages"]
    notes = state.get("notes", [])
    configurable = WorkflowConfiguration.from_runnable_config(config)
    if state.get("research_iterations", 1) > configurable.max_search_depth:
        return Command(goto=END)
    reflection_model_config = {
        "model": configurable.reflection_model,
        "max_tokens": configurable.reflection_model_max_tokens,
        "tags": ["langsmith:nostream"]
    }
    reflection_model = configurable_model.with_config(reflection_model_config).with_structured_output(ReflectionResult).with_retry(stop_after_attempt=configurable.max_structured_output_retries)
    findings = "\n".join(notes)
    reflection_prompt = upfront_model_provider_reflection_system_prompt.format(
        messages=get_buffer_string(messages),
        findings=findings,
    )
    try:
        response = await reflection_model.ainvoke([HumanMessage(content=reflection_prompt)])
        if response.is_satisfied:
            return Command(goto=END)
        else:
            knowledge_gaps = "\n".join([f"- {gap}" for gap in response.knowledge_gaps])
            focus_areas = "\n".join([f"- {query}" for query in response.suggested_queries])
            conversation_history = get_conversation_history(state.get("messages", []))
            research_messages = [
                AIMessage(content=f"{conversation_history}\n\nCurrent research status:\n\n{findings}"),
                HumanMessage(content=gap_context_prompt.format(
                    knowledge_gaps=knowledge_gaps,
                    focus_areas=focus_areas,
                    reasoning=response.reasoning
                ))
            ]
            return Command(
                goto="research",
                update={
                    "research_messages": research_messages
                }
            )
    except Exception as e:
        print(f"Error in reflection phase: {e}")
        return Command(goto="model_provider_reflection", update={"research_iterations": state.get("research_iterations", 1) + 1})


upfront_researcher_builder = StateGraph(ResearchUnitState)
upfront_researcher_builder.add_node("research", research)
upfront_researcher_builder.add_node("reflection", reflection)
upfront_researcher_builder.add_edge(START, "research")
upfront_researcher = upfront_researcher_builder.compile()


async def generate_outline(state: GeneralResearcherState, config: RunnableConfig):
    messages = state["messages"]
    notes = state.get("notes", [])
    configurable = WorkflowConfiguration.from_runnable_config(config)
    planner_model_config = {
        "model": configurable.outliner_model,
        "max_tokens": configurable.outliner_model_max_tokens,
        "tags": ["langsmith:nostream"]
    }
    planner_model = configurable_model.with_config(planner_model_config).with_structured_output(Outline).with_retry(stop_after_attempt=configurable.max_structured_output_retries)
    try:
        outline_results = await asyncio.wait_for(
            planner_model.ainvoke(
                [HumanMessage(
                    content=response_structure_instructions.format(
                        messages=get_buffer_string(messages),
                        context="\n".join(notes),
                    ))
                ]
            ),
            timeout=45.0
        )
        return {
            "outline": outline_results.outline
        }
    except Exception as e:
        print(f"Error generating outline: {e}")


async def final_report_generation(state: GeneralResearcherState, config: RunnableConfig):
    messages = state["messages"]
    outline = state["outline"]
    notes = state.get("notes", [])
    collected_sources = state.get("collected_sources", [])
    configurable = WorkflowConfiguration.from_runnable_config(config)
    writer_model_config = {
        "model": configurable.final_report_model,
        "max_tokens": configurable.final_report_model_max_tokens,
    }
    final_report_prompt = final_report_generation_prompt.format(
        messages=get_buffer_string(messages),
        findings="\n".join(notes),
        source_list="\n".join([f"{source.title} ({source.url})" for source in collected_sources]),
        outline="\n\n".join([f"## {section.name}\n{section.description}" for section in outline])
    )
    try:
        final_report = await asyncio.wait_for(
            configurable_model.with_config(writer_model_config).ainvoke([HumanMessage(content=final_report_prompt)]), 
            timeout=300.0
        )
        print("Final report successfully generated")
        return {"final_report": final_report, "messages": [final_report]}
    except Exception as e:
        print(f"Error generating final report: {e}")
        return {"final_report": "Error generating final report"}


general_researcher_builder = StateGraph(GeneralResearcherState, input=GeneralResearcherStateInput, output=GeneralResearcherStateOutput, config_schema=WorkflowConfiguration)
general_researcher_builder.add_node("upfront_researcher", upfront_researcher)
general_researcher_builder.add_node("generate_outline", generate_outline)
general_researcher_builder.add_node("final_report_generation", final_report_generation)
general_researcher_builder.add_node("clarify_with_user", clarify_with_user)
general_researcher_builder.add_conditional_edges(START, initial_router, {
    "research": "upfront_researcher",
    "clarify_with_user": "clarify_with_user"
})
general_researcher_builder.add_edge("clarify_with_user", END)
general_researcher_builder.add_edge("upfront_researcher", "generate_outline")
general_researcher_builder.add_edge("generate_outline", "final_report_generation")
general_researcher_builder.add_edge("final_report_generation", END)

general_researcher = general_researcher_builder.compile()