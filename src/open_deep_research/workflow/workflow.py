from typing import Literal
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, get_buffer_string
from langchain_core.runnables import RunnableConfig
from langgraph.constants import Send
from langgraph.graph import START, END, StateGraph
from langgraph.types import interrupt, Command

from open_deep_research.workflow.configuration import WorkflowConfiguration
from open_deep_research.workflow.state import (
    ReportStateInput,
    ReportStateOutput,
    ReportState,
    SectionState,
    SectionOutputState,
    ClarifyWithUser,
    SectionOutput
)
from open_deep_research.state import (
    Sections,
    Queries,
    Feedback,
)
from open_deep_research.workflow.prompts import (
    clarify_with_user_instructions,
    report_planner_query_writer_instructions,
    report_planner_instructions,
    query_writer_instructions, 
    section_writer_instructions,
    final_section_writer_instructions,
    section_grader_instructions,
    section_writer_inputs
)
from open_deep_research.utils import (
    format_sections, 
    get_config_value, 
    get_search_params, 
    select_and_execute_search,
    get_today_str
)

## Nodes
def initial_router(state: ReportState, config: RunnableConfig):
    configurable = WorkflowConfiguration.from_runnable_config(config)
    if configurable.clarify_with_user and not state.get("already_clarified_topic", False):
        return "clarify_with_user"
    else:
        return "generate_report_plan"


async def clarify_with_user(state: ReportState, config: RunnableConfig):
    messages = state["messages"]
    configurable = WorkflowConfiguration.from_runnable_config(config)
    writer_provider = get_config_value(configurable.writer_provider)
    writer_model_name = get_config_value(configurable.writer_model)
    writer_model_kwargs = get_config_value(configurable.writer_model_kwargs or {})
    writer_model = init_chat_model(model=writer_model_name, model_provider=writer_provider, **writer_model_kwargs) 
    structured_llm = writer_model.with_structured_output(ClarifyWithUser)
    system_instructions = clarify_with_user_instructions.format(messages=get_buffer_string(messages))
    results = await structured_llm.ainvoke([SystemMessage(content=system_instructions),
                                     HumanMessage(content="Generate search queries that will help with planning the sections of the report.")])
    return {"messages": [AIMessage(content=results.question)], "already_clarified_topic": True}


async def generate_report_plan(state: ReportState, config: RunnableConfig) -> Command[Literal["human_feedback","build_section_with_web_research"]]:
    messages = state["messages"]
    feedback_list = state.get("feedback_on_report_plan", [])
    feedback = " /// ".join(feedback_list) if feedback_list else ""

    configurable = WorkflowConfiguration.from_runnable_config(config)
    report_structure = configurable.report_structure
    number_of_queries = configurable.number_of_queries
    search_api = get_config_value(configurable.search_api)
    search_api_config = configurable.search_api_config or {}  # Get the config dict, default to empty
    params_to_pass = get_search_params(search_api, search_api_config)  # Filter parameters
    sections_user_approval = configurable.sections_user_approval

    if isinstance(report_structure, dict):
        report_structure = str(report_structure)

    writer_provider = get_config_value(configurable.writer_provider)
    writer_model_name = get_config_value(configurable.writer_model)
    writer_model_kwargs = get_config_value(configurable.writer_model_kwargs or {})
    writer_model = init_chat_model(model=writer_model_name, model_provider=writer_provider, **writer_model_kwargs) 
    structured_llm = writer_model.with_structured_output(Queries)

    system_instructions_query = report_planner_query_writer_instructions.format(
        messages=get_buffer_string(messages),
        report_organization=report_structure,
        number_of_queries=number_of_queries,
        today=get_today_str()
    )
    results = await structured_llm.ainvoke([SystemMessage(content=system_instructions_query),
                                     HumanMessage(content="Generate search queries that will help with planning the sections of the report.")])
    
    query_list = [query.search_query for query in results.queries]
    source_str = await select_and_execute_search(search_api, query_list, params_to_pass)
    system_instructions_sections = report_planner_instructions.format(messages=get_buffer_string(messages), report_organization=report_structure, context=source_str, feedback=feedback)

    planner_provider = get_config_value(configurable.planner_provider)
    planner_model = get_config_value(configurable.planner_model)
    planner_model_kwargs = get_config_value(configurable.planner_model_kwargs or {})
    planner_message = """Generate the sections of the report. Your response must include a 'sections' field containing a list of sections. 
                        Each section must have: name, description, research, and content fields."""
    
    if planner_model == "claude-3-7-sonnet-latest":
        # Allocate a thinking budget for claude-3-7-sonnet-latest as the planner model
        planner_llm = init_chat_model(model=planner_model, 
                                      model_provider=planner_provider, 
                                      max_tokens=20_000, 
                                      thinking={"type": "enabled", "budget_tokens": 16_000})
    else:
        # With other models, thinking tokens are not specifically allocated
        planner_llm = init_chat_model(model=planner_model, 
                                      model_provider=planner_provider,
                                      **planner_model_kwargs)
    
    structured_llm = planner_llm.with_structured_output(Sections)
    report_sections = await structured_llm.ainvoke([SystemMessage(content=system_instructions_sections),
                                             HumanMessage(content=planner_message)])
    sections = report_sections.sections

    if sections_user_approval:
        return Command(goto="human_feedback", update={"sections": sections})
    else:
        return Command(goto=[
            Send("build_section_with_web_research", {"messages": messages, "section": s, "search_iterations": 0}) 
            for s in sections 
            if s.research
        ], update={"sections": sections})


async def human_feedback(state: ReportState, config: RunnableConfig) -> Command[Literal["generate_report_plan","build_section_with_web_research"]]:
    messages = state["messages"]
    sections = state['sections']
    sections_str = "\n\n".join(
        f"Section: {section.name}\n"
        f"Description: {section.description}\n"
        f"Research needed: {'Yes' if section.research else 'No'}\n"
        for section in sections
    )
    interrupt_message = f"""Please provide feedback on the following report plan. 
                        \n\n{sections_str}\n
                        \nDoes the report plan meet your needs?\nPass 'true' to approve the report plan.\nOr, provide feedback to regenerate the report plan:"""
    feedback = interrupt(interrupt_message)
    if (isinstance(feedback, bool) and feedback is True) or (isinstance(feedback, str) and feedback.lower() == "true"):
        return Command(goto=[
            Send("build_section_with_web_research", {"messages": messages, "section": s, "search_iterations": 0}) 
            for s in sections 
            if s.research
        ])
    elif isinstance(feedback, str):
        return Command(goto="generate_report_plan", 
                       update={"feedback_on_report_plan": [feedback]})
    else:
        raise TypeError(f"Interrupt value of type {type(feedback)} is not supported.")


async def generate_queries(state: SectionState, config: RunnableConfig):
    messages = state["messages"]
    section = state["section"]
    configurable = WorkflowConfiguration.from_runnable_config(config)
    number_of_queries = configurable.number_of_queries
    writer_provider = get_config_value(configurable.writer_provider)
    writer_model_name = get_config_value(configurable.writer_model)
    writer_model_kwargs = get_config_value(configurable.writer_model_kwargs or {})
    writer_model = init_chat_model(model=writer_model_name, model_provider=writer_provider, **writer_model_kwargs) 
    structured_llm = writer_model.with_structured_output(Queries)
    system_instructions = query_writer_instructions.format(messages=get_buffer_string(messages), 
                                                           section_topic=section.description, 
                                                           number_of_queries=number_of_queries,
                                                           today=get_today_str())

    queries = await structured_llm.ainvoke([SystemMessage(content=system_instructions),
                                     HumanMessage(content="Generate search queries on the provided topic.")])
    return {"search_queries": queries.queries}


async def search_web(state: SectionState, config: RunnableConfig):
    search_queries = state["search_queries"]
    configurable = WorkflowConfiguration.from_runnable_config(config)
    search_api = get_config_value(configurable.search_api)
    search_api_config = configurable.search_api_config or {}
    params_to_pass = get_search_params(search_api, search_api_config)

    query_list = [query.search_query for query in search_queries]
    source_str = await select_and_execute_search(search_api, query_list, params_to_pass)

    return {"source_str": source_str, "search_iterations": state["search_iterations"] + 1}


async def write_section(state: SectionState, config: RunnableConfig):
    messages = state["messages"]
    section = state["section"]
    source_str = state["source_str"]
    configurable = WorkflowConfiguration.from_runnable_config(config)
    section_writer_inputs_formatted = section_writer_inputs.format(messages=get_buffer_string(messages), 
                                                             section_name=section.name, 
                                                             section_topic=section.description, 
                                                             context=source_str, 
                                                             section_content=section.content)
    writer_provider = get_config_value(configurable.writer_provider)
    writer_model_name = get_config_value(configurable.writer_model)
    writer_model_kwargs = get_config_value(configurable.writer_model_kwargs or {})
    writer_model = init_chat_model(
        model=writer_model_name,
        model_provider=writer_provider,
        **writer_model_kwargs,
        max_retries=configurable.max_structured_output_retries
    ).with_structured_output(SectionOutput)

    section_content = await writer_model.ainvoke([SystemMessage(content=section_writer_instructions),
                                           HumanMessage(content=section_writer_inputs_formatted)])
    section.content = section_content.section_content

    section_grader_message = ("Grade the report and consider follow-up questions for missing information. "
                              "If the grade is 'pass', return empty strings for all follow-up queries. "
                              "If the grade is 'fail', provide specific search queries to gather missing information.")
    
    section_grader_instructions_formatted = section_grader_instructions.format(messages=get_buffer_string(messages), 
                                                                               section_topic=section.description,
                                                                               section=section.content, 
                                                                               number_of_follow_up_queries=configurable.number_of_queries)

    planner_provider = get_config_value(configurable.planner_provider)
    planner_model = get_config_value(configurable.planner_model)
    planner_model_kwargs = get_config_value(configurable.planner_model_kwargs or {})
    if planner_model == "claude-3-7-sonnet-latest":
        # Allocate a thinking budget for claude-3-7-sonnet-latest as the planner model
        reflection_model = init_chat_model(model=planner_model, 
                                           model_provider=planner_provider, 
                                           max_tokens=20_000, 
                                           thinking={"type": "enabled", "budget_tokens": 16_000}).with_structured_output(Feedback)
    else:
        reflection_model = init_chat_model(model=planner_model, 
                                           model_provider=planner_provider,
                                           max_retries=configurable.max_structured_output_retries,
                                           **planner_model_kwargs).with_structured_output(Feedback)

    feedback = await reflection_model.ainvoke([SystemMessage(content=section_grader_instructions_formatted),
                                        HumanMessage(content=section_grader_message)])

    if feedback.grade == "pass" or state["search_iterations"] >= configurable.max_search_depth:
        update = {"completed_sections": [section]}
        if configurable.include_source_str:
            update["source_str"] = source_str
        return Command(update=update, goto=END)
    else:
        return Command(
            update={"search_queries": feedback.follow_up_queries, "section": section},
            goto="search_web"
        )


async def write_final_sections(state: SectionState, config: RunnableConfig):
    configurable = WorkflowConfiguration.from_runnable_config(config)
    writer_provider = get_config_value(configurable.writer_provider)
    writer_model_name = get_config_value(configurable.writer_model)
    writer_model_kwargs = get_config_value(configurable.writer_model_kwargs or {})
    writer_model = init_chat_model(model=writer_model_name, model_provider=writer_provider, **writer_model_kwargs) 
    messages = state["messages"]
    section = state["section"]
    completed_report_sections = state["report_sections_from_research"]
    system_instructions = final_section_writer_instructions.format(messages=get_buffer_string(messages), 
                                                                   section_name=section.name, 
                                                                   section_topic=section.description, 
                                                                   context=completed_report_sections)
    section_content = await writer_model.ainvoke([SystemMessage(content=system_instructions),
                                           HumanMessage(content="Generate a report section based on the provided sources.")])   
    section.content = section_content.content
    return {"completed_sections": [section]}


async def gather_completed_sections(state: ReportState):
    completed_sections = state["completed_sections"]
    completed_report_sections = format_sections(completed_sections)

    return {"report_sections_from_research": completed_report_sections}


async def compile_final_report(state: ReportState, config: RunnableConfig):
    configurable = WorkflowConfiguration.from_runnable_config(config)
    sections = state["sections"]
    completed_sections = {s.name: s.content for s in state["completed_sections"]}
    for section in sections:
        section.content = completed_sections[section.name]
    all_sections = "\n\n".join([s.content for s in sections])

    if configurable.include_source_str:
        return {"final_report": all_sections, "source_str": state["source_str"], "messages": [AIMessage(content=all_sections)]}
    else:
        return {"final_report": all_sections, "messages": [AIMessage(content=all_sections)]}


async def initiate_final_section_writing(state: ReportState):
    return [
        Send("write_final_sections", {"messages": state["messages"], "section": s, "report_sections_from_research": state["report_sections_from_research"]}) 
        for s in state["sections"] 
        if not s.research
    ]


## Graph
section_builder = StateGraph(SectionState, output=SectionOutputState)
section_builder.add_node("generate_queries", generate_queries)
section_builder.add_node("search_web", search_web)
section_builder.add_node("write_section", write_section)
section_builder.add_edge(START, "generate_queries")
section_builder.add_edge("generate_queries", "search_web")
section_builder.add_edge("search_web", "write_section")

builder = StateGraph(ReportState, input=ReportStateInput, output=ReportStateOutput, config_schema=WorkflowConfiguration)
builder.add_node("clarify_with_user", clarify_with_user)
builder.add_node("generate_report_plan", generate_report_plan)
builder.add_node("human_feedback", human_feedback)
builder.add_node("build_section_with_web_research", section_builder.compile())
builder.add_node("gather_completed_sections", gather_completed_sections)
builder.add_node("write_final_sections", write_final_sections)
builder.add_node("compile_final_report", compile_final_report)
builder.add_conditional_edges(START, initial_router, ["clarify_with_user", "generate_report_plan"])
builder.add_edge("clarify_with_user", END)
builder.add_edge("build_section_with_web_research", "gather_completed_sections")
builder.add_conditional_edges("gather_completed_sections", initiate_final_section_writing, ["write_final_sections"])
builder.add_edge("write_final_sections", "compile_final_report")
builder.add_edge("compile_final_report", END)
workflow = builder.compile()