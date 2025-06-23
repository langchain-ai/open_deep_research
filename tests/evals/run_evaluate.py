from langsmith import Client
from tests.evals.evaluators import eval_overall_quality, eval_relevance, eval_structure
from dotenv import load_dotenv
import asyncio
from typing import Literal
from langchain_core.messages import MessageLikeRepresentation
from open_deep_research.general_researcher.general_researcher import general_researcher_builder
from open_deep_research.multi_agent import supervisor_builder
from open_deep_research.workflow.workflow import builder
from langgraph.checkpoint.memory import MemorySaver
import uuid

load_dotenv("../.env")

client = Client()

# TODO: Configure these variables
dataset_name = "ODR: Workflow Examples"
evaluators = [eval_overall_quality, eval_relevance, eval_structure]
process_search_results = "summarize"
include_source = False
summarization_model = "claude-3-5-haiku-latest"
summarization_model_provider = "anthropic"
supervisor_model = "claude-3-7-sonnet-latest"
researcher_model = "claude-3-7-sonnet-latest"
one_shot_mode = True
writer_model = "claude-3-7-sonnet-latest"
writer_model_provider = "anthropic"
writer_model_kwargs = {"max_tokens": 20000}
planner_model = "gpt-4.1"
planner_model_provider = "openai"
planner_model_kwargs = None
clarify_with_user = False
sections_user_approval = False
max_search_depth = 3
max_structured_output_retries = 3

async def generate_report_multi_agent(
    messages: list[MessageLikeRepresentation],
):
    """Generate a report using the open deep research multi-agent architecture"""
    graph = supervisor_builder.compile()
    config = {"configurable": {}}
    if include_source:
        config["configurable"]["include_source_str"] = True
    if process_search_results:
        config["configurable"]["process_search_results"] = process_search_results
    config["configurable"]["summarization_model"] = summarization_model
    config["configurable"]["summarization_provider"] = summarization_model_provider
    config["configurable"]["supervisor_model"] = supervisor_model
    config["configurable"]["researcher_model"] = researcher_model

    final_state = await graph.ainvoke(
        # TODO: This is a hack, find workaround at some point
        {"messages": messages + [{"role": "user", "content": "Generate the report now and don't ask any more follow-up questions"}]},
        config
    )
    return {
        "messages": [
            {"role": "assistant", "content": final_state["final_report"]}
        ]
    }

async def generate_report_workflow(
    messages: list[MessageLikeRepresentation],
):
    """Generate a report using the open deep research workflow"""
    graph = builder.compile(checkpointer=MemorySaver())
    config = {
        "configurable": {
            "thread_id": str(uuid.uuid4()),
        }
    }
    config["configurable"]["include_source_str"] = include_source
    config["configurable"]["process_search_results"] = process_search_results
    config["configurable"]["summarization_model"] = summarization_model
    config["configurable"]["summarization_provider"] = summarization_model_provider
    config["configurable"]["writer_model"] = writer_model
    config["configurable"]["writer_provider"] = writer_model_provider
    config["configurable"]["clarify_with_user"] = clarify_with_user
    config["configurable"]["sections_user_approval"] = sections_user_approval

    final_state =await graph.ainvoke(
        {"messages": messages},
        config
    )
    return final_state

async def generate_report_general_researcher(
    messages: list[MessageLikeRepresentation],
):
    """Generate a report using the open deep research general researcher"""
    graph = general_researcher_builder.compile(checkpointer=MemorySaver())
    config = {
        "configurable": {
            "thread_id": str(uuid.uuid4()),
        }
    }
    config["configurable"]["search_api"] = "openai"
    config["configurable"]["process_search_results"] = process_search_results
    config["configurable"]["summarization_model"] = "openai:gpt-4.1-nano"
    config["configurable"]["summarization_model_max_tokens"] = 10000
    config["configurable"]["research_model"] = "openai:gpt-4.1"
    config["configurable"]["research_model_max_tokens"] = 10000
    config["configurable"]["reflection_model"] = "openai:gpt-4.1"
    config["configurable"]["reflection_model_max_tokens"] = 10000
    config["configurable"]["outliner_model"] = "openai:gpt-4.1"
    config["configurable"]["outliner_model_max_tokens"] = 10000
    config["configurable"]["final_report_model"] = "openai:gpt-4.1"
    config["configurable"]["final_report_model_max_tokens"] = 10000
    config["configurable"]["max_search_depth"] = max_search_depth
    config["configurable"]["max_structured_output_retries"] = max_structured_output_retries
    config["configurable"]["outline_user_approval"] = sections_user_approval

    final_state = await graph.ainvoke(
        {"messages": messages},
        config
    )
    return final_state

async def target(inputs: dict):
    # return await generate_report_multi_agent(inputs["messages"])
    # return await generate_report_workflow(inputs["messages"])
    return await generate_report_general_researcher(inputs["messages"])

async def main():
    return await client.aevaluate(
        target,
        data=client.list_examples(dataset_name=dataset_name, splits=["enumeration"]),
        evaluators=evaluators,
        # experiment_prefix=f"ODR: GR - SM:{summarization_model} WM:{writer_model}  #",
        experiment_prefix=f"GR2.0 - OpenAI Search, One Shot Mode  #",
        max_concurrency=1,
    )

if __name__ == "__main__":
    results = asyncio.run(main())
    print(results)