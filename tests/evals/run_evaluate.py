from langsmith import Client
from tests.evals.evaluators import eval_overall_quality, eval_relevance, eval_structure
from tests.evals.target import generate_report_multi_agent
from dotenv import load_dotenv
import os
import asyncio
from typing import Literal
from langchain_core.messages import MessageLikeRepresentation
from open_deep_research.multi_agent import supervisor_builder

load_dotenv("../.env")

client = Client()

dataset_name = "ODR: Multi Agent Examples"
evaluators = [eval_overall_quality, eval_relevance, eval_structure]
# TODO: Configure these variables
process_search_results = "summarize"
include_source = False
summarization_model = "gpt-4.1"
summarization_model_provider = "openai"
supervisor_model = "openai:gpt-4.1"
researcher_model = "openai:gpt-4.1"
final_report_model = "openai:gpt-4.1"


async def generate_report_multi_agent(
    messages: list[MessageLikeRepresentation],
    process_search_results: Literal["summarize", "split_and_rerank"] | None = None,
    include_source: bool = True,
    summarization_model: str = summarization_model,
    summarization_model_provider: str = summarization_model_provider,
    supervisor_model: str = supervisor_model,
    researcher_model: str = researcher_model,
    final_report_model: str = final_report_model,
):
    """Generate a report using the open deep research multi-agent architecture"""
    graph = supervisor_builder.compile()
    config = {"configurable": {}}
    if include_source:
        config["configurable"]["include_source_str"] = True
    if process_search_results:
        config["configurable"]["process_search_results"] = process_search_results
    config["configurable"]["summarization_model"] = summarization_model
    config["configurable"]["summarization_model_provider"] = summarization_model_provider
    config["configurable"]["supervisor_model"] = supervisor_model
    config["configurable"]["researcher_model"] = researcher_model
    config["configurable"]["final_report_model"] = final_report_model

    final_state = await graph.ainvoke(
        # this is a hack
        # TODO: Find workaround at some point
        {"messages": messages + [{"role": "user", "content": "Generate the report now and don't ask any more follow-up questions"}]},
        config
    )
    return {
        "messages": [
            {"role": "assistant", "content": final_state["final_report"]}
        ]
    }

async def target(inputs: dict):
    return await generate_report_multi_agent(
        inputs["messages"],
        process_search_results,
        include_source, 
        summarization_model,
        summarization_model_provider,
        supervisor_model,
        researcher_model
    )

async def main():
    return await client.aevaluate(
        target,
        data=dataset_name,
        evaluators=evaluators,
        experiment_prefix=f"ODR: Multi Agent - PSR:{process_search_results}, IS:{include_source}",
        max_concurrency=1,
        metadata={"process_search_results": process_search_results, "include_source": include_source},
    )

if __name__ == "__main__":
    results = asyncio.run(main())
    print(results)