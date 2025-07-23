from langsmith import Client
from tests.evaluators import eval_overall_quality, eval_relevance, eval_structure, eval_correctness, eval_groundedness, eval_completeness
from dotenv import load_dotenv
import asyncio
from apex_deep_research.deep_researcher import deep_researcher_builder
from langgraph.checkpoint.memory import MemorySaver
import uuid

load_dotenv("../.env")

client = Client()

# NOTE: Configure the right dataset and evaluators
dataset_name = "ODR: Comprehensive Test"
evaluators = [eval_groundedness, eval_completeness, eval_structure]

async def target(
    inputs: dict,
):
    """Generate a report using the open deep research general researcher"""
    graph = deep_researcher_builder.compile(checkpointer=MemorySaver())
    config = {
        "configurable": {
            "thread_id": str(uuid.uuid4()),
        }
    }
    # NOTE: Configure the right dataset and evaluators
    config["configurable"]["max_structured_output_retries"] = 3
    config["configurable"]["allow_clarification"] = False
    config["configurable"]["max_concurrent_research_units"] = 10
    config["configurable"]["search_api"] = "tavily"     # NOTE: We use Tavily to stay consistent
    config["configurable"]["max_researcher_iterations"] = 3
    config["configurable"]["max_react_tool_calls"] = 10
    config["configurable"]["summarization_model"] = "openai:gpt-4.1-nano"
    config["configurable"]["summarization_model_max_tokens"] = 8192
    config["configurable"]["research_model"] = "openai:gpt-4.1"
    config["configurable"]["research_model_max_tokens"] = 10000
    config["configurable"]["compression_model"] = "openai:gpt-4.1-mini"
    config["configurable"]["compression_model_max_tokens"] = 10000
    config["configurable"]["final_report_model"] = "openai:gpt-4.1"
    config["configurable"]["final_report_model_max_tokens"] = 10000
    # NOTE: We do not use MCP tools to stay consistent
    final_state = await graph.ainvoke(
        {"messages": [{"role": "user", "content": inputs["messages"][0]["content"]}]},
        config
    )
    return final_state

async def main():
    return await client.aevaluate(
        target,
        data=client.list_examples(dataset_name=dataset_name, splits=["test2"]),
        evaluators=evaluators,
        experiment_prefix=f"DR Supervisor: Multi Agent (v3) - Tavily #",
        max_concurrency=1,
    )

if __name__ == "__main__":
    results = asyncio.run(main())
    print(results)