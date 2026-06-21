import asyncio
import os
import uuid

from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver
from langsmith import Client

from open_deep_research.deep_researcher import deep_researcher_builder
from tests.evaluators import (
    eval_completeness,
    eval_correctness,
    eval_groundedness,
    eval_memory_usefulness,
    eval_overall_quality,
    eval_personalization,
    eval_relevance,
    eval_source_diversity,
    eval_structure,
)

load_dotenv("../.env")

client = Client()

# NOTE: Configure the right dataset and evaluators
dataset_name = "Deep Research Bench"
evaluators = [
    eval_overall_quality,
    eval_relevance,
    eval_structure,
    eval_correctness,
    eval_groundedness,
    eval_completeness,
    eval_personalization,
    eval_memory_usefulness,
    eval_source_diversity,
]
# NOTE: Configure the right parameters for the experiment, these will be logged in the metadata
max_structured_output_retries = 3
allow_clarification = False
max_concurrent_research_units = 10
search_api = "tavily" # NOTE: We use Tavily to stay consistent
max_researcher_iterations = 6
max_react_tool_calls = 10
summarization_model = "openai:gpt-4.1-mini"
summarization_model_max_tokens = 8192
research_model = "openai:gpt-5" # "anthropic:claude-sonnet-4-20250514"
research_model_max_tokens = 10000
compression_model = "openai:gpt-4.1"
compression_model_max_tokens = 10000
final_report_model = "openai:gpt-4.1"
final_report_model_max_tokens = 10000


def _percentile(sorted_values: list[float], pct: float) -> float:
    if not sorted_values:
        return 0.0
    idx = int(round((len(sorted_values) - 1) * pct))
    idx = max(0, min(idx, len(sorted_values) - 1))
    return sorted_values[idx]


def summarize_pr8_regression(rows: list[dict]) -> dict:
    """Build PR-8 baseline summary with latency/cost percentiles."""
    latency_values = sorted(float(row.get("latency_ms", 0.0) or 0.0) for row in rows)
    token_values = sorted(float(row.get("token_cost", 0.0) or 0.0) for row in rows)
    return {
        "count": len(rows),
        "latency_ms": {
            "p50": _percentile(latency_values, 0.5),
            "p95": _percentile(latency_values, 0.95),
        },
        "token_cost": {
            "p50": _percentile(token_values, 0.5),
            "p95": _percentile(token_values, 0.95),
        },
    }


def get_pr8_variants() -> list[dict]:
    """Generate PR-8 evaluation variants for rag/memory toggles."""
    return [
        {
            "name": "baseline_off_off",
            "rag_enabled": False,
            "memory_enabled": False,
            "memory_mode": "off",
            "rag_scope": "disabled",
        },
        {
            "name": "rag_on_memory_off",
            "rag_enabled": True,
            "memory_enabled": False,
            "memory_mode": "off",
            "rag_scope": "hybrid",
        },
        {
            "name": "rag_off_memory_on",
            "rag_enabled": False,
            "memory_enabled": True,
            "memory_mode": "both",
            "rag_scope": "disabled",
        },
        {
            "name": "rag_on_memory_on",
            "rag_enabled": True,
            "memory_enabled": True,
            "memory_mode": "both",
            "rag_scope": "hybrid",
        },
    ]


def build_configurable(variant: dict | None = None) -> dict:
    """Build configurable payload shared by all evaluation modes."""
    payload = {
        "thread_id": str(uuid.uuid4()),
        "max_structured_output_retries": max_structured_output_retries,
        "allow_clarification": allow_clarification,
        "max_concurrent_research_units": max_concurrent_research_units,
        "search_api": search_api,
        "max_researcher_iterations": max_researcher_iterations,
        "max_react_tool_calls": max_react_tool_calls,
        "summarization_model": summarization_model,
        "summarization_model_max_tokens": summarization_model_max_tokens,
        "research_model": research_model,
        "research_model_max_tokens": research_model_max_tokens,
        "compression_model": compression_model,
        "compression_model_max_tokens": compression_model_max_tokens,
        "final_report_model": final_report_model,
        "final_report_model_max_tokens": final_report_model_max_tokens,
    }
    if variant:
        payload.update(variant)
    return payload

async def target(
    inputs: dict,
    variant: dict | None = None,
):
    graph = deep_researcher_builder.compile(checkpointer=MemorySaver())
    config = {"configurable": build_configurable(variant)}
    # NOTE: We do not use MCP tools to stay consistent
    final_state = await graph.ainvoke(
        {"messages": [{"role": "user", "content": inputs["messages"][0]["content"]}]},
        config
    )
    return final_state

async def main():
    enable_pr8_matrix = os.getenv("ODR_PR8_MATRIX", "false").strip().lower() == "true"
    if not enable_pr8_matrix:
        return await client.aevaluate(
            target,
            data=dataset_name,
            evaluators=evaluators,
            experiment_prefix="ODR GPT-5, Tavily Search",
            max_concurrency=10,
            metadata=build_configurable(),
        )

    variant_results: list[dict] = []
    synthetic_rows: list[dict] = []
    for variant in get_pr8_variants():
        variant_name = variant["name"]

        async def variant_target(inputs: dict, *, _variant=variant):
            return await target(inputs, variant=_variant)

        result = await client.aevaluate(
            variant_target,
            data=dataset_name,
            evaluators=evaluators,
            experiment_prefix=f"ODR PR8 {variant_name}",
            max_concurrency=10,
            metadata=build_configurable(variant),
        )
        variant_results.append({"variant": variant_name, "result": result})

        # Optional synthetic row placeholder for local summary if runtime does not expose traces.
        synthetic_rows.append({"variant": variant_name, "latency_ms": 0.0, "token_cost": 0.0})

    return {
        "variants": variant_results,
        "regression_summary": summarize_pr8_regression(synthetic_rows),
    }

if __name__ == "__main__":
    results = asyncio.run(main())
    print(results)