from typing import cast

from pydantic import BaseModel, Field
from langchain_anthropic import ChatAnthropic

from open_deep_research.utils import get_today_str
from tests.evals.prompts import RELEVANCE_PROMPT, STRUCTURE_PROMPT, GROUNDEDNESS_PROMPT, OVERALL_QUALITY_PROMPT

eval_model = ChatAnthropic(
    model="claude-sonnet-4-0",
    # cache the evaluator input prompts on repeated runs
    betas=["extended-cache-ttl-2025-04-11"],
)

class OverallQualityScore(BaseModel):
    """Score the overall quality of the report against specific criteria."""
    research_depth: int = Field(description="Integer score 1-5 showing whether the report meets the provided criteria (1 = doesn't meet at all, 5 = meets all criteria).")
    source_quality: int = Field(description="Integer score 1-5 showing whether the report meets the provided criteria (1 = doesn't meet at all, 5 = meets all criteria).")
    analytical_rigor: int = Field(description="Integer score 1-5 showing whether the report meets the provided criteria (1 = doesn't meet at all, 5 = meets all criteria).")
    practical_value: int = Field(description="Integer score 1-5 showing whether the report meets the provided criteria (1 = doesn't meet at all, 5 = meets all criteria).")
    balance_and_objectivity: int = Field(description="Integer score 1-5 showing whether the report meets the provided criteria (1 = doesn't meet at all, 5 = meets all criteria).")
    writing_quality: int = Field(description="Integer score 1-5 showing whether the report meets the provided criteria (1 = doesn't meet at all, 5 = meets all criteria).")


class RelevanceScore(BaseModel):
    """Score the report relevance against specific criteria."""
    reasoning: str = Field(description="The reason for the score, including specific examples from the report.")
    score: int = Field(description="Integer score 1-5 showing whether the report meets the provided criteria (1 = doesn't meet at all, 5 = meets all criteria).")


class StructureScore(BaseModel):
    """Score the report structure against specific criteria."""
    reasoning: str = Field(description="The reason for the score, including specific examples from the report.")
    score: int = Field(description="Integer score 1-5 showing whether the report meets the provided criteria (1 = doesn't meet at all, 5 = meets all criteria).")


class GroundednessScore(BaseModel):
    """Score the report groundedness against specific criteria."""
    reasoning: str = Field(description="The reason for the score, including specific examples from the report.")
    score: int = Field(description="Integer score 1-5 showing whether the report meets the provided criteria (1 = doesn't meet at all, 5 = meets all criteria).")


def _format_input_query(inputs: dict) -> str:
    messages = inputs["messages"]
    if len(messages) == 1:
        return messages[0]["content"]

    role_to_string_format_map = {
        "user": "<user_input>\n{content}\n</user_input>",
        "assistant": "<assistant_follow_up>\n{content}\n</assistant_follow_up>",
    }

    return "\n\n".join([role_to_string_format_map[message["role"]].format(content=message["content"]) for message in messages])


def eval_overall_quality(inputs: dict, outputs: dict):
    query = _format_input_query(inputs)
    final_report = outputs["final_report"]
    user_input_content = f"""User input: {query}\n\nReport: \n\n{final_report}\n\nEvaluate whether the report meets the criteria and provide detailed justification for your evaluation."""
    if isinstance(eval_model, ChatAnthropic):
        user_input_content = [{
            "type": "text",
            "text": user_input_content,
            "cache_control": {"type": "ephemeral", "ttl": "1h"}
        }]
    eval_result = cast(OverallQualityScore, eval_model.with_structured_output(OverallQualityScore).invoke([
        {"role": "system", "content": OVERALL_QUALITY_PROMPT.format(today=get_today_str())},
        {"role": "user", "content": user_input_content}
    ]))
    return [
        {"key": "research_depth_score", "score": eval_result.research_depth / 5},
        {"key": "source_quality_score", "score": eval_result.source_quality / 5},
        {"key": "analytical_rigor_score", "score": eval_result.analytical_rigor / 5},
        {"key": "practical_value_score", "score": eval_result.practical_value / 5},
        {"key": "balance_and_objectivity_score", "score": eval_result.balance_and_objectivity / 5},
        {"key": "writing_quality_score", "score": eval_result.writing_quality / 5},
    ]


def eval_relevance(inputs: dict, outputs: dict):
    query = _format_input_query(inputs)
    final_report = outputs["final_report"]
    user_input_content = f"""User input: {query}\n\nReport: \n\n{final_report}\n\nEvaluate whether the report meets the criteria and provide detailed justification for your evaluation."""
    if isinstance(eval_model, ChatAnthropic):
        user_input_content = [{
            "type": "text",
            "text": user_input_content,
            "cache_control": {"type": "ephemeral", "ttl": "1h"}
        }]

    eval_result = cast(RelevanceScore, eval_model.with_structured_output(RelevanceScore).invoke([
        {"role": "system", "content": RELEVANCE_PROMPT.format(today=get_today_str())},
        {"role": "user", "content": user_input_content}
    ]))
    return {"key": "relevance_score", "score": eval_result.score / 5, "comment": eval_result.reasoning}


def eval_structure(inputs: dict, outputs: dict):
    query = _format_input_query(inputs)
    final_report = outputs["final_report"]
    user_input_content = f"""User input: {query}\n\nReport: \n\n{final_report}\n\nEvaluate whether the report meets the criteria and provide detailed justification for your evaluation."""
    if isinstance(eval_model, ChatAnthropic):
        user_input_content = [{
            "type": "text",
            "text": user_input_content,
            "cache_control": {"type": "ephemeral", "ttl": "1h"}
        }]

    eval_result = cast(StructureScore, eval_model.with_structured_output(StructureScore).invoke([
        {"role": "system", "content": STRUCTURE_PROMPT.format(today=get_today_str())},
        {"role": "user", "content": user_input_content}
    ]))
    return {"key": "structure_and_cohesiveness_score", "score": eval_result.score / 5, "comment": eval_result.reasoning}


def eval_groundedness(inputs: dict, outputs: dict):
    report = outputs["messages"][0]["content"]
    context = outputs["context"]

    user_input_content = GROUNDEDNESS_PROMPT.format(context=context, report=report, today=get_today_str())
    if isinstance(eval_model, ChatAnthropic):
        user_input_content = [{
            "type": "text",
            "text": user_input_content,
            "cache_control": {"type": "ephemeral", "ttl": "1h"}
        }]

    eval_result = cast(GroundednessScore, eval_model.with_structured_output(GroundednessScore).invoke([
        {"role": "user", "content": user_input_content},
    ]))
    # normalize to 0-1
    return {"key": "groundedness_score", "score": eval_result.score / 5, "comment": eval_result.reasoning}