import re
from urllib.parse import urlparse
from typing import cast
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from open_deep_research.utils import get_today_str
from tests.prompts import RELEVANCE_PROMPT, STRUCTURE_PROMPT, GROUNDEDNESS_PROMPT, OVERALL_QUALITY_PROMPT, CORRECTNESS_PROMPT, COMPLETENESS_PROMPT

eval_model: ChatOpenAI | ChatAnthropic | None = None


def _get_eval_model() -> ChatOpenAI | ChatAnthropic:
    global eval_model
    if eval_model is None:
        eval_model = ChatOpenAI(model="gpt-4.1")
    return eval_model

def _format_input_query(inputs: dict) -> str:
    messages = inputs["messages"]
    if len(messages) == 1:
        return messages[0]["content"]

    role_to_string_format_map = {
        "user": "<user_input>\n{content}\n</user_input>",
        "assistant": "<assistant_follow_up>\n{content}\n</assistant_follow_up>",
    }

    return "\n\n".join([role_to_string_format_map[message["role"]].format(content=message["content"]) for message in messages])


class OverallQualityScore(BaseModel):
    """Score the overall quality of the report against specific criteria."""
    research_depth: int = Field(description="Integer score 1-5 showing whether the report meets the provided criteria (1 = doesn't meet at all, 5 = meets all criteria).")
    source_quality: int = Field(description="Integer score 1-5 showing whether the report meets the provided criteria (1 = doesn't meet at all, 5 = meets all criteria).")
    analytical_rigor: int = Field(description="Integer score 1-5 showing whether the report meets the provided criteria (1 = doesn't meet at all, 5 = meets all criteria).")
    practical_value: int = Field(description="Integer score 1-5 showing whether the report meets the provided criteria (1 = doesn't meet at all, 5 = meets all criteria).")
    balance_and_objectivity: int = Field(description="Integer score 1-5 showing whether the report meets the provided criteria (1 = doesn't meet at all, 5 = meets all criteria).")
    writing_quality: int = Field(description="Integer score 1-5 showing whether the report meets the provided criteria (1 = doesn't meet at all, 5 = meets all criteria).")

def eval_overall_quality(inputs: dict, outputs: dict):
    model = _get_eval_model()
    query = _format_input_query(inputs)
    final_report = outputs["final_report"]
    user_input_content = f"""User input: {query}\n\nReport: \n\n{final_report}\n\nEvaluate whether the report meets the criteria and provide detailed justification for your evaluation."""
    if isinstance(model, ChatAnthropic):
        user_input_content = [{
            "type": "text",
            "text": user_input_content,
            "cache_control": {"type": "ephemeral", "ttl": "1h"}
        }]
    eval_result = cast(OverallQualityScore, model.with_structured_output(OverallQualityScore).invoke([
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


class RelevanceScore(BaseModel):
    """Score the report relevance against specific criteria."""
    reasoning: str = Field(description="The reason for the score, including specific examples from the report.")
    score: int = Field(description="Integer score 1-5 showing whether the report meets the provided criteria for relevance (1 = doesn't meet at all, 5 = meets all criteria).")

def eval_relevance(inputs: dict, outputs: dict):
    model = _get_eval_model()
    query = _format_input_query(inputs)
    final_report = outputs["final_report"]
    user_input_content = f"""User input: {query}\n\nReport: \n\n{final_report}\n\nEvaluate whether the report meets the criteria and provide detailed justification for your evaluation."""
    if isinstance(model, ChatAnthropic):
        user_input_content = [{
            "type": "text",
            "text": user_input_content,
            "cache_control": {"type": "ephemeral", "ttl": "1h"}
        }]

    eval_result = cast(RelevanceScore, model.with_structured_output(RelevanceScore).invoke([
        {"role": "system", "content": RELEVANCE_PROMPT.format(today=get_today_str())},
        {"role": "user", "content": user_input_content}
    ]))
    return {"key": "relevance_score", "score": eval_result.score / 5, "comment": eval_result.reasoning}


class StructureScore(BaseModel):
    """Score the report structure against specific criteria."""
    reasoning: str = Field(description="The reason for the score, including specific examples from the report.")
    score: int = Field(description="Integer score 1-5 showing whether the report meets the provided criteria for structure and flow (1 = doesn't meet at all, 5 = meets all criteria).")

def eval_structure(inputs: dict, outputs: dict):
    model = _get_eval_model()
    query = _format_input_query(inputs)
    final_report = outputs["final_report"]
    user_input_content = STRUCTURE_PROMPT.format(user_question=query, report=final_report, today=get_today_str())
    if isinstance(model, ChatAnthropic):
        user_input_content = [{
            "type": "text",
            "text": user_input_content,
            "cache_control": {"type": "ephemeral", "ttl": "1h"}
        }]

    eval_result = cast(StructureScore, model.with_structured_output(StructureScore).invoke([
        {"role": "user", "content": user_input_content}
    ]))
    return {"key": "structure_and_cohesiveness_score", "score": eval_result.score / 5, "comment": eval_result.reasoning}


class CorrectnessScore(BaseModel):
    """Score the report correctness against specific criteria."""
    reasoning: str = Field(description="The reason for the score, including specific examples from the report.")
    score: int = Field(description="Integer score 1-5 showing whether the report meets the provided criteria for correctness (1 = doesn't meet at all, 5 = meets all criteria).")

def eval_correctness(inputs: dict, outputs: dict, reference_outputs: dict):
    model = _get_eval_model()
    query = _format_input_query(inputs)
    final_report = outputs["final_report"]
    answer = reference_outputs["answer"]
    user_input_content = CORRECTNESS_PROMPT.format(user_question=query, report=final_report, answer=answer, today=get_today_str())
    if isinstance(model, ChatAnthropic):
        user_input_content = [{
            "type": "text",
            "text": user_input_content,
            "cache_control": {"type": "ephemeral", "ttl": "1h"}
        }]

    eval_result = cast(CorrectnessScore, model.with_structured_output(CorrectnessScore).invoke([
        {"role": "user", "content": user_input_content}
    ]))
    return {"key": "correctness_score", "score": eval_result.score / 5, "comment": eval_result.reasoning}

class GroundednessClaim(BaseModel):
    """A claim from the report, and whether or not it is grounded in the context"""
    claim: str = Field(description="The claim extracted from the report.")
    grounded: bool = Field(description="Whether the claim is grounded in the context.")

class GroundednessScore(BaseModel):
    """Extract the claims and whether they are grounded in the context"""
    claims: list[GroundednessClaim] = Field(description="All claims extracted from the report, and whether or not they are grounded in the context.")

def eval_groundedness(inputs: dict, outputs: dict):
    model = _get_eval_model()
    final_report = outputs["final_report"]
    context = str(outputs["raw_notes"])

    user_input_content = GROUNDEDNESS_PROMPT.format(context=context, report=final_report, today=get_today_str())
    if isinstance(model, ChatAnthropic):
        user_input_content = [{
            "type": "text",
            "text": user_input_content,
            "cache_control": {"type": "ephemeral", "ttl": "1h"}
        }]

    eval_result = cast(GroundednessScore, model.with_structured_output(GroundednessScore).with_retry(stop_after_attempt=3).invoke([
        {"role": "user", "content": user_input_content},
    ]))
    # normalize to 0-1
    grounded_claims = [claim for claim in eval_result.claims if claim.grounded]
    return {"key": "groundedness_score", "score": len(grounded_claims) / len(eval_result.claims), "comment": str(eval_result.claims)}


class CompletenessScore(BaseModel):
    """Score the report completeness against specific criteria."""
    reasoning: str = Field(description="The reason for the score, including specific examples from the report.")
    score: int = Field(description="Integer score 1-5 showing whether the report meets the provided criteria for completeness (1 = doesn't meet at all, 5 = meets all criteria).")

def eval_completeness(inputs: dict, outputs: dict):
    model = _get_eval_model()
    query = _format_input_query(inputs)
    final_report = outputs["final_report"]
    research_brief = outputs["research_brief"]
    user_input_content = COMPLETENESS_PROMPT.format(user_question=query, research_brief=research_brief, report=final_report, today=get_today_str())
    if isinstance(model, ChatAnthropic):
        user_input_content = [{
            "type": "text",
            "text": user_input_content,
            "cache_control": {"type": "ephemeral", "ttl": "1h"}
        }]

    eval_result = cast(CompletenessScore, model.with_structured_output(CompletenessScore).invoke([
        {"role": "user", "content": user_input_content}
    ]))
    return {"key": "completeness_score", "score": eval_result.score / 5, "comment": eval_result.reasoning}


def _extract_user_text(inputs: dict) -> str:
    messages = inputs.get("messages", [])
    if not messages:
        return ""
    return "\n".join(str(message.get("content", "")) for message in messages if isinstance(message, dict)).strip()


def _extract_report(outputs: dict) -> str:
    return str(outputs.get("final_report", "") or "")


def _extract_sources(outputs: dict) -> list[dict]:
    api_response = outputs.get("api_response", {})
    if not isinstance(api_response, dict):
        return []
    sources = api_response.get("sources", [])
    if not isinstance(sources, list):
        return []
    return [source for source in sources if isinstance(source, dict)]


def _contains_any(text: str, patterns: list[str]) -> bool:
    lowered = text.lower()
    return any(pattern in lowered for pattern in patterns)


def eval_personalization(inputs: dict, outputs: dict) -> dict:
    """Heuristic personalization metric from user intent -> report alignment (0-1)."""
    user_text = _extract_user_text(inputs)
    report_text = _extract_report(outputs)

    checks = [
        (_contains_any(user_text, ["中文", "chinese"]), _contains_any(report_text, ["的", "是", "在"])),
        (_contains_any(user_text, ["english", "英文"]), _contains_any(report_text, [" the ", " and ", " is "])),
        (_contains_any(user_text, ["简洁", "concise", "简短"]), len(report_text) <= 3500),
        (_contains_any(user_text, ["详细", "detailed", "深入"]), len(report_text) >= 1500),
        (_contains_any(user_text, ["不要表格", "no table"]), "|" not in report_text),
    ]

    active_checks = [check for check in checks if check[0]]
    if not active_checks:
        return {"key": "personalization_score", "score": 0.5, "comment": "No explicit personalization constraints found."}

    satisfied = sum(1 for _, hit in active_checks if hit)
    score = satisfied / len(active_checks)
    return {
        "key": "personalization_score",
        "score": score,
        "comment": f"Matched {satisfied}/{len(active_checks)} explicit personalization constraints.",
    }


def eval_memory_usefulness(inputs: dict, outputs: dict) -> dict:
    """Estimate whether memory channel materially contributes to output (0-1)."""
    del inputs
    confirmed_preferences = outputs.get("confirmed_long_term_preferences", [])
    if not isinstance(confirmed_preferences, list):
        confirmed_preferences = []

    report_text = _extract_report(outputs).lower()
    sources = _extract_sources(outputs)
    has_memory_source = any(str(source.get("channel", "")).lower() == "memory" for source in sources)

    preference_hits = 0
    for pref in confirmed_preferences:
        value = str(pref).split(":", 1)[-1].strip().lower()
        if value and value in report_text:
            preference_hits += 1

    signals = [
        1.0 if has_memory_source else 0.0,
        min(preference_hits / max(len(confirmed_preferences), 1), 1.0) if confirmed_preferences else 0.0,
    ]
    score = sum(signals) / len(signals)
    return {
        "key": "memory_usefulness_score",
        "score": score,
        "comment": f"memory_sources={has_memory_source}, preference_hits={preference_hits}/{len(confirmed_preferences)}",
    }


def eval_source_diversity(inputs: dict, outputs: dict) -> dict:
    """Measure source channel/domain diversity from structured API sources (0-1)."""
    del inputs
    sources = _extract_sources(outputs)
    if not sources:
        return {"key": "source_diversity_score", "score": 0.0, "comment": "No structured sources available."}

    channels: set[str] = set()
    domains: set[str] = set()
    for source in sources:
        channel = str(source.get("channel", "")).strip().lower()
        if channel:
            channels.add(channel)

        source_value = str(source.get("source", "")).strip()
        if source_value.startswith("http://") or source_value.startswith("https://"):
            parsed = urlparse(source_value)
            if parsed.netloc:
                domains.add(parsed.netloc.lower())
        else:
            match = re.match(r"^[A-Za-z]:[/\\]", source_value)
            if match:
                domains.add("local_fs")

    channel_component = min(len(channels) / 3.0, 1.0)
    domain_component = min(len(domains) / 5.0, 1.0)
    score = (channel_component + domain_component) / 2.0
    return {
        "key": "source_diversity_score",
        "score": score,
        "comment": f"channels={sorted(channels)}, domains={sorted(domains)}",
    }