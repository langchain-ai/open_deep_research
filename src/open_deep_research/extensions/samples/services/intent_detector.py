"""
Intent Detection Agent — Determines if the user wants data analysis and visualization.

Uses a dedicated LLM call to classify the user's query into:
  - Research only (default)
  - Research + Analysis + Visualization (when user explicitly asks)

Analysis and visualization are always coupled — if one is requested, both run.
"""

import json
import logging
import os

from langchain_core.messages import HumanMessage, SystemMessage

logger = logging.getLogger(__name__)

INTENT_DETECTION_PROMPT = """You are an intent detection agent. Your ONLY job is to determine whether the user wants:

1. **Research only** — The user wants information, a report, a summary, comparison, or overview.
2. **Research + Analysis** — The user wants research AND also wants data analysis, visualization, charts, graphs, or dashboards generated from the research data.

Rules:
- If the user mentions analysis, analyze, visualization, visualize, charts, graphs, plots, dashboards, data analysis, or similar — return analysis_required: true
- If the user is simply asking for research, information, details, comparison, summary, overview — return analysis_required: false
- When in doubt, return false (research only is the safe default)

Examples:
- "Research environment pollution in Delhi" → false
- "Tell me about the solar system" → false
- "Comparative research on EV vs Hydrogen" → false
- "Details on quantum computing advancements" → false
- "Research pollution in Delhi and analyze the data" → true
- "Pollution trends with visualizations" → true
- "AI market analysis with charts" → true
- "Healthcare costs — perform analysis and generate visualizations" → true
- "Compare EV companies and show me graphs" → true
- "Climate change data — analyze it" → true
- "Research renewable energy with data analysis and dashboards" → true
- "What is the current state of AI?" → false
- "Deep dive into SaaS metrics, visualize the trends" → true

Respond with ONLY valid JSON:
{"analysis_required": true}
or
{"analysis_required": false}
"""


async def detect_intent(query: str, provider: str = None, model: str = None) -> dict:
    """
    Detect whether the user's query requires analysis and visualization.

    Args:
        query: The user's raw research query
        provider: LLM provider to use (e.g., "gemini", "openai", "groq")
        model: LLM model name

    Returns:
        {"analysis_required": bool}
    """
    try:
        from llm_clients import get_llm_client

        # Use the user's selected provider/model, fall back to env defaults
        llm_provider = provider or os.environ.get("LLM_PROVIDER", "azure")
        llm_model = model or os.environ.get("LLM_MODEL", None)

        logger.info(
            f"[INTENT DETECTION] Detecting intent for query: {query[:100]}..."
        )
        logger.info(
            f"[INTENT DETECTION] Using provider={llm_provider}, model={llm_model}"
        )

        llm = get_llm_client(llm_provider, llm_model)

        messages = [
            SystemMessage(content=INTENT_DETECTION_PROMPT),
            HumanMessage(content=f"User query: {query}"),
        ]

        response = await llm.ainvoke(messages)
        content = response.content.strip()

        # Parse JSON response
        # Handle cases where LLM wraps JSON in markdown code blocks
        if "```" in content:
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
            content = content.strip()

        result = json.loads(content)
        analysis_required = result.get("analysis_required", False)

        logger.info(
            f"[INTENT DETECTION] Result: analysis_required={analysis_required}"
        )

        return {"analysis_required": analysis_required}

    except Exception as e:
        # Safe fallback — if anything fails, default to research only
        logger.warning(
            f"[INTENT DETECTION] Failed ({type(e).__name__}: {e}). "
            f"Defaulting to research only."
        )
        return {"analysis_required": False}
