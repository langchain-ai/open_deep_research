"""LLM factory for extensions - reuses main project's LLM client configuration.

Ensures the same model selected by the user (e.g., gemini-2.5-flash) is used
for both the core research pipeline and all extension tools/agents.
"""
import os
import logging

logger = logging.getLogger(__name__)


def get_extensions_llm(provider: str = None, model: str = None, temperature: float = 0.0):
    """Get an LLM client using the project's configured provider and model.

    Accepts an explicit provider/model (passed from the user's frontend selection).
    Falls back to LLM_PROVIDER / LLM_MODEL env vars when not provided.

    Args:
        provider: LLM provider ('azure' or 'gemini'). Falls back to LLM_PROVIDER env.
        model:    Model name. Falls back to LLM_MODEL env or provider default.
        temperature: LLM temperature (default 0.0 for deterministic output)

    Returns:
        A LangChain chat model instance
    """
    from llm_clients import get_llm_client, MODEL_CONFIGS

    if not provider:
        provider = os.getenv("LLM_PROVIDER", "azure")
    if not model:
        model = os.getenv("LLM_MODEL") or MODEL_CONFIGS.get(provider, {}).get("default_model")

    logger.info(f"[EXTENSIONS] Creating LLM client: provider={provider}, model={model}")
    return get_llm_client(provider, model)
