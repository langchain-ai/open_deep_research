"""LLM factory for extensions - uses langchain's init_chat_model for provider-agnostic setup.

Ensures extensions use the same model configuration pattern as open_deep_research.
Supports any provider via the 'provider:model_name' format (e.g., 'openai:gpt-4.1').
Falls back to environment variables when explicit values are not provided.
"""
import os
import logging

from langchain.chat_models import init_chat_model

logger = logging.getLogger(__name__)

# Default model if nothing is configured
_DEFAULT_MODEL = "openai:gpt-4.1"


def get_extensions_llm(provider: str = None, model: str = None, temperature: float = 0.0):
    """Get an LLM client using langchain's init_chat_model.

    Accepts an explicit provider/model (passed from the user's frontend selection).
    Falls back to LLM_PROVIDER / LLM_MODEL env vars when not provided.

    The model string should be in the format 'provider:model_name' (e.g., 'openai:gpt-4.1').
    If provider is given separately, it will be combined: '{provider}:{model}'.

    Args:
        provider: LLM provider ('openai', 'anthropic', 'google-genai', etc.).
                  Falls back to LLM_PROVIDER env var.
        model:    Model name or 'provider:model' string.
                  Falls back to LLM_MODEL env var or _DEFAULT_MODEL.
        temperature: LLM temperature (default 0.0 for deterministic output).

    Returns:
        A LangChain chat model instance.
    """
    # Resolve provider
    if not provider:
        provider = os.getenv("LLM_PROVIDER", "")

    # Resolve model
    if not model:
        model = os.getenv("LLM_MODEL", "")

    # Build the full model string
    if model and ":" in model:
        # Model already in 'provider:model_name' format
        full_model = model
    elif model and provider:
        # Combine provider + model
        full_model = f"{provider}:{model}"
    elif model:
        # Model name only, assume openai
        full_model = f"openai:{model}"
    elif provider:
        # Provider only -- use a reasonable default
        provider_defaults = {
            "openai": "openai:gpt-4.1",
            "anthropic": "anthropic:claude-sonnet-4-20250514",
            "google-genai": "google-genai:gemini-2.0-flash",
            "azure": "azure:gpt-4.1",
        }
        full_model = provider_defaults.get(provider, _DEFAULT_MODEL)
    else:
        full_model = _DEFAULT_MODEL

    logger.info(f"[EXTENSIONS] Creating LLM client: model={full_model}, temperature={temperature}")

    return init_chat_model(model=full_model, temperature=temperature)


__all__ = ['get_extensions_llm']
