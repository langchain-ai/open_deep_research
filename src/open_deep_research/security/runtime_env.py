from __future__ import annotations

import os

from .logging_setup import install_safe_logging
from .model_override import install_init_chat_model_override

def _bool_env(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "on"}

def bootstrap_runtime_env() -> None:
    """Bootstrap runtime env:
    - Load .env if python-dotenv is available
    - Install safe logging filter (redaction)
    - Bridge LLM_* env -> OpenAI-compatible env (OpenRouter)
    - Patch init_chat_model to use LLM_MODEL automatically
    """
    try:
        from dotenv import load_dotenv  # type: ignore
        load_dotenv(override=False)
    except Exception:
        pass

    install_safe_logging()

    # Tracing off by default
    if not _bool_env("LANGSMITH_TRACING", default=False):
        os.environ.setdefault("LANGSMITH_TRACING", "false")
        os.environ.setdefault("LANGCHAIN_TRACING_V2", "false")

    mode = (os.getenv("LLM_MODE") or "").strip().lower()

    if mode == "api":
        api_key = (os.getenv("LLM_API_KEY") or "").strip()
        if not api_key:
            raise RuntimeError(
                "Fail-fast: LLM_MODE=api but LLM_API_KEY is empty.\n"
                "Create .env from .env.example and set LLM_API_KEY to your OpenRouter key.\n"
                "Do NOT commit .env."
            )

        base_url = (os.getenv("LLM_BASE_URL") or "").strip()
        # Set OpenAI-compatible env vars expected by many clients
        os.environ.setdefault("OPENAI_API_KEY", api_key)
        if base_url:
            os.environ.setdefault("OPENAI_BASE_URL", base_url)
            os.environ.setdefault("OPENAI_API_BASE", base_url)

    # Install model override wrapper (best-effort)
    install_init_chat_model_override()
