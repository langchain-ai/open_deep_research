from __future__ import annotations

import os
from typing import Any, Callable

def _bool_env(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "on"}

def install_init_chat_model_override() -> None:
    """Patch langchain's init_chat_model to respect LLM_MODEL (OpenRouter) without editing app code.

    Behavior:
    - If LLM_MODE=api and LLM_MODEL is set:
      - any model string like 'openai:XYZ' becomes 'openai:{LLM_MODEL}'
    - Does not log or print secrets.
    """
    mode = (os.getenv("LLM_MODE") or "").strip().lower()
    llm_model = (os.getenv("LLM_MODEL") or "").strip()
    if mode != "api" or not llm_model:
        return

    try:
        # Most common location
        from langchain.chat_models.base import init_chat_model as _orig  # type: ignore
        import langchain.chat_models.base as _base  # type: ignore
    except Exception:
        try:
            from langchain.chat_models import init_chat_model as _orig  # type: ignore
            import langchain.chat_models as _base  # type: ignore
        except Exception:
            return

    if getattr(_base, "__odr_patched__", False):
        return

    def _wrap(model: Any, *args: Any, **kwargs: Any):
        try:
            if isinstance(model, str) and model.startswith("openai:"):
                model = f"openai:{llm_model}"
        except Exception:
            pass
        # Temperature default: only if not provided
        if "temperature" not in kwargs:
            try:
                kwargs["temperature"] = float(os.getenv("LLM_TEMPERATURE") or "0")
            except Exception:
                pass
        return _orig(model, *args, **kwargs)

    setattr(_base, "init_chat_model", _wrap)
    setattr(_base, "__odr_patched__", True)
