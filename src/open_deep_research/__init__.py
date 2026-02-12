
# --- security bootstrap (OpenRouter env + safe logging) ---
try:
    from .security.runtime_env import bootstrap_runtime_env
    bootstrap_runtime_env()
except Exception:
    # Fail-fast is expected if LLM_MODE=api but LLM_API_KEY is empty.
    raise
