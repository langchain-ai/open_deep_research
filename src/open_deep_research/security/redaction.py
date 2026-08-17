from __future__ import annotations

import os
import re
from typing import Iterable

# Conservative token-like patterns (prefer over-redaction to avoid leaks).
_PATTERNS = [
    re.compile(r"Bearer\s+[A-Za-z0-9._\-]{10,}", re.IGNORECASE),
    re.compile(r"\bsk-[A-Za-z0-9]{10,}\b"),
    re.compile(r"\bSYNTH[_-]TOKEN[_-]?[A-Za-z0-9._\-]{6,}\b", re.IGNORECASE),
]

def _secret_values_from_env() -> list[str]:
    vals: list[str] = []
    for k, v in os.environ.items():
        if not v:
            continue
        ku = k.upper()
        if ku.endswith("_API_KEY") or ku in {"OPENAI_API_KEY", "LANGSMITH_API_KEY"}:
            vals.append(v)
    return sorted(set(vals), key=len, reverse=True)

def redact_text(text: str | None, extra_secrets: Iterable[str] | None = None) -> str | None:
    if text is None:
        return None

    out = text

    secrets = _secret_values_from_env()
    if extra_secrets:
        secrets.extend([s for s in extra_secrets if s])
    for s in sorted(set(secrets), key=len, reverse=True):
        if s and s in out:
            out = out.replace(s, "[REDACTED]")

    for p in _PATTERNS:
        out = p.sub("[REDACTED]", out)

    return out
