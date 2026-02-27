from __future__ import annotations

import logging
import traceback
from typing import Any

from .redaction import redact_text

class RedactionFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        try:
            msg = record.getMessage()
            record.msg = redact_text(msg)
            record.args = ()

            if record.exc_info:
                exc_text = "".join(traceback.format_exception(*record.exc_info))
                record.exc_text = redact_text(exc_text)
        except Exception:
            # never break logging
            pass
        return True

_INSTALLED = False

def install_safe_logging() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    logging.getLogger().addFilter(RedactionFilter())
    _INSTALLED = True

def safe_log_dict(logger: logging.Logger, level: int, msg: str, data: dict[str, Any]) -> None:
    safe: dict[str, Any] = {}
    for k, v in data.items():
        ku = k.upper()
        if ku.endswith("_API_KEY") or ku in {"OPENAI_API_KEY", "AUTHORIZATION"}:
            safe[k] = "[REDACTED]"
        else:
            safe[k] = v if isinstance(v, (str, int, float, bool, type(None))) else f"<{type(v).__name__}>"
    logger.log(level, "%s %s", msg, safe)
