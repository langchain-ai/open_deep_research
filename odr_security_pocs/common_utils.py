#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple


@dataclass(frozen=True)
class ParsedOutput:
    raw_text: str
    json_obj: Optional[Dict[str, Any]]
    json_error: Optional[str]


def try_parse_json(text: str) -> ParsedOutput:
    """Try to parse JSON from a model response.

    Supports:
    - pure JSON response
    - JSON embedded in text (extract first {...} block)
    """
    t = text.strip()
    if not t:
        return ParsedOutput(raw_text=text, json_obj=None, json_error="empty")

    # direct parse
    try:
        obj = json.loads(t)
        if isinstance(obj, dict):
            return ParsedOutput(raw_text=text, json_obj=obj, json_error=None)
        return ParsedOutput(raw_text=text, json_obj=None, json_error="json is not an object")
    except Exception as e:
        direct_err = str(e)

    # extract first JSON object-ish block
    m = re.search(r"\{[\s\S]*\}\s*$", t)
    if not m:
        # try any {...} region
        m = re.search(r"\{[\s\S]*\}", t)
    if m:
        candidate = m.group(0)
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return ParsedOutput(raw_text=text, json_obj=obj, json_error=None)
            return ParsedOutput(raw_text=text, json_obj=None, json_error="json is not an object")
        except Exception as e:
            return ParsedOutput(raw_text=text, json_obj=None, json_error=f"embedded json parse error: {e}")
    return ParsedOutput(raw_text=text, json_obj=None, json_error=f"direct json parse error: {direct_err}")
