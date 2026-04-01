#!/usr/bin/env python3
from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Tuple


JsonObj = Dict[str, Any]


def request_json(
    url: str,
    payload: Optional[JsonObj] = None,
    method: str = "POST",
    timeout_s: int = 120,
    headers: Optional[Dict[str, str]] = None,
) -> Any:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    hdrs = {"Content-Type": "application/json", "Accept": "application/json"}
    if headers:
        hdrs.update(headers)

    req = urllib.request.Request(url, data=data, method=method, headers=hdrs)
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            body = resp.read().decode("utf-8") or "null"
            return json.loads(body)
    except urllib.error.HTTPError as e:
        body = ""
        try:
            body = e.read().decode("utf-8")
        except Exception:
            pass
        raise RuntimeError(f"HTTP {e.code} for {url}\nResponse body:\n{body}") from None


def search_assistants(base_url: str) -> List[JsonObj]:
    url = f"{base_url.rstrip('/')}/assistants/search"
    res = request_json(url, payload={})
    if isinstance(res, dict) and isinstance(res.get("data"), list):
        return res["data"]
    if isinstance(res, list):
        return res

    url2 = f"{base_url.rstrip('/')}/assistants"
    try:
        res2 = request_json(url2, payload=None, method="GET")
        if isinstance(res2, dict) and isinstance(res2.get("data"), list):
            return res2["data"]
        if isinstance(res2, list):
            return res2
    except Exception:
        pass
    raise RuntimeError(f"Could not list assistants from {url} (and fallback {url2})")


def pick_assistant_id(assistants: List[JsonObj], user_value: str) -> str:
    for a in assistants:
        if a.get("assistant_id") == user_value:
            return user_value

    matches: List[str] = []
    for a in assistants:
        if a.get("graph_id") == user_value or a.get("name") == user_value:
            if a.get("assistant_id"):
                matches.append(a["assistant_id"])

    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise RuntimeError(
            f"Ambiguous assistant selector '{user_value}'. Matched multiple assistants: {matches}. "
            "Pass an explicit assistant_id."
        )
    return user_value


def create_thread(base_url: str) -> str:
    url = f"{base_url.rstrip('/')}/threads"
    res = request_json(url, payload={})
    thread_id = res.get("thread_id") if isinstance(res, dict) else None
    if not thread_id:
        raise RuntimeError(f"Unexpected /threads response (no thread_id): {res}")
    return thread_id


def run_wait(
    base_url: str,
    thread_id: str,
    assistant_id: str,
    user_prompt: str,
) -> JsonObj:
    url = f"{base_url.rstrip('/')}/threads/{thread_id}/runs/wait"
    payload: JsonObj = {
        "assistant_id": assistant_id,
        "input": {"messages": [{"type": "human", "content": user_prompt}]},
    }
    res = request_json(url, payload=payload)
    if not isinstance(res, dict):
        raise RuntimeError(f"Unexpected runs/wait response: {res}")
    return res


def _iter_text_blobs(obj: Any) -> List[str]:
    blobs: List[str] = []
    if obj is None:
        return blobs

    if isinstance(obj, str):
        return [obj]

    if isinstance(obj, dict):
        # message-like
        if isinstance(obj.get("content"), str):
            blobs.append(obj["content"])
        if isinstance(obj.get("text"), str):
            blobs.append(obj["text"])
        # some frameworks store message chunks as {"content":[{"type":"text","text":"..."}]}
        if isinstance(obj.get("content"), list):
            for item in obj["content"]:
                blobs.extend(_iter_text_blobs(item))
        for v in obj.values():
            blobs.extend(_iter_text_blobs(v))
        return blobs

    if isinstance(obj, list):
        for it in obj:
            blobs.extend(_iter_text_blobs(it))
        return blobs

    return blobs


def extract_assistant_text(run_result: JsonObj) -> str:
    """Best-effort extraction of assistant text from a LangGraph server run result."""
    blobs = _iter_text_blobs(run_result)

    # Heuristic: prefer longer blobs and those that look like the final answer
    blobs = [b for b in blobs if isinstance(b, str) and b.strip()]
    if not blobs:
        return ""

    # Deduplicate while preserving order
    seen = set()
    uniq: List[str] = []
    for b in blobs:
        key = b.strip()
        if key in seen:
            continue
        seen.add(key)
        uniq.append(b)

    # Prefer the longest blob
    uniq.sort(key=lambda s: len(s))
    return uniq[-1]
