import json
import os
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional


SESSIONS_DIR = Path.cwd() / ".sessions"


def _ensure_dir() -> None:
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)


def _session_path(session_id: str) -> Path:
    return SESSIONS_DIR / f"{session_id}.json"


def list_sessions() -> List[Dict[str, Any]]:
    _ensure_dir()
    sessions: List[Dict[str, Any]] = []
    for p in sorted(SESSIONS_DIR.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
        try:
            with p.open("r", encoding="utf-8") as f:
                data = json.load(f)
            sessions.append({
                "id": data.get("id", p.stem),
                "title": data.get("title") or "Untitled",
                "updated_at": data.get("updated_at"),
            })
        except Exception:
            continue
    return sessions


def new_session(title: Optional[str] = None) -> Dict[str, Any]:
    _ensure_dir()
    session_id = str(uuid.uuid4())
    now = int(time.time())
    data: Dict[str, Any] = {
        "id": session_id,
        "title": (title or "New Chat").strip()[:80],
        "created_at": now,
        "updated_at": now,
        "messages": [],
        "research_brief": None,
        "notes": [],
        "raw_notes": [],
        "final_report": None,
        "config": {},
    }
    save_session(session_id, data)
    return data


def load_session(session_id: str) -> Optional[Dict[str, Any]]:
    path = _session_path(session_id)
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_session(session_id: str, data: Dict[str, Any]) -> None:
    _ensure_dir()
    data["updated_at"] = int(time.time())
    with _session_path(session_id).open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def rename_session(session_id: str, title: str) -> None:
    data = load_session(session_id)
    if not data:
        return
    data["title"] = title.strip()[:80]
    save_session(session_id, data)


def delete_session(session_id: str) -> None:
    path = _session_path(session_id)
    if path.exists():
        path.unlink()

