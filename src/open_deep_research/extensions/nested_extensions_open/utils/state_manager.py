"""State manager for persisting agent state for UI access."""
import os
import json
import sqlite3
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List


class StateManager:
    """Manages state persistence for UI access.

    Supports both SQLite (default) and JSON file storage.
    Supports session-based grouping of conversations.
    """

    def __init__(self, storage_type: str = "sqlite"):
        """Initialize state manager.

        Args:
            storage_type: 'sqlite' or 'json'
        """
        self.storage_type = storage_type

        if storage_type == "sqlite":
            self.db_path = "outputs/states/agent_state.db"
            self._init_db()
        elif storage_type == "json":
            self.state_dir = Path("outputs/states/")
            self.state_dir.mkdir(parents=True, exist_ok=True)
        else:
            raise ValueError(f"Unsupported storage_type: {storage_type}")

    def _init_db(self):
        """Initialize SQLite database with session_id support."""
        os.makedirs("outputs/states", exist_ok=True)

        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS agent_states (
                conversation_id TEXT PRIMARY KEY,
                session_id TEXT,
                query TEXT,
                status TEXT,
                state_data TEXT,
                created_at TEXT,
                updated_at TEXT
            )
        """)
        # Add session_id column if table already exists without it
        try:
            conn.execute("ALTER TABLE agent_states ADD COLUMN session_id TEXT")
        except sqlite3.OperationalError:
            pass  # Column already exists
        # Index for fast session lookups
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_session_id
            ON agent_states(session_id)
        """)
        conn.commit()
        conn.close()

    def save_state(self, conversation_id: str, state: Dict[str, Any]):
        """Save state for UI access.

        Args:
            conversation_id: Unique conversation identifier
            state: State dictionary to save (should include session_id if available)
        """
        if self.storage_type == "sqlite":
            self._save_to_db(conversation_id, state)
        elif self.storage_type == "json":
            self._save_to_json(conversation_id, state)

    def _save_to_db(self, conversation_id: str, state: Dict[str, Any]):
        """Save state to SQLite database."""
        conn = sqlite3.connect(self.db_path)
        now = datetime.now(timezone.utc).isoformat()

        # Check if exists
        cursor = conn.execute(
            "SELECT created_at FROM agent_states WHERE conversation_id = ?",
            (conversation_id,)
        )
        row = cursor.fetchone()
        created_at = row[0] if row else now

        # Insert or update
        conn.execute("""
            INSERT OR REPLACE INTO agent_states
            (conversation_id, session_id, query, status, state_data, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            conversation_id,
            state.get("session_id", ""),
            state.get("query", ""),
            state.get("status", "pending"),
            json.dumps(state),
            created_at,
            now
        ))
        conn.commit()
        conn.close()

    def _save_to_json(self, conversation_id: str, state: Dict[str, Any]):
        """Save state to JSON file.

        Filename format: {session_id}_{conversation_id}.json
        Falls back to {conversation_id}.json if no session_id.
        """
        session_id = state.get("session_id", "")
        if session_id:
            file_path = self.state_dir / f"{session_id}_{conversation_id}.json"
        else:
            file_path = self.state_dir / f"{conversation_id}.json"
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(state, f, indent=2)

    def get_state(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve state for UI.

        Args:
            conversation_id: Unique conversation identifier

        Returns:
            State dictionary or None if not found
        """
        if self.storage_type == "sqlite":
            return self._get_from_db(conversation_id)
        elif self.storage_type == "json":
            return self._get_from_json(conversation_id)

    def _get_from_db(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """Get state from SQLite database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute(
            "SELECT state_data FROM agent_states WHERE conversation_id = ?",
            (conversation_id,)
        )
        row = cursor.fetchone()
        conn.close()

        return json.loads(row[0]) if row else None

    def _get_from_json(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """Get state from JSON file. Searches for both naming patterns."""
        # Try direct match first
        file_path = self.state_dir / f"{conversation_id}.json"
        if file_path.exists():
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        # Try session-prefixed pattern: {session_id}_{conversation_id}.json
        for f in self.state_dir.glob(f"*_{conversation_id}.json"):
            with open(f, 'r', encoding='utf-8') as fh:
                return json.load(fh)
        return None

    # =================================================================
    # SESSION HISTORY — retrieve all conversations for a given session
    # =================================================================

    def get_session_history(self, session_id: str) -> List[Dict[str, Any]]:
        """Get all conversation states for a session, ordered by time.

        Args:
            session_id: Session identifier that groups conversations

        Returns:
            List of state dicts ordered by creation time (oldest first)
        """
        if self.storage_type == "sqlite":
            return self._get_session_history_db(session_id)
        elif self.storage_type == "json":
            return self._get_session_history_json(session_id)
        return []

    def _get_session_history_db(self, session_id: str) -> List[Dict[str, Any]]:
        """Get session history from SQLite."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute(
            "SELECT state_data FROM agent_states WHERE session_id = ? ORDER BY created_at ASC",
            (session_id,)
        )
        rows = cursor.fetchall()
        conn.close()
        return [json.loads(row[0]) for row in rows]

    def _get_session_history_json(self, session_id: str) -> List[Dict[str, Any]]:
        """Get session history from JSON files.

        Looks for files matching {session_id}_*.json pattern.
        """
        files = sorted(
            self.state_dir.glob(f"{session_id}_*.json"),
            key=lambda f: f.stat().st_mtime
        )
        history = []
        for file in files:
            with open(file, 'r', encoding='utf-8') as f:
                history.append(json.load(f))
        return history

    def build_session_context(self, session_id: str, max_chars: int = 16000) -> str:
        """Build a rich context string from session history for LLM injection.

        Loads all previous conversations in the session and formats them
        with full research report, analysis output, extracted data, data
        profile, and chart explanations so the LLM can answer follow-up
        questions about any aspect of the prior work.

        Args:
            session_id: Session identifier
            max_chars: Maximum total characters for the context string

        Returns:
            Formatted context string, or empty string if no history
        """
        history = self.get_session_history(session_id)
        # Only include completed conversations
        completed = [h for h in history if h.get("status") == "completed"]
        if not completed:
            return ""

        context_parts = []
        chars_used = 0
        # Budget per entry — distribute evenly, minimum 1500 chars each
        per_entry_budget = max(1500, max_chars // max(len(completed), 1))

        for i, entry in enumerate(completed, 1):
            query = entry.get("query", "Unknown query")
            report = entry.get("final_report", "")
            analysis = entry.get("analysis_output", "")
            charts = entry.get("charts", [])
            chart_explanations = entry.get("chart_explanations", {})
            extracted_data = entry.get("extracted_data", "")
            data_profile = entry.get("data_profile", "")
            sources = entry.get("sources", [])

            # Allocate budget across sections:
            # report ~40%, analysis ~25%, extracted_data ~15%, rest ~20%
            report_budget = int(per_entry_budget * 0.40)
            analysis_budget = int(per_entry_budget * 0.25)
            data_budget = int(per_entry_budget * 0.15)

            part = f"--- Query {i}: {query} ---\n"

            if report:
                r = report if len(report) <= report_budget else report[:report_budget] + "... [truncated]"
                part += f"Report:\n{r}\n"

            if analysis:
                a = analysis if len(analysis) <= analysis_budget else analysis[:analysis_budget] + "... [truncated]"
                part += f"Analysis output:\n{a}\n"

            if extracted_data:
                d = extracted_data if len(extracted_data) <= data_budget else extracted_data[:data_budget] + "... [truncated]"
                part += f"Extracted data:\n{d}\n"

            if data_profile:
                p = data_profile if len(data_profile) <= 1000 else data_profile[:1000] + "... [truncated]"
                part += f"Data profile:\n{p}\n"

            if chart_explanations and isinstance(chart_explanations, dict):
                # Actual format from DataAnalysisAgent._extract_chart_explanations():
                #   {path: {"title": "...", "explanation": "..."}}
                # Also handle possible Pydantic-style {"charts": [...]} format.
                expl_lines = []
                if "charts" in chart_explanations and isinstance(chart_explanations["charts"], list):
                    # Pydantic ChartExplanations style
                    for ce in chart_explanations["charts"]:
                        if isinstance(ce, dict):
                            path = ce.get("chart_path", "")
                            title = ce.get("title", "")
                            explanation = ce.get("explanation", "")
                            expl_lines.append(f"  - {title} ({path}): {explanation}")
                else:
                    # Dict[str, Dict[str, str]] style — the actual stored format
                    for path, meta in chart_explanations.items():
                        if isinstance(meta, dict):
                            title = meta.get("title", "")
                            explanation = meta.get("explanation", "")
                            expl_lines.append(f"  - {title} ({path}): {explanation}")
                if expl_lines:
                    part += "Chart explanations:\n" + "\n".join(expl_lines) + "\n"
            elif charts:
                part += f"Chart files: {', '.join(charts)}\n"

            if sources:
                src_lines = []
                for s in sources[:10]:
                    if isinstance(s, dict):
                        src_lines.append(f"  - {s.get('title', s.get('url', ''))}")
                    else:
                        src_lines.append(f"  - {s}")
                remaining = len(sources) - len(src_lines)
                part += "Sources:\n" + "\n".join(src_lines)
                if remaining > 0:
                    part += f"\n  ... and {remaining} more"
                part += "\n"

            # Check total budget
            if chars_used + len(part) > max_chars:
                context_parts.append(f"... [{len(completed) - i + 1} earlier queries omitted]")
                break
            context_parts.append(part)
            chars_used += len(part)

        if not context_parts:
            return ""

        header = f"=== PREVIOUS CONVERSATION CONTEXT ({len(completed)} prior queries) ===\n"
        footer = "\nYou can answer follow-up questions about the research, data, charts, or analysis above.\n"
        return header + "\n".join(context_parts) + footer

    def list_conversations(self, limit: int = 50) -> List[Dict[str, Any]]:
        """List recent conversations for UI.

        Args:
            limit: Maximum number of conversations to return

        Returns:
            List of conversation metadata
        """
        if self.storage_type == "sqlite":
            return self._list_from_db(limit)
        elif self.storage_type == "json":
            return self._list_from_json(limit)

    def _list_from_db(self, limit: int) -> List[Dict[str, Any]]:
        """List conversations from SQLite."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute("""
            SELECT conversation_id, session_id, query, status, updated_at
            FROM agent_states
            ORDER BY updated_at DESC
            LIMIT ?
        """, (limit,))
        rows = cursor.fetchall()
        conn.close()

        return [
            {
                "conversation_id": r[0],
                "session_id": r[1],
                "query": r[2],
                "status": r[3],
                "updated_at": r[4]
            }
            for r in rows
        ]

    def _list_from_json(self, limit: int) -> List[Dict[str, Any]]:
        """List conversations from JSON files."""
        files = sorted(
            self.state_dir.glob("*.json"),
            key=lambda f: f.stat().st_mtime,
            reverse=True
        )[:limit]

        conversations = []
        for file in files:
            with open(file, 'r', encoding='utf-8') as f:
                state = json.load(f)
                conversations.append({
                    "conversation_id": state.get("conversation_id"),
                    "session_id": state.get("session_id"),
                    "query": state.get("query"),
                    "status": state.get("status"),
                    "updated_at": state.get("timestamp")
                })

        return conversations

    def delete_state(self, conversation_id: str):
        """Delete a conversation state.

        Args:
            conversation_id: Unique conversation identifier
        """
        if self.storage_type == "sqlite":
            conn = sqlite3.connect(self.db_path)
            conn.execute(
                "DELETE FROM agent_states WHERE conversation_id = ?",
                (conversation_id,)
            )
            conn.commit()
            conn.close()
        elif self.storage_type == "json":
            # Try direct match
            file_path = self.state_dir / f"{conversation_id}.json"
            if file_path.exists():
                file_path.unlink()
                return
            # Try session-prefixed pattern
            for f in self.state_dir.glob(f"*_{conversation_id}.json"):
                f.unlink()
                return


__all__ = ['StateManager']
