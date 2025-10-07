# Structured Activity Log Schema v1.1

This document defines the JSON schema for entries in `logs/activity.log.jsonl`. Each line in the log file is a JSON object that must conform to this schema. The goal of this schema is to provide a rich, structured dataset for post-mortem analysis and to power the proactive task generation in Phase 7.

## Schema Definition

```json
{
  "type": "object",
  "properties": {
    "log_id": {
      "type": "string",
      "format": "uuid",
      "description": "A unique identifier for this specific log entry."
    },
    "session_id": {
      "type": "string",
      "description": "A unique identifier for the entire work session. Allows grouping of all logs for a single session."
    },
    "timestamp": {
      "type": "string",
      "format": "date-time",
      "description": "The ISO 8601 timestamp of the event."
    },
    "phase": {
      "type": "string",
      "enum": ["Phase 0", "Phase 1", "Phase 2", "Phase 3", "Phase 4", "Phase 5", "Phase 6", "Phase 7", "Phase 8"],
      "description": "The protocol phase in which the action occurred."
    },
    "task": {
      "type": "object",
      "properties": {
        "id": {
          "type": "string",
          "description": "A unique identifier for the current task."
        },
        "plan_step": {
          "type": "integer",
          "description": "The specific step number in the plan that this action corresponds to."
        }
      },
      "required": ["id", "plan_step"]
    },
    "action": {
      "type": "object",
      "properties": {
        "type": {
          "type": "string",
          "enum": [
            "TASK_START",
            "FILE_READ",
            "FILE_WRITE",
            "TOOL_EXEC",
            "EXTERNAL_RAG_QUERY",
            "PLAN_UPDATE",
            "CRITIC_FEEDBACK",
            "POST_MORTEM",
            "TASK_END",
            "INFO",
            "SYSTEM_FAILURE"
          ],
          "description": "The type of action being logged."
        },
        "details": {
          "type": "object",
          "description": "A flexible object containing details specific to the action type. See 'Action Details Examples' for best practices."
        }
      },
      "required": ["type", "details"]
    },
    "outcome": {
      "type": "object",
      "properties": {
        "status": {
          "type": "string",
          "enum": ["SUCCESS", "FAILURE", "IN_PROGRESS"],
          "description": "The outcome of the action."
        },
        "message": {
          "type": "string",
          "description": "A human-readable message describing the outcome."
        },
        "error": {
            "type": "object",
            "properties": {
                "message": {"type": "string"},
                "stack_trace": {"type": "string"}
            },
            "description": "Structured error information, present only on FAILURE."
        }
      },
      "required": ["status"]
    },
    "evidence_citation": {
      "type": "string",
      "description": "A citation to the source (e.g., external documentation, internal artifact) that justifies the action, as per protocol."
    }
  },
  "required": ["log_id", "session_id", "timestamp", "phase", "task", "action", "outcome"]
}
```

## Action Details Examples

- **TASK_START**: `{"origin": "user" | "proactive", "description": "High-level task description.", "justification": "Analysis that led to this proactive task."}`
- **FILE_WRITE**: `{"path": "/path/to/file.md", "content_hash": "sha256_hash_of_content"}`
- **TOOL_EXEC**: `{"command": "ls -l", "stdout": "...", "stderr": "..."}`
- **EXTERNAL_RAG_QUERY**: `{"query": "React best practices 2025", "results_summary": "Top 3 results summarized..."}`
- **POST_MORTEM**: `{"summary": "What worked, what failed, root cause analysis."}`
- **TASK_END**: `{"summary": "Signals the formal end of the development phase of a Finite Development Cycle, post-mortem complete."}`
- **INFO**: `{"summary": "An informational message or observation."}`
- **SYSTEM_FAILURE**: `{"error_message": "...", "stack_trace": "..."}`

## Example Entry

```json
{"log_id":"...","session_id":"...","timestamp":"2025-10-05T18:00:00Z","phase":"Phase 2","task":{"id":"improve-logging-01","plan_step":3},"action":{"type":"FILE_WRITE","details":{"path":"LOGGING_SCHEMA.md","content_hash":"..."}},"outcome":{"status":"SUCCESS","message":"Updated logging schema to v1.1."},"evidence_citation":"Agent.md, Phase 7 analysis"}
```