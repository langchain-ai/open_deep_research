#!/usr/bin/env bash
set -euo pipefail
# Grep for token-like strings in common artifact locations.
# Safe: does not print secrets; only reports matches/paths.
rg -n --hidden -S "(sk-[A-Za-z0-9]{10,}|Bearer\s+[A-Za-z0-9._\-]{10,}|LLM_API_KEY=|OPENAI_API_KEY=|LANGSMITH_API_KEY=)" results/ .langsmith/ *.log 2>/dev/null || true
