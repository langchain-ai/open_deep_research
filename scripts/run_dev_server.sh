#!/usr/bin/env bash
set -euo pipefail
python3 -m venv .venv 2>/dev/null || true
source .venv/bin/activate
python -m pip install -U pip wheel >/dev/null
python -m pip install -e . >/dev/null
python -m pip install -U python-dotenv >/dev/null
langgraph dev
