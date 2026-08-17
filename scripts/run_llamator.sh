#!/usr/bin/env bash
set -euo pipefail
source .venv/bin/activate 2>/dev/null || true
python -m pip install -U python-dotenv >/dev/null 2>&1 || true
python security/llamator/run_llamator.py
