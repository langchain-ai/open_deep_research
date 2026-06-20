#!/bin/bash
set -e

echo "=== Open Deep Research Harness Verification ==="

echo "=== Compile Python sources ==="
python -m compileall -q src

echo "=== Ruff ==="
python -m ruff check .

echo "=== Mypy ==="
python -m mypy src

echo "=== Collect legacy tests without external API execution ==="
python -m pytest --collect-only -q src/legacy/tests

echo "=== Verification Complete ==="
echo "Read feature_list.json, select one feature, and record evidence in progress.md."
