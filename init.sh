#!/bin/bash
set -e

echo "=== Open Deep Research Harness 验证 ==="

echo "=== 编译 Python 源码 ==="
python -m compileall -q src

echo "=== Ruff 检查 ==="
python -m ruff check .

echo "=== mypy 类型检查 ==="
python -m mypy src

echo "=== 收集遗留测试（不调用外部 API）==="
python -m pytest --collect-only -q src/legacy/tests

echo "=== 验证完成 ==="
echo "读取 feature_list.json，选择一个功能，并在 progress.md 记录证据。"
