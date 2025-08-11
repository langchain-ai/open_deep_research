# Repository Guidelines

## Project Structure & Module Organization
- `src/open_deep_research/`: Core agent modules (`configuration.py`, `deep_researcher.py`, `state.py`, `prompts.py`, `utils.py`).
- `src/legacy/`: Prior implementations; useful for reference and experiments.
- `src/security/`: Security-related helpers and checks.
- `tests/`: Evaluation scripts and utilities (e.g., `run_evaluate.py`, `extract_langsmith_data.py`).
- `examples/`: Example tasks and prompts.
- Root configs: `pyproject.toml` (build, deps, lint), `.env(.example)` (secrets), `langgraph.json` (studio config), `uv.lock`.

## Build, Test, and Development Commands
- Create env: `uv venv && source .venv/bin/activate`
- Install deps: `uv sync` (or `uv pip install -r pyproject.toml`)
- Run locally (LangGraph Studio + API):
  - `uvx --refresh --from "langgraph-cli[inmem]" --with-editable . --python 3.11 langgraph dev --allow-blocking`
- Lint + format: `uv run ruff check .` and `uv run ruff format` (fixes import order and style).
- Type check (optional): `uv run mypy src` (install with `uv sync --extra dev`).
- Evaluation: `uv run python tests/run_evaluate.py` then `uv run python tests/extract_langsmith_data.py --project-name "NAME" --model-name "MODEL" --dataset-name "deep_research_bench"`.

## Coding Style & Naming Conventions
- Python 3.10+; 4‑space indentation; type hints for public APIs.
- Naming: modules/files `snake_case.py`; functions/vars `snake_case`; classes `PascalCase`; constants `UPPER_SNAKE`.
- Docstrings: Google style; first line imperative (see `pyproject.toml` pydocstyle `D401`).
- Linting: `ruff` enforces pycodestyle/pyflakes/isort/docs. Prefer small, focused modules under `src/open_deep_research/`.

## Testing Guidelines
- Frameworks: `pytest` available; current repo focuses on evaluation scripts in `tests/`.
- Add unit tests under `tests/` with pattern `test_*.py`; run via `uv run pytest`.
- Keep tests deterministic; mock network/model calls where possible.
- If contributing benchmarks, stash outputs under `tests/expt_results/` (git-ignored) and summarize in PR.

## Commit & Pull Request Guidelines
- Commits: imperative mood, clear scope; group logical changes. Example: `research: bound tool retries in supervisor`.
- PRs: include purpose, approach, testing steps, config/env vars touched, and links to related issues. Add screenshots or Studio run links when UI/graph behavior changes.
- CI hygiene: run lint, type check, and (if added) unit tests locally before requesting review.

## Security & Configuration Tips
- Never commit secrets. Copy `cp .env.example .env` and set keys. `Configuration.from_runnable_config` maps fields from uppercase env vars (e.g., `RESEARCH_MODEL`).
- MCP/Search tools may require additional keys; document any new variables in `.env.example` when adding integrations.
