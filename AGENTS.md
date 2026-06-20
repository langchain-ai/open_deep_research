# Repository Guidelines

## Startup Workflow

Before writing code:

1. Confirm the repository root and read `README.md`.
2. Read `feature_list.json`, `progress.md`, and any current `session-handoff.md`.
3. Select exactly one feature with an actionable status.
4. Run `./init.sh`. If the baseline fails, record the failure before changing code.
5. Review `git status --short` and preserve unrelated user changes.

If dependencies are missing, create the development environment with `uv sync --extra dev`.

## Project Boundaries

Production code is in `src/open_deep_research/`; deployment authentication is in `src/security/`. Evaluation tooling is in `tests/`, while legacy code and its pytest suite are in `src/legacy/`. `deep_research_from_scratch/` is a separate tutorial package and must not be changed unless the active feature explicitly targets it.

Work on one feature at a time. Stay in scope, do not rewrite unrelated files, and never commit `.env`, API keys, private MCP settings, or sensitive generated reports.

## State Artifacts

- `feature_list.json` is the source of truth for feature status, dependencies, and evidence.
- `progress.md` records current state, decisions, blockers, and the next action.
- `session-handoff.md` provides a restart path for unfinished work.
- `init.sh` is the standard verification entrypoint.

Add a concrete feature entry before starting new implementation work. Allowed status values are `not-started`, `in-progress`, `blocked`, and `completed`.

## Verification Commands

Run the full local baseline with:

```bash
./init.sh
```

It performs:

- `python -m compileall -q src`
- `python -m ruff check .`
- `python -m mypy src`
- `python -m pytest --collect-only -q src/legacy/tests`

The comprehensive evaluation, `python tests/run_evaluate.py`, uses external services and may incur cost. Run it only when explicitly required and credentials are available.

## Definition of Done

A feature is done only when its scoped behavior is complete, relevant verification has run, evidence is recorded, documentation is updated when necessary, and the repository has a clear restart path.

## End of Session

Before ending:

1. Update the active feature status and evidence.
2. Update `progress.md` with files changed, verification results, blockers, and the recommended next step.
3. Refresh `session-handoff.md` when work remains unfinished.
4. Check `git status --short` and leave unrelated changes untouched.

Leave a clean restart path: the next session must recover the active objective and rerun verification from these files without relying on chat history.
