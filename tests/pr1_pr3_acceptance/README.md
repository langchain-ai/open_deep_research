# PR1-PR3 Acceptance

This folder contains final acceptance checks for the recent personalization foundation work:

- PR1: configuration validation behavior
- PR2: ingestion behavior for empty and non-empty local knowledge folders
- PR3: RAG tool registration in runtime tool assembly

## Files

- run_acceptance.ps1: one-command runner for local acceptance
- validate_pr1_pr3.py: Python validation script
- fixtures/: sample folders and files used by tests (add-only)

## Run

From repository root on Windows PowerShell:

    ./tests/pr1_pr3_acceptance/run_acceptance.ps1

## Notes

- The script is add-only for fixtures and does not delete files.
- The script creates a new timestamped Chroma test directory for every run under fixtures/chroma_runs.
- Empty-folder ingestion is expected to return non-zero exit code.
- Full acceptance requires OPENAI_API_KEY plus dependencies installed by uv sync.
