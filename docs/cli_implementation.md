# CLI implementation

This document explains the motivation for adding a command line interface and how the new functionality works.

The repository originally relied on LangGraph Studio running locally to provide a web UI. To make the research tool more flexible, the codebase now exposes a simple `run_multi_agent` helper and a CLI command named `odr`.

`run_multi_agent` can be imported from the `open_deep_research` package. It accepts the human prompt and an optional `auto_accept_plan` flag. When called, it runs the multi-agent workflow and returns the final report.

The `odr` CLI wraps this helper so users can invoke the assistant directly from the terminal:

```bash
odr path/to/prompt.txt --auto-accept-plan
```

The prompt file's content is passed to `run_multi_agent` and the report is printed using Rich formatting. The `--auto-accept-plan` flag bypasses the plan confirmation step.

The CLI is registered as an entry point in `pyproject.toml`, allowing installation of the package with `pip install -e .` and subsequent invocation of `odr` from anywhere.

Google Custom Search credentials must be provided as environment variables (`GOOGLE_API_KEY`, `GOOGLE_CX`). This is documented in the README.
