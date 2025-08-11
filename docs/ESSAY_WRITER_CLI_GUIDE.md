# Essay Writer and CLI Guide

This guide documents the new Essay Writer graph, the parametric CLI (`scripts/call_api.py`), and the Makefile targets for running and testing from the terminal.

## Essay Writer Graph
- Location: `src/open_deep_research/essay_writer.py`
- Graph name: `Essay Writer` (registered in `langgraph.json`).
- Flow: `clarify_with_user → write_outline → essay_supervisor (subgraph) → compose_essay`.
- Behavior:
  - Outline: Reuses the brief transformation to produce an outline-like “research_brief”.
  - Supervisor: Delegates section/topic research via `ConductResearch` and `think_tool` with concurrency limits.
  - Researcher: Uses configured search/MCP tools; synthesizes a compressed note per delegation.
  - Compose: Uses the “final report” composer as a placeholder essay writer.
- State: Reuses `AgentState` and `SupervisorState`; compatible with the existing server and tooling.

## Running the Server
- Start dev server (LangGraph):
  - `make dev`
- API base: `http://127.0.0.1:2024`
- Docs/Playground: `http://127.0.0.1:2024/docs`
- Graphs available: `Deep Researcher`, `Essay Writer`.

## HTTP API (direct)
- Non‑stream invoke:
  - `POST /api/graphs/{graph}/invoke`
  - Body: `{ "input": {"messages": [{"type": "human", "content": "..."}]}, "config": {"configurable": {"thread_id": "t1"}} }`
- Streaming events:
  - `POST /api/graphs/{graph}/stream` (keep‑alive)
  - Use `curl -sN` to view node events (on_node_start/on_node_end, final state).

## Parametric CLI: `scripts/call_api.py`
- Non‑stream:
  - `uv run python scripts/call_api.py --graph "Essay Writer" --message "AI in education" --thread-id essay-1`
- Stream node events:
  - `uv run python scripts/call_api.py --graph "Essay Writer" --message "Nuclear energy" --thread-id essay-2 --stream`
- Model/search overrides (per run):
  - `--research-model`, `--final-model`, `--summarization-model`, `--compression-model`, `--search-api` (tavily|openai|anthropic|none)
  - Agenticity: `--max-researcher-iterations`, `--max-react-tool-calls`, `--max-concurrent-research-units`
- API keys via request (optional):
  - Requires server env `GET_API_KEYS_FROM_CONFIG=true`.
  - `--openai-key … --anthropic-key … --tavily-key …`
- Save final output:
  - `--output out/essay.md` writes the `final_report` to a file.

## Makefile Targets
- `make dev`: Start LangGraph server.
- `make invoke MSG="..." [GRAPH=...] [THREAD=...] [RESEARCH_MODEL=...] [FINAL_MODEL=...] [SEARCH_API=...] [OUT=path]`
- `make stream MSG="..." [GRAPH=...] [THREAD=...] [RESEARCH_MODEL=...] [FINAL_MODEL=...] [SEARCH_API=...]`
- Shortcuts:
  - `make essay` / `make essay-stream` (graph = `Essay Writer`)
  - `make deep` / `make deep-stream` (graph = `Deep Researcher`)
- Compare models (A/B):
  - `make compare-models GRAPH="Essay Writer" MSG="AI in education" FINAL_MODEL="openai:gpt-4.1" FINAL_MODEL_B="anthropic:claude-sonnet-4-20250514"`
  - Outputs: `out/run_A.md`, `out/run_B.md` for diffing.

## Configuration & Keys
- Default: Read API keys from environment (e.g., `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `TAVILY_API_KEY`).
- Inline keys (optional): Set `GET_API_KEYS_FROM_CONFIG=true` in server env and pass `--openai-key`/`--anthropic-key`/`--tavily-key` in the CLI.
- Per‑run configuration: Override any `Configuration` field via `config.configurable` using the CLI flags above.

## Next Steps
- Swap in essay‑specific prompts for outline and composition.
- Add a dedicated `WriteSection` tool to draft per‑section text.
- Introduce a structured outline schema (sections, claims, evidence) for more controlled drafting.

