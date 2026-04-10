# Open Deep Research Repository Overview

## Project Description
Open Deep Research is a configurable, fully open-source deep research agent that works across multiple model providers, search tools, and MCP (Model Context Protocol) servers. It enables automated research with parallel processing and comprehensive report generation.

## Repository Structure

### Root Directory
- `README.md` - Comprehensive project documentation with quickstart guide
- `pyproject.toml` - Python project configuration and dependencies
- `langgraph.json` - LangGraph configuration defining the main graph entry point
- `uv.lock` - UV package manager lock file
- `LICENSE` - MIT license
- `.env.example` - Environment variables template (not tracked)

### Core Implementation (`src/open_deep_research/`)
- `deep_researcher.py` - Main LangGraph implementation (entry point: `deep_researcher`)
- `configuration.py` - Configuration management and settings
- `state.py` - Graph state definitions and data structures  
- `prompts.py` - System prompts and prompt templates
- `utils.py` - Utility functions and helpers
- `files/` - Research output and example files

### Legacy Implementations (`src/legacy/`)
Contains two earlier research implementations:
- `graph.py` - Plan-and-execute workflow with human-in-the-loop
- `multi_agent.py` - Supervisor-researcher multi-agent architecture
- `legacy.md` - Documentation for legacy implementations
- `CLAUDE.md` - Legacy-specific Claude instructions
- `tests/` - Legacy-specific tests

### Security (`src/security/`)
- `auth.py` - Authentication handler for LangGraph deployment

### Testing (`tests/`)
- `run_evaluate.py` - Main evaluation script configured to run on deep research bench
- `evaluators.py` - Specialized evaluation functions  
- `prompts.py` - Evaluation prompts and criteria
- `pairwise_evaluation.py` - Comparative evaluation tools
- `supervisor_parallel_evaluation.py` - Multi-threaded evaluation

### Examples (`examples/`)
- `arxiv.md` - ArXiv research example
- `pubmed.md` - PubMed research example
- `inference-market.md` - Inference market analysis examples

## Key Technologies
- **LangGraph** - Workflow orchestration and graph execution
- **LangChain** - LLM integration and tool calling
- **Multiple LLM Providers** - OpenAI, Anthropic, Google, Groq, DeepSeek support
- **Search APIs** - Tavily, OpenAI/Anthropic native search, DuckDuckGo, Exa
- **MCP Servers** - Model Context Protocol for extended capabilities

## Development Commands
- `uvx langgraph dev` - Start development server with LangGraph Studio
- `python tests/run_evaluate.py` - Run comprehensive evaluations
- `ruff check` - Code linting
- `mypy` - Type checking

## Configuration
All settings configurable via:
- Environment variables (`.env` file)
- Web UI in LangGraph Studio
- Direct configuration modification

Key settings include model selection, search API choice, concurrency limits, and MCP server configurations.

## Agent Graph Architecture

The system has **3 nested graph layers**: a main graph, a supervisor subgraph, and a researcher subgraph.

### Main Graph Flow (`deep_researcher` in `deep_researcher.py`)

```
User Query
    │
    ▼
[clarify_with_user]
    │  Asks LLM: "Does this need clarification?"
    │  → If yes: returns question to user and STOPS (waits for reply)
    │  → If no: proceeds
    │
    ▼
[write_research_brief]
    │  Transforms conversation into a structured research brief
    │  (a focused paragraph describing exactly what to research)
    │
    ▼
[research_supervisor]  ← supervisor subgraph (loop)
    │
    ▼
[final_report_generation]
    │  Synthesizes all collected notes into the final report
    │
    ▼
  END
```

### Supervisor Subgraph (inside `research_supervisor`)

The supervisor loops between two nodes:
- `[supervisor]` — LLM decides next action using one of 3 tools:
  - `think_tool` — internal scratchpad/reflection, no side effects
  - `ConductResearch` — spawns researcher subgraphs (can spawn multiple in parallel)
  - `ResearchComplete` — signals research is done, exits loop
- `[supervisor_tools]` — executes tool calls; runs `ConductResearch` calls in parallel via `asyncio.gather()`

Loop exits when: `ResearchComplete` is called, `max_researcher_iterations` exceeded, or no tool calls made.

### Researcher Subgraph (spawned in parallel by supervisor)

Each researcher loops between:
- `[researcher]` — LLM searches using bound tools (Tavily, OpenAI web search, MCP tools, think_tool)
- `[researcher_tools]` — executes search tool calls in parallel

Loop exits when: no tool calls made, `max_react_tool_calls` exceeded, or `ResearchComplete` called.

After loop: `[compress_research]` distills all findings into a concise summary returned to supervisor.

### Key Design Decisions

1. **Three loops, not one.** Supervisor loops to plan what to research next. Each researcher loops to search until satisfied. This enables thorough, adaptive research.
2. **Parallel researchers.** `asyncio.gather()` runs up to `max_concurrent_research_units` (default: 5) researchers simultaneously.
3. **Compression before aggregation.** Each researcher compresses its own findings before returning to supervisor, preventing context overflow.
4. **Notes vs. compressed_research.** Raw search results go to `raw_notes` (for debugging); `notes` holds compressed summaries that feed the final report.

## State Structures (`state.py`)

| State | Used by | Key fields |
|---|---|---|
| `AgentState` | Main graph | `messages`, `research_brief`, `notes`, `final_report` |
| `SupervisorState` | Supervisor subgraph | `supervisor_messages`, `research_iterations`, `notes` |
| `ResearcherState` | Each researcher | `researcher_messages`, `tool_call_iterations`, `compressed_research` |

The `override_reducer` pattern allows nodes to explicitly replace list fields (instead of appending) by passing `{"type": "override", "value": [...]}`.

## Minimum Required API Keys

- `OPENAI_API_KEY` — covers all default models (`gpt-4.1`, `gpt-4.1-mini`)
- `TAVILY_API_KEY` — default search API (free tier sufficient); OR switch `search_api` to `"openai"` to use OpenAI native web search
- `LANGSMITH_API_KEY` + `LANGSMITH_TRACING=true` — strongly recommended for debugging traces (free at smith.langchain.com)

Do not set unused keys to `""` — leave them out entirely to avoid library errors.