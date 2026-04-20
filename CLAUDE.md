# Open Deep Research — AI Assistant Guide

## Project Overview

Open Deep Research is a configurable, fully open-source deep research agent implemented as a LangGraph workflow. It orchestrates a hierarchy of LLM agents (supervisor + parallel sub-researchers) to conduct multi-step web research and synthesize findings into comprehensive markdown reports.

- **Package**: `open_deep_research` v0.0.16
- **Entry point**: `src/open_deep_research/deep_researcher.py:deep_researcher`
- **LangGraph graph name**: `Deep Researcher` (defined in `langgraph.json`)
- **Python**: ≥ 3.10 (3.11 used in deployment)
- **Package manager**: `uv` (lock file: `uv.lock`)

---

## Repository Structure

```
open_deep_research/
├── CLAUDE.md                        # This file
├── README.md                        # User-facing documentation
├── pyproject.toml                   # Dependencies and tool configuration
├── langgraph.json                   # LangGraph server config (graph entry + auth)
├── .env.example                     # Required environment variables template
├── uv.lock                          # Pinned dependency versions
├── src/
│   ├── open_deep_research/          # PRIMARY implementation (use this)
│   │   ├── deep_researcher.py       # Main LangGraph graph + all nodes
│   │   ├── configuration.py         # Pydantic Configuration class + SearchAPI enum
│   │   ├── state.py                 # All TypedDict/Pydantic state classes
│   │   ├── prompts.py               # All prompt strings (no logic)
│   │   └── utils.py                 # Tools, search helpers, token utilities
│   ├── legacy/                      # DEPRECATED — do not modify
│   │   ├── graph.py                 # Old plan-and-execute workflow
│   │   ├── multi_agent.py           # Old supervisor-researcher architecture
│   │   ├── configuration.py
│   │   ├── state.py
│   │   ├── prompts.py
│   │   ├── utils.py
│   │   ├── legacy.md
│   │   ├── CLAUDE.md
│   │   └── tests/
│   └── security/
│       └── auth.py                  # Supabase JWT auth for LangGraph deployment
├── tests/
│   ├── run_evaluate.py              # Main evaluation runner (LangSmith)
│   ├── evaluators.py                # Evaluation functions (quality, relevance, etc.)
│   ├── prompts.py                   # Evaluation prompt templates
│   ├── pairwise_evaluation.py       # A/B comparison evaluation
│   ├── supervisor_parallel_evaluation.py
│   └── extract_langsmith_data.py
└── examples/
    ├── arxiv.md
    ├── pubmed.md
    └── inference-market.md
```

> **Important**: Only modify files under `src/open_deep_research/` and `tests/`. The `src/legacy/` directory is deprecated and should not receive new features.

---

## Graph Architecture

The main graph (`deep_researcher`) is a three-layer nested LangGraph workflow:

### Layer 1: Main Graph (`deep_researcher`)

```
START
  └─► clarify_with_user
        ├─► END (if clarification needed — returns question to user)
        └─► write_research_brief
              └─► research_supervisor (SubGraph)
                    └─► final_report_generation
                          └─► END
```

**Nodes:**

| Node | Function | Purpose |
|------|----------|---------|
| `clarify_with_user` | `clarify_with_user()` | Optionally ask a clarifying question before researching |
| `write_research_brief` | `write_research_brief()` | Converts user messages → structured `ResearchQuestion` brief |
| `research_supervisor` | `supervisor_subgraph` | Manages parallel research delegation |
| `final_report_generation` | `final_report_generation()` | Synthesizes all notes into final markdown report |

### Layer 2: Supervisor Subgraph

```
START → supervisor ⇄ supervisor_tools → END
```

The supervisor loops calling `ConductResearch` (spawning sub-researchers) or `think_tool` until it calls `ResearchComplete` or exceeds `max_researcher_iterations`.

**Termination conditions** (checked in `supervisor_tools`):
- `ResearchComplete` tool called
- No tool calls in response
- `research_iterations > max_researcher_iterations`

### Layer 3: Researcher Subgraph (runs in parallel)

```
START → researcher ⇄ researcher_tools → compress_research → END
```

Each researcher runs a ReAct loop using search tools and `think_tool`, then compresses its findings.

**Termination conditions** (checked in `researcher_tools`):
- No tool calls and no native web search detected
- `tool_call_iterations >= max_react_tool_calls`
- `ResearchComplete` tool called

---

## State Definitions (`state.py`)

### Structured Output Models (Pydantic)

| Class | Used By | Purpose |
|-------|---------|---------|
| `ConductResearch` | Supervisor tool | Wraps a `research_topic` string |
| `ResearchComplete` | Supervisor + Researcher tools | Signals end of research |
| `ClarifyWithUser` | `clarify_with_user` node | `need_clarification`, `question`, `verification` |
| `ResearchQuestion` | `write_research_brief` node | `research_brief` string |
| `Summary` | Tavily summarization | `summary` + `key_excerpts` |

### State TypedDicts

| Class | Used In | Key Fields |
|-------|---------|-----------|
| `AgentInputState` | Main graph input | `messages` (inherited from `MessagesState`) |
| `AgentState` | Main graph | `messages`, `supervisor_messages`, `research_brief`, `raw_notes`, `notes`, `final_report` |
| `SupervisorState` | Supervisor subgraph | `supervisor_messages`, `research_brief`, `notes`, `research_iterations`, `raw_notes` |
| `ResearcherState` | Researcher subgraph | `researcher_messages`, `tool_call_iterations`, `research_topic`, `compressed_research`, `raw_notes` |
| `ResearcherOutputState` | Researcher subgraph output | `compressed_research`, `raw_notes` |

### `override_reducer`

Many list fields use `override_reducer` instead of the default `operator.add`. Pass `{"type": "override", "value": [...]}` to replace rather than append a list field.

---

## Configuration (`configuration.py`)

All settings are managed via the `Configuration` Pydantic model. Values are loaded from (in priority order): environment variables (uppercased field name) → `config["configurable"]` → field defaults.

### Key Settings

| Field | Default | Description |
|-------|---------|-------------|
| `allow_clarification` | `True` | Whether to ask the user a clarifying question |
| `max_concurrent_research_units` | `5` | Max parallel researcher sub-agents |
| `max_researcher_iterations` | `6` | Max supervisor loop iterations |
| `max_react_tool_calls` | `10` | Max tool calls per researcher |
| `max_structured_output_retries` | `3` | LLM retries for structured output |
| `search_api` | `SearchAPI.TAVILY` | Search provider (see `SearchAPI` enum) |
| `research_model` | `openai:gpt-4.1` | Model for supervisor + researchers |
| `research_model_max_tokens` | `10000` | Max output tokens for research model |
| `compression_model` | `openai:gpt-4.1` | Model for compressing researcher output |
| `compression_model_max_tokens` | `8192` | Max output tokens for compression |
| `final_report_model` | `openai:gpt-4.1` | Model for writing the final report |
| `final_report_model_max_tokens` | `10000` | Max output tokens for final report |
| `summarization_model` | `openai:gpt-4.1-mini` | Model for summarizing Tavily results |
| `summarization_model_max_tokens` | `8192` | Max output tokens for summarization |
| `max_content_length` | `50000` | Max chars of webpage content before summarization |
| `mcp_config` | `None` | `MCPConfig(url, tools, auth_required)` |
| `mcp_prompt` | `None` | Additional instructions about MCP tools |

### `SearchAPI` Enum

```python
SearchAPI.ANTHROPIC  # "anthropic" — Anthropic native web search (web_search_20250305)
SearchAPI.OPENAI     # "openai"    — OpenAI web search preview
SearchAPI.TAVILY     # "tavily"    — Tavily search API (default, fetches + summarizes)
SearchAPI.NONE       # "none"      — No search; only MCP tools
```

> **Note**: When using `SearchAPI.ANTHROPIC`, the `research_model` must be an Anthropic model. When using `SearchAPI.OPENAI`, it must be an OpenAI model.

### Model String Format

Models use LangChain's `init_chat_model` format: `"provider:model-name"`, e.g.:
- `"openai:gpt-4.1"`, `"openai:gpt-4.1-mini"`, `"openai:o3"`
- `"anthropic:claude-opus-4"`, `"anthropic:claude-sonnet-4"`
- `"google:gemini-1.5-pro"`
- `"bedrock:us.anthropic.claude-sonnet-4-20250514-v1:0"`

---

## Utilities (`utils.py`)

### Tools

| Tool | Type | Description |
|------|------|-------------|
| `tavily_search` | `@tool` (async) | Searches Tavily for `List[str]` queries, deduplicates results, and summarizes each page |
| `think_tool` | `@tool` (sync) | Strategic reflection — records a string; used between searches |

### Key Functions

| Function | Description |
|----------|-------------|
| `get_all_tools(config)` | Returns `[ResearchComplete, think_tool, <search_tool>, ...mcp_tools]` |
| `get_search_tool(search_api)` | Returns the configured search tool list |
| `load_mcp_tools(config, existing_names)` | Loads tools from MCP server with auth |
| `get_api_key_for_model(model_name, config)` | Returns API key from env or config |
| `get_tavily_api_key(config)` | Returns Tavily key from env or config |
| `is_token_limit_exceeded(exception, model_name)` | Detects context limit errors for OpenAI/Anthropic/Gemini |
| `get_model_token_limit(model_string)` | Looks up context window size from `MODEL_TOKEN_LIMITS` dict |
| `remove_up_to_last_ai_message(messages)` | Truncates message history for token limit recovery |
| `anthropic_websearch_called(response)` | Detects native Anthropic web search usage |
| `openai_websearch_called(response)` | Detects native OpenAI web search usage |
| `get_today_str()` | Returns formatted date string for prompts |
| `wrap_mcp_authenticate_tool(tool)` | Wraps MCP tool with auth error handling |

### `MODEL_TOKEN_LIMITS`

A static dictionary in `utils.py` mapping model strings to context window sizes (in tokens). **Update this dict when adding support for new models.** Token limit recovery uses `model_token_limit * 4` as a character approximation.

### API Key Resolution

The `GET_API_KEYS_FROM_CONFIG` environment variable controls where API keys come from:
- `"false"` (default): Read from environment variables (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, etc.)
- `"true"`: Read from `config["configurable"]["apiKeys"]` dict (used in Open Agent Platform deployments)

---

## Environment Variables

Copy `.env.example` to `.env` and fill in the required keys:

```bash
# LLM Providers (add the ones you need)
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
GOOGLE_API_KEY=

# Search
TAVILY_API_KEY=          # Required if search_api=tavily (default)

# Observability
LANGSMITH_API_KEY=
LANGSMITH_PROJECT=
LANGSMITH_TRACING=true

# Open Agent Platform (production deployment only)
SUPABASE_KEY=
SUPABASE_URL=
GET_API_KEYS_FROM_CONFIG=false   # Set to "true" for OAP production
```

---

## Development Commands

```bash
# Start local LangGraph Studio dev server
uvx langgraph dev

# Linting (ruff with Google docstring style)
ruff check
ruff check --fix

# Type checking
mypy

# Run evaluation against LangSmith "Deep Research Bench" dataset
python tests/run_evaluate.py
```

### Development Server

`uvx langgraph dev` starts a local server at `http://127.0.0.1:2024` and opens LangGraph Studio where you can interact with the graph, configure parameters via UI, and trace execution.

---

## Prompts (`prompts.py`)

All prompt strings are stored as module-level constants. They use Python `.format()` with named placeholders:

| Prompt | Placeholders | Purpose |
|--------|-------------|---------|
| `clarify_with_user_instructions` | `{messages}`, `{date}` | Determines if clarification is needed |
| `transform_messages_into_research_topic_prompt` | `{messages}`, `{date}` | Converts conversation to research brief |
| `lead_researcher_prompt` | `{date}`, `{max_concurrent_research_units}`, `{max_researcher_iterations}` | Supervisor system prompt |
| `research_system_prompt` | `{mcp_prompt}`, `{date}` | Individual researcher system prompt |
| `compress_research_system_prompt` | `{date}` | Compression/synthesis system prompt |
| `compress_research_simple_human_message` | *(none)* | Human turn for compression |
| `final_report_generation_prompt` | `{research_brief}`, `{messages}`, `{findings}`, `{date}` | Final report generation prompt |
| `summarize_webpage_prompt` | `{webpage_content}`, `{date}` | Tavily result summarization |

---

## Security (`src/security/auth.py`)

JWT-based authentication for LangGraph Server deployments using Supabase:

- `auth.authenticate` → `get_current_user()`: Validates Bearer tokens via Supabase
- `@auth.on.threads.*` handlers: Enforce per-user thread ownership
- `@auth.on.assistants.*` handlers: Enforce per-user assistant ownership
- `@auth.on.store()`: Enforces namespace-based store access control (`namespace[0] == user.identity`)
- `StudioUser` instances (LangGraph Studio) bypass all access controls

Referenced in `langgraph.json` as `"auth": {"path": "./src/security/auth.py:auth"}`.

---

## Evaluation (`tests/`)

Uses LangSmith's `client.aevaluate()` against the **"Deep Research Bench"** dataset.

**Evaluators** (from `tests/evaluators.py`):
- `eval_overall_quality` — overall report quality
- `eval_relevance` — topic and section relevance
- `eval_structure` — logical flow and markdown formatting
- `eval_correctness` — factual accuracy
- `eval_groundedness` — claims backed by sources
- `eval_completeness` — coverage of the research question

**Running evaluations**: Edit `tests/run_evaluate.py` to configure the experiment parameters (model, search API, iteration counts, etc.) before running. The experiment prefix and metadata are logged to LangSmith.

---

## Key Conventions

### Async-First

All graph nodes and most utility functions are `async`. Use `asyncio.gather()` for parallel operations (researcher invocations, Tavily queries, tool executions).

### Structured Output Pattern

```python
model = (
    configurable_model
    .with_structured_output(SomePydanticModel)
    .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
    .with_config(model_config)
)
```

All structured output calls use `.with_retry()` with `max_structured_output_retries`.

### Token Limit Recovery

When token limit errors occur:
1. `compress_research`: Calls `remove_up_to_last_ai_message()` and retries (max 3 attempts)
2. `final_report_generation`: Progressively truncates `findings` string (−10% per retry, max 3 retries)
3. `supervisor_tools`: Exits research phase gracefully on any exception

### Command Pattern

All graph nodes return `Command(goto="node_name", update={...})` rather than plain dicts. This is a LangGraph pattern for explicit routing with state updates.

### MCP Tool Authentication

MCP tools with `auth_required=True` use a Supabase-to-MCP token exchange flow:
1. Supabase token from `config["configurable"]["x-supabase-access-token"]`
2. Exchanged via OAuth at `<mcp_url>/oauth/token`
3. Cached in LangGraph store under `(user_id, "tokens")` with expiration checking
4. Bearer token passed as `Authorization` header to `<mcp_url>/mcp`

### Ruff Configuration

The project uses ruff with Google docstring style. Key rules:
- `E`, `F`, `I` (pycodestyle, pyflakes, isort)
- `D` with `D401` (Google convention, imperative mood)
- `T201` (no print statements)
- Ignored: `UP006`, `UP007`, `UP035` (older type hint forms), `D417`, `E501` (long lines ok)
- `tests/*` files exempt from `D` and `UP` rules

---

## Common Patterns for Modifications

### Adding a New Search Provider

1. Add a new value to `SearchAPI` enum in `configuration.py`
2. Add a handler branch in `get_search_tool()` in `utils.py`
3. Update `Configuration.search_api` field's `x_oap_ui_config.options` list

### Adding a New Model

1. Add the model string and token limit to `MODEL_TOKEN_LIMITS` in `utils.py`
2. Ensure the provider's LangChain package is listed in `pyproject.toml`

### Modifying Prompts

Edit the relevant string constant in `prompts.py`. All prompts use `.format()` — add new `{placeholder}` variables and pass them at the call site in `deep_researcher.py`.

### Adding a New Graph Node

1. Define an `async def node_name(state: SomeState, config: RunnableConfig)` function in `deep_researcher.py`
2. Add it to the appropriate `StateGraph` builder with `builder.add_node()`
3. Wire edges with `builder.add_edge()` or return `Command(goto=...)` from connected nodes
