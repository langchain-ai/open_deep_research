# Raochra fork notes

This repository is a Raochra-owned fork of
[`langchain-ai/open_deep_research`](https://github.com/langchain-ai/open_deep_research),
realizing the deep-research engine behind the consulting-workbench Researcher
role contract. Engine code stays generic; consulting flavor is supplied via
deployment overlay assets (Raochra issue RSM-308).

## Upstream baseline

* Upstream: `langchain-ai/open_deep_research`
* Branch baseline: `main` at commit `0dd30bd47ed6ed3ac4d2b678997662830f227a14`
  (snapshot 2026-04-29).
* Upstream remote configured locally as `upstream`.

To sync upstream changes:

```bash
git fetch upstream main
git merge upstream/main
```

## Engine HTTP boundary

The `open_deep_research.engine_server` module exposes a FastAPI app:

* `POST /research` — neutral DeepResearchEngine contract (see
  `engine_models.py`). Bearer auth, X-Request-ID echoed as
  `runtime_trace_ref`.
* `GET /health` — liveness probe.

Run locally:

```bash
export OPEN_DEEP_RESEARCH_API_KEY=<choose-a-strong-token>
# Cheap LLM provider via OpenAI-compatible endpoint (see §LLM provider config).
export OPENAI_API_BASE_URL=https://dashscope-intl.aliyuncs.com/compatible-mode/v1
export OPENAI_API_KEY=<dashscope-key>
export TAVILY_API_KEY=<tavily-key>

uv run uvicorn open_deep_research.engine_server:app --host 0.0.0.0 --port 2024
```

Then point sensemaker-studio's `.env` at it:

```bash
OPEN_DEEP_RESEARCH_BASE_URL=http://localhost:2024
OPEN_DEEP_RESEARCH_API_KEY=<the-token-above>
RSM_RUN_RESEARCHER_LIVE_BV=1
```

## LLM provider config

Engine code is generic; provider choice is deployment-time. Per Raochra's
cost posture, default to cheaper OpenAI-compatible providers:

* **Qwen** (Alibaba DashScope) — base URL
  `https://dashscope-intl.aliyuncs.com/compatible-mode/v1`, models like
  `qwen-plus`, `qwen-max`.
* **Kimi** (Moonshot) — base URL `https://api.moonshot.cn/v1`, models like
  `moonshot-v1-32k`, `kimi-k1.5`.
* **GLM** (Zhipu) — base URL `https://open.bigmodel.cn/api/paas/v4`, models
  like `glm-4-air`, `glm-4-flash`.
* **MiniMax** — base URL `https://api.minimaxi.chat/v1`, models like
  `abab6.5-chat`, `minimax-text-01`.

Override the model identifiers ODR uses by exporting the matching env vars
before launch:

```bash
export RESEARCH_MODEL=openai:qwen-plus
export SUMMARIZATION_MODEL=openai:qwen-turbo
export COMPRESSION_MODEL=openai:qwen-plus
export FINAL_REPORT_MODEL=openai:qwen-plus
```

`ConsultingResearcher` request fields (`topic_lens_scope`,
`source_preferences`, `research_target`) flow through as generic engine
parameters and do not require provider-specific handling.

## Out of scope here

* Consulting prompt overlays (planning/synthesis tone, disconfirming
  discipline) — Raochra issue RSM-308.
* Cloud Run production deploy — separate slice.
* Closeout BV across consulting use cases (UC-001..005) — Raochra issue
  RSM-185.
