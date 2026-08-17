# Open Deep Research behavioural tests (Monocle Test Tools)

Trace-based tests that lock in Open Deep Research's behaviour. Monocle records
each run as a structured trace -- the agent invocation, LLM token usage, and
timings -- and each test asserts against that trace: which agent ran, what it was
asked, what it produced, and its token/duration cost. A later prompt, model, or
config change that regresses the behaviour fails here.

## Layout

- `test_opendeepresearch.py` — the suite: four offline tests (one per curated question) + one live test
- `conftest.py` — Monocle setup, `.env` loading, and `run_opendeepresearch()`
- `traces/` — recorded good-trace fixtures the offline tests replay
- `requirements.txt` — dependencies

## Tests

| Test | Scenario | What it shows |
|---|---|---|
| `test_earth_seasons` | What causes Earth's seasons (explainer) | agent, verbatim output, token + duration budget |
| `test_renewable_vs_nonrenewable` | Renewable vs. nonrenewable energy (comparison) | agent, output, `contains_any_output`, budget |
| `test_ocean_tides` | What causes ocean tides (explainer) | agent, output, `contains_any_output`, budget |
| `test_tcp_vs_udp` | TCP vs. UDP (comparison) | agent, output, `contains_any_output`, budget |
| `test_tcp_vs_udp_live` | TCP vs. UDP, run live | live run, structure + budget only |

The offline tests replay recorded traces with budgets measured from those runs
(rounded up with headroom). The live test drives the agent end-to-end and asserts
structure and budget only, since the output legitimately varies run to run.

Open Deep Research runs its search inside the model call (OpenAI-native web
search in the `openai.resources.responses` model-api spans), so its traces
contain no `agentic.tool.invocation` spans. The tests assert the agent
invocation, output, and budgets that exist in the trace, and do not assert tool
calls.

## Run

```bash
pip install -r requirements.txt
pytest tests/monocle/ -k "not live"   # offline, no network, no keys
pytest tests/monocle/                 # includes the live runs (needs OPENAI_API_KEY)
```

The live tests skip unless `OPENAI_API_KEY` is set. They use OpenAI-native
search (so no Tavily/other search key is needed) and are cost-capped to a single
researcher iteration on `gpt-4o-mini`.

## Add your own test

1. Run Open Deep Research under Monocle and capture a trace of a run you're happy
   with (Monocle writes trace JSON to `.monocle/` by default).
2. Move it into `traces/` and load it with
   `monocle_trace_asserter.validator.add_remote_spans(JSONSpanLoader.from_json(path))`.
3. Assert with the fluent API — `called_agent(...)`, `contains_output(...)`,
   `contains_any_output(...)`, `under_token_limit(...)`,
   `under_duration(..., span_type="workflow")` — then add it alongside the others.

## Evaluations (optional)

Each test carries a commented-out `check_eval("hallucination", ...)` chain.
Monocle can run evaluation checks against a trace; set `OKAHU_API_KEY` and
uncomment to enable.
