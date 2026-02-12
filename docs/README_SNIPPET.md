<!-- ODR_OPENROUTER_SNIPPET_START -->
## OpenRouter (OpenAI-compatible) quickstart

This project is configured to work without OpenAI API access by using an OpenAI-compatible endpoint (default: OpenRouter).
Keys are read from `.env` / environment variables only (never committed).

### Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip wheel
python -m pip install -e .
python -m pip install -U python-dotenv
cp .env.example .env
# edit .env: set LLM_API_KEY (and optionally LLAMATOR_ATTACK_API_KEY), keep LIVE_API=0 by default
```

### Run (dev-server)
```bash
langgraph dev
# default: http://127.0.0.1:2024
```

### LLAMATOR
Safe default:
```bash
python security/llamator/run_llamator.py
# LIVE_API=0 => SKIP (dry run) with a results/llamator/DRY_RUN.json artifact.
```

Live mode:
```bash
# set LIVE_API=1 in .env
python -m pip install -U llamator
python security/llamator/run_llamator.py
```

### Non-leak check
```bash
bash scripts/non_leak_check.sh
```
<!-- ODR_OPENROUTER_SNIPPET_END -->
