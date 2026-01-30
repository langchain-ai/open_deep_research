<!-- ODR_SECURITY_SNIPPET_START -->
## Runtime LLM configuration (MODE-A via OpenRouter)

This repository supports MODE-A (API) using an OpenAI-compatible endpoint.
Default provider in `.env.example` is OpenRouter (`https://openrouter.ai/api/v1`).

### Secrets hygiene (mandatory)
- Put keys only in `.env` (gitignored), environment variables, or CI secrets.
- Never paste keys into chat, issues, PRs, or logs.

### LIVE_API safety switch
Default is `LIVE_API=0`.
- `LIVE_API=0` => LLAMATOR live calls are skipped (dry run) with an explanation.
- `LIVE_API=1` => enables real API calls (requires valid keys).

### Non-leak verification
Run:
```bash
bash scripts/non_leak_check.sh
```
<!-- ODR_SECURITY_SNIPPET_END -->
