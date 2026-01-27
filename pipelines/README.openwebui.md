# OpenWebUI Integration for Open Deep Research

This guide explains how to integrate Open Deep Research with OpenWebUI using the custom pipeline.

## Prerequisites

- OpenWebUI running in Docker
- Open Deep Research deployed on the same Docker network
- API keys configured in Open Deep Research:
  - `ANTHROPIC_API_KEY` (required for default models)
  - `TAVILY_API_KEY` (required for default search)

## Architecture

```
┌─────────────────────┐         ┌─────────────────────────┐
│                     │         │                         │
│      OpenWebUI      │────────▶│   Open Deep Research    │
│                     │         │    (LangGraph API)      │
│   Your Browser      │         │                         │
│                     │         │   Container Name:       │
│                     │         │   open-deep-research    │
│                     │         │   Internal Port: 2024   │
└─────────────────────┘         └─────────────────────────┘
          │                               │
          └───────────┬───────────────────┘
                      │
              Docker Network
               (webui-net)
```

Both containers communicate over the internal Docker network. No ports are exposed externally for security.

## Installation Steps

### Step 1: Ensure Network Connectivity

Both containers must be on the same Docker network. By default, Open Deep Research connects to a network named `webui-net`.

**Check your OpenWebUI network name:**
```bash
docker network ls | grep -i webui
```

**If your OpenWebUI uses a different network**, update the `.env` file:
```bash
EXTERNAL_NETWORK=your-network-name
```

### Step 2: Configure API Keys

Edit the `.env` file in the Open Deep Research directory:

```bash
# Required for default configuration
ANTHROPIC_API_KEY=sk-ant-...
TAVILY_API_KEY=tvly-...

# Optional: For tracing/debugging
LANGSMITH_API_KEY=lsv2_...
LANGSMITH_PROJECT=deep-research
LANGSMITH_TRACING=true
```

### Step 3: Deploy Open Deep Research

```bash
cd open_deep_research

# First time or after changes
docker-compose up -d --build

# Verify it's running
docker logs -f open-deep-research
```

You should see:
```
Welcome to
╦  ┌─┐┌┐┌┌─┐╔═╗┬─┐┌─┐┌─┐┬ ┬
║  ├─┤││││ ┬║ ╦├┬┘├─┤├─┘├─┤
╩═╝┴ ┴┘└┘└─┘╚═╝┴└─┴ ┴┴  ┴ ┴

- API: http://0.0.0.0:2024
```

**No auth errors should appear** (authentication has been disabled for internal network use).

### Step 4: Install the Pipeline in OpenWebUI

1. Open OpenWebUI in your browser
2. Log in as an administrator
3. Go to **Admin Panel** (gear icon) → **Settings** → **Pipelines**
4. Click the **"+"** button to add a new pipeline
5. Select **"Upload from file"**
6. Navigate to and select: `pipelines/deep_research_pipeline.py`
7. Click **Save**

The pipeline should now appear in the list as "Deep Research".

### Step 5: Configure Pipeline Settings (Optional)

After installation, you can customize the pipeline:

1. Go to **Admin Panel** → **Settings** → **Pipelines**
2. Click on **"Deep Research"** to expand settings
3. Adjust the **Valves** as needed (see Configuration Reference below)
4. Click **Save**

### Step 6: Use Deep Research

1. Start a **new chat** in OpenWebUI
2. Click on the model selector dropdown
3. Select **"Deep Research"**
4. Enter your research query
5. Wait for the comprehensive research report (typically 1-5 minutes)

## Configuration Reference

### Connection Settings

| Valve | Default | Description |
|-------|---------|-------------|
| `LANGGRAPH_URL` | `http://open-deep-research:2024` | LangGraph server URL. Use container name for Docker network. |

> **Note:** The `ASSISTANT_ID` is set statically to `"Deep Researcher"` to match the graph name in `langgraph.json` and cannot be changed via Valves.

### Polling Settings

| Valve | Default | Description |
|-------|---------|-------------|
| `POLL_INTERVAL` | `3` | Seconds between status checks |
| `MAX_WAIT_TIME` | `600` | Maximum wait time in seconds (10 min) |

### Model Configuration

| Valve | Default | Description |
|-------|---------|-------------|
| `RESEARCH_MODEL` | `anthropic:claude-sonnet-4-20250514` | Model for conducting research |
| `RESEARCH_MODEL_MAX_TOKENS` | `16000` | Max output tokens for research |
| `COMPRESSION_MODEL` | `anthropic:claude-sonnet-4-20250514` | Model for compressing findings |
| `COMPRESSION_MODEL_MAX_TOKENS` | `8192` | Max output tokens for compression |
| `FINAL_REPORT_MODEL` | `anthropic:claude-sonnet-4-20250514` | Model for final report |
| `FINAL_REPORT_MODEL_MAX_TOKENS` | `16000` | Max output tokens for report |
| `SUMMARIZATION_MODEL` | `anthropic:claude-3-5-haiku-latest` | Model for summarizing search results |
| `SUMMARIZATION_MODEL_MAX_TOKENS` | `4096` | Max output tokens for summaries |

### Research Behavior

| Valve | Default | Description |
|-------|---------|-------------|
| `SEARCH_API` | `tavily` | Search provider: `tavily`, `openai`, `anthropic`, `none` |
| `ALLOW_CLARIFICATION` | `true` | Ask clarifying questions before research |
| `MAX_CONCURRENT_RESEARCH_UNITS` | `5` | Parallel sub-agents (higher = faster, may hit rate limits) |
| `MAX_RESEARCHER_ITERATIONS` | `6` | Research depth iterations |
| `MAX_REACT_TOOL_CALLS` | `10` | Tool calls per researcher step |
| `MAX_CONTENT_LENGTH` | `50000` | Max webpage content before summarization |

## Usage Tips

### Ideal Research Queries

Deep Research excels at open-ended research questions:

**Good queries:**
- "What are the latest developments in quantum computing for 2025?"
- "Compare the environmental impact of electric vs hydrogen vehicles with recent data"
- "Explain the current state of AI regulation in the European Union"
- "What are the most promising cancer immunotherapy approaches being researched?"

**Less optimal queries:**
- "What is 2+2?" (too simple)
- "Write me a poem" (creative task, not research)
- "Summarize this document" (use a regular model)

### Understanding the Research Process

When you submit a query, the system:

1. **Clarification** (if enabled): May ask follow-up questions to refine scope
2. **Research Brief**: Generates a structured research plan
3. **Parallel Research**: Spawns multiple sub-researchers to gather information
4. **Compression**: Synthesizes findings from each researcher
5. **Final Report**: Generates a comprehensive, well-structured report

This typically takes **1-5 minutes** depending on query complexity and settings.

### Performance Tuning

**For faster results:**
- Decrease `MAX_RESEARCHER_ITERATIONS` (e.g., 3-4)
- Decrease `MAX_CONCURRENT_RESEARCH_UNITS` (e.g., 3)
- Use faster models (Haiku for summarization)

**For deeper research:**
- Increase `MAX_RESEARCHER_ITERATIONS` (e.g., 8-10)
- Increase `MAX_CONCURRENT_RESEARCH_UNITS` (e.g., 7-10)
- Increase `MAX_WAIT_TIME` accordingly

**For cost optimization:**
- Use Haiku for summarization (default)
- Use Sonnet for research (default)
- Only use Opus for final reports if highest quality needed

## Troubleshooting

### "Connection Error: Cannot connect to LangGraph server"

**Causes:**
- Open Deep Research container not running
- Containers on different Docker networks
- Wrong `LANGGRAPH_URL` in pipeline settings

**Solutions:**
```bash
# Check if container is running
docker ps | grep open-deep-research

# Check container logs
docker logs open-deep-research

# Verify network connectivity
docker network inspect webui-net

# Ensure both containers are on the same network
docker network connect webui-net open-deep-research
```

### "Research timed out"

**Causes:**
- Complex query requiring extended research
- Rate limiting from API providers
- Network issues

**Solutions:**
- Increase `MAX_WAIT_TIME` in pipeline valves (e.g., 900 for 15 minutes)
- Reduce `MAX_CONCURRENT_RESEARCH_UNITS` to avoid rate limits
- Simplify the research query

### "No research results received"

**Causes:**
- Query too vague or unclear
- API key issues
- Search API problems

**Solutions:**
- Check Open Deep Research logs: `docker logs open-deep-research`
- Verify API keys are set in `.env`
- Try a clearer, more specific query
- Check if `ALLOW_CLARIFICATION` is enabled

### Pipeline not appearing in OpenWebUI

**Solutions:**
- Ensure you uploaded the file as an administrator
- Check OpenWebUI logs for pipeline loading errors
- Try restarting OpenWebUI: `docker restart open-webui`

## Model Format Reference

Models are specified as `provider:model-name`:

### Anthropic Claude
- `anthropic:claude-sonnet-4-20250514` - Claude Sonnet 4 (balanced)
- `anthropic:claude-opus-4-5-20251101` - Claude Opus 4.5 (most capable)
- `anthropic:claude-3-5-haiku-latest` - Claude Haiku 3.5 (fast/cheap)

### OpenAI
- `openai:gpt-4.1` - GPT-4.1
- `openai:gpt-4.1-mini` - GPT-4.1 Mini (fast/cheap)
- `openai:gpt-4o` - GPT-4o

### Google
- `google:gemini-2.0-flash` - Gemini 2.0 Flash

## Search API Options

| Value | Description | Required Key |
|-------|-------------|--------------|
| `tavily` | Tavily Search (recommended) | `TAVILY_API_KEY` |
| `openai` | OpenAI native web search | `OPENAI_API_KEY` |
| `anthropic` | Anthropic native web search | `ANTHROPIC_API_KEY` |
| `none` | Disable web search | None |

## Security Notes

- The LangGraph API port is **not exposed externally**
- All communication happens within the Docker network
- API keys are stored in the Open Deep Research container only
- The pipeline does not store or transmit API keys
- No authentication is required because access is limited to internal network

## Updating

To update the pipeline after changes:

1. In OpenWebUI: **Admin Panel** → **Settings** → **Pipelines**
2. Delete the existing "Deep Research" pipeline
3. Upload the updated `deep_research_pipeline.py`
4. Reconfigure valves if needed

To update Open Deep Research:

```bash
cd open_deep_research
git pull  # or apply your changes
docker-compose down
docker-compose up -d --build
```
