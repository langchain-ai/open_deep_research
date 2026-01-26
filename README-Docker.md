# Open Deep Research - Docker Setup

This guide covers running Open Deep Research using Docker and Docker Compose.

## Prerequisites

- Docker (version 20.10 or higher)
- Docker Compose (version 2.0 or higher)
- At least one LLM API key (OpenAI, Anthropic, or Google)
- Tavily API key for search functionality

## Quick Start

### 1. Clone and Setup

```bash
git clone https://github.com/langchain-ai/open_deep_research.git
cd open_deep_research
```

### 2. Configure Environment

Copy the example environment file and add your API keys:

```bash
cp .env.docker.example .env
```

Edit `.env` and add your API keys:

```bash
# Minimum required for basic functionality
OPENAI_API_KEY=sk-your-key-here
TAVILY_API_KEY=tvly-your-key-here

# Optional: Add other providers
ANTHROPIC_API_KEY=sk-ant-your-key-here
GOOGLE_API_KEY=your-key-here
```

### 3. Build and Run

```bash
docker-compose up -d
```

This will:
- Build the Docker image (first run only)
- Start the container
- Expose the LangGraph API on port 2024

### 4. Access the Application

**Development Mode (default):**
- API: http://localhost:2024
- Studio UI: https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024
- API Docs: http://localhost:2024/docs

**Production Mode:**
- API only: http://localhost:2024

## Configuration

### Modes

Set the `MODE` environment variable in `.env` or docker-compose.yml:

```bash
MODE=development  # Includes LangGraph Studio UI (default)
MODE=production   # API-only mode
```

### Custom Port

Change the port mapping in docker-compose.yml:

```yaml
ports:
  - "8080:2024"  # Maps host port 8080 to container port 2024
```

Or set via environment variable:
```bash
PORT=8080 docker-compose up
```

## Usage Examples

### Starting the Container

```bash
# Start in background
docker-compose up -d

# Start with logs
docker-compose up

# Rebuild image
docker-compose up -d --build
```

### Stopping the Container

```bash
# Stop containers
docker-compose down

# Stop and remove volumes
docker-compose down -v
```

### Viewing Logs

```bash
# View all logs
docker-compose logs -f

# View specific service
docker-compose logs -f open-deep-research
```

### Interactive Shell

```bash
docker-compose exec open-deep-research bash
```

## Environment Variables Reference

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | Yes* | OpenAI API key (one LLM provider required) |
| `ANTHROPIC_API_KEY` | No | Anthropic API key |
| `GOOGLE_API_KEY` | No | Google/Vertex AI API key |
| `TAVILY_API_KEY` | Yes | Tavily search API key |
| `EXA_API_KEY` | No | Exa search API key |
| `LANGSMITH_API_KEY` | No | LangSmith tracing API key |
| `LANGSMITH_PROJECT` | No | LangSmith project name |
| `LANGSMITH_TRACING` | No | Enable LangSmith tracing |
| `MODE` | No | `development` or `production` (default: development) |
| `PORT` | No | Host port to expose (default: 2024) |

*At least one LLM provider API key is required.

## Troubleshooting

### Container Won't Start

Check logs:
```bash
docker-compose logs open-deep-research
```

Common issues:
- Missing API keys in `.env` file
- Port 2024 already in use
- Insufficient system resources

### Health Check Failing

The health check probes the API docs endpoint. If it fails:
- Check if port is accessible: `curl http://localhost:2024/docs`
- Verify environment variables are set correctly
- Check container logs for errors

### Slow Build Times

First build takes longer due to dependency installation. Subsequent builds use Docker layer caching.

### Volume Mount Issues

If source code changes aren't reflected:
- Stop the container: `docker-compose down`
- Restart: `docker-compose up -d`

## Development Workflow

For development with hot-reload:

1. Keep the container running
2. Edit source files locally (mounted volume)
3. Restart the container to apply changes:
   ```bash
   docker-compose restart
   ```

## Production Deployment

For production deployment:

1. Set `MODE=production` in `.env`
2. Remove or comment out the `volumes` section in docker-compose.yml
3. Consider using secrets for API keys
4. Add resource limits:
   ```yaml
   deploy:
     resources:
       limits:
         cpus: '2'
         memory: 4G
   ```

## Additional Resources

- [Open Deep Research Documentation](https://github.com/langchain-ai/open_deep_research)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [Docker Documentation](https://docs.docker.com/)

## License

MIT License - See [LICENSE](LICENSE) file.
