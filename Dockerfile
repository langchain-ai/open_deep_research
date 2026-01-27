FROM python:3.11-slim AS base

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

FROM base AS runtime

RUN pip install --no-cache-dir uv

RUN groupadd -r appuser && useradd -r -g appuser -m appuser

RUN mkdir -p /home/appuser/.cache /home/appuser/.local && \
    chown -R appuser:appuser /home/appuser && \
    chown -R appuser:appuser /app

COPY pyproject.toml uv.lock ./
COPY src/ ./src/
COPY tests/ ./tests/

RUN uv sync

COPY --chown=appuser:appuser langgraph.json ./
RUN chown -R appuser:appuser /app

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV HOME=/home/appuser
ENV UV_CACHE_DIR=/home/appuser/.cache/uv

USER appuser

EXPOSE 2024

ARG MODE=development
ENV MODE=${MODE}

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://127.0.0.1:2024/health', timeout=5)" || exit 1

CMD uv run langgraph dev --allow-blocking --host 0.0.0.0
