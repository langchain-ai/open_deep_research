# ============================================================
# Stage 1: Build React frontend
# ============================================================
FROM node:18-alpine AS frontend-builder

WORKDIR /frontend

# Copy package files first (Docker layer caching)
COPY ai-research-assistant/package.json ai-research-assistant/package-lock.json ./
COPY ai-research-assistant/.npmrc ./.npmrc

# Install dependencies (clean install for reproducible builds)
RUN npm ci --ignore-scripts

# Copy frontend source and config
COPY ai-research-assistant/src/ ./src/
COPY ai-research-assistant/public/ ./public/
COPY ai-research-assistant/tailwind.config.js ./
COPY ai-research-assistant/postcss.config.js ./

# Build production bundle
ENV NODE_OPTIONS=--max-old-space-size=2048
RUN npm run build


# ============================================================
# Stage 2: Python application runtime
# ============================================================
FROM python:3.11-slim AS runtime

# System dependencies:
#   - libmagic1: python-magic file type detection
#   - tesseract-ocr: OCR for image-based PDFs
#   - ffmpeg: audio/video processing (moviepy, whisper)
#   - libgl1: OpenCV headless rendering
#   - gcc/build-essential: compile native Python extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    libmagic1 \
    tesseract-ocr \
    ffmpeg \
    libgl1 \
    gcc \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv (fast Python package manager)
RUN pip install --no-cache-dir uv

# Create non-root user (required by OpenShift security context)
RUN groupadd -g 1001 appgroup && \
    useradd -u 1001 -g appgroup -m -s /bin/bash appuser

WORKDIR /app

# Copy dependency files first (Docker layer caching)
COPY pyproject.toml uv.lock ./

# Install Python dependencies via uv
RUN uv sync --frozen --no-dev

# Copy backend application source
COPY app.py ./
COPY llm_clients.py ./
COPY orchestrator.py ./
COPY session_store.py ./
COPY src/ ./src/
COPY services/ ./services/
COPY routers/ ./routers/
COPY extensions/ ./extensions/
COPY models/ ./models/

# Copy React production build from Stage 1
COPY --from=frontend-builder /frontend/build/ ./ai-research-assistant/build/

# Create runtime directories (writable by appuser)
RUN mkdir -p outputs/reports outputs/charts uploads && \
    chown -R appuser:appgroup /app

# Switch to non-root user
USER appuser

# Expose FastAPI port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/')" || exit 1

# Start the application
# - workers=1: LLM calls are long-running; scale via pod replicas, not workers
# - host=0.0.0.0: accept connections from outside the container
CMD ["uv", "run", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
