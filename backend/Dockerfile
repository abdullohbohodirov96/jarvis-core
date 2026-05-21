# =============================================================================
# JARVIS API — Multi-stage Dockerfile
# =============================================================================
#
# Stage 1 (builder): Install Python deps and compile wheels.
# Stage 2 (production): Lean runtime image with non-root user.
#
# Build:
#   docker build -t jarvis-api:latest ./backend
# Run:
#   docker run -p 8000:8000 --env-file ../.env jarvis-api:latest
# =============================================================================

# ---------------------------------------------------------------------------
# Stage 1 — builder
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS builder

# Keeps Python from generating .pyc files and enables unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Build-time dependencies (compilers, headers required by some Python packages)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        g++ \
        libffi-dev \
        libssl-dev \
        libpq-dev \
        libsndfile1-dev \
        portaudio19-dev \
        ffmpeg \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copy only requirements first to leverage layer caching — the big pip install
# only re-runs when requirements.txt actually changes.
COPY requirements.txt .

RUN pip install --upgrade pip setuptools wheel && \
    pip install --prefix=/install --no-cache-dir -r requirements.txt


# ---------------------------------------------------------------------------
# Stage 2 — production runtime
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS production

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    # Whisper downloads models here; mount a volume in prod to persist them.
    WHISPER_CACHE_DIR=/app/data/whisper_cache \
    PATH="/install/bin:$PATH"

# Runtime-only system packages (no compilers needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libsndfile1 \
        portaudio19-dev \
        libpq5 \
        curl \
        tini \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder stage
COPY --from=builder /install /usr/local

# Create a non-root application user
RUN groupadd --gid 10001 jarvis && \
    useradd --uid 10001 --gid jarvis --shell /bin/bash --create-home jarvis

WORKDIR /app

# Create directories the app expects to exist and give ownership to jarvis
RUN mkdir -p /app/data/whisper_cache \
             /app/data/sessions \
             /app/static \
             /app/logs && \
    chown -R jarvis:jarvis /app

# Copy application source code
COPY --chown=jarvis:jarvis . /app/

# Drop privileges
USER jarvis

# Document the port (actual binding is handled by the command/entrypoint)
EXPOSE 8000

# ----------------------------------------------------------
# Healthcheck — polls the lightweight root endpoint
# ----------------------------------------------------------
HEALTHCHECK \
    --interval=30s \
    --timeout=10s \
    --start-period=40s \
    --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# ----------------------------------------------------------
# Entrypoint
# ----------------------------------------------------------
# Use tini as PID 1 to handle signal forwarding and zombie reaping.
ENTRYPOINT ["tini", "--"]

# Default command — override in docker-compose for workers/beat/flower.
CMD ["uvicorn", "app.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--loop", "uvloop", \
     "--http", "httptools", \
     "--log-level", "info", \
     "--access-log"]
