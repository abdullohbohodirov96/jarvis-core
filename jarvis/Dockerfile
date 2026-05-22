# =============================================================================
# JARVIS Desktop AI Assistant — Dockerfile
#
# Multi-stage build:
#   Stage 1 (builder): Install Python dependencies in a virtual environment.
#   Stage 2 (runtime): Copy the venv and application code into a slim image.
#
# Default CMD runs in --api-only mode because Docker containers typically do
# not have access to a microphone or display.  For local voice use, run JARVIS
# directly on the host with:
#   python jarvis/main.py
# =============================================================================

# ---------------------------------------------------------------------------
# Stage 1 — builder
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS builder

WORKDIR /build

# Install C build dependencies needed by PyAudio, soundfile, scipy, etc.
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        g++ \
        portaudio19-dev \
        libsndfile1-dev \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy only the requirements first so Docker can cache the pip layer.
COPY requirements.txt .

# Create a venv and install everything into it.
RUN python -m venv /venv \
    && /venv/bin/pip install --upgrade pip wheel \
    && /venv/bin/pip install --no-cache-dir -r requirements.txt


# ---------------------------------------------------------------------------
# Stage 2 — runtime
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS runtime

# Runtime OS packages:
#   ffmpeg        — required by Whisper for audio decoding
#   portaudio19-dev, libsndfile1 — shared libs for PyAudio / soundfile
#   libgl1        — required by PyAutoGUI on Linux (even in headless mode)
#   xvfb          — virtual framebuffer for PyAutoGUI when no display present
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        portaudio19-dev \
        libsndfile1 \
        libgl1 \
        xvfb \
    && rm -rf /var/lib/apt/lists/*

# Copy the virtual environment from the builder stage.
COPY --from=builder /venv /venv

# Make the venv's Python the default.
ENV PATH="/venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Copy application source.
COPY . .

# Create directories that JARVIS writes to at runtime.
RUN mkdir -p /tmp/jarvis_screenshots

# Expose the FastAPI REST port.
EXPOSE 8765

# ---------------------------------------------------------------------------
# Default command
# ---------------------------------------------------------------------------
# Runs JARVIS in API-only mode (no microphone, no display needed).
# Voice features require running on the host; see README.md.
#
# Override at runtime:
#   docker run jarvis python main.py --no-voice
CMD ["python", "main.py", "--no-voice", "--api-only"]
