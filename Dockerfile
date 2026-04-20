# ============================================================
# Bias-Free Hiring Pipeline — Multi-Stage Dockerfile
# Sprint 2 — Production image with pre-cached ML models
#
# Stages:
#   builder  — installs all Python deps into a venv
#   models   — pre-downloads spaCy + SBERT models into cache
#   runtime  — lean final image: venv + models + app code
#
# Build:
#   docker build -t hiring-pipeline .
#
# Run:
#   docker run -p 5000:5000 \
#     -v $(pwd)/raw_cvs:/app/raw_cvs \
#     -v $(pwd)/data:/app/data \
#     hiring-pipeline
#
# Skip large models (CI / resource-constrained builds):
#   docker build --build-arg SKIP_LARGE_MODELS=1 -t hiring-pipeline .
# ============================================================

# ── Stage 1: builder ─────────────────────────────────────────
FROM python:3.11-slim AS builder

# Build-time arg: set to 1 in CI to use the small spaCy model
ARG SKIP_LARGE_MODELS=0

WORKDIR /build

# Install OS-level build dependencies (needed for some wheels)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        g++ \
        libffi-dev \
        libssl-dev \
        git \
    && rm -rf /var/lib/apt/lists/*

# Create a virtualenv so we can copy it cleanly into the runtime stage
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip first — avoids resolver bugs with older pip
RUN pip install --no-cache-dir --upgrade pip wheel setuptools

# Copy only requirements first — Docker layer-caches this step so a
# code change doesn't re-install all packages on rebuild
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Stage 2: model downloader ─────────────────────────────────
FROM builder AS models

ARG SKIP_LARGE_MODELS=0
ENV SKIP_LARGE_MODELS=${SKIP_LARGE_MODELS}
ENV PATH="/opt/venv/bin:$PATH"

# HuggingFace / sentence-transformers cache directory
ENV HF_HOME=/opt/models/huggingface
ENV SENTENCE_TRANSFORMERS_HOME=/opt/models/sentence_transformers
ENV TRANSFORMERS_CACHE=/opt/models/huggingface

COPY download_models.py .

# Pre-download all ML models — cached in /opt/models inside the image.
# This layer is invalidated only when download_models.py changes,
# not on every code change.
RUN python download_models.py ; exit 0
# Note: we use '; exit 0' so a network timeout during build does NOT
# fail the entire image — the pipeline will fall back to runtime download.

# ── Stage 3: runtime (final, lean image) ─────────────────────
FROM python:3.11-slim AS runtime

LABEL maintainer="Anish Idhayan"
LABEL description="Bias-Free Hiring Pipeline — Flask Dashboard"
LABEL version="1.0.0"

# Runtime OS dependencies only (no build tools)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the pre-built virtualenv from the builder stage
COPY --from=builder /opt/venv /opt/venv

# Copy the pre-downloaded models from the models stage
COPY --from=models /opt/models /opt/models

# Set model cache paths so the pipeline finds the pre-downloaded models
ENV PATH="/opt/venv/bin:$PATH"
ENV HF_HOME=/opt/models/huggingface
ENV SENTENCE_TRANSFORMERS_HOME=/opt/models/sentence_transformers
ENV TRANSFORMERS_CACHE=/opt/models/huggingface
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Create non-root user for security
RUN groupadd --gid 1001 pipeline \
    && useradd --uid 1001 --gid pipeline --shell /bin/bash --create-home pipeline

# Copy application source
COPY --chown=pipeline:pipeline . /app

# Create writable data directories and set ownership
RUN mkdir -p \
        /app/raw_cvs \
        /app/anonymized_cvs \
        /app/parsed \
        /app/vault \
        /app/explanations \
        /app/audit \
        /app/logs \
        /app/data \
    && chown -R pipeline:pipeline /app

# Switch to non-root user
USER pipeline

# Expose Flask port
EXPOSE 5000

# Docker health check — polls /health every 30 s
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Default command: launch the Flask dashboard
# Override with: docker run hiring-pipeline python main.py --jd-file jd.txt
CMD ["python", "module5.py", "--host", "0.0.0.0", "--port", "5000"]
