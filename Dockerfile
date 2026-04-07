# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 — Builder: install all Python dependencies
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy

WORKDIR /build

# OS build deps
RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential curl \
 && rm -rf /var/lib/apt/lists/*

# Install uv (fast package installer)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Install Python deps into an isolated prefix — kept in /build/.venv
COPY requirements.txt .
# Install CPU-only torch first to prevent sentence-transformers from
# pulling in 2GB+ of CUDA/GPU wheels (nvidia-nccl, nvidia-cublas, etc.)
RUN uv venv /build/.venv \
 && uv pip install --python /build/.venv/bin/python \
      torch --index-url https://download.pytorch.org/whl/cpu \
 && uv pip install --python /build/.venv/bin/python -r requirements.txt

# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 — Runtime: slim final image, no build tools
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH="/app" \
    PATH="/app/.venv/bin:$PATH" \
    PORT=8080 \
    ENV=production

WORKDIR /app

# Runtime OS deps (PDF extraction, libgomp for sentence-transformers)
RUN apt-get update \
 && apt-get install -y --no-install-recommends poppler-utils libgomp1 \
 && rm -rf /var/lib/apt/lists/*

# Copy the venv from builder — no build tools in final image
COPY --from=builder /build/.venv /app/.venv

# Copy application source
COPY . .

# Non-root user for security
RUN addgroup --system appgroup \
 && adduser --system --ingroup appgroup --no-create-home appuser \
 && chown -R appuser:appgroup /app

USER appuser

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')" || exit 1

# Production: 2 workers, no reload
# Development: override with `--reload --workers 1`
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "2"]
