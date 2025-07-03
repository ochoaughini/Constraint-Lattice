# syntax=docker/dockerfile:1

FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_CACHE_DIR=/root/.cache/pip

WORKDIR /app

# --- Builder stage: install dependencies into a venv ---
FROM base AS builder

# Copy only dependency files first for better cache usage
COPY --link pyproject.toml ./
COPY --link requirements-lock.txt ./

# Create venv and install dependencies (core + web server)
RUN --mount=type=cache,target=$PIP_CACHE_DIR \
    python -m venv .venv && \
    .venv/bin/pip install -r requirements-lock.txt gunicorn uvicorn[standard]

# --- Final stage: copy app code and venv, set up non-root user ---
FROM base AS final

# Create a non-root user and group
RUN addgroup --system appgroup && adduser --system --ingroup appgroup appuser

WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /app/.venv /app/.venv

# Copy the rest of the application code (excluding .git, .env, etc. via .dockerignore)
COPY --link . .

# Set PATH to use the venv
ENV PATH="/app/.venv/bin:$PATH"

# Set the project root to the Python path
# Include project src directory so "import constraint_lattice" works at runtime
ENV PYTHONPATH=/app/src:/app

# Expose the default port
EXPOSE 8000

# Container health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 CMD curl -f http://localhost:8000/health || exit 1

USER appuser

# Start FastAPI via Gunicorn with 4 Uvicorn workers, using $PORT if set (Cloud Run convention)
CMD ["sh", "-c", "exec gunicorn -k uvicorn.workers.UvicornWorker -w 4 -b 0.0.0.0:${PORT:-8000} saas.main:app"]
