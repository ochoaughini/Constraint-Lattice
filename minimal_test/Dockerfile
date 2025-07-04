# syntax=docker/dockerfile:1

FROM python:3.11-slim AS base

# Set working directory
WORKDIR /app

FROM base AS builder

# Install dependencies in a virtual environment, using pip cache
COPY --link requirements.txt ./
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m venv .venv && \
    .venv/bin/pip install -r requirements.txt

# Copy application code
COPY --link . .

FROM base AS final

# Create a non-root user
RUN addgroup --system appuser && adduser --system --ingroup appuser appuser

# Copy virtual environment from builder
COPY --from=builder /app/.venv /app/.venv
# Copy application code from builder
COPY --from=builder /app /app

# Set environment to use venv
ENV PATH="/app/.venv/bin:$PATH"

USER appuser

EXPOSE 8000

CMD ["sh", "-c", "exec gunicorn -k uvicorn.workers.UvicornWorker -w 4 -b 0.0.0.0:${PORT:-8000} main:app"]
