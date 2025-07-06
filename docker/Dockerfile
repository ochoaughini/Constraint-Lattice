# syntax=docker/dockerfile:1

# Specify the platform explicitly
ARG TARGETPLATFORM
FROM --platform=$TARGETPLATFORM python:3.9-slim AS base

# Print build information for debugging
RUN echo "Building for platform: $TARGETPLATFORM" && \
    uname -a && \
    dpkg --print-architecture

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_CACHE_DIR=/root/.cache/pip

WORKDIR /app

# --- Builder stage: install dependencies into a venv ---
FROM base AS builder

# Copy only dependency files first for better cache usage
COPY requirements.txt ./

# Create venv and install dependencies (core + web server)
RUN --mount=type=cache,target=$PIP_CACHE_DIR \
    python -m venv .venv && \
    .venv/bin/pip install --no-cache-dir -r requirements.txt fastapi uvicorn requests streamlit

# --- Final stage: copy app code and venv, set up non-root user ---
FROM base AS final

# Create a non-root user and group
RUN addgroup --system appgroup && adduser --system --ingroup appgroup appuser

WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /app/.venv /app/.venv

# Copy the rest of the application code (excluding .git, .env, etc. via .dockerignore)
COPY . .

# Set PATH to use the venv
ENV PATH="/app/.venv/bin:$PATH"

# Set the project root to the Python path
# Include project src directory so "import constraint_lattice" works at runtime
ENV PYTHONPATH=/app/src:/app

# Expose ports
EXPOSE 8501
EXPOSE 8000

# Container health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 CMD /bin/sh -c 'curl -f http://localhost:${PORT:-8000}/health || exit 1'

USER appuser

# Start services
CMD ["sh", "-c", "streamlit run ui/audit_viewer.py --server.port=8501 & uvicorn api.main:app --host 0.0.0.0 --port 8000"]
