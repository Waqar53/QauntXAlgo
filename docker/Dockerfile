# QuantXalgo Dockerfile
# Multi-stage build for optimized production image

# =============================================================================
# STAGE 1: Builder
# =============================================================================
FROM python:3.11-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY pyproject.toml README.md ./
COPY quantxalgo/__init__.py ./quantxalgo/

RUN pip install --no-cache-dir --upgrade pip && \
    pip wheel --no-cache-dir --wheel-dir /app/wheels -e .

# =============================================================================
# STAGE 2: Production
# =============================================================================
FROM python:3.11-slim as production

WORKDIR /app

# Create non-root user for security
RUN groupadd --gid 1000 quantxalgo && \
    useradd --uid 1000 --gid quantxalgo --shell /bin/bash --create-home quantxalgo

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy wheels from builder and install
COPY --from=builder /app/wheels /wheels
RUN pip install --no-cache-dir /wheels/*.whl && rm -rf /wheels

# Copy application code
COPY --chown=quantxalgo:quantxalgo . .

# Create necessary directories
RUN mkdir -p /app/data /app/logs /app/data/cache && \
    chown -R quantxalgo:quantxalgo /app

# Switch to non-root user
USER quantxalgo

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    QUANTXALGO_ENV=production \
    API_HOST=0.0.0.0 \
    API_PORT=8000

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command
CMD ["python", "-m", "uvicorn", "quantxalgo.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

# =============================================================================
# STAGE 3: Development
# =============================================================================
FROM python:3.11-slim as development

WORKDIR /app

# Install dev dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy and install dependencies
COPY pyproject.toml README.md ./
COPY quantxalgo/__init__.py ./quantxalgo/

RUN pip install --no-cache-dir -e ".[dev,research]"

# Copy application code
COPY . .

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    QUANTXALGO_ENV=development \
    DEBUG=true

# Expose port
EXPOSE 8000

# Default command for development
CMD ["uvicorn", "quantxalgo.api.main:app", "--reload", "--host", "0.0.0.0", "--port", "8000"]
