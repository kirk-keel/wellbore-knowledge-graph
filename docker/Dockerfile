# ==============================================================================
# Wellbore Engineering Knowledge Graph - Production Containerfile
# ==============================================================================
FROM python:3.12-slim AS runtime

# Prevent Python from writing .pyc and buffer stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency specifications
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create non-root application user for container security
RUN groupadd -g 10001 appgroup && \
    useradd -u 10001 -g appgroup -s /bin/bash -m appuser

# Copy application code, configs, and assets
COPY src/ ./src/
COPY configs/ ./configs/
COPY data/ ./data/
COPY scripts/ ./scripts/
COPY docker/entrypoint.sh ./entrypoint.sh
COPY pyproject.toml .

RUN chmod +x ./entrypoint.sh && \
    mkdir -p logs && \
    chown -R appuser:appgroup /app

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=20s --timeout=5s --retries=3 --start-period=10s \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

ENTRYPOINT ["./entrypoint.sh"]
