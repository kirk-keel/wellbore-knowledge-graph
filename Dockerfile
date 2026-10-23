# ==============================================================================
# Wellbore Engineering Knowledge Graph - Container Image
# ==============================================================================
FROM python:3.12-slim AS runtime

# Prevent Python from writing .pyc and buffer stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

WORKDIR /app

# Install minimal system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy dependencies manifest and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create non-root application user for container security
RUN groupadd -g 10001 appgroup && \
    useradd -u 10001 -g appgroup -s /bin/bash -m appuser

# Copy application assets
COPY src/ ./src/
COPY configs/ ./configs/
COPY data/ ./data/
COPY scripts/ ./scripts/
COPY pyproject.toml .

# Create logs directory and assign permissions
RUN mkdir -p logs && chown -R appuser:appgroup /app

USER appuser

EXPOSE 8000

# Healthcheck for FastAPI service
HEALTHCHECK --interval=20s --timeout=5s --retries=3 --start-period=10s \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

# Launch uvicorn server
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
