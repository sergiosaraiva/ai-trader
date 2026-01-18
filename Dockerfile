# AI Trader Backend - Production Dockerfile
# Optimized for Railway deployment with volume support

FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PORT=8001

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements-api.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements-api.txt

# Copy application code
COPY src/ ./src/
COPY models/ ./models/
COPY configs/ ./configs/

# Copy read-only data (historical price and sentiment data for predictions)
# These don't change at runtime
COPY data/forex/ ./data/forex/
COPY data/sentiment/ ./data/sentiment/

# Create directories for runtime data
# The /app/data/db directory will be overlaid by Railway volume mount
RUN mkdir -p /app/logs /app/data/db /app/data/cache /app/data/storage

# Expose port
EXPOSE 8001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8001/health', timeout=5)" || exit 1

# Run the application
# Listen on all interfaces (:: for IPv4/IPv6 compatibility with Railway)
CMD ["uvicorn", "src.api.main:app", "--host", "::", "--port", "8001"]
