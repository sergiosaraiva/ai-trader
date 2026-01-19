# Simple Backend Dockerfile
FROM python:3.12-slim

WORKDIR /app

# Install dependencies
RUN apt-get update && apt-get install -y --no-install-recommends gcc g++ && rm -rf /var/lib/apt/lists/*
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# Copy application
COPY src/ ./src/
COPY models/ ./models/
COPY configs/ ./configs/
COPY scripts/ ./scripts/
COPY data/forex/ ./data/forex/
COPY data/sentiment/ ./data/sentiment/

# Copy and prepare entrypoint
COPY docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh

# Create data directories
RUN mkdir -p /app/logs /app/data/db

ENV PYTHONPATH=/app
EXPOSE 8001

ENTRYPOINT ["/docker-entrypoint.sh"]
