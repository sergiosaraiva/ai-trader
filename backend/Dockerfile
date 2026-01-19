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
COPY data/forex/ ./data/forex/
COPY data/sentiment/ ./data/sentiment/

# Create data directories
RUN mkdir -p /app/logs /app/data/db

ENV PYTHONPATH=/app
EXPOSE 8001

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8001"]
