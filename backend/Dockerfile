# CPU-Only Backend Dockerfile
FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends gcc g++ && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install wheel
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install core scientific packages first (large downloads)
RUN pip install --no-cache-dir --default-timeout=600 numpy "pandas>=2.0.0,<4.0.0" "scipy>=1.10.0,<2.0.0"
RUN pip install --no-cache-dir --default-timeout=600 "pyarrow>=14.0.0,<25.0.0"
RUN pip install --no-cache-dir --default-timeout=600 "scikit-learn>=1.3.0,<2.0.0"

# Install XGBoost (includes CUDA libs but will use CPU if no GPU present)
# Split into smaller retries to handle network timeouts
RUN pip install --no-cache-dir --default-timeout=900 --retries=10 "xgboost>=2.0.0,<3.0.0"

# Install remaining packages
COPY requirements-docker.txt .
RUN pip install --no-cache-dir --default-timeout=300 -r requirements-docker.txt

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
