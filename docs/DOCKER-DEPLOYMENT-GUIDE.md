# Docker Deployment Guide

Complete guide for deploying the AI Trading Agent system using Docker.

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Service Details](#service-details)
- [Development Workflow](#development-workflow)
- [Production Deployment](#production-deployment)
- [Monitoring & Troubleshooting](#monitoring--troubleshooting)
- [Security Considerations](#security-considerations)

## Architecture Overview

The system consists of four containerized services:

```
┌─────────────────────────────────────────────────────────────────┐
│                    DOCKER ARCHITECTURE                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   PostgreSQL (Port 5432)                                         │
│   └── Shared database for all services                          │
│                                                                  │
│   Backend API (Port 8001)                                        │
│   ├── FastAPI application                                       │
│   ├── Serves predictions and market data                        │
│   ├── Manages paper trading accounts                            │
│   └── Health check: /health                                     │
│                                                                  │
│   Agent (Port 8002)                                              │
│   ├── Autonomous trading agent                                  │
│   ├── Monitors predictions → Executes trades                    │
│   ├── Modes: simulation, paper, live                            │
│   └── Health check: /health                                     │
│                                                                  │
│   Frontend (Port 3001)                                           │
│   ├── React dashboard (nginx)                                   │
│   ├── Displays predictions and trading status                   │
│   └── Proxies API requests to backend                           │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**Dependency Order:**
```
postgres → backend → agent
             ↓
          frontend
```

All services communicate via the `ai-trader-network` Docker bridge network.

## Prerequisites

- **Docker**: Version 20.10 or higher
- **Docker Compose**: Version 2.0 or higher
- **System Resources**:
  - 4 GB RAM minimum (8 GB recommended)
  - 10 GB disk space (for models + data)
  - CPU: 2 cores minimum (4 cores recommended)

### Installation

**Linux:**
```bash
# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Install Docker Compose
sudo apt-get install docker-compose-plugin
```

**macOS:**
```bash
# Install Docker Desktop
brew install --cask docker
```

**Windows:**
Download and install [Docker Desktop for Windows](https://www.docker.com/products/docker-desktop)

## Quick Start

### 1. Clone Repository

```bash
git clone https://github.com/yourusername/ai-trader.git
cd ai-trader
```

### 2. Configure Environment

```bash
# Copy example environment file
cp .env.example .env

# Edit with your configuration
nano .env
```

**Minimum required changes:**
```bash
# Set a secure database password
POSTGRES_PASSWORD=your_secure_password_here

# Add your API keys
OPENAI_API_KEY=sk-...
FRED_API_KEY=...
```

### 3. Start Services

```bash
# Using Makefile (recommended)
make up

# Or directly with docker-compose
docker-compose up -d
```

### 4. Verify Deployment

```bash
# Check service health
make health

# View logs
make logs

# Check specific service
make logs-agent
```

### 5. Access Services

- **Frontend Dashboard**: http://localhost:3001
- **Backend API Docs**: http://localhost:8001/docs
- **Agent Health Check**: http://localhost:8002/health

## Configuration

### Environment Variables

Edit `.env` file to configure the system:

#### Database Configuration

```bash
POSTGRES_USER=aitrader
POSTGRES_PASSWORD=your_secure_password  # CHANGE THIS!
POSTGRES_DB=aitrader
DATABASE_URL=postgresql://aitrader:your_secure_password@postgres:5432/aitrader
```

#### API Keys

```bash
# OpenAI API (for trade explanations)
OPENAI_API_KEY=sk-...

# FRED API (for sentiment data)
FRED_API_KEY=...
```

#### Agent Configuration

```bash
# Trading Mode
AGENT_MODE=simulation  # simulation | paper | live

# Trading Parameters
AGENT_SYMBOL=EURUSD
AGENT_CONFIDENCE_THRESHOLD=0.70
AGENT_MAX_POSITION_SIZE=0.1
AGENT_USE_KELLY_SIZING=true
AGENT_INITIAL_CAPITAL=100000.0

# Timing
AGENT_CYCLE_INTERVAL=60  # Check every 60 seconds

# MT5 Credentials (paper/live modes only)
AGENT_MT5_LOGIN=
AGENT_MT5_PASSWORD=
AGENT_MT5_SERVER=

# Safety Limits
AGENT_MAX_CONSECUTIVE_LOSSES=5
AGENT_MAX_DRAWDOWN_PERCENT=10.0
AGENT_MAX_DAILY_LOSS_PERCENT=5.0
```

## Service Details

### PostgreSQL Database

**Container:** `ai-trader-postgres`
**Port:** 5432
**Image:** postgres:16-alpine

**Volumes:**
- `postgres_data:/var/lib/postgresql/data` (persistent)

**Health Check:**
```bash
docker-compose exec postgres pg_isready -U aitrader
```

**Access Shell:**
```bash
make db-shell
# Or: docker-compose exec postgres psql -U aitrader -d aitrader
```

### Backend API

**Container:** `ai-trader-backend`
**Port:** 8001
**Build:** `backend/Dockerfile`

**Volumes:**
- `./backend/models:/app/models:ro` (read-only)
- `./backend/data:/app/data` (read-write)

**Health Check:**
```bash
curl http://localhost:8001/health
```

**API Documentation:**
- Swagger UI: http://localhost:8001/docs
- ReDoc: http://localhost:8001/redoc

### Agent

**Container:** `ai-trader-agent`
**Port:** 8002
**Build:** `backend/Dockerfile.agent`

**Volumes:**
- `./backend/models:/app/models:ro` (read-only)
- `./backend/data/forex:/app/data/forex:ro` (read-only)
- `./backend/data/sentiment:/app/data/sentiment:ro` (read-only)
- `./backend/logs:/app/logs` (read-write)

**Health Check:**
```bash
curl http://localhost:8002/health
```

**Status Endpoint:**
```bash
curl http://localhost:8002/status | jq
```

**Important Notes:**
- **MT5 on Linux**: MT5 requires Windows. Docker containers run Linux, so only **simulation mode** works in Docker.
- **Paper/Live Trading**: Run agent natively on Windows or use WSL2 with proper MT5 setup.

### Frontend

**Container:** `ai-trader-frontend`
**Port:** 3001 (external), 80 (internal)
**Build:** `frontend/Dockerfile` (multi-stage: node → nginx)

**Health Check:**
```bash
curl http://localhost:3001/health
```

## Development Workflow

### Development Mode with Hot Reload

The `docker-compose.override.yml` file enables hot reload in development:

```bash
# Start with development overrides
make dev-up

# Source code changes are automatically reflected:
# - backend/src/* → backend container
# - frontend/src/* → requires npm rebuild (see below)
```

**Frontend Hot Reload:**
For live frontend development, run Vite locally instead:

```bash
# Stop frontend container
docker-compose stop frontend

# Run Vite dev server
cd frontend
npm install
npm run dev
# Access at http://localhost:5173
```

### Rebuilding Containers

```bash
# Rebuild all containers
make build

# Rebuild specific service
make build-backend
make build-agent
make build-frontend

# Rebuild and restart
make build && make up
```

### Viewing Logs

```bash
# All services
make logs

# Specific service
make logs-backend
make logs-agent
make logs-frontend

# Follow agent startup
make agent-logs-startup
```

### Database Operations

```bash
# Open PostgreSQL shell
make db-shell

# Run migrations
make db-migrate

# Backup database
docker-compose exec postgres pg_dump -U aitrader aitrader > backup.sql

# Restore database
docker-compose exec -T postgres psql -U aitrader aitrader < backup.sql
```

### Agent Operations

```bash
# View agent status
make agent-status

# Restart agent
make restart-agent

# Stop agent (for testing)
make agent-stop

# Start agent
make agent-start

# View agent logs with timestamps
docker-compose logs -f --timestamps agent
```

## Production Deployment

### 1. Production Environment File

Create `.env.production`:

```bash
# Strong passwords
POSTGRES_PASSWORD=<strong-random-password>

# Production database
DATABASE_URL=postgresql://aitrader:<password>@postgres:5432/aitrader

# API Keys
OPENAI_API_KEY=sk-...
FRED_API_KEY=...

# Environment
ENVIRONMENT=production

# Agent: Start with simulation mode
AGENT_MODE=simulation

# Safety limits (conservative)
AGENT_MAX_CONSECUTIVE_LOSSES=3
AGENT_MAX_DRAWDOWN_PERCENT=5.0
AGENT_MAX_DAILY_LOSS_PERCENT=2.0
```

### 2. Production Compose File

Create `docker-compose.prod.yml`:

```yaml
version: '3.8'

services:
  backend:
    restart: always
    volumes:
      - ./backend/models:/app/models:ro  # Read-only
      - backend_data:/app/data           # Named volume
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"

  agent:
    restart: always
    volumes:
      - ./backend/models:/app/models:ro
      - ./backend/data/forex:/app/data/forex:ro
      - ./backend/data/sentiment:/app/data/sentiment:ro
      - agent_logs:/app/logs
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"

  frontend:
    restart: always
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"

volumes:
  backend_data:
  agent_logs:
```

### 3. Deploy to Production

```bash
# Load production environment
export $(cat .env.production | xargs)

# Deploy with production overrides
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d

# Verify deployment
docker-compose ps
docker-compose logs -f
```

### 4. SSL/TLS with Reverse Proxy

**Using Nginx:**

```nginx
# /etc/nginx/sites-available/ai-trader
server {
    listen 443 ssl http2;
    server_name yourdomain.com;

    ssl_certificate /etc/letsencrypt/live/yourdomain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/yourdomain.com/privkey.pem;

    # Frontend
    location / {
        proxy_pass http://localhost:3001;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }

    # Backend API
    location /api/ {
        proxy_pass http://localhost:8001/api/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

## Monitoring & Troubleshooting

### Health Checks

```bash
# Automated health check
make health

# Manual checks
curl http://localhost:8001/health
curl http://localhost:8002/health
curl http://localhost:3001/health
```

### Container Status

```bash
# Show all containers
make ps

# Detailed status
docker-compose ps -a

# Resource usage
docker stats
```

### Common Issues

#### Agent Fails to Start

**Symptom:** Agent container exits immediately

**Solution:**
```bash
# Check logs
make logs-agent

# Common causes:
# 1. Database not ready → Wait longer (healthcheck will retry)
# 2. Missing models → Ensure models/ directory is mounted
# 3. Invalid configuration → Check .env file
```

#### Database Connection Refused

**Symptom:** Backend/agent can't connect to PostgreSQL

**Solution:**
```bash
# Verify PostgreSQL is healthy
docker-compose exec postgres pg_isready -U aitrader

# Check network
docker network inspect ai-trader-network

# Restart PostgreSQL
docker-compose restart postgres
```

#### Frontend Can't Reach Backend

**Symptom:** Frontend shows "Network Error"

**Solution:**
```bash
# Check backend is running
curl http://localhost:8001/health

# Check nginx proxy configuration
docker-compose exec frontend cat /etc/nginx/conf.d/default.conf

# Verify BACKEND_URL environment variable
docker-compose exec frontend env | grep BACKEND
```

#### Agent Not Trading

**Symptom:** Agent runs but doesn't execute trades

**Solution:**
```bash
# Check agent status
curl http://localhost:8002/status | jq

# Verify configuration
docker-compose exec agent env | grep AGENT_

# Common causes:
# 1. Mode is simulation → Expected behavior
# 2. Confidence threshold too high → Lower AGENT_CONFIDENCE_THRESHOLD
# 3. Safety limits triggered → Check logs for circuit breaker messages
```

### Log Analysis

```bash
# Search logs for errors
docker-compose logs | grep -i error

# Filter by service
docker-compose logs backend | grep -i "error\|exception"

# Show timestamps
docker-compose logs -f --timestamps agent

# Last 100 lines
docker-compose logs --tail=100 agent
```

### Performance Monitoring

```bash
# Resource usage by container
docker stats ai-trader-backend ai-trader-agent ai-trader-frontend

# Disk usage
docker system df

# Network inspection
docker network inspect ai-trader-network
```

## Security Considerations

### Production Checklist

- [ ] Change default PostgreSQL password
- [ ] Use strong passwords (minimum 16 characters, random)
- [ ] Store `.env` file securely (never commit to git)
- [ ] Enable SSL/TLS for public access
- [ ] Restrict database port (5432) to internal network only
- [ ] Use read-only volume mounts where possible
- [ ] Enable container resource limits
- [ ] Implement firewall rules
- [ ] Regular security updates: `docker-compose pull && docker-compose up -d`
- [ ] Monitor logs for suspicious activity
- [ ] Backup database regularly

### Secrets Management

**Development:**
- Use `.env` file (gitignored)

**Production:**
- Use Docker secrets (Swarm mode)
- Use environment-specific secrets management (AWS Secrets Manager, HashiCorp Vault)
- Never log sensitive values

### Network Security

```yaml
# Restrict PostgreSQL to internal network only
services:
  postgres:
    ports: []  # Remove external port mapping
    expose:
      - "5432"  # Only accessible within Docker network
```

### Container Hardening

```dockerfile
# Use non-root user
RUN useradd -m -u 1000 appuser
USER appuser

# Read-only root filesystem
services:
  backend:
    read_only: true
    tmpfs:
      - /tmp
```

## Backup & Recovery

### Database Backup

```bash
# Manual backup
docker-compose exec postgres pg_dump -U aitrader -F c aitrader > backup.dump

# Automated backup (cron)
0 2 * * * cd /path/to/ai-trader && docker-compose exec -T postgres pg_dump -U aitrader -F c aitrader > backups/aitrader_$(date +\%Y\%m\%d).dump
```

### Restore Database

```bash
# Stop services
docker-compose down

# Start only PostgreSQL
docker-compose up -d postgres

# Restore
docker-compose exec -T postgres pg_restore -U aitrader -d aitrader < backup.dump

# Start all services
docker-compose up -d
```

### Volume Backup

```bash
# Backup PostgreSQL data volume
docker run --rm -v ai-trader_postgres_data:/data -v $(pwd):/backup alpine tar czf /backup/postgres_data.tar.gz /data

# Restore PostgreSQL data volume
docker run --rm -v ai-trader_postgres_data:/data -v $(pwd):/backup alpine tar xzf /backup/postgres_data.tar.gz -C /
```

## Cleanup

```bash
# Stop all services
make down

# Remove volumes (CAUTION: deletes all data!)
make clean

# Remove unused Docker resources
docker system prune -a --volumes
```

---

**Version:** 1.0.0
**Last Updated:** 2026-01-22
**Maintainer:** AI Trading Agent Team
