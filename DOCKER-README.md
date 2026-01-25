# AI Trading Agent - Docker Deployment

Complete Docker orchestration for the AI Trading Agent system.

## Quick Start

```bash
# 1. Configure environment
cp .env.example .env
nano .env  # Edit with your settings

# 2. Start all services
make up

# 3. Verify deployment
make health

# 4. View logs
make logs
```

**Access Points:**
- Frontend: http://localhost:3001
- Backend API: http://localhost:8001/docs
- Agent Health: http://localhost:8002/health

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    AI TRADING AGENT SYSTEM                   │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  PostgreSQL (5432)                                           │
│  └── Persistent storage for trades and predictions          │
│                                                              │
│  Backend API (8001)                                          │
│  ├── FastAPI application                                    │
│  ├── MTF Ensemble predictions                               │
│  ├── Paper trading management                               │
│  └── Market data service (yfinance)                         │
│                                                              │
│  Agent (8002)                                                │
│  ├── Autonomous trading loop                                │
│  ├── Monitors predictions → Executes trades                 │
│  ├── Modes: simulation, paper, live                         │
│  └── Safety circuit breakers                                │
│                                                              │
│  Frontend (3001)                                             │
│  ├── React dashboard (nginx)                                │
│  ├── Real-time predictions display                          │
│  └── Trading account status                                 │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Services

| Service | Port | Purpose | Build |
|---------|------|---------|-------|
| postgres | 5432 | Database | postgres:16-alpine |
| backend | 8001 | API Server | backend/Dockerfile |
| agent | 8002 | Trading Agent | backend/Dockerfile.agent |
| frontend | 3001 | Web Dashboard | frontend/Dockerfile |

**Network:** All services communicate via `ai-trader-network` bridge.

## Makefile Commands

| Command | Description |
|---------|-------------|
| `make up` | Start all services |
| `make down` | Stop all services |
| `make logs` | View all logs (follow) |
| `make logs-agent` | View agent logs |
| `make logs-backend` | View backend logs |
| `make build` | Rebuild all containers |
| `make restart` | Restart all services |
| `make restart-agent` | Restart agent only |
| `make health` | Check service health |
| `make ps` | Show container status |
| `make db-shell` | Open PostgreSQL shell |
| `make clean` | Stop and remove volumes |
| `make agent-status` | Detailed agent status |

Run `make help` for complete list.

## Configuration

### Required Environment Variables

Edit `.env` file:

```bash
# Database (CHANGE PASSWORD!)
POSTGRES_PASSWORD=your_secure_password_here

# API Keys
OPENAI_API_KEY=sk-...
FRED_API_KEY=...

# Agent Mode
AGENT_MODE=simulation  # simulation | paper | live
```

See `.env.example` for full configuration options.

### Agent Modes

| Mode | Description | MT5 Required |
|------|-------------|--------------|
| **simulation** | Fully simulated (no real data feed) | No |
| **paper** | Real price feed, no real money | Yes (Windows) |
| **live** | REAL MONEY TRADING | Yes (Windows) |

**Important:** MT5 requires Windows. Docker containers run Linux, so only **simulation mode** works in Docker. For paper/live trading, run agent natively on Windows.

## Volumes

| Volume | Mount Point | Purpose | Mode |
|--------|-------------|---------|------|
| `postgres_data` | `/var/lib/postgresql/data` | Database persistence | RW |
| `./backend/models` | `/app/models` | ML models | RO |
| `./backend/data` | `/app/data` | Market/sentiment data | RW (backend), RO (agent) |
| `./backend/logs` | `/app/logs` | Agent logs | RW |

## Health Checks

All services have automatic health checks:

```bash
# Automated
make health

# Manual
curl http://localhost:8001/health  # Backend
curl http://localhost:8002/health  # Agent
curl http://localhost:3001/health  # Frontend
```

## Development Workflow

### Hot Reload

```bash
# Start with development overrides
make dev-up

# Source code changes in backend/src/ are automatically reflected
```

### Frontend Development

For live frontend development with Vite:

```bash
# Stop frontend container
docker-compose stop frontend

# Run Vite locally
cd frontend
npm install
npm run dev
# Access at http://localhost:5173
```

### Rebuilding After Changes

```bash
# Rebuild specific service
make build-agent
docker-compose up -d agent

# Rebuild all
make build && make up
```

## Monitoring

### View Logs

```bash
# Follow all logs
make logs

# Follow specific service
make logs-agent

# Last 50 lines
docker-compose logs --tail=50 agent

# Search for errors
docker-compose logs | grep -i error
```

### Agent Status

```bash
# Detailed status
make agent-status

# Returns JSON:
{
  "status": "running",
  "mode": "simulation",
  "cycle_count": 42,
  "model_loaded": true,
  "last_prediction": {...},
  "position": {...}
}
```

### Resource Usage

```bash
# View container resources
docker stats

# Disk usage
docker system df
```

## Troubleshooting

### Agent Not Starting

```bash
# Check logs
make logs-agent

# Common issues:
# 1. Database not ready → Wait for healthcheck
# 2. Missing models → Verify models/ directory exists
# 3. Invalid config → Check .env file
```

### Database Connection Errors

```bash
# Verify PostgreSQL
docker-compose exec postgres pg_isready -U aitrader

# Test connectivity
docker-compose exec agent ping postgres

# Restart database
docker-compose restart postgres
```

### Agent Not Trading

```bash
# Check status
curl http://localhost:8002/status | jq

# Verify predictions available
curl http://localhost:8001/api/v1/predictions/current

# Common causes:
# - Confidence threshold too high
# - Safety limits triggered
# - No new predictions available
```

## Database Operations

```bash
# Open PostgreSQL shell
make db-shell

# View trades
SELECT * FROM trades ORDER BY entry_time DESC LIMIT 10;

# Count trades by status
SELECT status, COUNT(*) FROM trades GROUP BY status;

# Exit
\q
```

### Backup & Restore

```bash
# Backup
docker-compose exec postgres pg_dump -U aitrader aitrader > backup.sql

# Restore
docker-compose exec -T postgres psql -U aitrader aitrader < backup.sql
```

## Production Deployment

### 1. Configure Production Environment

```bash
# Create production env file
cp .env.example .env.production

# Edit with production settings
nano .env.production
```

**Key settings:**
- Strong database password
- Production API keys
- Conservative safety limits
- Proper resource limits

### 2. Deploy

```bash
# Load production environment
export $(cat .env.production | xargs)

# Deploy
docker-compose up -d

# Verify
make health
make logs
```

### 3. SSL/TLS

Use a reverse proxy (nginx, Caddy) for HTTPS:

```nginx
server {
    listen 443 ssl http2;
    server_name yourdomain.com;

    location / {
        proxy_pass http://localhost:3001;
    }

    location /api/ {
        proxy_pass http://localhost:8001/api/;
    }
}
```

## Security Considerations

- [ ] Change default PostgreSQL password
- [ ] Use strong, random passwords (16+ characters)
- [ ] Never commit `.env` file
- [ ] Enable SSL/TLS for public access
- [ ] Restrict database port to internal network
- [ ] Use read-only volume mounts where possible
- [ ] Regular security updates: `docker-compose pull && docker-compose up -d`
- [ ] Monitor logs for suspicious activity

## Performance Tuning

### Resource Limits

Edit `docker-compose.yml`:

```yaml
services:
  agent:
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 2G
        reservations:
          cpus: '1.0'
          memory: 1G
```

### Logging

Limit log file size:

```yaml
services:
  agent:
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"
```

## Cleanup

```bash
# Stop services
make down

# Remove volumes (CAUTION: deletes data!)
make clean

# Remove all Docker resources
docker system prune -a --volumes
```

## Documentation

- [Complete Deployment Guide](docs/DOCKER-DEPLOYMENT-GUIDE.md)
- [Quick Reference](docs/DOCKER-QUICK-REFERENCE.md)
- [Project Documentation](docs/01-current-state-of-the-art.md)

## Support

For issues or questions:
1. Check logs: `make logs-agent`
2. Verify health: `make health`
3. Review [Troubleshooting Guide](docs/DOCKER-DEPLOYMENT-GUIDE.md#troubleshooting)

---

**Version:** 1.0.0
**Last Updated:** 2026-01-22
