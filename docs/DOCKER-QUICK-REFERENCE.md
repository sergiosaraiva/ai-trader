# Docker Quick Reference

Quick reference for common Docker commands for the AI Trading Agent system.

## Table of Contents

- [Starting & Stopping](#starting--stopping)
- [Logs & Monitoring](#logs--monitoring)
- [Health Checks](#health-checks)
- [Database Operations](#database-operations)
- [Agent Operations](#agent-operations)
- [Development](#development)
- [Troubleshooting](#troubleshooting)

## Starting & Stopping

```bash
# Start all services (detached)
make up
# Or: docker-compose up -d

# Stop all services
make down
# Or: docker-compose down

# Restart all services
make restart
# Or: docker-compose restart

# Restart specific service
make restart-agent
make restart-backend
make restart-frontend
```

## Logs & Monitoring

```bash
# View all logs (follow mode)
make logs
# Or: docker-compose logs -f

# View specific service logs
make logs-agent      # Agent logs
make logs-backend    # Backend logs
make logs-frontend   # Frontend logs

# View last 50 lines
docker-compose logs --tail=50 agent

# View logs with timestamps
docker-compose logs -f --timestamps agent

# Search logs for errors
docker-compose logs | grep -i error

# View agent startup sequence
make agent-logs-startup
```

## Health Checks

```bash
# Check all services
make health

# Check individual services
curl http://localhost:8001/health  # Backend
curl http://localhost:8002/health  # Agent
curl http://localhost:3001/health  # Frontend

# Get detailed agent status
make agent-status
# Or: curl http://localhost:8002/status | jq

# View container status
make ps
# Or: docker-compose ps

# View resource usage
docker stats
```

## Database Operations

```bash
# Open PostgreSQL shell
make db-shell
# Or: docker-compose exec postgres psql -U aitrader -d aitrader

# Run database migrations
make db-migrate

# Check database connectivity
docker-compose exec postgres pg_isready -U aitrader

# Backup database
docker-compose exec postgres pg_dump -U aitrader aitrader > backup.sql

# Restore database
docker-compose exec -T postgres psql -U aitrader aitrader < backup.sql

# View database tables
docker-compose exec postgres psql -U aitrader -d aitrader -c "\dt"

# Count trades
docker-compose exec postgres psql -U aitrader -d aitrader -c "SELECT COUNT(*) FROM trades;"

# View recent trades
docker-compose exec postgres psql -U aitrader -d aitrader -c "SELECT * FROM trades ORDER BY entry_time DESC LIMIT 10;"
```

## Agent Operations

```bash
# View agent status
make agent-status
# Returns: mode, status, cycle_count, model_loaded, etc.

# Restart agent
make restart-agent

# Stop agent
make agent-stop
# Or: docker-compose stop agent

# Start agent
make agent-start
# Or: docker-compose start agent

# View agent configuration
docker-compose exec agent env | grep AGENT_

# Check agent health
curl http://localhost:8002/health

# Monitor agent activity
docker-compose logs -f --tail=100 agent
```

## Development

```bash
# Start with hot reload
make dev-up

# Rebuild containers
make build               # All containers
make build-backend       # Backend only
make build-agent         # Agent only
make build-frontend      # Frontend only

# Execute command in container
docker-compose exec backend python -m src.api.main --help
docker-compose exec agent python -c "from src.agent.config import AgentConfig; print(AgentConfig.from_env())"

# Access container shell
docker-compose exec backend bash
docker-compose exec agent bash
docker-compose exec postgres bash

# View container environment
docker-compose exec agent env
```

## Troubleshooting

### Service Won't Start

```bash
# Check container logs
docker-compose logs agent

# Check container exit code
docker-compose ps

# Verify dependencies are healthy
docker-compose ps postgres backend

# Rebuild container
docker-compose build agent
docker-compose up -d agent
```

### Database Connection Issues

```bash
# Verify PostgreSQL is running
docker-compose ps postgres

# Check PostgreSQL health
docker-compose exec postgres pg_isready -U aitrader

# Verify network connectivity
docker-compose exec backend ping -c 3 postgres
docker-compose exec agent ping -c 3 postgres

# Check database URL
docker-compose exec agent env | grep DATABASE_URL

# Restart PostgreSQL
docker-compose restart postgres
```

### Agent Not Trading

```bash
# Check agent status
curl http://localhost:8002/status | jq

# Verify configuration
docker-compose exec agent env | grep AGENT_

# Check if models are loaded
docker-compose exec agent ls -lh /app/models/mtf_ensemble/

# Verify prediction service
curl http://localhost:8001/api/v1/predictions/current

# Review agent logs for errors
docker-compose logs --tail=200 agent | grep -i "error\|exception\|failed"
```

### Network Issues

```bash
# List Docker networks
docker network ls

# Inspect ai-trader network
docker network inspect ai-trader-network

# Verify containers are on network
docker network inspect ai-trader-network | jq '.[].Containers'

# Test backend connectivity from agent
docker-compose exec agent curl -f http://backend:8001/health

# Test frontend to backend
docker-compose exec frontend wget -O- http://backend:8001/health
```

### Performance Issues

```bash
# View resource usage
docker stats

# Check container resource limits
docker inspect ai-trader-agent | jq '.[].HostConfig.Memory'

# View disk usage
docker system df

# Clean up unused resources
docker system prune -f

# Check logs for performance warnings
docker-compose logs | grep -i "slow\|timeout\|performance"
```

### Container Restart Loop

```bash
# View recent logs
docker-compose logs --tail=100 agent

# Check exit code
docker-compose ps agent

# Verify configuration
docker-compose config

# Try running with explicit logs
docker-compose up agent

# Check health check configuration
docker inspect ai-trader-agent | jq '.[].State.Health'
```

## Clean Up

```bash
# Stop all services
make down

# Stop and remove volumes (CAUTION: deletes data!)
make clean

# Remove unused Docker resources
docker system prune -f

# Remove all containers, images, volumes (NUCLEAR OPTION)
docker system prune -a --volumes
```

## Environment Variables

```bash
# View all environment variables for a service
docker-compose exec agent env

# View specific variable
docker-compose exec agent printenv AGENT_MODE

# Set variable temporarily (restart required)
docker-compose exec agent export AGENT_MODE=simulation

# Permanently change: edit .env file, then:
docker-compose up -d
```

## Building & Deployment

```bash
# Build without cache
docker-compose build --no-cache

# Pull latest base images
docker-compose pull

# Build and start
docker-compose up -d --build

# Deploy with specific compose file
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d

# Scale services (if needed)
docker-compose up -d --scale agent=0  # Disable agent
docker-compose up -d --scale agent=1  # Re-enable agent
```

## Useful Aliases

Add these to your `~/.bashrc` or `~/.zshrc`:

```bash
# AI Trader aliases
alias ai-up='cd /path/to/ai-trader && make up'
alias ai-down='cd /path/to/ai-trader && make down'
alias ai-logs='cd /path/to/ai-trader && make logs'
alias ai-logs-agent='cd /path/to/ai-trader && make logs-agent'
alias ai-health='cd /path/to/ai-trader && make health'
alias ai-restart-agent='cd /path/to/ai-trader && make restart-agent'
alias ai-status='curl -s http://localhost:8002/status | jq'
```

## Common Workflows

### Initial Setup

```bash
# 1. Clone and configure
git clone <repo>
cd ai-trader
cp .env.example .env
nano .env  # Edit configuration

# 2. Start services
make up

# 3. Verify
make health
make logs
```

### Daily Development

```bash
# Morning: Start services
make up

# Development: View logs
make logs-agent

# Changes: Rebuild and restart
make build && make restart

# Evening: Stop services
make down
```

### Debugging Agent Issue

```bash
# 1. Check status
curl http://localhost:8002/status | jq

# 2. View logs
make logs-agent

# 3. Check configuration
docker-compose exec agent env | grep AGENT_

# 4. Restart with fresh logs
docker-compose restart agent && docker-compose logs -f agent

# 5. If still failing, rebuild
docker-compose build agent && docker-compose up -d agent
```

### Database Maintenance

```bash
# 1. Backup
docker-compose exec postgres pg_dump -U aitrader aitrader > backup_$(date +%Y%m%d).sql

# 2. Open shell
make db-shell

# 3. Check trades
SELECT COUNT(*), status FROM trades GROUP BY status;

# 4. Exit
\q
```

---

**Tip:** Run `make help` to see all available Makefile commands.
