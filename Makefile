# Makefile for AI Trading Agent Docker Operations
# Provides convenient shortcuts for common docker-compose commands

.PHONY: help up down logs logs-backend logs-agent logs-frontend build clean restart restart-agent restart-backend db-shell ps health

# Default target: show help
help:
	@echo "AI Trading Agent - Docker Commands"
	@echo ""
	@echo "Basic Commands:"
	@echo "  make up              - Start all services in detached mode"
	@echo "  make down            - Stop all services"
	@echo "  make logs            - View logs from all services (follow mode)"
	@echo "  make build           - Rebuild all containers"
	@echo "  make clean           - Stop services and remove volumes"
	@echo "  make ps              - Show status of all services"
	@echo ""
	@echo "Service-Specific Logs:"
	@echo "  make logs-backend    - View backend API logs"
	@echo "  make logs-agent      - View agent logs"
	@echo "  make logs-frontend   - View frontend logs"
	@echo ""
	@echo "Restart Commands:"
	@echo "  make restart         - Restart all services"
	@echo "  make restart-agent   - Restart agent only"
	@echo "  make restart-backend - Restart backend only"
	@echo ""
	@echo "Database Commands:"
	@echo "  make db-shell        - Open PostgreSQL shell"
	@echo "  make db-migrate      - Run database migrations"
	@echo ""
	@echo "Health & Status:"
	@echo "  make health          - Check health of all services"
	@echo ""
	@echo "Development:"
	@echo "  make dev-up          - Start with override (hot reload)"
	@echo "  make dev-down        - Stop development environment"

# Start all services
up:
	@echo "Starting all services..."
	docker-compose up -d
	@echo ""
	@echo "Services started. View logs with: make logs"
	@echo "Backend:  http://localhost:8001"
	@echo "Frontend: http://localhost:3001"
	@echo "Agent:    http://localhost:8002/health"

# Start with development overrides (hot reload)
dev-up:
	@echo "Starting development environment with hot reload..."
	docker-compose up -d
	@echo ""
	@echo "Development services started."
	@echo "Source code changes will be reflected automatically."

# Stop all services
down:
	@echo "Stopping all services..."
	docker-compose down

# Stop development environment
dev-down:
	docker-compose down

# View logs (all services, follow mode)
logs:
	docker-compose logs -f

# View backend logs
logs-backend:
	docker-compose logs -f backend

# View agent logs
logs-agent:
	docker-compose logs -f agent

# View frontend logs
logs-frontend:
	docker-compose logs -f frontend

# Rebuild all containers
build:
	@echo "Building all containers..."
	docker-compose build

# Build specific service
build-backend:
	docker-compose build backend

build-agent:
	docker-compose build agent

build-frontend:
	docker-compose build frontend

# Clean: stop and remove volumes
clean:
	@echo "Stopping services and removing volumes..."
	docker-compose down -v
	@echo "Cleaning Docker system..."
	docker system prune -f
	@echo "Cleanup complete."

# Restart all services
restart:
	@echo "Restarting all services..."
	docker-compose restart

# Restart agent only
restart-agent:
	@echo "Restarting agent..."
	docker-compose restart agent

# Restart backend only
restart-backend:
	@echo "Restarting backend..."
	docker-compose restart backend

# Restart frontend only
restart-frontend:
	@echo "Restarting frontend..."
	docker-compose restart frontend

# Open PostgreSQL shell
db-shell:
	@echo "Opening PostgreSQL shell..."
	docker-compose exec postgres psql -U ${POSTGRES_USER} -d ${POSTGRES_DB}

# Run database migrations
db-migrate:
	@echo "Running database migrations..."
	docker-compose exec backend python -c "from src.api.database.session import init_db; init_db()"

# Show service status
ps:
	docker-compose ps

# Check health of all services
health:
	@echo "Checking service health..."
	@echo ""
	@echo "Backend Health:"
	@curl -f http://localhost:8001/health 2>/dev/null && echo " ✓ Backend healthy" || echo " ✗ Backend unhealthy"
	@echo ""
	@echo "Agent Health:"
	@curl -f http://localhost:8002/health 2>/dev/null && echo " ✓ Agent healthy" || echo " ✗ Agent unhealthy"
	@echo ""
	@echo "Frontend Health:"
	@curl -f http://localhost:3001/health 2>/dev/null && echo " ✓ Frontend healthy" || echo " ✗ Frontend unhealthy"
	@echo ""
	@echo "Database:"
	@docker-compose exec postgres pg_isready 2>/dev/null && echo " ✓ PostgreSQL healthy" || echo " ✗ PostgreSQL unhealthy"

# View agent status (detailed)
agent-status:
	@echo "Agent Status:"
	@curl -s http://localhost:8002/status | python -m json.tool

# Stop agent (for testing manual control)
agent-stop:
	docker-compose stop agent

# Start agent
agent-start:
	docker-compose start agent

# Follow agent startup logs
agent-logs-startup:
	docker-compose logs -f --tail=50 agent
