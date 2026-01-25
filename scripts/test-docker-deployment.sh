#!/bin/bash
# Test script for Docker deployment verification

set -e

echo "=========================================="
echo "Docker Deployment Test Suite"
echo "=========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test counters
TESTS_PASSED=0
TESTS_FAILED=0

# Helper functions
pass() {
    echo -e "${GREEN}✓${NC} $1"
    TESTS_PASSED=$((TESTS_PASSED + 1))
}

fail() {
    echo -e "${RED}✗${NC} $1"
    TESTS_FAILED=$((TESTS_FAILED + 1))
}

warn() {
    echo -e "${YELLOW}⚠${NC} $1"
}

# Test 1: Docker installed
echo "Test 1: Docker installation"
if command -v docker &> /dev/null; then
    DOCKER_VERSION=$(docker --version)
    pass "Docker installed: $DOCKER_VERSION"
else
    fail "Docker not installed"
    exit 1
fi
echo ""

# Test 2: Docker Compose installed
echo "Test 2: Docker Compose installation"
if docker compose version &> /dev/null; then
    COMPOSE_VERSION=$(docker compose version)
    pass "Docker Compose installed: $COMPOSE_VERSION"
elif command -v docker-compose &> /dev/null; then
    COMPOSE_VERSION=$(docker-compose --version)
    pass "Docker Compose (standalone) installed: $COMPOSE_VERSION"
else
    fail "Docker Compose not installed"
    exit 1
fi
echo ""

# Test 3: docker-compose.yml syntax
echo "Test 3: docker-compose.yml validation"
if docker compose config --quiet; then
    pass "docker-compose.yml syntax valid"
else
    fail "docker-compose.yml syntax invalid"
fi
echo ""

# Test 4: Required files exist
echo "Test 4: Required files"
FILES=(
    "docker-compose.yml"
    "backend/Dockerfile"
    "backend/Dockerfile.agent"
    "backend/docker-entrypoint.sh"
    "backend/docker-entrypoint-agent.sh"
    "frontend/Dockerfile"
    ".env.example"
)

for file in "${FILES[@]}"; do
    if [ -f "$file" ]; then
        pass "File exists: $file"
    else
        fail "File missing: $file"
    fi
done
echo ""

# Test 5: Environment file
echo "Test 5: Environment configuration"
if [ -f ".env" ]; then
    pass ".env file exists"

    # Check required variables
    REQUIRED_VARS=(
        "POSTGRES_PASSWORD"
        "DATABASE_URL"
        "AGENT_MODE"
    )

    for var in "${REQUIRED_VARS[@]}"; do
        if grep -q "^${var}=" .env; then
            pass "Required variable set: $var"
        else
            warn "Variable not set in .env: $var"
        fi
    done
else
    warn ".env file not found (using .env.example defaults)"
fi
echo ""

# Test 6: Entrypoint scripts are executable
echo "Test 6: Entrypoint permissions"
ENTRYPOINTS=(
    "backend/docker-entrypoint.sh"
    "backend/docker-entrypoint-agent.sh"
)

for script in "${ENTRYPOINTS[@]}"; do
    if [ -x "$script" ]; then
        pass "Executable: $script"
    else
        warn "Not executable: $script (will be set in Dockerfile)"
    fi
done
echo ""

# Test 7: Required directories exist
echo "Test 7: Required directories"
DIRS=(
    "backend/models"
    "backend/data"
    "backend/src"
    "frontend/src"
)

for dir in "${DIRS[@]}"; do
    if [ -d "$dir" ]; then
        pass "Directory exists: $dir"
    else
        fail "Directory missing: $dir"
    fi
done
echo ""

# Test 8: Models exist
echo "Test 8: ML Models"
MODEL_DIR="backend/models/mtf_ensemble"
if [ -d "$MODEL_DIR" ]; then
    pass "Model directory exists: $MODEL_DIR"

    MODELS=(
        "1H_model.pkl"
        "4H_model.pkl"
        "D_model.pkl"
        "stacking_meta_learner.pkl"
    )

    for model in "${MODELS[@]}"; do
        if [ -f "$MODEL_DIR/$model" ]; then
            pass "Model exists: $model"
        else
            warn "Model missing: $model (agent will fail to start)"
        fi
    done
else
    fail "Model directory missing: $MODEL_DIR"
fi
echo ""

# Test 9: Network configuration
echo "Test 9: Network configuration"
if docker compose config | grep -q "ai-trader-network"; then
    pass "Network 'ai-trader-network' defined"
else
    fail "Network 'ai-trader-network' not defined"
fi
echo ""

# Test 10: Service definitions
echo "Test 10: Service definitions"
SERVICES=("postgres" "backend" "agent" "frontend")

for service in "${SERVICES[@]}"; do
    if docker compose config --services | grep -q "^${service}$"; then
        pass "Service defined: $service"
    else
        fail "Service missing: $service"
    fi
done
echo ""

# Test 11: Health checks defined
echo "Test 11: Health checks"
for service in "${SERVICES[@]}"; do
    if docker compose config | grep -A 20 "^  ${service}:" | grep -q "healthcheck:"; then
        pass "Health check defined: $service"
    else
        warn "No health check for: $service"
    fi
done
echo ""

# Test 12: Port mappings
echo "Test 12: Port mappings"
PORTS=(
    "5432:5432"  # PostgreSQL
    "8001:8001"  # Backend
    "8002:8002"  # Agent
    "3001:80"    # Frontend
)

for port_mapping in "${PORTS[@]}"; do
    if docker compose config | grep -q "$port_mapping"; then
        pass "Port mapped: $port_mapping"
    else
        warn "Port not mapped: $port_mapping"
    fi
done
echo ""

# Test 13: Volume mounts
echo "Test 13: Volume mounts"
if docker compose config | grep -q "postgres_data"; then
    pass "PostgreSQL data volume defined"
else
    fail "PostgreSQL data volume missing"
fi

# Check read-only mounts
if docker compose config | grep -q "models:ro"; then
    pass "Models mounted read-only"
else
    warn "Models not mounted read-only"
fi
echo ""

# Test 14: Agent configuration
echo "Test 14: Agent environment variables"
AGENT_VARS=(
    "AGENT_MODE"
    "AGENT_SYMBOL"
    "AGENT_CONFIDENCE_THRESHOLD"
    "AGENT_HEALTH_PORT"
)

for var in "${AGENT_VARS[@]}"; do
    if docker compose config | grep -q "$var"; then
        pass "Agent variable defined: $var"
    else
        fail "Agent variable missing: $var"
    fi
done
echo ""

# Test 15: Dependency order
echo "Test 15: Service dependencies"
if docker compose config | grep -A 10 "^  backend:" | grep -q "depends_on:"; then
    pass "Backend has dependencies defined"
else
    fail "Backend dependencies missing"
fi

if docker compose config | grep -A 10 "^  agent:" | grep -q "depends_on:"; then
    pass "Agent has dependencies defined"
else
    fail "Agent dependencies missing"
fi
echo ""

# Optional: Test Docker build (if --build flag)
if [ "$1" == "--build" ]; then
    echo "Test 16: Docker build (optional)"
    echo "Building containers (this may take several minutes)..."

    if docker compose build --quiet; then
        pass "All containers built successfully"
    else
        fail "Container build failed"
    fi
    echo ""
fi

# Optional: Test Docker startup (if --start flag)
if [ "$1" == "--start" ]; then
    echo "Test 17: Container startup (optional)"
    echo "Starting containers..."

    if docker compose up -d; then
        pass "Containers started"

        echo "Waiting for services to be healthy (60s timeout)..."
        sleep 10

        # Check health
        HEALTHY=0
        for i in {1..12}; do
            if curl -f http://localhost:8001/health &> /dev/null && \
               curl -f http://localhost:8002/health &> /dev/null && \
               curl -f http://localhost:3001/health &> /dev/null; then
                HEALTHY=1
                break
            fi
            echo "  Waiting... ($i/12)"
            sleep 5
        done

        if [ $HEALTHY -eq 1 ]; then
            pass "All services healthy"
        else
            warn "Services not healthy after 60s (may still be starting)"
        fi

        echo ""
        echo "Container status:"
        docker compose ps

        echo ""
        echo "To stop containers: docker compose down"
    else
        fail "Failed to start containers"
    fi
    echo ""
fi

# Summary
echo "=========================================="
echo "Test Summary"
echo "=========================================="
echo -e "${GREEN}Passed:${NC} $TESTS_PASSED"
echo -e "${RED}Failed:${NC} $TESTS_FAILED"
echo ""

if [ $TESTS_FAILED -eq 0 ]; then
    echo -e "${GREEN}✓ All tests passed!${NC}"
    echo ""
    echo "Your Docker configuration is ready."
    echo ""
    echo "Next steps:"
    echo "  1. Configure .env file (if not done)"
    echo "  2. Start services: make up"
    echo "  3. Check health: make health"
    echo "  4. View logs: make logs"
    exit 0
else
    echo -e "${RED}✗ Some tests failed${NC}"
    echo ""
    echo "Please fix the issues above before deploying."
    exit 1
fi
