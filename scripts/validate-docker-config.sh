#!/bin/bash
# Docker Configuration Validation Script
# Validates all Docker-related configuration files for the AI Trading Agent

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Counters
TOTAL_CHECKS=0
PASSED_CHECKS=0
FAILED_CHECKS=0
WARNING_CHECKS=0

# Test results
FAILED_TESTS=()
WARNING_MESSAGES=()

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Docker Configuration Validation${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Helper functions
check_pass() {
    TOTAL_CHECKS=$((TOTAL_CHECKS + 1))
    PASSED_CHECKS=$((PASSED_CHECKS + 1))
    echo -e "${GREEN}✓${NC} $1"
}

check_fail() {
    TOTAL_CHECKS=$((TOTAL_CHECKS + 1))
    FAILED_CHECKS=$((FAILED_CHECKS + 1))
    FAILED_TESTS+=("$1")
    echo -e "${RED}✗${NC} $1"
}

check_warning() {
    TOTAL_CHECKS=$((TOTAL_CHECKS + 1))
    WARNING_CHECKS=$((WARNING_CHECKS + 1))
    WARNING_MESSAGES+=("$1")
    echo -e "${YELLOW}⚠${NC} $1"
}

# ========================================
# 1. File Existence Checks
# ========================================
echo -e "${BLUE}1. Checking File Existence...${NC}"

if [ -f "backend/Dockerfile.agent" ]; then
    check_pass "backend/Dockerfile.agent exists"
else
    check_fail "backend/Dockerfile.agent not found"
fi

if [ -f "backend/docker-entrypoint-agent.sh" ]; then
    check_pass "backend/docker-entrypoint-agent.sh exists"
else
    check_fail "backend/docker-entrypoint-agent.sh not found"
fi

if [ -f "docker-compose.yml" ]; then
    check_pass "docker-compose.yml exists"
else
    check_fail "docker-compose.yml not found"
fi

if [ -f "docker-compose.override.yml" ]; then
    check_pass "docker-compose.override.yml exists"
else
    check_fail "docker-compose.override.yml not found"
fi

if [ -f ".env.example" ]; then
    check_pass ".env.example exists"
else
    check_fail ".env.example not found"
fi

if [ -f "Makefile" ]; then
    check_pass "Makefile exists"
else
    check_fail "Makefile not found"
fi

echo ""

# ========================================
# 2. Dockerfile.agent Validation
# ========================================
echo -e "${BLUE}2. Validating backend/Dockerfile.agent...${NC}"

if [ -f "backend/Dockerfile.agent" ]; then
    # Check base image
    if grep -q "^FROM python:3.12" backend/Dockerfile.agent; then
        check_pass "Base image is python:3.12-slim"
    else
        check_fail "Base image is not python:3.12-slim"
    fi

    # Check WORKDIR
    if grep -q "^WORKDIR /app" backend/Dockerfile.agent; then
        check_pass "WORKDIR is set to /app"
    else
        check_fail "WORKDIR not set correctly"
    fi

    # Check required files are copied
    if grep -q "COPY src/" backend/Dockerfile.agent; then
        check_pass "Source code (src/) is copied"
    else
        check_fail "Source code (src/) is not copied"
    fi

    if grep -q "COPY models/" backend/Dockerfile.agent; then
        check_pass "Models directory is copied"
    else
        check_fail "Models directory is not copied"
    fi

    if grep -q "COPY docker-entrypoint-agent.sh" backend/Dockerfile.agent; then
        check_pass "Entrypoint script is copied"
    else
        check_fail "Entrypoint script is not copied"
    fi

    # Check entrypoint is set
    if grep -q "^ENTRYPOINT.*docker-entrypoint-agent.sh" backend/Dockerfile.agent; then
        check_pass "ENTRYPOINT is configured"
    else
        check_fail "ENTRYPOINT is not configured"
    fi

    # Check health check port is exposed
    if grep -q "^EXPOSE 8002" backend/Dockerfile.agent; then
        check_pass "Health check port 8002 is exposed"
    else
        check_fail "Health check port 8002 is not exposed"
    fi

    # Check chmod for entrypoint
    if grep -q "chmod +x.*docker-entrypoint-agent.sh" backend/Dockerfile.agent; then
        check_pass "Entrypoint script permissions are set"
    else
        check_fail "Entrypoint script permissions are not set"
    fi

    # Check PYTHONPATH is set
    if grep -q "ENV PYTHONPATH=/app" backend/Dockerfile.agent; then
        check_pass "PYTHONPATH environment variable is set"
    else
        check_fail "PYTHONPATH environment variable is not set"
    fi
fi

echo ""

# ========================================
# 3. Entrypoint Script Validation
# ========================================
echo -e "${BLUE}3. Validating backend/docker-entrypoint-agent.sh...${NC}"

if [ -f "backend/docker-entrypoint-agent.sh" ]; then
    # Check shebang
    if head -n 1 backend/docker-entrypoint-agent.sh | grep -q "^#!/bin/bash"; then
        check_pass "Shebang is correct (#!/bin/bash)"
    else
        check_fail "Shebang is missing or incorrect"
    fi

    # Check error handling
    if grep -q "^set -e" backend/docker-entrypoint-agent.sh; then
        check_pass "Error handling enabled (set -e)"
    else
        check_fail "Error handling not enabled (set -e missing)"
    fi

    # Check executable permissions
    if [ -x "backend/docker-entrypoint-agent.sh" ]; then
        check_pass "Entrypoint script is executable"
    else
        check_warning "Entrypoint script is not executable (will be set in Dockerfile)"
    fi

    # Check PostgreSQL wait logic
    if grep -q "pg_isready" backend/docker-entrypoint-agent.sh; then
        check_pass "PostgreSQL wait logic implemented"
    else
        check_fail "PostgreSQL wait logic missing"
    fi

    # Check backend API wait logic
    if grep -q "BACKEND_URL" backend/docker-entrypoint-agent.sh; then
        check_pass "Backend API wait logic implemented"
    else
        check_fail "Backend API wait logic missing"
    fi

    # Check live mode validation
    if grep -q "AGENT_MODE.*live" backend/docker-entrypoint-agent.sh; then
        check_pass "Live mode validation implemented"
    else
        check_fail "Live mode validation missing"
    fi

    # Check MT5 credentials validation for live mode
    if grep -q "AGENT_MT5_LOGIN" backend/docker-entrypoint-agent.sh && \
       grep -q "AGENT_MT5_PASSWORD" backend/docker-entrypoint-agent.sh && \
       grep -q "AGENT_MT5_SERVER" backend/docker-entrypoint-agent.sh; then
        check_pass "MT5 credentials validation implemented"
    else
        check_fail "MT5 credentials validation missing"
    fi

    # Check agent is started with exec
    if grep -q "^exec python -m src.agent.main" backend/docker-entrypoint-agent.sh; then
        check_pass "Agent is started with exec"
    else
        check_fail "Agent is not started with exec (signal handling may not work)"
    fi
fi

echo ""

# ========================================
# 4. docker-compose.yml Validation
# ========================================
echo -e "${BLUE}4. Validating docker-compose.yml...${NC}"

if [ -f "docker-compose.yml" ]; then
    # Check YAML syntax
    if command -v docker-compose &> /dev/null; then
        if docker-compose config > /dev/null 2>&1; then
            check_pass "YAML syntax is valid"
        else
            check_fail "YAML syntax is invalid"
        fi
    else
        check_warning "docker-compose not installed, skipping YAML validation"
    fi

    # Check all services are defined
    for service in postgres backend agent frontend; do
        if grep -q "^  $service:" docker-compose.yml; then
            check_pass "Service '$service' is defined"
        else
            check_fail "Service '$service' is not defined"
        fi
    done

    # Check agent service configuration
    if grep -A 50 "^  agent:" docker-compose.yml | grep -q "dockerfile: Dockerfile.agent"; then
        check_pass "Agent uses Dockerfile.agent"
    else
        check_fail "Agent does not use Dockerfile.agent"
    fi

    # Check agent depends on postgres and backend
    if grep -A 50 "^  agent:" docker-compose.yml | grep -q "postgres:" && \
       grep -A 50 "^  agent:" docker-compose.yml | grep -q "backend:"; then
        check_pass "Agent depends on postgres and backend"
    else
        check_fail "Agent dependencies are not correct"
    fi

    # Check health checks
    for service in postgres backend agent frontend; do
        if grep -A 50 "^  $service:" docker-compose.yml | grep -q "healthcheck:"; then
            check_pass "Service '$service' has health check"
        else
            check_fail "Service '$service' is missing health check"
        fi
    done

    # Check agent port 8002
    if grep -A 50 "^  agent:" docker-compose.yml | grep -q '"8002:8002"'; then
        check_pass "Agent port 8002 is mapped"
    else
        check_fail "Agent port 8002 is not mapped"
    fi

    # Check volumes are defined
    if grep -q "^volumes:" docker-compose.yml; then
        check_pass "Volumes section is defined"
    else
        check_fail "Volumes section is missing"
    fi

    if grep -q "postgres_data:" docker-compose.yml; then
        check_pass "postgres_data volume is defined"
    else
        check_fail "postgres_data volume is missing"
    fi

    # Check networks are defined
    if grep -q "^networks:" docker-compose.yml; then
        check_pass "Networks section is defined"
    else
        check_fail "Networks section is missing"
    fi

    if grep -q "ai-trader-network:" docker-compose.yml; then
        check_pass "ai-trader-network is defined"
    else
        check_fail "ai-trader-network is missing"
    fi

    # Check all services use the network
    for service in postgres backend agent frontend; do
        if grep -A 50 "^  $service:" docker-compose.yml | grep -q "ai-trader-network"; then
            check_pass "Service '$service' uses ai-trader-network"
        else
            check_fail "Service '$service' does not use ai-trader-network"
        fi
    done

    # Check agent environment variables
    agent_env_vars=(
        "AGENT_MODE"
        "AGENT_SYMBOL"
        "AGENT_CONFIDENCE_THRESHOLD"
        "AGENT_CYCLE_INTERVAL"
        "AGENT_MAX_POSITION_SIZE"
        "AGENT_USE_KELLY_SIZING"
        "AGENT_HEALTH_PORT"
        "AGENT_INITIAL_CAPITAL"
        "AGENT_MAX_CONSECUTIVE_LOSSES"
        "AGENT_MAX_DRAWDOWN_PERCENT"
        "AGENT_MAX_DAILY_LOSS_PERCENT"
    )

    for var in "${agent_env_vars[@]}"; do
        if grep -A 50 "^  agent:" docker-compose.yml | grep -q "$var"; then
            check_pass "Agent environment variable '$var' is set"
        else
            check_fail "Agent environment variable '$var' is missing"
        fi
    done

    # Check agent volume mounts
    if grep -A 50 "^  agent:" docker-compose.yml | grep -q "./backend/models:/app/models:ro"; then
        check_pass "Agent models directory is mounted read-only"
    else
        check_fail "Agent models directory mount is missing or not read-only"
    fi

    if grep -A 50 "^  agent:" docker-compose.yml | grep -q "./backend/logs:/app/logs"; then
        check_pass "Agent logs directory is mounted"
    else
        check_fail "Agent logs directory mount is missing"
    fi
fi

echo ""

# ========================================
# 5. docker-compose.override.yml Validation
# ========================================
echo -e "${BLUE}5. Validating docker-compose.override.yml...${NC}"

if [ -f "docker-compose.override.yml" ]; then
    # Check services override
    if grep -q "^  backend:" docker-compose.override.yml; then
        check_pass "Backend override is defined"
    else
        check_warning "Backend override is not defined"
    fi

    if grep -q "^  agent:" docker-compose.override.yml; then
        check_pass "Agent override is defined"
    else
        check_warning "Agent override is not defined"
    fi

    # Check development environment variable
    if grep -q "ENVIRONMENT=development" docker-compose.override.yml; then
        check_pass "ENVIRONMENT=development is set in override"
    else
        check_warning "ENVIRONMENT=development is not set in override"
    fi

    # Check agent is forced to simulation mode in development
    if grep -A 20 "^  agent:" docker-compose.override.yml | grep -q "AGENT_MODE=simulation"; then
        check_pass "Agent forced to simulation mode in development"
    else
        check_warning "Agent not forced to simulation mode in development"
    fi

    # Check source code mounting for hot reload
    if grep -A 20 "^  backend:" docker-compose.override.yml | grep -q "./backend/src:/app/src"; then
        check_pass "Backend source code mounted for hot reload"
    else
        check_warning "Backend source code not mounted for hot reload"
    fi

    if grep -A 20 "^  agent:" docker-compose.override.yml | grep -q "./backend/src:/app/src"; then
        check_pass "Agent source code mounted for hot reload"
    else
        check_warning "Agent source code not mounted for hot reload"
    fi
fi

echo ""

# ========================================
# 6. .env.example Validation
# ========================================
echo -e "${BLUE}6. Validating .env.example...${NC}"

if [ -f ".env.example" ]; then
    # Check required database variables
    required_vars=(
        "DATABASE_URL"
        "POSTGRES_USER"
        "POSTGRES_PASSWORD"
        "POSTGRES_DB"
        "AGENT_MODE"
        "AGENT_SYMBOL"
        "AGENT_CONFIDENCE_THRESHOLD"
        "AGENT_CYCLE_INTERVAL"
        "AGENT_MAX_POSITION_SIZE"
        "AGENT_USE_KELLY_SIZING"
        "AGENT_INITIAL_CAPITAL"
        "AGENT_HEALTH_PORT"
        "AGENT_MAX_CONSECUTIVE_LOSSES"
        "AGENT_MAX_DRAWDOWN_PERCENT"
        "AGENT_MAX_DAILY_LOSS_PERCENT"
    )

    for var in "${required_vars[@]}"; do
        if grep -q "^$var=" .env.example; then
            check_pass "Environment variable '$var' is documented"
        else
            check_fail "Environment variable '$var' is missing from .env.example"
        fi
    done

    # Check no secrets have actual values (except development defaults)
    if grep -q "^AGENT_MT5_PASSWORD=.*[a-zA-Z0-9]" .env.example && \
       ! grep -q "^AGENT_MT5_PASSWORD=$" .env.example; then
        check_fail "AGENT_MT5_PASSWORD has a value in .env.example (should be empty)"
    else
        check_pass "AGENT_MT5_PASSWORD has no value in .env.example"
    fi

    # Check security warnings are present
    if grep -q "SECURITY WARNING" .env.example; then
        check_pass "Security warnings are present"
    else
        check_warning "Security warnings are missing"
    fi

    # Check valid variable names (no spaces, valid format)
    if grep -E "^[A-Z_]+=.*" .env.example > /dev/null; then
        check_pass "All variable names follow convention (UPPERCASE_WITH_UNDERSCORES)"
    else
        check_warning "Some variable names may not follow convention"
    fi

    # Check default mode is simulation
    if grep -q "^AGENT_MODE=simulation" .env.example; then
        check_pass "Default AGENT_MODE is simulation"
    else
        check_fail "Default AGENT_MODE is not simulation (should be simulation for safety)"
    fi
fi

echo ""

# ========================================
# 7. Makefile Validation
# ========================================
echo -e "${BLUE}7. Validating Makefile...${NC}"

if [ -f "Makefile" ]; then
    # Check required targets
    required_targets=(
        "help"
        "up"
        "down"
        "logs"
        "logs-agent"
        "build"
        "clean"
        "restart-agent"
        "health"
    )

    for target in "${required_targets[@]}"; do
        if grep -q "^$target:" Makefile; then
            check_pass "Makefile target '$target' is defined"
        else
            check_fail "Makefile target '$target' is missing"
        fi
    done

    # Check .PHONY declaration
    if grep -q "^.PHONY:" Makefile; then
        check_pass "Makefile has .PHONY declaration"
    else
        check_warning "Makefile is missing .PHONY declaration"
    fi

    # Check agent-specific targets
    if grep -q "agent-status:" Makefile; then
        check_pass "Makefile has agent-status target"
    else
        check_warning "Makefile is missing agent-status target"
    fi
fi

echo ""

# ========================================
# 8. Port Conflict Checks
# ========================================
echo -e "${BLUE}8. Checking Port Assignments...${NC}"

if [ -f "docker-compose.yml" ]; then
    # Extract port mappings and check for conflicts
    ports=$(grep -E '^\s*-\s*"[0-9]+:[0-9]+"' docker-compose.yml | sed 's/.*"\([0-9]*\):.*/\1/' | sort)

    # Check for duplicate ports
    duplicate_ports=$(echo "$ports" | uniq -d)
    if [ -z "$duplicate_ports" ]; then
        check_pass "No duplicate port mappings found"
    else
        check_fail "Duplicate port mappings found: $duplicate_ports"
    fi

    # Check standard ports
    if echo "$ports" | grep -q "^5432$"; then
        check_pass "PostgreSQL port 5432 is mapped"
    else
        check_fail "PostgreSQL port 5432 is not mapped"
    fi

    if echo "$ports" | grep -q "^8001$"; then
        check_pass "Backend port 8001 is mapped"
    else
        check_fail "Backend port 8001 is not mapped"
    fi

    if echo "$ports" | grep -q "^8002$"; then
        check_pass "Agent port 8002 is mapped"
    else
        check_fail "Agent port 8002 is not mapped"
    fi

    if echo "$ports" | grep -q "^3001$"; then
        check_pass "Frontend port 3001 is mapped"
    else
        check_fail "Frontend port 3001 is not mapped"
    fi
fi

echo ""

# ========================================
# 9. Volume Path Validation
# ========================================
echo -e "${BLUE}9. Validating Volume Paths...${NC}"

# Check if required directories exist
required_dirs=(
    "backend/models"
    "backend/data/forex"
    "backend/data/sentiment"
    "backend/logs"
)

for dir in "${required_dirs[@]}"; do
    if [ -d "$dir" ]; then
        check_pass "Directory '$dir' exists"
    else
        check_warning "Directory '$dir' does not exist (will be created by Docker)"
    fi
done

echo ""

# ========================================
# 10. Security Checks
# ========================================
echo -e "${BLUE}10. Security Checks...${NC}"

# Check .env is gitignored
if [ -f ".gitignore" ]; then
    if grep -q "^\.env$" .gitignore; then
        check_pass ".env is in .gitignore"
    else
        check_fail ".env is not in .gitignore (SECURITY RISK)"
    fi
fi

# Check actual .env doesn't exist in repo
if [ -f ".env" ]; then
    check_warning ".env file exists (should not be in version control)"
else
    check_pass ".env file does not exist (good - use .env.example as template)"
fi

# Check for hardcoded credentials in docker-compose.yml
if grep -E "(password|secret|key).*:.*['\"].*[a-zA-Z0-9]{8,}" docker-compose.yml > /dev/null; then
    check_warning "Potential hardcoded credentials found in docker-compose.yml"
else
    check_pass "No obvious hardcoded credentials in docker-compose.yml"
fi

# Check agent Dockerfile doesn't copy secrets
if grep -E "COPY.*(\.env|credentials|secrets)" backend/Dockerfile.agent > /dev/null; then
    check_fail "Agent Dockerfile copies potential secret files"
else
    check_pass "Agent Dockerfile doesn't copy secret files"
fi

echo ""

# ========================================
# 11. Integration Validation
# ========================================
echo -e "${BLUE}11. Integration Checks...${NC}"

if [ -f "docker-compose.yml" ]; then
    # Check agent can reach backend
    if grep -A 50 "^  agent:" docker-compose.yml | grep -q "BACKEND_URL=http://backend:8001"; then
        check_pass "Agent configured to reach backend via Docker network"
    else
        check_fail "Agent BACKEND_URL not configured correctly"
    fi

    # Check backend and agent share same DATABASE_URL pattern
    backend_db=$(grep -A 30 "^  backend:" docker-compose.yml | grep "DATABASE_URL" | head -1)
    agent_db=$(grep -A 50 "^  agent:" docker-compose.yml | grep "DATABASE_URL" | head -1)

    if [ "$backend_db" = "$agent_db" ]; then
        check_pass "Backend and agent use same DATABASE_URL"
    else
        check_fail "Backend and agent have different DATABASE_URL configurations"
    fi

    # Check all services use same network
    services_on_network=0
    for service in postgres backend agent frontend; do
        if grep -A 50 "^  $service:" docker-compose.yml | grep -q "ai-trader-network"; then
            services_on_network=$((services_on_network + 1))
        fi
    done

    if [ $services_on_network -eq 4 ]; then
        check_pass "All services are on the same Docker network"
    else
        check_fail "Not all services are on the same Docker network ($services_on_network/4)"
    fi
fi

echo ""

# ========================================
# Summary Report
# ========================================
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Validation Summary${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo -e "Total Checks:    $TOTAL_CHECKS"
echo -e "${GREEN}Passed:          $PASSED_CHECKS${NC}"
echo -e "${RED}Failed:          $FAILED_CHECKS${NC}"
echo -e "${YELLOW}Warnings:        $WARNING_CHECKS${NC}"
echo ""

# Calculate success rate
if [ $TOTAL_CHECKS -gt 0 ]; then
    success_rate=$((PASSED_CHECKS * 100 / TOTAL_CHECKS))
    echo -e "Success Rate:    $success_rate%"
    echo ""
fi

# Print failed tests
if [ $FAILED_CHECKS -gt 0 ]; then
    echo -e "${RED}Failed Checks:${NC}"
    for test in "${FAILED_TESTS[@]}"; do
        echo -e "${RED}  ✗${NC} $test"
    done
    echo ""
fi

# Print warnings
if [ $WARNING_CHECKS -gt 0 ]; then
    echo -e "${YELLOW}Warnings:${NC}"
    for warning in "${WARNING_MESSAGES[@]}"; do
        echo -e "${YELLOW}  ⚠${NC} $warning"
    done
    echo ""
fi

# Final status
echo -e "${BLUE}========================================${NC}"
if [ $FAILED_CHECKS -eq 0 ]; then
    echo -e "${GREEN}✓ All critical checks passed!${NC}"
    echo ""
    echo -e "The Docker configuration is valid and ready for deployment."
    exit 0
else
    echo -e "${RED}✗ Validation failed with $FAILED_CHECKS errors${NC}"
    echo ""
    echo -e "Please fix the failed checks before deploying."
    exit 1
fi
