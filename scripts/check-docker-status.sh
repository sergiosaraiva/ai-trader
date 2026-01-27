#!/bin/bash
# Docker Build Status Checker
# Run this in the morning to see what happened overnight

echo "=========================================="
echo "AI Trader Docker Status Report"
echo "=========================================="
echo "Generated: $(date)"
echo ""

# Check if docker-compose processes are still running
echo "1. Build Process Status"
echo "----------------------------------------"
if pgrep -f "docker-compose up" > /dev/null; then
    echo "⏳ Docker Compose is still building..."
    echo "   Processes:"
    ps aux | grep "docker-compose\|docker build" | grep -v grep | head -5
else
    echo "✅ Docker Compose build process completed"
fi
echo ""

# Check container status
echo "2. Container Status"
echo "----------------------------------------"
docker-compose ps 2>&1
echo ""

# Check running containers
echo "3. All Docker Containers"
echo "----------------------------------------"
docker ps -a --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | grep -E "NAME|ai-trader|postgres"
echo ""

# Check images
echo "4. Docker Images"
echo "----------------------------------------"
docker images | grep -E "REPOSITORY|ai-trader|postgres"
echo ""

# Check if services are responding
echo "5. Service Health Checks"
echo "----------------------------------------"

# Backend API
if curl -s http://localhost:8001/health > /dev/null 2>&1; then
    echo "✅ Backend API: HEALTHY (http://localhost:8001)"
else
    echo "❌ Backend API: NOT RESPONDING (http://localhost:8001)"
fi

# Frontend
if curl -s http://localhost:3001 > /dev/null 2>&1; then
    echo "✅ Frontend: HEALTHY (http://localhost:3001)"
else
    echo "❌ Frontend: NOT RESPONDING (http://localhost:3001)"
fi

# Agent
if curl -s http://localhost:8002/health > /dev/null 2>&1; then
    echo "✅ Agent: HEALTHY (http://localhost:8002)"
else
    echo "❌ Agent: NOT RESPONDING (http://localhost:8002)"
fi

# Postgres
if docker-compose exec -T postgres pg_isready > /dev/null 2>&1; then
    echo "✅ PostgreSQL: HEALTHY"
else
    echo "❌ PostgreSQL: NOT RESPONDING"
fi
echo ""

# Check logs for errors
echo "6. Recent Errors (last 20 lines)"
echo "----------------------------------------"
if [ -f /tmp/docker-build-final.log ]; then
    echo "Build log errors:"
    grep -i "error\|failed\|timeout" /tmp/docker-build-final.log | tail -20
    echo ""
fi

# Check disk usage
echo "7. Disk Space Status"
echo "----------------------------------------"
echo "Docker disk usage:"
docker system df
echo ""
echo "Project directories:"
du -sh backend/data backend/models 2>/dev/null
echo ""

# Provide next steps
echo "=========================================="
echo "Next Steps"
echo "=========================================="
echo ""

CONTAINERS_RUNNING=$(docker-compose ps | grep -c "Up")
if [ "$CONTAINERS_RUNNING" -ge 3 ]; then
    echo "✅ SUCCESS! Containers are running."
    echo ""
    echo "Access your application:"
    echo "  Frontend:  http://localhost:3001"
    echo "  Backend:   http://localhost:8001"
    echo "  API Docs:  http://localhost:8001/docs"
    echo "  Agent:     http://localhost:8002/health"
    echo ""
    echo "View logs:"
    echo "  docker-compose logs -f"
    echo "  docker-compose logs -f backend"
    echo "  docker-compose logs -f agent"
elif pgrep -f "docker-compose up" > /dev/null; then
    echo "⏳ Build still in progress. Wait a bit longer or check logs:"
    echo "  tail -f /tmp/docker-build-final.log"
else
    echo "❌ Build appears to have failed. Check logs:"
    echo "  cat /tmp/docker-build-final.log | tail -100"
    echo ""
    echo "Try these recovery options:"
    echo ""
    echo "Option 1: Use existing images (if available)"
    echo "  docker-compose up -d"
    echo ""
    echo "Option 2: Rebuild with cached layers"
    echo "  docker-compose build --parallel"
    echo "  docker-compose up -d"
    echo ""
    echo "Option 3: Run without agent (if agent build failed)"
    echo "  docker-compose up -d postgres backend frontend"
    echo ""
    echo "Option 4: Check detailed logs"
    echo "  docker-compose logs backend | tail -100"
    echo "  docker-compose logs agent | tail -100"
fi
echo ""
echo "=========================================="
