#!/bin/bash
set -e

# Trap signals for graceful shutdown
trap 'echo "Received shutdown signal, stopping agent..."; exit 0' SIGTERM SIGINT

echo "=========================================="
echo "Starting AI Trading Agent..."
echo "=========================================="

# Display configuration (mask sensitive values)
echo "Configuration:"
echo "  Mode: ${AGENT_MODE:-simulation}"
echo "  Symbol: ${AGENT_SYMBOL:-EURUSD}"
echo "  Confidence Threshold: ${AGENT_CONFIDENCE_THRESHOLD:-0.70}"
echo "  Cycle Interval: ${AGENT_CYCLE_INTERVAL:-60}s"
echo "  Health Port: ${AGENT_HEALTH_PORT:-8002}"
# Never log credentials or API keys
if [ -n "$AGENT_MT5_LOGIN" ]; then
    echo "  MT5 Login: ***configured***"
fi
if [ -n "$OPENAI_API_KEY" ]; then
    echo "  OpenAI API: ***configured***"
fi

# Wait for PostgreSQL to be ready
if [ -n "$DATABASE_URL" ]; then
    echo "Waiting for PostgreSQL..."
    # Extract host and port from DATABASE_URL (format: postgresql://user:pass@host:port/db)
    # More robust parsing using Python for complex URLs
    if command -v python3 >/dev/null 2>&1; then
        DB_HOST=$(python3 -c "from urllib.parse import urlparse; u = urlparse('$DATABASE_URL'); print(u.hostname or 'localhost')")
        DB_PORT=$(python3 -c "from urllib.parse import urlparse; u = urlparse('$DATABASE_URL'); print(u.port or 5432)")
    else
        # Fallback to sed parsing
        DB_HOST=$(echo "$DATABASE_URL" | sed -E 's|.*@([^:]+):.*|\1|')
        DB_PORT=$(echo "$DATABASE_URL" | sed -E 's|.*:([0-9]+)/.*|\1|')
    fi

    # Wait up to 60 seconds for database
    MAX_RETRIES=60
    RETRY_COUNT=0

    while ! pg_isready -h "$DB_HOST" -p "$DB_PORT" -U "${POSTGRES_USER:-aitrader}" > /dev/null 2>&1; do
        RETRY_COUNT=$((RETRY_COUNT + 1))
        if [ $RETRY_COUNT -ge $MAX_RETRIES ]; then
            echo "ERROR: PostgreSQL is not ready after ${MAX_RETRIES} seconds"
            exit 1
        fi
        echo "  Waiting for PostgreSQL... (attempt $RETRY_COUNT/$MAX_RETRIES)"
        sleep 1
    done

    echo "PostgreSQL is ready!"
fi

# Wait for backend API to be healthy (agent depends on backend services)
if [ -n "$BACKEND_URL" ]; then
    BACKEND_HOST="${BACKEND_URL:-http://backend:8001}"
    echo "Waiting for backend API at $BACKEND_HOST..."

    MAX_RETRIES=60
    RETRY_COUNT=0

    while ! curl -f -s "$BACKEND_HOST/health" > /dev/null 2>&1; do
        RETRY_COUNT=$((RETRY_COUNT + 1))
        if [ $RETRY_COUNT -ge $MAX_RETRIES ]; then
            echo "WARNING: Backend API not responding after ${MAX_RETRIES} seconds"
            echo "Continuing anyway, agent will retry connections..."
            break
        fi
        echo "  Waiting for backend API... (attempt $RETRY_COUNT/$MAX_RETRIES)"
        sleep 1
    done

    if [ $RETRY_COUNT -lt $MAX_RETRIES ]; then
        echo "Backend API is ready!"
    fi
fi

# Validate required environment variables for live mode
if [ "$AGENT_MODE" = "live" ]; then
    # Check for required credentials
    if [ -z "$AGENT_MT5_LOGIN" ] || [ -z "$AGENT_MT5_PASSWORD" ] || [ -z "$AGENT_MT5_SERVER" ]; then
        echo "ERROR: Live mode requires MT5 credentials (AGENT_MT5_LOGIN, AGENT_MT5_PASSWORD, AGENT_MT5_SERVER)"
        exit 1
    fi

    # Require explicit confirmation for live trading
    if [ "$AGENT_LIVE_TRADING_CONFIRMED" != "true" ]; then
        echo "======================================================="
        echo "ERROR: Live trading requires explicit confirmation"
        echo "======================================================="
        echo "To enable live trading, you must:"
        echo "1. Complete thorough testing in simulation mode"
        echo "2. Complete thorough testing in paper mode"
        echo "3. Review all safety settings and risk limits"
        echo "4. Set AGENT_LIVE_TRADING_CONFIRMED=true"
        echo "======================================================="
        echo "LIVE TRADING INVOLVES REAL MONEY AND RISK OF LOSS"
        echo "======================================================="
        exit 1
    fi

    echo "======================================================="
    echo "WARNING: LIVE TRADING MODE - REAL MONEY AT RISK!"
    echo "======================================================="
    echo "Ensure all safety limits are properly configured:"
    echo "  Max consecutive losses: ${AGENT_MAX_CONSECUTIVE_LOSSES}"
    echo "  Max drawdown: ${AGENT_MAX_DRAWDOWN_PERCENT}%"
    echo "  Max daily loss: ${AGENT_MAX_DAILY_LOSS_PERCENT}%"
    echo "======================================================="
fi

# Display startup warnings
if [ "$AGENT_MODE" = "simulation" ]; then
    echo "INFO: Running in SIMULATION mode (no real trades)"
elif [ "$AGENT_MODE" = "paper" ]; then
    echo "INFO: Running in PAPER mode (real price feed, no real money)"
elif [ "$AGENT_MODE" = "live" ]; then
    echo "WARNING: Running in LIVE mode - REAL MONEY AT RISK!"
    echo "WARNING: Ensure you have tested thoroughly in simulation/paper mode first"
fi

echo "=========================================="
echo "Starting agent main loop..."
echo "=========================================="

# Start the agent
exec python -m src.agent.main
