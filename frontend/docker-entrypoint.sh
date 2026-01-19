#!/bin/sh
set -e

echo "========================================"
echo "=== Frontend Container Starting ==="
echo "========================================"
echo "Date: $(date)"
echo "PORT from Railway: ${PORT:-not set}"
echo "BACKEND_URL from Railway: ${BACKEND_URL:-not set}"
echo ""

# Default values if not set
BACKEND_URL=${BACKEND_URL:-http://backend:8001}
PORT=${PORT:-80}

# Export for envsubst
export BACKEND_URL
export PORT

# Substitute environment variables in nginx config
envsubst '${BACKEND_URL} ${PORT}' < /etc/nginx/conf.d/default.conf.template > /etc/nginx/conf.d/default.conf

# Log configuration
echo "Starting nginx with:"
echo "  - BACKEND_URL: ${BACKEND_URL}"
echo "  - PORT: ${PORT}"
echo ""

# Show nginx config
echo "=== Generated nginx config ==="
cat /etc/nginx/conf.d/default.conf
echo "==============================="
echo ""

# Test nginx config
echo "Testing nginx configuration..."
nginx -t

echo "Starting nginx..."
exec nginx -g 'daemon off;'
