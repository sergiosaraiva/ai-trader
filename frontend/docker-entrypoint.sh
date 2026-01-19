#!/bin/sh
set -e

# Debug: Print all environment variables related to backend
echo "=== Environment Debug ==="
echo "BACKEND_URL from env: ${BACKEND_URL}"
env | grep -i backend || true
echo "========================="

# Default values if not set
BACKEND_URL=${BACKEND_URL:-http://backend:8001}
PORT=${PORT:-80}

# Export for envsubst
export BACKEND_URL
export PORT

# Substitute environment variables in nginx config
envsubst '${BACKEND_URL} ${PORT}' < /etc/nginx/conf.d/default.conf.template > /etc/nginx/conf.d/default.conf

# Log configuration for debugging
echo "Starting nginx with:"
echo "  - BACKEND_URL: ${BACKEND_URL}"
echo "  - PORT: ${PORT}"

# Debug: Show the actual nginx config proxy_pass line
echo "=== Nginx proxy_pass config ==="
grep -i proxy_pass /etc/nginx/conf.d/default.conf || echo "No proxy_pass found"
echo "==============================="

# Start nginx
exec nginx -g 'daemon off;'
