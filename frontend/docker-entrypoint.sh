#!/bin/sh
set -e

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

# Start nginx
exec nginx -g 'daemon off;'
