#!/bin/sh
set -e

# Default backend URL if not set
BACKEND_URL=${BACKEND_URL:-http://backend:8001}

# Substitute environment variables in nginx config
envsubst '${BACKEND_URL}' < /etc/nginx/conf.d/default.conf.template > /etc/nginx/conf.d/default.conf

# Start nginx
exec nginx -g 'daemon off;'
