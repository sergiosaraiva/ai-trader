# AI Trader Frontend - Production Dockerfile
# Multi-stage build for optimized production image
# Optimized for Railway deployment

# Stage 1: Build
FROM node:20-alpine AS builder

WORKDIR /app

# Build argument for API URL (set at build time)
ARG VITE_API_URL
ENV VITE_API_URL=${VITE_API_URL}

# Copy package files
COPY package*.json ./

# Install dependencies
RUN npm ci

# Copy source code
COPY . .

# Build the application (VITE_API_URL is embedded at build time)
RUN echo "Building with VITE_API_URL=${VITE_API_URL}" && npm run build

# Stage 2: Production
FROM nginx:alpine

# Install envsubst (gettext package)
RUN apk add --no-cache gettext

# Copy nginx config template
COPY nginx.conf.template /etc/nginx/conf.d/default.conf.template

# Copy entrypoint script
COPY docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh

# Copy built assets from builder
COPY --from=builder /app/dist /usr/share/nginx/html

# Expose port (Railway uses PORT env variable)
EXPOSE 80

# Default backend URL - override with BACKEND_URL env var in Railway
# For Railway: Set BACKEND_URL to your backend service's public URL
# Example: https://ai-trader-backend-production.up.railway.app
ENV BACKEND_URL=http://backend:8001

# No Docker HEALTHCHECK - Railway handles this externally
# The Railway healthcheck in railway.json checks /health on the public port

# Run entrypoint script
ENTRYPOINT ["/docker-entrypoint.sh"]
