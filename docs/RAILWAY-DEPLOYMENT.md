# Railway Deployment Guide

This guide explains how to deploy the AI Trader application to Railway cloud platform.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    RAILWAY DEPLOYMENT                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌─────────────────┐         ┌─────────────────┐              │
│   │    Frontend     │         │    Backend      │              │
│   │   (nginx)       │ ──────► │   (FastAPI)     │              │
│   │                 │  HTTPS  │                 │              │
│   │   Port: $PORT   │         │   Port: $PORT   │              │
│   └─────────────────┘         └────────┬────────┘              │
│                                        │                        │
│                               ┌────────▼────────┐              │
│                               │     Volume      │              │
│                               │  /app/data/db   │              │
│                               │  (SQLite DB)    │              │
│                               └─────────────────┘              │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Prerequisites

1. A Railway account (https://railway.app)
2. Railway CLI installed (optional, for CLI deployment)
3. Git repository with the AI Trader code

## Deployment Steps

### Step 1: Create a New Project in Railway

1. Go to https://railway.app/dashboard
2. Click "New Project"
3. Select "Empty Project"

### Step 2: Deploy the Backend Service

1. In your project, click "New" → "GitHub Repo"
2. Select your AI Trader repository
3. Railway will auto-detect the `Dockerfile` in the root
4. Configure the service:

   **Service Settings:**
   - Name: `backend` (or `ai-trader-backend`)
   - Root Directory: `/` (leave empty for root)

   **Environment Variables:**
   ```
   PORT=8001
   DATABASE_URL=sqlite:////app/data/db/trading.db
   FRED_API_KEY=your_fred_api_key (optional, for sentiment data)
   ```

5. **Add a Volume** (IMPORTANT for database persistence):
   - Click on the backend service
   - Go to "Settings" → "Volumes"
   - Click "Add Volume"
   - Mount Path: `/app/data/db`
   - This ensures your SQLite database persists across deployments

6. Generate a public domain:
   - Go to "Settings" → "Networking"
   - Click "Generate Domain"
   - Note this URL (e.g., `https://ai-trader-backend-production.up.railway.app`)

### Step 3: Deploy the Frontend Service

1. In the same project, click "New" → "GitHub Repo"
2. Select the same repository
3. Configure the service:

   **Service Settings:**
   - Name: `frontend` (or `ai-trader-frontend`)
   - Root Directory: `/frontend`

   **Environment Variables:**
   ```
   BACKEND_URL=https://ai-trader-backend-production.up.railway.app
   ```
   (Use the backend URL from Step 2)

4. Generate a public domain:
   - Go to "Settings" → "Networking"
   - Click "Generate Domain"
   - This is your public application URL

### Step 4: Verify Deployment

1. Check backend health:
   ```
   curl https://your-backend-url.railway.app/health
   ```

2. Check frontend:
   - Open `https://your-frontend-url.railway.app` in a browser

3. Verify API proxy:
   ```
   curl https://your-frontend-url.railway.app/api/v1/predictions/current
   ```

## Environment Variables Reference

### Backend Service

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `PORT` | Yes | 8001 | Port for the API server (Railway sets this) |
| `DATABASE_URL` | No | `sqlite:////app/data/db/trading.db` | Database connection string |
| `FRED_API_KEY` | No | - | FRED API key for sentiment data updates |

### Frontend Service

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `PORT` | Yes | 80 | Port for nginx (Railway sets this) |
| `BACKEND_URL` | Yes | - | Full URL to the backend service |

## Volume Configuration

### Why Volumes are Needed

The backend uses SQLite to store:
- Paper trading account balance
- Trade history
- Signal predictions log

Without a volume, this data is lost on every deployment.

### Volume Setup

| Service | Volume Mount Path | Purpose |
|---------|------------------|---------|
| Backend | `/app/data/db` | SQLite database persistence |
| Frontend | None | Static files, no persistence needed |

### Important Notes

1. **Railway Free Tier**: 0.5GB volume storage
2. **Volume mounting**: Happens at runtime, not build time
3. **Single volume per service**: Railway limitation
4. **No replicas with volumes**: Railway limitation

## Troubleshooting

### Backend won't start

1. Check logs in Railway dashboard
2. Verify `PORT` environment variable is set
3. Check health endpoint: `/health`

### Frontend can't reach backend

1. Verify `BACKEND_URL` is set correctly (must be full URL with https://)
2. Check backend is running and healthy
3. Test backend directly: `curl https://backend-url/health`

### Database resets on deployment

1. Ensure volume is attached to backend service
2. Mount path must be `/app/data/db`
3. Check volume is properly configured in Railway dashboard

### CORS errors

The backend already has CORS configured for all origins. If you need to restrict:
1. Set `CORS_ORIGINS` environment variable
2. Format: comma-separated list of allowed origins

## Cost Estimation

Railway pricing (as of 2024):

| Plan | Resources | Volume | Cost |
|------|-----------|--------|------|
| Free/Trial | Limited | 0.5GB | $0 |
| Hobby | $5 credit | 5GB | $5/month |
| Pro | Usage-based | 50GB | ~$20+/month |

Estimated usage for AI Trader:
- Backend: ~0.5 GB RAM, minimal CPU
- Frontend: ~0.1 GB RAM, minimal CPU
- Volume: ~100 MB (database)

## Alternative: CLI Deployment

If you prefer using the Railway CLI:

```bash
# Install Railway CLI
npm install -g @railway/cli

# Login
railway login

# Create project
railway init

# Deploy backend
railway up

# Link frontend (from frontend directory)
cd frontend
railway link
railway up
```

## Updating the Deployment

Railway automatically deploys when you push to your connected GitHub branch.

For manual redeployment:
1. Go to the service in Railway dashboard
2. Click "Deploy" → "Redeploy"

## Monitoring

Railway provides built-in monitoring:
- **Logs**: Real-time application logs
- **Metrics**: CPU, Memory, Network usage
- **Alerts**: Configure notifications for failures

Access via the "Observability" tab in each service.
