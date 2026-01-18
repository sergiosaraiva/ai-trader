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

## Deployment Modes

The backend supports two deployment modes:

### Option 1: Always-On (Default)
- **Best for**: Continuous monitoring, real-time updates
- **Cost**: ~$5-10/month (runs 24/7)
- **How it works**: APScheduler runs background jobs automatically
- **Environment**: `SCHEDULER_ENABLED=true` (default)

### Option 2: Cron Mode (Cost-Optimized)
- **Best for**: Cost savings, serverless-style deployment
- **Cost**: Pay only when running (~$1-3/month)
- **How it works**: Railway Cron Jobs trigger HTTP endpoints
- **Environment**: `SCHEDULER_ENABLED=false`

## Prerequisites

1. A Railway account (https://railway.app)
2. Railway CLI installed (optional, for CLI deployment)
3. Git repository with the AI Trader code
4. FRED API key (free, for sentiment data): https://fredaccount.stlouisfed.org/apikeys

## Deployment Steps

### Step 1: Create a New Project in Railway

1. Go to https://railway.app/dashboard
2. Click "New Project"
3. Select "Empty Project"

### Step 2: Deploy the Backend Service

1. In your project, click "New" → "GitHub Repo"
2. Select your AI Trader repository
3. Railway will auto-detect the `Dockerfile` in the backend folder
4. Configure the service:

   **Service Settings:**
   - Name: `backend` (or `ai-trader-backend`)
   - Root Directory: `/backend`

   **Environment Variables (Always-On Mode):**
   ```
   PORT=8001
   DATABASE_URL=sqlite:////app/data/db/trading.db
   FRED_API_KEY=your_fred_api_key
   SCHEDULER_ENABLED=true
   ```

   **Environment Variables (Cron Mode):**
   ```
   PORT=8001
   DATABASE_URL=sqlite:////app/data/db/trading.db
   FRED_API_KEY=your_fred_api_key
   SCHEDULER_ENABLED=false
   CRON_API_KEY=your_secret_key_here
   ```

5. **Configure Memory** (Settings → Resources):
   - **Recommended**: 512 MB
   - **Minimum**: 384 MB (tight, may OOM during heavy operations)
   - The backend loads 3 XGBoost models + price data (~400 MB typical usage)

6. **Add a Volume** (IMPORTANT for database persistence):
   - Click on the backend service
   - Go to "Settings" → "Volumes"
   - Click "Add Volume"
   - Mount Path: `/app/data/db`
   - This ensures your SQLite database persists across deployments

7. Generate a public domain:
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

4. **Configure Memory** (Settings → Resources):
   - **Recommended**: 128 MB (nginx is lightweight)

5. Generate a public domain:
   - Go to "Settings" → "Networking"
   - Click "Generate Domain"
   - This is your public application URL

### Step 4: Configure Cron Jobs (Cron Mode Only)

If using `SCHEDULER_ENABLED=false`, set up Railway Cron Jobs:

1. In your project, click "New" → "Cron Job"
2. Create two cron jobs:

   **Hourly Update (Main Cycle):**
   - Name: `hourly-tick`
   - Schedule: `1 * * * *` (every hour at :01)
   - Command:
     ```bash
     curl -X POST https://your-backend-url.railway.app/api/v1/cron/tick \
       -H "X-Cron-Key: your_secret_key_here"
     ```

   **Position Check (Every 5 Minutes):**
   - Name: `position-check`
   - Schedule: `*/5 * * * *` (every 5 minutes)
   - Command:
     ```bash
     curl -X POST https://your-backend-url.railway.app/api/v1/cron/check \
       -H "X-Cron-Key: your_secret_key_here"
     ```

### Step 5: Verify Deployment

1. Check backend health:
   ```bash
   curl https://your-backend-url.railway.app/health
   ```

2. Check operating mode:
   ```bash
   curl https://your-backend-url.railway.app/api/v1/cron/status
   ```
   Should show `"mode": "scheduler"` or `"mode": "cron"`

3. Check detailed health:
   ```bash
   curl https://your-backend-url.railway.app/health/detailed
   ```

4. Check frontend:
   - Open `https://your-frontend-url.railway.app` in a browser

5. Test prediction endpoint:
   ```bash
   curl https://your-backend-url.railway.app/api/v1/predictions/latest
   ```

## Environment Variables Reference

### Backend Service

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `PORT` | Yes | 8001 | Port for the API server (Railway sets this) |
| `DATABASE_URL` | No | `sqlite:////app/data/db/trading.db` | Database connection string |
| `FRED_API_KEY` | No | - | FRED API key for VIX/EPU sentiment data |
| `SCHEDULER_ENABLED` | No | `true` | `true` for always-on, `false` for cron mode |
| `CRON_API_KEY` | No | - | Secret key to secure cron endpoints (recommended for cron mode) |

### Frontend Service

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `PORT` | Yes | 80 | Port for nginx (Railway sets this) |
| `BACKEND_URL` | Yes | - | Full URL to the backend service |

## Resource Recommendations

### Memory

| Service | Minimum | Recommended | Notes |
|---------|---------|-------------|-------|
| Backend | 384 MB | **512 MB** | Loads 3 ML models + price data |
| Frontend | 64 MB | **128 MB** | Static nginx server |

### Storage

| Service | Volume Mount | Size | Purpose |
|---------|--------------|------|---------|
| Backend | `/app/data/db` | ~100 MB | SQLite database (trades, predictions) |
| Frontend | None | - | Static files, no persistence needed |

## Volume Configuration

### Why Volumes are Needed

The backend uses SQLite to store:
- Paper trading account balance
- Trade history
- Prediction logs

Without a volume, this data is lost on every deployment.

### Important Notes

1. **Railway Hobby Plan**: 5GB volume storage included
2. **Volume mounting**: Happens at runtime, not build time
3. **Single volume per service**: Railway limitation
4. **No replicas with volumes**: Railway limitation

## Cost Estimation

Railway pricing (as of 2024):

| Plan | Resources | Volume | Cost |
|------|-----------|--------|------|
| Hobby | $5 credit/month | 5GB | $5/month |
| Pro | Usage-based | 50GB | ~$20+/month |

### Estimated Monthly Cost

**Always-On Mode (SCHEDULER_ENABLED=true):**
| Service | Memory | Hours | Est. Cost |
|---------|--------|-------|-----------|
| Backend | 512 MB | 720 | ~$3-5 |
| Frontend | 128 MB | 720 | ~$1-2 |
| **Total** | | | **~$5-7/month** |

**Cron Mode (SCHEDULER_ENABLED=false):**
| Service | Memory | Hours | Est. Cost |
|---------|--------|-------|-----------|
| Backend | 512 MB | ~50* | ~$0.50-1 |
| Frontend | 128 MB | 720 | ~$1-2 |
| Cron Jobs | - | - | ~$0 |
| **Total** | | | **~$2-3/month** |

*Backend sleeps when idle, wakes on requests

## API Endpoints for Cron

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/cron/status` | GET | Check current mode (scheduler/cron) |
| `/api/v1/cron/tick` | POST | Run full hourly cycle (pipeline + prediction + trade) |
| `/api/v1/cron/check` | POST | Check positions for TP/SL exits |

### Cron Tick Response Example
```json
{
  "status": "success",
  "results": {
    "timestamp": "2024-01-15T10:01:00Z",
    "pipeline": {"status": "success"},
    "prediction": {
      "direction": "long",
      "confidence": 0.72,
      "should_trade": true
    },
    "trade": {"executed": true, "direction": "long"},
    "errors": []
  }
}
```

## Troubleshooting

### Backend won't start

1. Check logs in Railway dashboard
2. Verify `PORT` environment variable is set
3. Check health endpoint: `/health`
4. **Memory issue?** Increase to 512 MB or higher

### Backend runs out of memory

1. Increase memory allocation in Settings → Resources
2. Minimum 384 MB, recommended 512 MB
3. Check for memory leaks in logs

### Scheduler not running jobs

1. Check `/api/v1/cron/status` - verify `scheduler_enabled: true`
2. Check logs for scheduler errors
3. If using cron mode, ensure cron jobs are configured

### Frontend can't reach backend

1. Verify `BACKEND_URL` is set correctly (must be full URL with https://)
2. Check backend is running and healthy
3. Test backend directly: `curl https://backend-url/health`

### Database resets on deployment

1. Ensure volume is attached to backend service
2. Mount path must be `/app/data/db`
3. Check volume is properly configured in Railway dashboard

### Cron endpoints return 401

1. If `CRON_API_KEY` is set, include `X-Cron-Key` header
2. Or remove `CRON_API_KEY` to disable auth

### CORS errors

The backend has CORS configured for all origins. If you need to restrict:
1. Set `CORS_ORIGINS` environment variable
2. Format: comma-separated list of allowed origins

## CLI Deployment

If you prefer using the Railway CLI:

```bash
# Install Railway CLI
npm install -g @railway/cli

# Login
railway login

# Create project
railway init

# Deploy backend
cd backend
railway link
railway up

# Deploy frontend
cd ../frontend
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

### Useful Log Queries

```
# Check scheduler status
"Scheduler started" OR "SCHEDULER_ENABLED"

# Check predictions
"Prediction:" AND "conf:"

# Check errors
level:error
```

## Security Recommendations

1. **Set CRON_API_KEY** when using cron mode to prevent unauthorized access
2. **Use HTTPS** (Railway provides this automatically)
3. **Don't expose FRED_API_KEY** in logs or frontend
4. **Restrict CORS** for production if needed

## Getting a FRED API Key

The FRED API key is used to fetch VIX and EPU sentiment data:

1. Go to https://fredaccount.stlouisfed.org/apikeys
2. Create a free account
3. Request an API key
4. Add to Railway environment variables as `FRED_API_KEY`

Without this key, sentiment features will be disabled but the system will still work.
