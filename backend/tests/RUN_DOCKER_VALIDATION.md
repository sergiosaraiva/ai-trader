# How to Run Docker Configuration Validation

## Quick Start

```bash
# From project root
./scripts/validate-docker-config.sh
```

## Expected Output

```
========================================
Docker Configuration Validation
========================================

1. Checking File Existence...
✓ backend/Dockerfile.agent exists
✓ backend/docker-entrypoint-agent.sh exists
✓ docker-compose.yml exists
✓ docker-compose.override.yml exists
✓ .env.example exists
✓ Makefile exists

2. Validating backend/Dockerfile.agent...
✓ Base image is python:3.12-slim
✓ WORKDIR is set to /app
✓ Source code (src/) is copied
✓ Models directory is copied
✓ Entrypoint script is copied
✓ ENTRYPOINT is configured
✓ Health check port 8002 is exposed
✓ Entrypoint script permissions are set
✓ PYTHONPATH environment variable is set

... (continues for 108 checks)

========================================
Validation Summary
========================================

Total Checks:    108
Passed:          103
Failed:          3
Warnings:        2

Success Rate:    95%
```

## Current Issues (2026-01-22)

### Critical Issues (Must Fix)
1. **Agent missing network configuration**
   - Add `networks: - ai-trader-network` to agent service in docker-compose.yml

### False Positives (Can Ignore)
2. **AGENT_MT5_PASSWORD validation**
   - Script incorrectly flags empty value as having content
   - Configuration is correct

### Warnings (Informational)
3. **backend/logs directory**
   - Will be created automatically by Docker
   - No action needed

4. **.env file exists**
   - Already in .gitignore
   - Just ensure it's not committed

## Fix Commands

### Fix Network Configuration (Required)

Edit `docker-compose.yml` and add networks section to agent service:

```yaml
agent:
  build:
    context: ./backend
    dockerfile: Dockerfile.agent
  # ... other configuration ...
  networks:
    - ai-trader-network
```

### Create Logs Directory (Optional)

```bash
mkdir -p backend/logs
```

### Verify .env Not Committed

```bash
git status | grep .env
# Should only show .env.example
```

## After Fixing

```bash
# Re-run validation
./scripts/validate-docker-config.sh

# Expected: 108/108 passed (or 107/108 with MT5 false positive)
# Exit code: 0
```

## Test with Docker

```bash
# Build and start services
make up

# Check health
make health

# Should show:
# ✓ Backend healthy
# ✓ Agent healthy
# ✓ Frontend healthy
# ✓ PostgreSQL healthy
```

## Validation Categories

| Category | Checks | Status |
|----------|--------|--------|
| File Existence | 6 | ✓ |
| Dockerfile.agent | 9 | ✓ |
| Entrypoint Script | 8 | ✓ |
| docker-compose.yml | 43 | ⚠ (2 failed) |
| Override File | 6 | ✓ |
| Environment Variables | 18 | ⚠ (1 false positive) |
| Makefile | 11 | ✓ |
| Port Assignments | 5 | ✓ |
| Volume Paths | 4 | ⚠ (1 warning) |
| Security Checks | 4 | ⚠ (1 warning) |
| Integration Checks | 3 | ⚠ (1 failed) |

## What Gets Validated

### Dockerfile Checks
- Base image (python:3.12-slim)
- WORKDIR (/app)
- Required files copied (src/, models/, configs/)
- Entrypoint configured
- Port 8002 exposed
- Permissions set (chmod +x)
- PYTHONPATH environment variable

### Entrypoint Script Checks
- Shebang (#!/bin/bash)
- Error handling (set -e)
- Executable permissions
- PostgreSQL wait logic
- Backend API wait logic
- Live mode validation
- MT5 credentials validation
- Exec usage (signal handling)

### Docker Compose Checks
- YAML syntax validation
- All services defined
- Agent uses Dockerfile.agent
- Dependencies (agent → backend → postgres)
- Health checks (all 4 services)
- Port mappings (no conflicts)
- Volume configuration
- **Network configuration** ⚠ FAILS HERE
- Environment variables (all 11 agent vars)

### Environment Variable Checks
- All required variables documented
- Security warnings present
- No hardcoded secrets
- Valid variable names
- Safe defaults (simulation mode)

### Security Checks
- .env in .gitignore
- No hardcoded credentials in docker-compose.yml
- Agent Dockerfile doesn't copy secrets
- .env file not in version control

## Integration with CI/CD

### GitHub Actions

```yaml
# .github/workflows/docker-validation.yml
name: Docker Validation

on: [push, pull_request]

jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Validate Docker Configuration
        run: |
          chmod +x scripts/validate-docker-config.sh
          ./scripts/validate-docker-config.sh
```

### Pre-commit Hook

```bash
# Add to .git/hooks/pre-commit
#!/bin/bash
./scripts/validate-docker-config.sh || exit 1
```

## Troubleshooting

### Script Won't Execute
```bash
chmod +x scripts/validate-docker-config.sh
```

### docker-compose Command Not Found
```bash
sudo apt-get install docker-compose
# Or use Docker Compose V2
alias docker-compose='docker compose'
```

### Colors Not Showing
```bash
# Colors require terminal support
# In CI/CD, colors may not display
# Exit codes still work: 0 = pass, 1 = fail
```

## Related Files

- **Validation Script:** `scripts/validate-docker-config.sh`
- **Detailed Report:** `backend/tests/DOCKER_CONFIG_VALIDATION_REPORT.md`
- **Quick Reference:** `backend/tests/QUICK_REFERENCE_DOCKER_VALIDATION.md`
- **Machine Output:** `backend/tests/DOCKER_VALIDATION_OUTPUT.yaml`

## Success Criteria

- [ ] 108 checks (or 107 with MT5 false positive)
- [ ] All critical issues fixed
- [ ] Exit code 0
- [ ] All services define networks section
- [ ] No port conflicts
- [ ] All environment variables documented
- [ ] Security checks pass

---

**Last Updated:** 2026-01-22
**Script Version:** 1.0.0
**Exit Code:** 0 = success, 1 = failure
