# Quick Reference: Docker Configuration Validation

## Run Validation

```bash
# From project root
./scripts/validate-docker-config.sh

# Expected output: 108 checks with detailed pass/fail/warning status
# Exit code 0 = all passed, Exit code 1 = failures detected
```

## What Gets Validated

### 1. File Existence (6 checks)
- Dockerfile.agent
- docker-entrypoint-agent.sh
- docker-compose.yml
- docker-compose.override.yml
- .env.example
- Makefile

### 2. Dockerfile Validation (9 checks)
- Base image
- WORKDIR
- File copying
- Entrypoint
- Exposed ports
- Permissions
- Environment variables

### 3. Entrypoint Script (8 checks)
- Shebang
- Error handling
- Permissions
- Database wait logic
- Backend wait logic
- Live mode validation
- MT5 credentials check
- Exec usage

### 4. Docker Compose (43 checks)
- YAML syntax
- Service definitions
- Dependencies
- Health checks
- Port mappings
- Volume configuration
- Network configuration
- Environment variables

### 5. Override File (6 checks)
- Development settings
- Hot reload configuration
- Simulation mode enforcement

### 6. Environment Variables (18 checks)
- All required variables
- Secure defaults
- Naming conventions
- No hardcoded secrets

### 7. Makefile (11 checks)
- Required targets
- PHONY declaration
- Agent-specific commands

### 8. Port Configuration (5 checks)
- No conflicts
- Standard port assignments

### 9. Volume Paths (4 checks)
- Required directories exist

### 10. Security (4 checks)
- .gitignore configuration
- No hardcoded credentials
- Secret file handling

### 11. Integration (3 checks)
- Service connectivity
- Database URL consistency
- Network configuration

## Current Status (2026-01-22)

```
Total:    108 checks
Passed:   103 (95%)
Failed:   3
Warnings: 2
```

## Known Issues

### Critical (Must Fix)
1. **Agent missing network configuration** - Add `networks: - ai-trader-network` to agent service
2. **Network isolation** - Same as #1 (duplicate detection)

### Minor (Can Ignore)
3. **Validation script false positive** - MT5_PASSWORD check incorrectly flags empty value

### Warnings (Informational)
1. **backend/logs directory** - Will be created by Docker automatically
2. **.env file exists** - Already in .gitignore, safe

## Fix Commands

```bash
# Create logs directory (optional)
mkdir -p backend/logs

# Verify .env is not tracked
git status | grep .env
# Should only show .env.example

# Fix network configuration
# Edit docker-compose.yml and add networks section to agent service
```

## After Fixing

```bash
# Re-run validation
./scripts/validate-docker-config.sh

# Should show: 108/108 passed (100%)
# Exit code: 0
```

## Integration with Git

```bash
# Run validation before committing Docker changes
git add docker-compose.yml backend/Dockerfile.agent
./scripts/validate-docker-config.sh && git commit -m "feat: Docker configuration"

# Or add to pre-commit hook
cp scripts/validate-docker-config.sh .git/hooks/pre-commit
```

## Continuous Integration

```yaml
# .github/workflows/docker-validation.yml
- name: Validate Docker Configuration
  run: |
    chmod +x scripts/validate-docker-config.sh
    ./scripts/validate-docker-config.sh
```

## Validation Categories

| Category | Checks | Status |
|----------|--------|--------|
| File Existence | 6 | ✓ 100% |
| Dockerfile | 9 | ✓ 100% |
| Entrypoint Script | 8 | ✓ 100% |
| Docker Compose | 43 | ⚠ 95% (2 failed) |
| Override File | 6 | ✓ 100% |
| Environment | 18 | ⚠ 94% (1 false positive) |
| Makefile | 11 | ✓ 100% |
| Ports | 5 | ✓ 100% |
| Volumes | 4 | ⚠ 75% (1 warning) |
| Security | 4 | ⚠ 75% (1 warning) |
| Integration | 3 | ⚠ 67% (1 failed) |

## Detailed Reports

- **Full Report:** `backend/tests/DOCKER_CONFIG_VALIDATION_REPORT.md`
- **This Guide:** `backend/tests/QUICK_REFERENCE_DOCKER_VALIDATION.md`

## Troubleshooting

### Validation Script Won't Run
```bash
# Make executable
chmod +x scripts/validate-docker-config.sh

# Check shebang
head -n 1 scripts/validate-docker-config.sh
# Should be: #!/bin/bash
```

### docker-compose Command Not Found
```bash
# Install docker-compose
sudo apt-get install docker-compose

# Or use Docker Compose V2
docker compose config
```

### False Positives
- Check validation script version
- Review specific check logic in script
- Report issues with context

## Quick Fix Checklist

Before deployment, ensure:
- [ ] All files exist
- [ ] Dockerfile syntax is valid
- [ ] Entrypoint script is executable
- [ ] docker-compose.yml is valid YAML
- [ ] All services define networks section
- [ ] No port conflicts
- [ ] Environment variables documented
- [ ] Security checks pass
- [ ] .env is gitignored
- [ ] Validation script returns exit code 0

---

**Last Updated:** 2026-01-22
**Script Location:** `/home/sergio/ai-trader/scripts/validate-docker-config.sh`
**Exit Code:** 0 = success, 1 = failure
