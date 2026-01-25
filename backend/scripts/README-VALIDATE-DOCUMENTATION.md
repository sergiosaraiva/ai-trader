# Documentation Validation Script

## Overview

The `validate-documentation.sh` script performs comprehensive validation of Phase 9 documentation to ensure completeness, correctness, and consistency.

## Usage

```bash
# Run validation
./scripts/validate-documentation.sh

# View the validation report
cat docs/DOCUMENTATION-VALIDATION-REPORT.md
```

## Validation Checks

### 1. File Existence
Verifies that all required documentation files exist:
- `docs/AI-TRADING-AGENT.md`
- `docs/AGENT-OPERATIONS-GUIDE.md`
- `docs/AGENT-API-REFERENCE.md`
- `docs/CHANGELOG.md`
- `docs/AGENT-QUICK-REFERENCE.md`
- `README.md` (project root)

### 2. Markdown Syntax
Checks for valid markdown syntax:
- Top-level headers exist
- Code blocks are balanced (even number of ``` markers)
- Tables are present and properly formatted

### 3. Internal Link Validation
Validates all internal markdown links:
- Verifies that referenced files exist
- Skips external HTTP/HTTPS links
- Checks relative path resolution

### 4. Code Block Language Tags
Ensures all code blocks have language tags:
```python  # Good
def example():
    pass
```

```       # Bad (no language tag)
code here
```

### 5. Required Sections
Verifies that key documentation sections exist:

**AI-TRADING-AGENT.md**:
- Overview
- Architecture
- Installation
- Configuration
- Usage

**AGENT-OPERATIONS-GUIDE.md**:
- Deployment
- Monitoring
- Troubleshooting

**AGENT-API-REFERENCE.md**:
- Endpoints
- Schemas
- Authentication

**CHANGELOG.md**:
- Version format `## [X.Y.Z]`

### 6. Environment Variables
Cross-references `.env.example` with documentation:
- Extracts all env vars from `.env.example`
- Checks that each is documented
- Reports coverage percentage

### 7. API Endpoints
Cross-references `src/api/routes/agent.py` with documentation:
- Extracts all API routes from code
- Checks that each is documented in AGENT-API-REFERENCE.md
- Reports coverage percentage

### 8. Version Consistency
Ensures version numbers are consistent:
- Extracts version numbers from all docs
- Verifies they match
- Warns if no version info found

## Output

The script produces:

1. **Console Output**: Real-time pass/fail status with colors
   - 🟢 Green: PASS
   - 🔴 Red: FAIL
   - 🟡 Yellow: WARN

2. **Validation Report**: `docs/DOCUMENTATION-VALIDATION-REPORT.md`
   - Executive summary with pass rate
   - Detailed check results
   - Recommendations for fixes
   - Validation metadata

## Exit Codes

- `0`: All checks passed
- `1`: One or more checks failed

## Example Output

```
[INFO] Starting documentation validation...
[INFO] Target: Phase 9 - Documentation

## 1. File Existence Validation

[FAIL] File missing: AI-TRADING-AGENT.md
[PASS] File exists: README.md

================================
   VALIDATION SUMMARY
================================
Total Checks:  10
Passed:        2
Failed:        8
Pass Rate:     20%
================================

Report saved to: docs/DOCUMENTATION-VALIDATION-REPORT.md
```

## Integration with CI/CD

Add to your CI pipeline:

```yaml
- name: Validate Documentation
  run: |
    cd backend
    ./scripts/validate-documentation.sh
```

## Troubleshooting

### "No version numbers found"
Add version info to your docs:
```markdown
**Version**: 1.0.0
```

### "Some environment variables not documented"
Ensure all vars from `.env.example` are mentioned in documentation.

### "Code blocks missing language tags"
Add language identifiers to all code blocks:
```python  # Not just ```
```

## Maintenance

Update the script when:
- New documentation files are added
- New required sections are defined
- New validation rules are needed

## Related

- `docs/DOCUMENTATION-VALIDATION-REPORT.md` - Latest validation results
- `.env.example` - Environment variable reference
- `src/api/routes/agent.py` - API endpoint definitions

---

**Version**: 1.0.0
**Last Updated**: 2026-01-22
