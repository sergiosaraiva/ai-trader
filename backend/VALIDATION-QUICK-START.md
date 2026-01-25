# Documentation Validation - Quick Start

**5-Minute Guide to Using the Validation System**

## Run Validation

```bash
cd backend
./scripts/validate-documentation.sh
```

## View Results

```bash
# Console output shows immediate pass/fail
# Report saved to:
cat docs/DOCUMENTATION-VALIDATION-REPORT.md
```

## Expected Output

```
[INFO] Starting documentation validation...

## 1. File Existence Validation

[PASS] File exists: README.md
[FAIL] File missing: AI-TRADING-AGENT.md

================================
   VALIDATION SUMMARY
================================
Total Checks:  10
Passed:        2
Failed:        8
Pass Rate:     20%
================================
```

## Exit Codes

- `0` = All checks passed
- `1` = Some checks failed

## CI/CD Integration

### GitHub Actions
```yaml
- name: Validate Docs
  run: cd backend && ./scripts/validate-documentation.sh
```

### GitLab CI
```yaml
validate-docs:
  script:
    - cd backend
    - ./scripts/validate-documentation.sh
```

## Files Validated

1. `docs/AI-TRADING-AGENT.md` - Main docs
2. `docs/AGENT-OPERATIONS-GUIDE.md` - Ops guide
3. `docs/AGENT-API-REFERENCE.md` - API ref
4. `docs/CHANGELOG.md` - Changelog
5. `docs/AGENT-QUICK-REFERENCE.md` - Quick ref
6. `README.md` - Project readme

## Checks Performed

- ✅ File existence
- ✅ Markdown syntax
- ✅ Internal links
- ✅ Code block language tags
- ✅ Required sections
- ✅ Environment variables
- ✅ API endpoints
- ✅ Version consistency

## Common Issues

### Missing Language Tags
```markdown
# Bad
```
code here
```

# Good
```python
code here
```
```

### Unbalanced Code Blocks
- Every ` ``` ` opening needs a ` ``` ` closing
- Count must be even

### Broken Links
- Check relative paths: `[link](../path/to/file.md)`
- Internal links must point to existing files

### Missing Sections
Each doc needs specific sections (see README-VALIDATE-DOCUMENTATION.md)

## More Info

- Full docs: `scripts/README-VALIDATE-DOCUMENTATION.md`
- Test results: `DOCUMENTATION-VALIDATION-TEST-REPORT.md`
- Implementation: `DOCUMENTATION-VALIDATION-SUMMARY.md`

---

**Quick Start Version**: 1.0.0
