# Framework Continuous Improvement System

This directory contains the continuous improvement infrastructure for the Agent-Skill framework.

## Purpose

Enable the framework to learn from mistakes and evolve alongside the codebase through structured error capture, analysis, and remediation.

## Directory Structure

```
.claude/improvement/
├── README.md                     # This file
├── error-template.md             # Template for reporting errors
├── maintenance-checklist.md      # Quarterly review checklist
├── errors/                       # Individual error reports
│   └── ERR-YYYY-MM-DD-NNN.md    # One file per error (with Error ID)
└── YYYY-QN-maintenance-report.md # Quarterly reports (when generated)

.claude/scripts/
└── validate-framework.sh         # YAML validation script

.claude/hooks/
└── pre-commit-framework-check.sh # Git pre-commit hook

.claude/skills/improving-framework-continuously/
└── SKILL.md                      # Continuous improvement skill
```

## Quick Start

### YAML Validation (Run First!)

Before any framework work, validate your skills and agents:

```bash
# Run full validation
.claude/scripts/validate-framework.sh

# Install pre-commit hook (one-time setup)
cp .claude/hooks/pre-commit-framework-check.sh .git/hooks/pre-commit
chmod +x .git/hooks/pre-commit
```

### Reporting an Error

When you encounter wrong agent/skill behavior:

```bash
# 1. Copy the template with Error ID
cp error-template.md errors/ERR-$(date +%Y-%m-%d)-001-brief-description.md

# 2. Open and fill out all sections
# Focus on:
#   - Error ID: ERR-YYYY-MM-DD-NNN
#   - Severity (auto-classify by error type)
#   - What went wrong (expected vs actual)
#   - Evidence (code snippets, outputs)
#   - Root cause analysis
#   - How you verified the correct behavior

# 3. Save and continue work
# Don't block on the report - it will be reviewed weekly
```

### Weekly Review Process

1. Collect all new error reports from `errors/`
2. Triage by severity (Critical → Low)
3. For each error:
   - Identify root cause
   - Update affected skill/agent
   - Add test case
   - Mark resolved
4. Update metrics

### Quarterly Maintenance

1. Copy `maintenance-checklist.md` to `YYYY-QN-maintenance-report.md`
2. Complete all sections
3. Generate changelog
4. Archive or update skills as needed

## Error Types

| Type | Severity | Description | Typical Fix |
|------|----------|-------------|-------------|
| YAML Format Issue | Critical | Agent/skill won't load | Fix frontmatter, run validation |
| Hallucination | High | Agent made up facts | Add grounding requirements |
| Agent Logic Error | High | Workflow bug | Fix agent definition |
| Outdated Pattern | Medium | Codebase changed | Re-scan and update skill |
| Missing Skill | Medium | No coverage for use case | Create new skill |
| Wrong Routing | Medium | Router selected wrong skill | Adjust scoring algorithm |
| Incomplete Guards | Low | Validation gap | Add to checklist |

### YAML Validation Rules

| Field | Skills | Agents |
|-------|--------|--------|
| `name` | Must match folder name | Must match filename (without .md) |
| `description` | Required, max 1024 chars | Required, max 1024 chars |
| `model` | Not required | Required (opus/sonnet/haiku/inherit) |
| `version` | Recommended | Optional |
| `color` | Not required | Recommended |
| Format | Lowercase, numbers, hyphens | Lowercase, numbers, hyphens |

## Metrics

### Targets

| Metric | Target | Current |
|--------|--------|---------|
| YAML validation pass rate | 100% | - |
| Error recurrence rate | <5% | - |
| Avg resolution time | <7 days | - |
| Backlog size | <10 | 0 |
| Critical errors | 0 | 0 |

### Tracking

Update weekly:
- Total errors reported
- Errors resolved
- Errors by type

## Integration Points

### With Validation

All framework updates must pass validation:

```bash
# Before any commit
.claude/scripts/validate-framework.sh

# Pre-commit hook (auto-runs)
.git/hooks/pre-commit
```

### With Skills

The `improving-framework-continuously` skill provides:
- Detailed workflow for processing errors
- Error type handling procedures
- Quality gates for resolutions
- YAML validation integration

Location: `.claude/skills/improving-framework-continuously/SKILL.md`

### With Agents

All agents can report errors:
- Code Engineer: Pattern mismatches during implementation
- Quality Guardian: Missed issues during review
- Test Automator: Missing test scenarios
- Solution Architect: Design pattern gaps

### With Skill Router

Routing errors are captured and used to:
- Improve scoring algorithm
- Add discriminating keywords
- Adjust confidence thresholds

## Files Reference

| File | Purpose | When to Use |
|------|---------|-------------|
| `error-template.md` | Structure for error reports | When error occurs |
| `maintenance-checklist.md` | Quarterly review guide | Every 3 months |
| `errors/*.md` | Individual error records | Review weekly |
| `.claude/scripts/validate-framework.sh` | YAML validation | Before every commit |
| `.claude/hooks/pre-commit-framework-check.sh` | Git hook | Auto-runs on commit |

## Best Practices

### Writing Good Error Reports

1. **Be specific**: Include exact outputs, not "it didn't work"
2. **Include evidence**: Code snippets, file contents, terminal output
3. **Verify ground truth**: Check actual codebase before claiming error
4. **Identify root cause**: Go beyond symptoms
5. **Propose fix**: Suggest what should change

### Effective Resolutions

1. **Fix the root cause**: Not just the symptom
2. **Check related skills**: Same error might exist elsewhere
3. **Add test case**: Prevent regression
4. **Update documentation**: If pattern changed
5. **Close the loop**: Mark resolved with summary

## Changelog

### 2026-01-18
- Validated all improvement system components
- Confirmed integration with v1.2.0 agents and skills
- YAML validation scripts verified working
- Pre-commit hook documentation updated

### 2026-01-16
- Added YAML validation integration to all components
- Added Error ID format (ERR-YYYY-MM-DD-NNN)
- Enhanced auto-classification rules
- Added git integration documentation

### 2026-01-07
- Initial creation of improvement system
- Created error template
- Created maintenance checklist
- Created continuous-improvement skill
