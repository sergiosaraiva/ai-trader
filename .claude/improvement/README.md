# Framework Continuous Improvement System

This directory contains the continuous improvement infrastructure for the Agent-Skill framework.

## Purpose

Enable the framework to learn from mistakes and evolve alongside the codebase through structured error capture, analysis, and remediation.

## Directory Structure

```
.claude/improvement/
├── README.md                    # This file
├── error-template.md            # Template for reporting errors
├── maintenance-checklist.md     # Quarterly review checklist
├── errors/                      # Individual error reports
│   └── YYYY-MM-DD-[desc].md    # One file per error
└── YYYY-QN-maintenance-report.md # Quarterly reports (when generated)
```

## Quick Start

### Reporting an Error

When you encounter wrong agent/skill behavior:

```bash
# 1. Copy the template
cp error-template.md errors/$(date +%Y-%m-%d)-brief-description.md

# 2. Open and fill out all sections
# Focus on:
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

| Type | Description | Typical Fix |
|------|-------------|-------------|
| Hallucination | Agent made up facts | Add grounding requirements |
| Outdated Pattern | Codebase changed | Re-scan and update skill |
| Missing Skill | No coverage for use case | Create new skill |
| Wrong Routing | Router selected wrong skill | Adjust scoring algorithm |
| Agent Logic | Workflow bug | Fix agent definition |
| Incomplete Guards | Validation gap | Add to checklist |

## Metrics

### Targets

| Metric | Target | Current |
|--------|--------|---------|
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

### With Skills

The `improving-framework-continuously` skill provides:
- Detailed workflow for processing errors
- Error type handling procedures
- Quality gates for resolutions

Location: `.claude/skills/continuous-improvement/SKILL.md`

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

### 2026-01-07
- Initial creation of improvement system
- Created error template
- Created maintenance checklist
- Created continuous-improvement skill
