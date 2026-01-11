# Framework Validation Suite

This directory contains validation scenarios for testing agent and skill effectiveness.

## Purpose

- Verify skills produce correct guidance
- Detect pattern drift early
- Establish baseline for framework quality
- Track improvement over time

## Structure

```
.claude/validation/
├── README.md                    # This file
├── scenarios/                   # Test scenarios by category
│   ├── backend-skills.md
│   ├── feature-skills.md
│   ├── data-skills.md
│   ├── trading-skills.md
│   ├── testing-skills.md
│   └── meta-skills.md
└── results/                     # Test results (when run)
    └── YYYY-MM-DD-results.md
```

## Running Validation

### Manual Validation

For each scenario:
1. Present the task to the appropriate agent
2. Record the skill selected and guidance given
3. Compare against expected behavior
4. Mark pass/fail with notes

### Validation Checklist

```
□ Skill selected matches expected
□ Guidance references correct files
□ Examples are syntactically valid
□ Quality checklist is appropriate
□ No hallucinated content
```

## Metrics

| Metric | Target |
|--------|--------|
| Skill accuracy | >95% |
| File reference validity | 100% |
| Guidance completeness | >90% |
| Zero hallucinations | 100% |

## Schedule

- **Weekly**: Quick validation of high-priority skills
- **Quarterly**: Full validation suite run
- **On change**: Validate affected skills after updates
