# Weekly Error Review Process

**Purpose:** Structured weekly review of framework errors with chain-of-thought analysis and anti-hallucination verification.

**Schedule:** Every Friday or first available day of the week
**Duration:** 1-2 hours depending on error volume
**Version:** 1.0.0

---

## Pre-Review Checklist

Before starting the weekly review:

- [ ] Block 1-2 hours for focused review
- [ ] Have access to error reports directory: `.claude/improvement/errors/`
- [ ] Have validation script ready: `.claude/scripts/validate-framework.sh`
- [ ] Have codebase access for verification

---

## Phase 1: Collect Errors (10 minutes)

### List All New Error Reports

```bash
# Find error reports from past 7 days
find .claude/improvement/errors/ -name "ERR-*.md" -mtime -7 -type f | sort
```

### Quick Triage Table

Fill out for each error:

| Error ID | Severity | Type | Agent/Skill | Status |
|----------|----------|------|-------------|--------|
| ERR-YYYY-MM-DD-001 | | | | [ ] New |
| ERR-YYYY-MM-DD-002 | | | | [ ] New |

### Count Summary

```
New errors this week: ___
Carryover from last week: ___
Total to process: ___
```

---

## Phase 2: Chain-of-Thought Analysis (30-45 minutes)

For each error, complete the 5-step analysis:

### Template (copy for each error)

```markdown
## Error: ERR-YYYY-MM-DD-NNN

### Step 1: What was the exact input/request?


### Step 2: What did the agent/skill actually do?
[Include evidence - output, generated code, etc.]

### Step 3: What should have happened instead?
[Include codebase evidence - actual file contents]

### Step 4: At what point did behavior diverge?
[Identify specific failure point]

### Step 5: Why did it diverge? (Root cause)
[Identify underlying cause - not symptoms]

### Classification
- Error Type: [ ] YAML / [ ] Hallucination / [ ] Outdated / [ ] Missing / [ ] Routing / [ ] Logic / [ ] Guards / [ ] Verification
- Default Severity: 
- Adjusted Severity (if escalation applies): 
```

### Escalation Check

For each error, check if escalation applies:

- [ ] Same error 3+ times in 30 days → Escalate to Critical
- [ ] Error affects >1 agent → Escalate one level
- [ ] Error in meta-skill → Escalate to High minimum
- [ ] YAML validation failure → Critical

---

## Phase 3: Anti-Hallucination Verification (15-20 minutes)

For each error's proposed fix, verify:

### Verification Checklist

```markdown
## Error: ERR-YYYY-MM-DD-NNN

### File Path Verification
- [ ] All referenced files exist (verified with `ls` or Read tool)
- Files checked:
  - `path/to/file1.py` → Exists: [ ] Yes / [ ] No
  - `path/to/file2.py` → Exists: [ ] Yes / [ ] No

### Line Number Verification
- [ ] All line numbers verified by reading actual file
- Lines checked:
  - `file1.py:45` → Content matches: [ ] Yes / [ ] No
  - `file2.py:120` → Content matches: [ ] Yes / [ ] No

### Pattern Verification
- [ ] Verbatim quotes provided for all pattern claims
- [ ] No assumptions made without verification

### Verbatim Evidence

**Source 1**: `filepath:line`
```
[EXACT content from file]
```

**Source 2**: `filepath:line`
```
[EXACT content from file]
```
```

---

## Phase 4: Prioritization (5 minutes)

### Priority Matrix

| Priority | Criteria | Errors |
|----------|----------|--------|
| P1 - Critical | YAML issues, recurring 3x | |
| P2 - High | Hallucination, logic errors | |
| P3 - Medium | Outdated, missing, routing | |
| P4 - Low | Guards, ambiguous | |

### Processing Order

```
1. ERR-______ (P_)
2. ERR-______ (P_)
3. ERR-______ (P_)
...
```

---

## Phase 5: Resolution (30-60 minutes)

For each prioritized error:

### Resolution Template

```markdown
## Resolving: ERR-YYYY-MM-DD-NNN

### Changes Made
1. File: `.claude/skills/[name]/SKILL.md`
   - Section: 
   - Before: [brief]
   - After: [brief]

2. File: [if additional]
   - Change: 

### Validation
- [ ] YAML validation passes: `.claude/scripts/validate-framework.sh`
- [ ] Re-ran original scenario → Correct behavior now: [ ] Yes / [ ] No
- [ ] No regressions in related scenarios

### Test Case Added
- Location: `.claude/improvement/validation/[test-name].md`
- Scenario: 
- Expected: 

### Recurrence Prevention Score
- Score: [1-5] (1=likely to recur, 5=definitely prevented)
- Rationale: 
```

---

## Phase 6: Update Error Reports (10 minutes)

For each resolved error:

1. Update status in error report: `[ ] Resolved`
2. Add resolution date
3. Add resolution summary
4. Add recurrence prevention score

---

## Phase 7: Metrics Update (5 minutes)

### Weekly Metrics

```markdown
## Week of YYYY-MM-DD

### Error Summary
- Errors at start of week: ___
- New errors this week: ___
- Errors resolved: ___
- Remaining backlog: ___

### Resolution Metrics
- Avg resolution time: ___ days
- Chain-of-thought completion: ___% 
- Anti-hallucination compliance: ___%
- Avg recurrence prevention score: ___/5

### By Error Type
| Type | Count | Resolved |
|------|-------|----------|
| YAML Format | | |
| Hallucination | | |
| Outdated Pattern | | |
| Missing Skill | | |
| Wrong Routing | | |
| Agent Logic | | |
| Missing Verification | | |
| Incomplete Guards | | |

### Trends
- Error rate vs last week: [ ] Up / [ ] Down / [ ] Same
- Recurrence rate: ___%
- Areas needing attention: 
```

---

## Phase 8: Self-Reflection Triggers

Check if any triggers require self-reflection analysis:

| Trigger | Condition Met? | Action |
|---------|----------------|--------|
| Same error 3+ times | [ ] Yes / [ ] No | Request: "Analyze why this keeps happening" |
| Error after recent fix | [ ] Yes / [ ] No | Request: "What did the fix miss?" |
| Multiple agents affected | [ ] Yes / [ ] No | Request: "What's the common factor?" |
| New error type | [ ] Yes / [ ] No | Request: "How should we categorize this?" |

### Self-Reflection Prompt (if triggered)

```
The [agent/skill] produced incorrect output for [task].

Observed: [what happened]
Expected: [what should happen]

Please analyze:
1. What information was missing or incorrect?
2. What verification step would have caught this?
3. What guardrail should be added?
4. How confident are you in this analysis? (1-5)
```

---

## Weekly Review Complete Checklist

- [ ] All new errors triaged
- [ ] Chain-of-thought analysis completed for each error
- [ ] Anti-hallucination verification done for all fixes
- [ ] Errors prioritized
- [ ] P1 and P2 errors resolved
- [ ] YAML validation passes
- [ ] Error reports updated with resolution
- [ ] Metrics updated
- [ ] Self-reflection triggers checked
- [ ] Carryover list updated for next week

---

## Output: Weekly Review Summary

Save to `.claude/improvement/weekly-reviews/YYYY-WNN.md`:

```markdown
# Weekly Review: YYYY-WNN

**Date**: YYYY-MM-DD
**Reviewer**: [Name]
**Duration**: [X hours]

## Summary
- Errors processed: ___
- Errors resolved: ___
- Remaining backlog: ___
- Health score impact: [+/-X points]

## Key Findings
1. 
2. 
3. 

## Resolutions
| Error ID | Type | Resolution | Prevention Score |
|----------|------|------------|------------------|
| | | | |

## Carryover to Next Week
| Error ID | Severity | Reason for Carryover |
|----------|----------|----------------------|
| | | |

## Process Improvements Identified
- 
- 

---
*Generated: YYYY-MM-DD*
```

---

## Quick Reference Commands

```bash
# List errors from past week
find .claude/improvement/errors/ -name "ERR-*.md" -mtime -7

# Run validation
.claude/scripts/validate-framework.sh

# Count total error backlog
ls -1 .claude/improvement/errors/*.md 2>/dev/null | wc -l

# Check for recurring errors (same description pattern)
grep -h "^## Error Information" .claude/improvement/errors/*.md | sort | uniq -c | sort -rn
```

---

*Template Version: 1.0.0*
*Created: 2026-01-23*
