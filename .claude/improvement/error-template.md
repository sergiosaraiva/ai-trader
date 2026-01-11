# Framework Error Report

Use this template when an agent or skill produces incorrect behavior. Save completed reports to `.claude/improvement/errors/YYYY-MM-DD-[description].md`.

---

## Error Information

| Field | Value |
|-------|-------|
| **Date** | YYYY-MM-DD |
| **Reporter** | [Your name] |
| **Severity** | [ ] Critical / [ ] High / [ ] Medium / [ ] Low |
| **Status** | [ ] New / [ ] In Progress / [ ] Resolved |

### Severity Guide

- **Critical**: Agent produces dangerous output (security risk, data loss, production breakage)
- **High**: Agent blocks workflow or produces completely wrong pattern
- **Medium**: Agent produces suboptimal pattern or misses important consideration
- **Low**: Minor issue, cosmetic, or edge case

---

## Context

### Task Being Attempted

```
[Describe what you were trying to accomplish]
```

### Agent Used

- [ ] Requirements Analyst
- [ ] Solution Architect
- [ ] Code Engineer
- [ ] Quality Guardian
- [ ] Test Automator
- [ ] Documentation Curator
- [ ] None (direct skill use)

### Skill Invoked

- **Skill Name**: [e.g., creating-technical-indicators]
- **Routed By**: [ ] Manual / [ ] Skill Router
- **Router Confidence**: [If routed, what was the confidence score?]

### Files Involved

```
[List all relevant files]
- src/path/to/file1.py
- src/path/to/file2.py
```

---

## What Went Wrong

### Expected Behavior

```
[Describe what should have happened based on codebase patterns]
```

### Actual Behavior

```
[Describe what actually happened]
```

### Evidence

```python
# Code snippet showing the problem
# Or paste agent/skill output here
```

### Impact

- [ ] Generated code failed to compile/run
- [ ] Generated code works but violates patterns
- [ ] Missing important consideration (security, performance, etc.)
- [ ] Wrong file modified/created
- [ ] Incomplete implementation
- [ ] Other: [describe]

---

## Root Cause Analysis

### Error Type

Check the primary category:

- [ ] **Hallucination**: Agent made up facts not in codebase or documentation
- [ ] **Outdated Pattern**: Codebase changed, skill/agent didn't update
- [ ] **Missing Skill**: No skill exists for this use case
- [ ] **Wrong Routing**: Skill-router selected incorrect skill
- [ ] **Agent Logic Error**: Workflow bug in agent definition
- [ ] **Incomplete Guardrails**: Validation didn't catch invalid approach
- [ ] **Ambiguous Instructions**: Skill instructions unclear or contradictory
- [ ] **Context Limit**: Important information fell out of context window

### Root Cause Details

```
[Detailed explanation of why the error occurred]

Example:
The skill references src/models/base.py:57 for the BaseModel pattern,
but the file was refactored in commit abc123 and the class is now at line 85.
The skill's example code no longer matches the actual implementation.
```

### Ground Truth Verification

How was the correct behavior confirmed?

- [ ] Checked actual codebase files
  - Files checked: [list]
- [ ] Consulted team expert
  - Expert: [name]
- [ ] Reviewed documentation
  - Docs: [list]
- [ ] Tested alternative approach
  - Result: [describe]
- [ ] Examined git history
  - Relevant commits: [list]

### Supporting Evidence

```
[Paste actual codebase content that shows the correct pattern]
```

---

## Remediation Plan

### Changes Needed

Check all that apply:

- [ ] Update skill: ________________________
- [ ] Create new skill: ________________________
- [ ] Fix agent workflow: ________________________
- [ ] Improve skill-router: ________________________
- [ ] Add guardrails: ________________________
- [ ] Update CLAUDE.md: ________________________
- [ ] No framework change (user error or edge case)

### Specific Updates Required

#### Change 1

| Field | Value |
|-------|-------|
| **File** | `.claude/skills/[category]/[name].md` |
| **Section** | [Which section to update] |
| **Line(s)** | [If known] |

**Before:**
```
[Current incorrect content]
```

**After:**
```
[Corrected content]
```

**Rationale:**
```
[Why this change fixes the issue]
```

#### Change 2 (if needed)

[Repeat structure above]

### Guardrails to Add

```
[What additional checks should prevent this error?]

Example:
- Add Quality Checklist item: "Verify file line numbers match current codebase"
- Add Common Mistakes entry: "Line numbers drift as codebase evolves"
```

---

## Test Case

### Regression Test

This scenario should be tested after the fix:

| Field | Value |
|-------|-------|
| **Scenario** | [Brief description of the original task] |
| **Input** | [What to provide to agent/skill] |
| **Expected Output** | [What correct behavior looks like] |
| **Validation Method** | [How to verify correctness] |

### Test Commands

```bash
# Commands to validate the fix
# Example:
# python -m py_compile src/features/technical/new_indicator.py
# pytest tests/unit/test_new_indicator.py -v
```

### Acceptance Criteria

- [ ] [Criterion 1: Specific checkable outcome]
- [ ] [Criterion 2: Specific checkable outcome]
- [ ] [Criterion 3: Specific checkable outcome]

---

## Implementation Checklist

### Before Committing

- [ ] Update affected skill/agent files
- [ ] Add example to skill documentation showing correct pattern
- [ ] Update skill-router if routing issue
- [ ] Add test case to validation suite
- [ ] Re-run validation on this specific scenario
- [ ] Verify no regressions in related scenarios

### Commit Information

| Field | Value |
|-------|-------|
| **Branch** | `fix/framework-[description]` |
| **Commit Message** | `fix(framework): [description]` |
| **Files Changed** | [list] |

### Post-Commit

- [ ] Mark this error report as Resolved
- [ ] Update error report with resolution date
- [ ] Share learnings with team (if significant)

---

## Resolution

### Resolution Date

YYYY-MM-DD

### Resolution Summary

```
[Brief summary of what was fixed and how]
```

### Lessons Learned

```
[What can be done to prevent similar errors?]
```

### Related Error Reports

- [Link to related reports, if any]

---

## Metadata

| Field | Value |
|-------|-------|
| **Created** | YYYY-MM-DD |
| **Last Updated** | YYYY-MM-DD |
| **Resolved** | YYYY-MM-DD |
| **Resolution PR** | [Link if applicable] |
| **Time to Resolution** | [X days] |
