Create or update a continuous improvement system for the Agent-Skill framework.

**Mode**: Create OR Update - If improvement system files already exist in `.claude/improvement/`, read them first and update/enhance them. If they don't exist, create them from scratch.

**First, consolidate your knowledge from these best practices:**
- https://www.anthropic.com/engineering/equipping-agents-for-the-real-world-with-agent-skills
- https://github.com/anthropics/claude-cookbooks/tree/main/skills
- https://platform.claude.com/docs/en/agents-and-tools/agent-skills/overview
- https://platform.claude.com/docs/en/agents-and-tools/agent-skills/best-practices
- https://platform.claude.com/docs/en/test-and-evaluate/strengthen-guardrails/reduce-hallucinations
- https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/overview

**If unclear about feedback mechanisms, review processes, or update frequency, ask for clarification.**

---

## CRITICAL: Format Validation on Every Update

**Every framework update MUST preserve valid YAML frontmatter.**

Before committing ANY change to skills or agents, run validation:

```bash
# Quick validation before commit
.claude/scripts/validate-framework.sh
```

If validation fails, the update MUST be fixed before merge.

---

## Part 1: Error Capture Template

Create or update `.claude/improvement/error-template.md`:

```markdown
# Framework Error Report

## Error Information
- **Date**: [YYYY-MM-DD]
- **Reporter**: [Name]
- **Severity**: [Critical/High/Medium/Low]
- **Error ID**: [Auto: ERR-YYYY-MM-DD-NNN]

## Severity Classification Guide
| Severity | Definition | Response Time |
|----------|------------|---------------|
| Critical | Agent/skill completely broken, blocks work | Same day |
| High | Wrong guidance that could cause bugs | 2 days |
| Medium | Suboptimal guidance, workaround exists | 1 week |
| Low | Minor improvement, nice-to-have | 2 weeks |

## Context
- **Task Description**: [What was being attempted]
- **Agent Used**: [Which agent]
- **Skill Invoked**: [Which skill, if any]
- **Files Involved**: [List]
- **Work Item ID**: [If applicable]

## What Went Wrong
**Expected Behavior**: [What should have happened]

**Actual Behavior**: [What actually happened]

**Evidence**: [Code snippet, screenshot, or output]

## Root Cause Analysis
**Error Type** (check one):
- [ ] Hallucination (agent made up facts)
- [ ] Outdated Pattern (codebase changed, skill didn't)
- [ ] Missing Skill (no skill exists for this case)
- [ ] Wrong Routing (skill-router selected wrong skill)
- [ ] Agent Logic Error (workflow bug)
- [ ] Incomplete Guardrails (didn't catch invalid approach)
- [ ] YAML Format Issue (agent/skill not loading)

**Root Cause**: [Detailed explanation]

**Verification**: [How was ground truth confirmed?]
- [ ] Checked actual codebase files
- [ ] Consulted team expert
- [ ] Reviewed documentation
- [ ] Tested alternative approach

## Remediation Plan
**Changes Needed**:
- [ ] Update skill: [Skill name]
- [ ] Create new skill: [Skill name]
- [ ] Fix agent workflow: [Agent name]
- [ ] Improve skill-router: [Specific change]
- [ ] Add guardrails: [What to check]
- [ ] Fix YAML format: [Which file]

**Specific Updates**:
```
File: .claude/skills/[name]/SKILL.md
Section: [Which section]
Change: [Before → After]
```

## Test Case
**Regression Test**:
- Scenario: [Task description]
- Expected: [Correct behavior]
- Validation: [How to verify]

## Implementation Checklist
- [ ] Update affected skill/agent files
- [ ] **Verify YAML frontmatter still valid** (run validation script)
- [ ] Add example to skill documentation
- [ ] Update skill-router if routing issue
- [ ] Increment version in file footer
- [ ] Add test case to validation suite
- [ ] Re-run validation on this scenario
- [ ] Commit changes with descriptive message
- [ ] Update changelog
```

---

## Part 2: Improvement Workflow Skill

Create or update `.claude/skills/improving-framework-continuously/SKILL.md`:

```yaml
---
name: improving-framework-continuously
description: Process error reports to evolve the Agent-Skill framework and prevent recurring mistakes. Use when reviewing error reports or performing scheduled maintenance.
---

# Improving Framework Continuously

## Quick Reference
- Process error reports weekly
- Validate YAML format on every update
- Track error recurrence rate
- Target <5% recurrence, <7 days resolution

## When to Use
- Weekly error report review
- After discovering incorrect agent/skill behavior
- During quarterly maintenance
- When error trends indicate systemic issues

## When NOT to Use
- For user errors (misunderstood the tool)
- For legitimate design trade-offs
- For known documented limitations

## Workflow

### Phase 1: Capture Error (When it happens)
1. Developer encounters wrong agent/skill behavior
2. Developer fills out `.claude/improvement/error-template.md`
3. Developer saves as `.claude/improvement/errors/YYYY-MM-DD-[description].md`
4. **Auto-classify severity** based on error type:
   - Hallucination → High
   - YAML Format Issue → Critical
   - Outdated Pattern → Medium
   - Missing Skill → Medium
   - Wrong Routing → Medium

### Phase 2: Root Cause Analysis (Weekly review)
1. Read all error reports from past week in parallel:
   ```
   Read errors/2024-03-18-*.md || Read errors/2024-03-19-*.md || ...
   ```
2. Classify error types
3. Identify patterns (multiple errors from same root cause?)
4. Prioritize by severity and frequency
5. Group related errors for batch fixes

### Phase 3: Framework Updates
For each error:
1. Locate affected skill/agent file
2. Search codebase for ground truth
3. Update file with correct pattern
4. Add example showing before/after
5. Strengthen guardrails to prevent recurrence
6. **Increment version number in file footer**
7. **Run YAML validation before committing**

### Phase 4: Validation
1. Re-run original scenario that failed
2. Verify correct behavior
3. Run full validation script:
   ```bash
   .claude/scripts/validate-framework.sh
   ```
4. Add to validation test suite
5. Mark error report as resolved

### Phase 5: Knowledge Sharing (Monthly)
1. Analyze error trends
2. Identify systemic issues
3. Update multiple skills if pattern is widespread
4. Share learnings with team

## Quality Gates
- [ ] Every error must have root cause identified
- [ ] Every fix must include test case
- [ ] Every update must reference actual codebase files
- [ ] **Every update must pass YAML validation**
- [ ] No error report older than 2 weeks unresolved

## Metrics to Track
- Errors reported per week
- Error recurrence rate (same error twice)
- Time to resolution
- Framework alignment score over time
- **YAML validation pass rate** (should be 100%)

**Targets:**
- <5% error recurrence
- <7 days resolution time
- 100% YAML validation pass rate

## Related Skills
- [routing-to-skills](../routing-to-skills/SKILL.md)
- [validating-framework-health](../validating-framework-health/SKILL.md)

<!-- Skill Metadata
Version: 1.0.0
Created: YYYY-MM-DD
-->
```

---

## Part 3: Git Integration (Pre-commit Hook)

Create `.claude/hooks/pre-commit-framework-check.sh`:

```bash
#!/bin/bash
# Pre-commit hook to validate framework files before commit
# Install: cp .claude/hooks/pre-commit-framework-check.sh .git/hooks/pre-commit && chmod +x .git/hooks/pre-commit

echo "Running framework validation..."

# Check if any framework files are being committed
framework_files=$(git diff --cached --name-only | grep -E "^\.claude/(skills|agents)/")

if [ -z "$framework_files" ]; then
  echo "No framework files in commit, skipping validation."
  exit 0
fi

echo "Framework files in commit:"
echo "$framework_files"
echo ""

errors=0

# Validate each modified skill
for file in $(echo "$framework_files" | grep "skills/.*/SKILL.md"); do
  if [ -f "$file" ]; then
    folder=$(dirname "$file" | xargs basename)
    echo "Validating skill: $file"

    # Check YAML frontmatter
    if [ "$(head -1 "$file")" != "---" ]; then
      echo "  ERROR: File doesn't start with ---"
      ((errors++))
    fi

    if ! grep -q "^name: $folder$" "$file"; then
      echo "  ERROR: name doesn't match folder '$folder'"
      ((errors++))
    fi

    if ! grep -q "^description:" "$file"; then
      echo "  ERROR: Missing description field"
      ((errors++))
    fi
  fi
done

# Validate each modified agent
for file in $(echo "$framework_files" | grep "agents/.*\.md"); do
  if [ -f "$file" ]; then
    filename=$(basename "$file" .md)
    echo "Validating agent: $file"

    # Check YAML frontmatter
    if [ "$(head -1 "$file")" != "---" ]; then
      echo "  ERROR: File doesn't start with ---"
      ((errors++))
    fi

    if ! grep -q "^name: $filename$" "$file"; then
      echo "  ERROR: name doesn't match filename '$filename'"
      ((errors++))
    fi

    if ! grep -q "^description:" "$file"; then
      echo "  ERROR: Missing description field"
      ((errors++))
    fi

    if ! grep -q "^model: \(inherit\|opus\|sonnet\|haiku\)$" "$file"; then
      echo "  ERROR: Missing or invalid model field"
      ((errors++))
    fi

    if ! grep -q "^color: \(blue\|cyan\|green\|yellow\|magenta\|red\)$" "$file"; then
      echo "  ERROR: Missing or invalid color field"
      ((errors++))
    fi
  fi
done

if [ $errors -gt 0 ]; then
  echo ""
  echo "❌ $errors validation errors found. Commit blocked."
  echo "Fix the errors above and try again."
  exit 1
fi

echo ""
echo "✅ All framework validations passed."
exit 0
```

**Installation instructions:**
```bash
# Make the hook executable and install it
cp .claude/hooks/pre-commit-framework-check.sh .git/hooks/pre-commit
chmod +x .git/hooks/pre-commit
```

---

## Part 4: Scheduled Maintenance Checklist

Create or update `.claude/improvement/maintenance-checklist.md`:

```markdown
# Quarterly Framework Maintenance

Run this checklist every 3 months:

## Pre-Maintenance Validation
- [ ] Run `.claude/scripts/validate-framework.sh`
- [ ] Fix any YAML format issues FIRST
- [ ] Record current metrics baseline

## Pattern Drift Analysis
- [ ] Run codebase pattern discovery (Step 1 prompt) again
- [ ] Compare new patterns vs existing skills
- [ ] Identify patterns that changed
- [ ] Identify new patterns that emerged
- [ ] Identify patterns no longer used

## Skill Health Check
For each skill:
- [ ] Does it reference actual current codebase files?
- [ ] Are examples still valid? (use Read tool to verify)
- [ ] Is it being used? (check invocation logs)
- [ ] Are there error reports against it?
- [ ] Does it need more examples?
- [ ] Is YAML frontmatter valid?
- [ ] Is version footer up to date?

## Agent Effectiveness
- [ ] Review agent invocation frequency
- [ ] Check success rates by agent
- [ ] Identify bottlenecks in workflows
- [ ] Gather developer feedback
- [ ] Verify all agents have valid YAML with model field

## Routing Accuracy
- [ ] Measure skill-router accuracy over past quarter
- [ ] Identify mis-routed tasks
- [ ] Update scoring algorithm if needed

## Framework Metrics
- [ ] Re-run validation suite (Step 5 prompt)
- [ ] Compare alignment scores vs last quarter
- [ ] Track trend (improving or degrading?)
- [ ] Error recurrence rate
- [ ] Average resolution time

## Updates
- [ ] Archive unused skills (move to `.claude/skills/_archived/`)
- [ ] Consolidate duplicate skills
- [ ] Split overly complex skills (>500 lines)
- [ ] Update all version numbers
- [ ] Generate changelog
- [ ] **Run full validation after all updates**

## Post-Maintenance Validation
- [ ] Run `.claude/scripts/validate-framework.sh` again
- [ ] Verify 100% pass rate
- [ ] Commit all changes with maintenance summary

## Output
Document in `.claude/improvement/YYYY-QN-maintenance-report.md`:

```markdown
# Q[N] [YYYY] Maintenance Report

**Date**: [Date]
**Performed By**: [Name]

## Pre-Maintenance State
- Skills: [count]
- Agents: [count]
- Error reports unresolved: [count]
- Validation pass rate: [%]

## Changes Made
### Skills Updated
- [skill-name]: [what changed]

### Skills Created
- [skill-name]: [why needed]

### Skills Archived
- [skill-name]: [why archived]

### Agents Updated
- [agent-name]: [what changed]

## Post-Maintenance State
- Skills: [count]
- Agents: [count]
- Error reports resolved: [count]
- Validation pass rate: [%]

## Metrics Comparison
| Metric | Last Quarter | This Quarter | Trend |
|--------|--------------|--------------|-------|
| Error recurrence | X% | Y% | ↑/↓ |
| Resolution time | X days | Y days | ↑/↓ |
| Skill coverage | X% | Y% | ↑/↓ |

## Next Quarter Focus
- [ ] [Priority item 1]
- [ ] [Priority item 2]
```
```

---

## Part 5: Integration with Development

Update your project documentation to include:

```markdown
## Framework Error Reporting

### When to Report
Report an error when:
- Agent gives wrong guidance
- Skill references outdated pattern
- Skill-router selects wrong skill
- Agent misses important consideration
- Quality checks don't catch real issue
- Agent or skill doesn't load (YAML issue)

Don't report:
- User error (misunderstood the tool)
- Legitimate design trade-offs
- Known limitations (documented)

### How to Report
1. Copy `.claude/improvement/error-template.md`
2. Fill out all sections (especially severity)
3. Save to `.claude/improvement/errors/[date]-[description].md`
4. Tag team lead for Critical/High severity
5. Continue work (don't block on report)

### Severity Response Times
| Severity | Response Time | Who Handles |
|----------|---------------|-------------|
| Critical | Same day | Framework maintainer |
| High | 2 days | Framework maintainer |
| Medium | 1 week | Weekly review |
| Low | 2 weeks | Monthly consolidation |
```

---

## Output Summary

This prompt creates or updates:

1. **Error Template**: `.claude/improvement/error-template.md`
2. **Improvement Skill**: `.claude/skills/improving-framework-continuously/SKILL.md`
3. **Pre-commit Hook**: `.claude/hooks/pre-commit-framework-check.sh`
4. **Maintenance Checklist**: `.claude/improvement/maintenance-checklist.md`
5. **Integration Documentation**: Update to project docs

All with YAML format validation integrated at every step.
