---
name: improving-framework-continuously
description: Processes error reports to evolve agents and skills, preventing recurring mistakes. Use when reviewing framework errors, performing maintenance, or analyzing improvement trends. Enables self-healing agent-skill framework.
version: 1.2.0
---

# Improving Framework Continuously

A meta-skill for evolving the agent-skill framework based on observed errors and changing codebase patterns.

## Quick Reference

```
1. Capture errors  → .claude/improvement/errors/YYYY-MM-DD-[desc].md
2. Validate YAML   → .claude/scripts/validate-framework.sh (EVERY update)
3. Analyze weekly  → Classify, prioritize, identify patterns
4. Update files    → Skills, agents, router, guardrails
5. Validate fixes  → Re-run scenario, add test case
6. Track metrics   → Error rate, recurrence, resolution time
```

## When to Use

- Processing accumulated error reports
- Performing weekly/quarterly framework review
- Analyzing why an agent/skill failed
- Updating framework after codebase refactoring
- Measuring framework health and improvement trends

## When NOT to Use

- Normal implementation tasks (use domain skills)
- One-off edge cases not worth generalizing
- User errors (misunderstanding vs framework bug)

---

## Critical: YAML Validation

**Every framework update MUST pass validation.**

```bash
# Run BEFORE every commit
.claude/scripts/validate-framework.sh

# Install pre-commit hook (one-time)
cp .claude/hooks/pre-commit-framework-check.sh .git/hooks/pre-commit
chmod +x .git/hooks/pre-commit
```

**YAML Format Rules:**

| Field | Skills | Agents |
|-------|--------|--------|
| `name` | Must match folder name | Must match filename |
| `description` | Required, max 1024 chars | Required, max 1024 chars |
| `model` | Not required | Required (opus/sonnet/haiku/inherit) |
| `version` | Recommended | Optional |
| `color` | Not required | Recommended |

---

## Improvement Workflow

### Phase 1: Error Capture (Real-time)

When an error occurs:

```
Developer encounters wrong behavior
        │
        ▼
Copy .claude/improvement/error-template.md
        │
        ▼
Fill out all sections with evidence
(Especially: Error ID, Severity, Error Type)
        │
        ▼
Save to .claude/improvement/errors/ERR-YYYY-MM-DD-NNN-[description].md
        │
        ▼
Continue work (don't block on report)
```

**Error ID Format:** `ERR-YYYY-MM-DD-NNN` (e.g., ERR-2026-01-16-001)

**Auto-Classification:**
| Error Type | Default Severity | Response Time |
|------------|------------------|---------------|
| YAML Format Issue | Critical | Same day |
| Hallucination | High | 2 days |
| Agent Logic Error | High | 2 days |
| Wrong Routing | Medium | 1 week |
| Outdated Pattern | Medium | 1 week |
| Missing Skill | Medium | 1 week |
| Incomplete Guardrails | Low | 2 weeks |

### Phase 2: Root Cause Analysis (Weekly)

Every week, review accumulated errors:

```
1. Collect all error reports from past week
   └─ Location: .claude/improvement/errors/

2. Classify by error type:
   ├─ YAML Format Issue  → Run validation script
   ├─ Hallucination      → Add grounding requirements
   ├─ Outdated Pattern   → Re-scan codebase
   ├─ Missing Skill      → Create new skill
   ├─ Wrong Routing      → Fix router algorithm
   ├─ Agent Logic Error  → Fix workflow
   └─ Incomplete Guards  → Add validation

3. Identify patterns:
   └─ Multiple errors with same root cause?
   └─ Specific skill/agent consistently failing?
   └─ Specific code area causing issues?

4. Prioritize by:
   └─ Severity (Critical > High > Medium > Low)
   └─ Frequency (recurring errors first)
   └─ Impact (blocking errors first)
```

### Phase 3: Framework Updates

For each prioritized error:

```
Step 1: Locate affected files
├─ .claude/skills/[category]/SKILL.md
├─ .claude/agents/[name].md
├─ .claude/skills/routing-to-skills/SKILL.md
└─ CLAUDE.md (if project-wide)

Step 2: Search codebase for ground truth
├─ Use Glob to find current file locations
├─ Use Grep to find pattern implementations
├─ Use Read to verify actual code
└─ Document exact file:line references

Step 3: Update with correct pattern
├─ Fix examples with actual code
├─ Update line number references
├─ Add missing considerations
└─ Strengthen guardrails

Step 4: VALIDATE YAML FORMAT
├─ Run: .claude/scripts/validate-framework.sh
├─ Fix any errors before proceeding
└─ Ensure name matches folder/filename

Step 5: Add preventive measures
├─ Add to Quality Checklist
├─ Add to Common Mistakes
├─ Add example showing the pitfall
└─ Update "When NOT to Use" if needed

Step 6: Commit changes
├─ Use descriptive commit message
├─ Reference error report ID
└─ Include test case
```

### Phase 4: Validation

After each update:

```
1. Run YAML validation
   └─ .claude/scripts/validate-framework.sh
   └─ Must pass with 0 errors

2. Re-run original failing scenario
   └─ Use exact same inputs from error report
   └─ Verify correct behavior now

3. Check for regressions
   └─ Run related scenarios
   └─ Ensure fix didn't break other cases

4. Add to validation suite
   └─ Create test case from error report
   └─ Location: .claude/improvement/validation/

5. Update error report
   └─ Mark as Resolved
   └─ Add resolution date
   └─ Document lessons learned
```

### Phase 5: Knowledge Sharing (Monthly)

Aggregate learnings:

```
1. Analyze error trends
   ├─ Which error types most common?
   ├─ Which skills most problematic?
   ├─ Which agents need attention?
   ├─ YAML validation pass rate
   └─ Is error rate improving?

2. Identify systemic issues
   └─ Same root cause across multiple skills?
   └─ Pattern that needs broader update?

3. Perform bulk updates
   └─ Update all affected skills
   └─ Add cross-cutting guardrails
   └─ Update shared documentation

4. Share with team
   └─ Document in monthly report
   └─ Highlight key learnings
   └─ Propose process improvements
```

---

## Error Type Handling

### YAML Format Issue (Critical)

**Symptoms:** Agent/skill not loading, "skill not found", frontmatter parsing errors

**Root Cause:** Invalid YAML frontmatter - name mismatch, missing fields, format errors

**Remediation:**
```bash
# 1. Run validation to find all issues
.claude/scripts/validate-framework.sh

# 2. Common fixes:
# - name must match folder name (skills) or filename (agents)
# - Must start with ---
# - Must have name: and description: fields
# - Agents must have model: field (opus/sonnet/haiku/inherit)
# - name: must be lowercase, numbers, hyphens only (max 64 chars)
# - description: max 1024 characters

# 3. Verify fix
.claude/scripts/validate-framework.sh
```

### Hallucination Errors

**Symptoms:** Agent states facts not in codebase, invents patterns, makes up file paths

**Root Cause:** Insufficient grounding, missing "I don't know" permission

**Remediation:**
```markdown
1. Add explicit grounding requirements:
   "Verify by reading actual file before citing"

2. Add uncertainty permission:
   "If pattern unclear, say 'I need to check the codebase'"

3. Require citations:
   "Include file:line reference for every pattern claim"

4. Add validation step:
   "After generating, verify all file references exist"
```

### Outdated Pattern Errors

**Symptoms:** Skill references old line numbers, deprecated patterns, moved files

**Root Cause:** Codebase evolved, skill didn't track changes

**Remediation:**
```markdown
1. Re-run codebase pattern discovery
2. Update all file:line references
3. Add "Last Verified" date to skill
4. Consider removing specific line numbers
   (use pattern descriptions instead)
5. Add to quarterly maintenance checklist
6. Run YAML validation after updates
```

### Missing Skill Errors

**Symptoms:** No skill matches task, router returns low confidence, agent improvises

**Root Cause:** New use case not covered by existing skills

**Remediation:**
```markdown
1. Create new skill for the use case
2. Follow standard structure with valid YAML frontmatter
3. Ensure name matches folder name
4. Run validation: .claude/scripts/validate-framework.sh
5. Add to skill-router registry
6. Update README index
7. Test routing to new skill
```

### Wrong Routing Errors

**Symptoms:** Router selects wrong skill, confidence doesn't match accuracy

**Root Cause:** Scoring algorithm gaps, keyword conflicts, ambiguous patterns

**Remediation:**
```markdown
1. Analyze routing decision log
2. Identify why wrong skill scored higher
3. Adjust weights or add discriminating keywords
4. Add negative keywords ("NOT for...")
5. Test with edge cases
6. Run validation after skill-router updates
```

---

## Quality Gates

Every error resolution must pass:

```
□ Root cause identified (not just symptoms)
□ Ground truth verified from actual codebase
□ Fix includes specific file changes
□ YAML validation passes: .claude/scripts/validate-framework.sh
□ Test case added for regression prevention
□ Related skills checked for same issue
□ Commit message references error report ID
□ Error report marked Resolved with date
```

**Time Limits:**
| Severity | Max Resolution Time |
|----------|-------------------|
| Critical | 24 hours |
| High | 3 days |
| Medium | 7 days |
| Low | 14 days |

---

## Metrics Dashboard

### Track Weekly

```
Errors Reported:       [count]
Errors Resolved:       [count]
Backlog:               [count]
Avg Resolution Time:   [days]
YAML Validation Rate:  [%] (should be 100%)
```

### Track Monthly

```
Error Rate Trend:        [improving/stable/degrading]
Recurrence Rate:         [% of errors that repeat]
By Type:
  - YAML Format Issue:   [count] (Critical - target: 0)
  - Hallucination:       [count]
  - Outdated Pattern:    [count]
  - Missing Skill:       [count]
  - Wrong Routing:       [count]
  - Agent Logic:         [count]
  - Incomplete Guards:   [count]
```

### Target Metrics

| Metric | Target | Action if Missed |
|--------|--------|-----------------|
| YAML validation pass | 100% | Fix immediately |
| Error recurrence rate | <5% | Review fix quality process |
| Resolution time (avg) | <7 days | Add resources or prioritize |
| Backlog size | <10 | Schedule cleanup sprint |
| Critical errors | 0 | Immediate attention |

---

## Git Integration

### Pre-commit Hook

Install the pre-commit hook to prevent invalid framework commits:

```bash
# Install hook
cp .claude/hooks/pre-commit-framework-check.sh .git/hooks/pre-commit
chmod +x .git/hooks/pre-commit
```

The hook will:
- Check all staged `.claude/skills/` and `.claude/agents/` files
- Validate YAML frontmatter format
- Block commit if validation fails
- Allow bypass with `git commit --no-verify` (not recommended)

### Commit Message Format

```
fix(framework): [description]

- Fixed: [what was wrong]
- Updated: [files changed]
- Validated: YAML frontmatter passes

Resolves: ERR-YYYY-MM-DD-NNN
```

---

## File Locations

| Purpose | Location |
|---------|----------|
| Error template | `.claude/improvement/error-template.md` |
| Error reports | `.claude/improvement/errors/*.md` |
| Maintenance checklist | `.claude/improvement/maintenance-checklist.md` |
| Quarterly reports | `.claude/improvement/YYYY-QN-maintenance-report.md` |
| Validation script | `.claude/scripts/validate-framework.sh` |
| Pre-commit hook | `.claude/hooks/pre-commit-framework-check.sh` |
| This skill | `.claude/skills/improving-framework-continuously/SKILL.md` |

---

## Examples

### Example 1: Fixing YAML Format Issue (Critical)

**Error Report:**
```
ERR-2026-01-16-001
Agent requirements-analyst not loading.
Error: "Agent not found in .claude/agents/"
```

**Resolution Steps:**
```bash
# 1. Run validation
.claude/scripts/validate-framework.sh

# Output:
# Checking: requirements-analyst.md
#   ERROR: name 'Requirements Analyst' doesn't match filename 'requirements-analyst'

# 2. Fix the YAML frontmatter
# Change: name: Requirements Analyst
# To:     name: requirements-analyst

# 3. Verify fix
.claude/scripts/validate-framework.sh
# Output: All validations passed!

# 4. Commit
git add .claude/agents/requirements-analyst.md
git commit -m "fix(framework): correct agent name field to match filename

Resolves: ERR-2026-01-16-001"
```

### Example 2: Weekly Error Triage Session

**Scenario:** 5 error reports accumulated over the week

**Triage Process:**
```markdown
## Week of 2026-01-16 Error Triage

### Error Reports Collected (5)
1. ERR-2026-01-13-001 - YAML format in new skill    - Critical
2. ERR-2026-01-14-001 - Missing test skill          - High
3. ERR-2026-01-14-002 - Router validation mismatch  - Medium
4. ERR-2026-01-15-001 - Hallucinated validator      - High
5. ERR-2026-01-16-001 - Incomplete checklist        - Low

### Prioritized Order (by severity)
1. [Critical] Fix YAML format (blocks loading)
2. [High] Fix hallucinated validator (trust issue)
3. [High] Create missing test skill (workflow gap)
4. [Medium] Fix router scoring (efficiency)
5. [Low] Add checklist item (edge case)

### Actions Taken
1. Fixed YAML frontmatter, ran validation ✓
2. Added grounding to validation skill ✓
3. Created planning-test-scenarios skill ✓
4. Updated router scoring weights ✓
5. Added checklist item to skill ✓

### Metrics Update
- Errors this week: 5
- Resolved: 5
- YAML Validation: 100% after fixes
- Backlog: 0
```

---

## Quality Checklist

- [ ] Error report completely filled out with Error ID
- [ ] Severity classified using auto-classification guide
- [ ] Root cause identified (not symptoms)
- [ ] Ground truth verified from actual codebase
- [ ] All affected files updated
- [ ] **YAML validation passes**
- [ ] Test case added
- [ ] Metrics updated
- [ ] Error report marked Resolved

## Common Mistakes

| Mistake | Impact | Fix |
|---------|--------|-----|
| Skipping YAML validation | Skills/agents don't load | Always run validation before commit |
| Fixing symptoms not cause | Error recurs | Do deeper root cause analysis |
| Missing related updates | Same error in other files | Search all skills for same pattern |
| No test case | Can't verify fix | Always add regression test |
| Vague resolution | "Fixed it" | Document exactly what changed |

## Related Skills

- [routing-to-skills](../routing-to-skills/SKILL.md) - For routing-related errors
- All domain skills - Targets of improvement updates
- All agents in `.claude/agents/` - Targets of workflow fixes

---

<!-- Skill Metadata
Version: 1.2.0
Created: 2026-01-07
Updated: 2026-01-18

Changes in 1.2.0:
- Verified integration with v1.2.0 agents
- Confirmed all validation components working
- Synced with routing-to-skills v1.2.0

Changes in 1.1.0:
- Added YAML validation integration
- Added Error ID format
- Added auto-classification rules
- Added pre-commit hook documentation
- Added validation script documentation
- Enhanced git integration section
-->
