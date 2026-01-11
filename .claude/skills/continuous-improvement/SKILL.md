---
name: improving-framework-continuously
description: Processes error reports to evolve agents and skills, preventing recurring mistakes. Use when reviewing framework errors, performing maintenance, or analyzing improvement trends. Enables self-healing agent-skill framework.
---

# Improving Framework Continuously

A meta-skill for evolving the agent-skill framework based on observed errors and changing codebase patterns.

## Quick Reference

```
1. Capture errors  → .claude/improvement/errors/YYYY-MM-DD-[desc].md
2. Analyze weekly  → Classify, prioritize, identify patterns
3. Update files    → Skills, agents, router, guardrails
4. Validate fixes  → Re-run scenario, add test case
5. Track metrics   → Error rate, recurrence, resolution time
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
        │
        ▼
Save to .claude/improvement/errors/YYYY-MM-DD-[description].md
        │
        ▼
Continue work (don't block on report)
```

**Critical Fields:**
- Error Type (hallucination, outdated, missing, routing, logic, guardrails)
- Ground Truth Verification (how was correct behavior confirmed?)
- Specific file/line references

### Phase 2: Root Cause Analysis (Weekly)

Every week, review accumulated errors:

```
1. Collect all error reports from past week
   └─ Location: .claude/improvement/errors/

2. Classify by error type:
   ├─ Hallucination     → Need stronger grounding
   ├─ Outdated Pattern  → Need codebase re-scan
   ├─ Missing Skill     → Need new skill creation
   ├─ Wrong Routing     → Need router algorithm fix
   ├─ Agent Logic Error → Need workflow fix
   └─ Incomplete Guards → Need validation additions

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
├─ .claude/skills/[category]/[name].md
├─ .claude/agents/[name].md
├─ .claude/skills/skill-router/SKILL.md
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

Step 4: Add preventive measures
├─ Add to Quality Checklist
├─ Add to Common Mistakes
├─ Add example showing the pitfall
└─ Update "When NOT to Use" if needed

Step 5: Commit changes
├─ Use descriptive commit message
├─ Reference error report
└─ Include test case
```

### Phase 4: Validation

After each update:

```
1. Re-run original failing scenario
   └─ Use exact same inputs from error report
   └─ Verify correct behavior now

2. Check for regressions
   └─ Run related scenarios
   └─ Ensure fix didn't break other cases

3. Add to validation suite
   └─ Create test case from error report
   └─ Location: .claude/improvement/validation/

4. Update error report
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
```

### Missing Skill Errors

**Symptoms:** No skill matches task, router returns low confidence, agent improvises

**Root Cause:** New use case not covered by existing skills

**Remediation:**
```markdown
1. Create new skill for the use case
2. Follow skill structure from existing skills
3. Add to skill-router registry
4. Update README index
5. Test routing to new skill
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
```

### Agent Logic Errors

**Symptoms:** Workflow produces wrong output, steps in wrong order, missing steps

**Root Cause:** Agent definition has bug, unclear decision tree, missing branch

**Remediation:**
```markdown
1. Review agent workflow definition
2. Trace through with failing scenario
3. Identify missing or wrong branch
4. Update workflow diagram
5. Add example conversation showing fix
```

### Incomplete Guardrails

**Symptoms:** Invalid approach not caught, quality check passed bad code

**Root Cause:** Checklist incomplete, validation gaps, missing "Common Mistakes"

**Remediation:**
```markdown
1. Add item to Quality Checklist
2. Add entry to Common Mistakes
3. Strengthen "When NOT to Use"
4. Add validation command
5. Consider adding pre-commit hook
```

---

## Quality Gates

Every error resolution must pass:

```
□ Root cause identified (not just symptoms)
□ Ground truth verified from actual codebase
□ Fix includes specific file changes
□ Test case added for regression prevention
□ Related skills checked for same issue
□ Commit message references error report
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
Errors Reported:     [count]
Errors Resolved:     [count]
Backlog:             [count]
Avg Resolution Time: [days]
```

### Track Monthly

```
Error Rate Trend:        [improving/stable/degrading]
Recurrence Rate:         [% of errors that repeat]
By Type:
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
| Error recurrence rate | <5% | Review fix quality process |
| Resolution time (avg) | <7 days | Add resources or prioritize |
| Backlog size | <10 | Schedule cleanup sprint |
| Critical errors | 0 | Immediate attention |

---

## Integration with Agents

### Agent Responsibilities

| Agent | Improvement Role |
|-------|-----------------|
| Code Engineer | Report errors during implementation |
| Quality Guardian | Identify framework gaps during review |
| Test Automator | Report missing test scenarios |
| Solution Architect | Report pattern mismatches |
| Documentation Curator | Report outdated documentation |

### Error Report Triggers

```
Code Engineer:
  └─ "Skill pattern didn't match what I found in codebase"
  └─ "Had to deviate from skill guidance"

Quality Guardian:
  └─ "Review found issue skill should have caught"
  └─ "Pattern checklist was incomplete"

Test Automator:
  └─ "Test scenario skill didn't cover edge case"
  └─ "Generated tests failed for valid code"

Solution Architect:
  └─ "Design pattern in skill was outdated"
  └─ "Missing skill for this architecture"
```

---

## Maintenance Integration

### Weekly Review Checklist

```
□ Collect error reports from .claude/improvement/errors/
□ Triage by severity (Critical first)
□ Assign to appropriate resolver
□ Track against resolution time targets
□ Update metrics dashboard
```

### Quarterly Audit

Reference: `.claude/improvement/maintenance-checklist.md`

```
□ Run full codebase pattern discovery
□ Compare patterns vs skill documentation
□ Identify drift and update skills
□ Archive unused skills
□ Generate quarterly report
```

---

## File Locations

| Purpose | Location |
|---------|----------|
| Error template | `.claude/improvement/error-template.md` |
| Error reports | `.claude/improvement/errors/*.md` |
| Maintenance checklist | `.claude/improvement/maintenance-checklist.md` |
| Quarterly reports | `.claude/improvement/YYYY-QN-maintenance-report.md` |
| Validation suite | `.claude/improvement/validation/*.md` |
| This skill | `.claude/skills/continuous-improvement/SKILL.md` |

---

## Examples

### Example 1: Processing an Outdated Pattern Error

**Error Report:**
```
Skill `implementing-prediction-models` references src/models/base.py:57
but BaseModel class is now at line 85 after refactoring.
```

**Resolution Steps:**
```bash
# 1. Find current location
grep -n "class BaseModel" src/models/base.py
# Output: 85:class BaseModel(ABC):

# 2. Update skill
# Edit .claude/skills/backend/implementing-prediction-models.md
# Change: src/models/base.py:57 → src/models/base.py:85

# 3. Search for other references to this file
grep -r "base.py:57" .claude/skills/

# 4. Update all found references

# 5. Add to maintenance notes
# "Line numbers in base.py updated 2026-01-07"

# 6. Commit
git add .claude/skills/
git commit -m "fix(skills): update base.py line references after refactor

Fixes error report 2026-01-07-basemodel-line-drift.md"
```

### Example 2: Adding Missing Skill

**Error Report:**
```
No skill exists for implementing WebSocket real-time feeds.
Router returned low confidence (0.35) for all skills.
```

**Resolution Steps:**
```
1. Confirm gap exists (search existing skills)
2. Research codebase for WebSocket patterns
3. Create new skill:
   .claude/skills/data-layer/implementing-websocket-feeds.md
4. Follow standard skill structure
5. Add to skill-router registry
6. Update README.md index
7. Test routing with original scenario
```

### Example 3: Fixing Wrong Routing

**Error Report:**
```
Task: "Add validation to data processor"
Router selected: creating-api-endpoints (0.72)
Should have selected: creating-data-processors (0.68)
```

**Analysis:**
```
Keyword "validation" matched API skill's description
File path "src/data/processors/" should have boosted correct skill
```

**Resolution:**
```markdown
# Update skill-router scoring

Add to creating-data-processors triggers:
- "validation" keyword (+15 when in data/processors path)

Add negative trigger to creating-api-endpoints:
- "validation" in data layer context (-10)

Update file path scoring:
- src/data/processors/** → creating-data-processors (+55, up from +50)
```

### Example 4: Fixing Hallucination Error

**Error Report:**
```
Agent stated "Use the TimeSeriesValidator class from src/validation/series.py"
but this file/class doesn't exist in the codebase.
```

**Root Cause Analysis:**
```
Error Type: Hallucination
The agent fabricated a class name that follows naming conventions
but doesn't exist. No similar class in codebase.
```

**Resolution Steps:**
```markdown
1. Verify ground truth:
   grep -r "TimeSeriesValidator" src/
   # No results - confirmed hallucination

2. Find actual validation approach:
   grep -r "validate.*time.*series" src/
   # Found: src/data/processors/ohlcv.py:validate_sequence()

3. Update affected skill:
   - File: .claude/skills/quality-testing/validating-time-series-data.md
   - Add grounding requirement:
     "Always verify class/function exists before referencing"
   - Add correct example:
     "Use validate_sequence() from src/data/processors/ohlcv.py"

4. Add to Common Mistakes:
   "Fabricating class names - Always grep codebase to verify existence"

5. Test: Ask agent same question, verify correct reference
```

### Example 5: Weekly Error Triage Session

**Scenario:** 5 error reports accumulated over the week

**Triage Process:**
```markdown
## Week of 2026-01-07 Error Triage

### Error Reports Collected (5)
1. 2026-01-03-basemodel-line-drift.md      - Outdated Pattern - Medium
2. 2026-01-04-missing-test-skill.md        - Missing Skill - Critical
3. 2026-01-05-router-validation-mismatch.md - Wrong Routing - High
4. 2026-01-06-hallucinated-validator.md    - Hallucination - High
5. 2026-01-07-incomplete-checklist.md      - Incomplete Guards - Low

### Prioritized Order (by severity + frequency)
1. [Critical] Missing test planning skills (blocks workflow)
2. [High] Hallucinated validator class (trust issue)
3. [High] Router selecting wrong skill (efficiency)
4. [Medium] Line number drift (minor inaccuracy)
5. [Low] Checklist missing item (edge case)

### Pattern Analysis
- 2 errors related to testing → Create test skills as priority
- 1 hallucination → Add grounding checks to affected skill
- No recurring errors from previous weeks ✓

### Actions Taken
1. Created planning-test-scenarios.md (resolves #2)
2. Created generating-test-data.md (resolves #2)
3. Added validation grounding to skill (resolves #4)
4. Updated router scoring (resolves #3)
5. Updated line references (resolves #1)
6. Added checklist item (resolves #5)

### Metrics Update
- Errors this week: 5
- Resolved: 5
- Backlog: 0
- Recurring: 0 (0%)
- Avg resolution: 2.4 days
```

---

## Quality Checklist

- [ ] Error report completely filled out
- [ ] Root cause identified (not symptoms)
- [ ] Ground truth verified from actual codebase
- [ ] All affected files updated
- [ ] Test case added
- [ ] Metrics updated
- [ ] Error report marked Resolved

## Common Mistakes

- **Fixing symptoms not cause**: Error recurs → Do deeper root cause analysis
- **Missing related updates**: Same error in other skills → Search all skills for same pattern
- **No test case**: Can't verify fix → Always add regression test
- **Vague resolution**: "Fixed it" → Document exactly what changed and why

## Related Skills

- [routing-to-skills](../skill-router/SKILL.md) - For routing-related errors
- All domain skills - Targets of improvement updates
- All agents in `.claude/agents/` - Targets of workflow fixes
