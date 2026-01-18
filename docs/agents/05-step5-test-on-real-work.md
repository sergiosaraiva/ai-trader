Validate the Agent-Skill framework by comparing development approaches on current work items.

**Mode**: Create OR Update - If `.claude/validation/` files already exist, read them first and update/enhance them. If they don't exist, create them from scratch.

**First, consolidate your knowledge from these best practices:**
- https://www.anthropic.com/engineering/equipping-agents-for-the-real-world-with-agent-skills
- https://github.com/anthropics/claude-cookbooks/tree/main/skills
- https://platform.claude.com/docs/en/agents-and-tools/agent-skills/overview
- https://platform.claude.com/docs/en/agents-and-tools/agent-skills/best-practices
- https://platform.claude.com/docs/en/test-and-evaluate/strengthen-guardrails/reduce-hallucinations
- https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/be-clear-and-direct

**Ask for clarification on test scope, success criteria, or team participation if needed.**

---

## PREREQUISITE: Framework Format Validation

**Before testing with real work, validate that all agents and skills are discoverable by Claude Code.**

### Step 0: Validate YAML Frontmatter (CRITICAL)

Run this validation FIRST - if format is wrong, agents/skills won't load:

**Validate Skills:**
```bash
echo "=== SKILL VALIDATION ==="
for skill in .claude/skills/*/SKILL.md; do
  folder=$(dirname "$skill" | xargs basename)
  echo "Checking: $skill"

  # Check file starts with ---
  first_line=$(head -1 "$skill")
  if [ "$first_line" != "---" ]; then
    echo "  ERROR: File doesn't start with ---"
  fi

  # Check name field exists and matches folder
  if ! grep -q "^name: $folder$" "$skill"; then
    echo "  ERROR: name field missing or doesn't match folder '$folder'"
  fi

  # Check description field exists
  if ! grep -q "^description:" "$skill"; then
    echo "  ERROR: description field missing"
  fi

  echo "  OK"
done
```

**Validate Agents:**
```bash
echo "=== AGENT VALIDATION ==="
for agent in .claude/agents/*.md; do
  filename=$(basename "$agent" .md)
  echo "Checking: $agent"
  errors=""

  # Check file starts with ---
  [ "$(head -1 "$agent")" != "---" ] && errors+="no-frontmatter "

  # Check name field matches filename (3-50 chars, lowercase-hyphen)
  grep -q "^name: $filename$" "$agent" || errors+="name-mismatch "

  # Check description field exists (should include <example> blocks)
  grep -q "^description:" "$agent" || errors+="no-description "

  # Check model field exists with valid value
  grep -q "^model: \(inherit\|opus\|sonnet\|haiku\)$" "$agent" || errors+="invalid-model "

  # Check color field exists with valid value (REQUIRED!)
  grep -q "^color: \(blue\|cyan\|green\|yellow\|magenta\|red\)$" "$agent" || errors+="missing-color "

  if [ -n "$errors" ]; then
    echo "  ❌ ERRORS: $errors"
  else
    echo "  ✅ OK"
  fi
done
```

**Fix any errors before proceeding. Framework cannot work if YAML format is wrong.**

---

## Comparative Validation Process

### Step 1: Record Baseline (For Future Regression Detection)

Before testing, record the current state:

**Create baseline file**: `.claude/validation/baseline-YYYY-MM-DD.md`

```markdown
# Framework Baseline

**Date**: [Today]
**Skills Count**: [X]
**Agents Count**: [6]

## Skills Inventory
| Skill Name | Version | Lines | Last Verified |
|------------|---------|-------|---------------|
| [name] | [version] | [lines] | [date] |

## Agents Inventory
| Agent Name | Model | Version |
|------------|-------|---------|
| requirements-analyst | sonnet | 1.0.0 |
| solution-architect | opus | 1.0.0 |
| [etc.] | | |

## Known Limitations
- [List any known gaps or issues]
```

---

### Step 2: Select Current Work Items

Choose 3 work items from your current sprint/backlog:

**Selection Criteria**:
- 1 small bug fix (1-2 hour estimate)
- 1 medium feature (4-8 hour estimate)
- 1 code refactoring or test addition (2-4 hour estimate)

**For each work item, document**:
- Work item ID and title
- Acceptance criteria
- Estimated effort
- Assigned developer (if any)

---

### Step 3: Split-Test Implementation

For each work item, conduct parallel development:

**Team A: Traditional Approach** (1 developer)
- Develop using current team practices
- No access to Agent-Skill framework
- Track actual time spent
- Document challenges encountered
- Note questions asked to team members

**Team B: Agent-Assisted Approach** (1 developer)
- Use Agent-Skill framework
- Start with Requirements Analyst agent
- Follow agent recommendations
- Track actual time spent
- Document agent interactions
- **Log which agents and skills were invoked**

**Comparison Template**:

```markdown
## Work Item: [ID] - [Title]

### Traditional Approach Results
- **Developer**: [Name]
- **Time Spent**: X hours
- **Files Modified**: [Count]
- **Questions to Team**: [Count and topics]
- **Challenges**: [List]
- **Rework Needed**: [Yes/No - details]
- **Code Review Issues**: [Count and severity]

### Agent-Assisted Results
- **Developer**: [Name]
- **Time Spent**: Y hours (% difference: Z%)
- **Files Modified**: [Count]
- **Agents Used**: [List with invocation count]
- **Skills Invoked**: [List with success/fail]
- **Skill Router Decisions**: [Log of routing choices]
- **Automated Guidance**: [Count of recommendations]
- **Rework Needed**: [Yes/No - details]
- **Code Review Issues**: [Count and severity]

### Quality Comparison
| Metric | Traditional | Agent-Assisted | Winner |
|--------|------------|----------------|---------|
| Time to Complete | X hrs | Y hrs | [Team] |
| Code Quality Score | [1-10] | [1-10] | [Team] |
| Test Coverage | X% | Y% | [Team] |
| Review Iterations | [Count] | [Count] | [Team] |
| Pattern Consistency | [Score] | [Score] | [Team] |
```

---

### Step 4: Measure Framework Value

Calculate improvement metrics:

**1. Time Savings:**
```
Formula: (Traditional Time - Agent Time) / Traditional Time × 100
Target: >20% time reduction
```

**2. Quality Improvement:**
- Code review issues reduced
- Test coverage increased
- Pattern consistency improved
- Target: >30% fewer review issues

**3. Developer Experience:**
- Fewer interruptions to ask questions
- Faster onboarding for unfamiliar code areas
- Reduced context switching
- Survey score: [1-10]

**4. Framework Effectiveness:**
- Agent invocation success rate
- Skill routing accuracy
- Recommendations followed vs ignored

**5. Knowledge Capture:**
- New patterns discovered during work
- Skills that were missing but needed
- Skills that worked perfectly

---

### Step 5: Framework Gaps Identification

Document where the framework fell short:

**Missing Skills:**
```markdown
| Task Attempted | Skill Needed | Workaround Used | Priority |
|----------------|--------------|-----------------|----------|
| [Description] | [Pattern missing] | [How proceeded] | High/Med/Low |
```

**Incorrect Guidance:**
```markdown
| Agent | Bad Recommendation | Correct Approach | Fix Required |
|-------|-------------------|------------------|--------------|
| [Name] | [What it suggested] | [What worked] | [Skill/agent update] |
```

**Routing Issues:**
```markdown
| Task Description | Skill Selected | Should Have Selected | Router Adjustment |
|------------------|----------------|---------------------|-------------------|
| [Request] | [Wrong skill] | [Correct skill] | [Scoring change] |
```

**YAML Format Issues Found:**
```markdown
| File | Issue | Fixed |
|------|-------|-------|
| [path] | [Missing model field] | [Yes/No] |
```

---

### Step 6: Immediate Improvements

Based on test results, prioritize fixes:

**Critical (Before Rollout):**
1. Fix any YAML format issues (agents/skills not loading)
2. Create skills for patterns that caused >30min delays
3. Fix routing errors that sent developers wrong direction

**Important (Week 1):**
1. Adjust agent prompts based on confusion points
2. Update scoring for misrouted tasks
3. Add examples from successful agent interactions

**Nice-to-Have (Month 1):**
1. Additional edge case handling
2. More examples for complex scenarios

---

## Output

Create or update `.claude/validation/comparative-validation-report.md`:

```markdown
# Comparative Validation Report

**Generated**: [Date]
**Framework Version**: 1.0.0

## Pre-Validation Checks

### YAML Format Validation
- Skills validated: X/Y passed
- Agents validated: 6/6 passed
- Issues fixed: [List any that were fixed]

## Executive Summary
- Work Items Tested: 3
- Total Time Saved: X hours (Y%)
- Quality Improvement: Z%
- Framework Ready: Yes/No

## Results by Work Item

### Bug Fix: [ID - Title]
- Traditional: X hours, Y review issues
- Agent-Assisted: A hours, B review issues
- **Winner**: [Approach] by [margin]
- **Agents Used**: [List]
- **Skills Invoked**: [List]

### Feature: [ID - Title]
- Traditional: X hours, Y review issues
- Agent-Assisted: A hours, B review issues
- **Winner**: [Approach] by [margin]
- **Agents Used**: [List]
- **Skills Invoked**: [List]

### Refactoring: [ID - Title]
- Traditional: X hours, Y review issues
- Agent-Assisted: A hours, B review issues
- **Winner**: [Approach] by [margin]
- **Agents Used**: [List]
- **Skills Invoked**: [List]

## Agent Performance

| Agent | Invocations | Success Rate | Avg Helpfulness |
|-------|-------------|--------------|-----------------|
| requirements-analyst | X | Y% | Z/10 |
| solution-architect | X | Y% | Z/10 |
| code-engineer | X | Y% | Z/10 |
| quality-guardian | X | Y% | Z/10 |
| test-automator | X | Y% | Z/10 |
| documentation-curator | X | Y% | Z/10 |

## Skill Performance

| Skill | Invocations | Routing Correct | Guidance Followed |
|-------|-------------|-----------------|-------------------|
| [name] | X | Y% | Z% |

## Developer Feedback

### Traditional Approach Pain Points
1. [Most time-consuming aspect]
2. [Most frustrating aspect]
3. [Where help was needed]

### Agent-Assisted Highlights
1. [Most helpful agent/skill]
2. [Biggest time saver]
3. [Unexpected benefit]

### Agent-Assisted Pain Points
1. [Where agents fell short]
2. [Confusing recommendations]
3. [Missing guidance]

## Framework Improvements Needed

### Critical (implement before rollout)
- [ ] [Issue and fix]

### Important (implement week 1)
- [ ] [Issue and fix]

### Nice-to-Have (implement month 1)
- [ ] [Issue and fix]

## Success Metrics

| Metric | Result | Target | Status |
|--------|--------|--------|--------|
| Time Savings | X% | >20% | ✅/❌ |
| Quality Improvement | Y% | >30% | ✅/❌ |
| Developer Satisfaction | Z/10 | >7 | ✅/❌ |
| YAML Format Valid | 100% | 100% | ✅/❌ |

## Recommendation

Based on results:
- [ ] Framework ready for team rollout
- [ ] Framework needs critical fixes first (list above)
- [ ] Run additional validation rounds

## Next Steps
1. [ ] Fix critical issues identified
2. [ ] Create missing high-priority skills
3. [ ] Schedule team training session
4. [ ] Plan phased rollout strategy
5. [ ] Set up weekly health monitoring (Step 7)
```

---

## Parallel Execution Guidance

When running validation:

**DO (parallel - faster):**
```
Read baseline.md || Read skills-index.md || Read agent-1.md
```

**DON'T (sequential - slower):**
```
Read baseline.md → Read skills-index.md → Read agent-1.md
```

---

## Validation Automation Script

Save as `.claude/scripts/validate-framework.sh`:

```bash
#!/bin/bash
echo "=== Framework Validation ==="
echo ""

# Count skills and agents
skill_count=$(ls .claude/skills/*/SKILL.md 2>/dev/null | wc -l)
agent_count=$(ls .claude/agents/*.md 2>/dev/null | wc -l)

echo "Skills found: $skill_count"
echo "Agents found: $agent_count"
echo ""

# Validate skills
echo "=== Validating Skills ==="
skill_errors=0
for skill in .claude/skills/*/SKILL.md; do
  folder=$(dirname "$skill" | xargs basename)
  errors=""

  [ "$(head -1 "$skill")" != "---" ] && errors+="no-frontmatter "
  grep -q "^name: $folder$" "$skill" || errors+="name-mismatch "
  grep -q "^description:" "$skill" || errors+="no-description "

  if [ -n "$errors" ]; then
    echo "FAIL: $skill ($errors)"
    ((skill_errors++))
  fi
done
echo "Skills with errors: $skill_errors"
echo ""

# Validate agents
echo "=== Validating Agents ==="
agent_errors=0
for agent in .claude/agents/*.md; do
  filename=$(basename "$agent" .md)
  errors=""

  [ "$(head -1 "$agent")" != "---" ] && errors+="no-frontmatter "
  grep -q "^name: $filename$" "$agent" || errors+="name-mismatch "
  grep -q "^description:" "$agent" || errors+="no-description "
  grep -q "^model: \(inherit\|opus\|sonnet\|haiku\)$" "$agent" || errors+="invalid-model "
  grep -q "^color: \(blue\|cyan\|green\|yellow\|magenta\|red\)$" "$agent" || errors+="missing-color "

  if [ -n "$errors" ]; then
    echo "FAIL: $agent ($errors)"
    ((agent_errors++))
  fi
done
echo "Agents with errors: $agent_errors"
echo ""

# Summary
total_errors=$((skill_errors + agent_errors))
if [ $total_errors -eq 0 ]; then
  echo "✅ All validations passed - framework ready for testing"
else
  echo "❌ $total_errors errors found - fix before testing"
fi
```
