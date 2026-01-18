Analyze the health and effectiveness of our Agent-Skill framework.

**Mode**: Create OR Update - If `.claude/metrics/weekly-health-report-*.md` files exist, read the most recent one first to compare trends. Create new report for this week.

**First, consolidate your knowledge from these best practices:**
- https://www.anthropic.com/engineering/equipping-agents-for-the-real-world-with-agent-skills
- https://github.com/anthropics/claude-cookbooks/tree/main/skills
- https://platform.claude.com/docs/en/agents-and-tools/agent-skills/overview
- https://platform.claude.com/docs/en/agents-and-tools/agent-skills/best-practices
- https://platform.claude.com/docs/en/test-and-evaluate/strengthen-guardrails/reduce-hallucinations

**Ask for clarification if unsure about priority metrics, problem thresholds, or action items.**

---

## CRITICAL FIRST: YAML Format Validation

**Run this BEFORE any other analysis. Format issues cause agents/skills to be invisible to Claude Code.**

### Validate All Skills
```bash
echo "=== SKILL FORMAT VALIDATION ==="
skill_errors=0
for skill in .claude/skills/*/SKILL.md; do
  folder=$(dirname "$skill" | xargs basename)
  errors=""

  [ "$(head -1 "$skill")" != "---" ] && errors+="no-frontmatter "
  grep -q "^name: $folder$" "$skill" || errors+="name-mismatch "
  grep -q "^description:" "$skill" || errors+="no-description "

  if [ -n "$errors" ]; then
    echo "❌ $skill: $errors"
    ((skill_errors++))
  fi
done
echo "Skills with format errors: $skill_errors"
```

### Validate All Agents
```bash
echo "=== AGENT FORMAT VALIDATION ==="
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
    echo "❌ $agent: $errors"
    ((agent_errors++))
  fi
done
echo "Agents with format errors: $agent_errors"
```

**If any format errors found: STOP and fix them immediately. They are Critical severity.**

---

## Data to Analyze

Read these in parallel for efficiency:

```
Read .claude/improvement/errors/*.md || Read .claude/metrics/feedback.json || Read .claude/validation/*.md
```

### 1. Usage Metrics (past week)
- Agent invocation counts by type
- Skill usage frequency
- Success rates (based on developer feedback)
- Average time to complete tasks

### 2. Quality Signals
- Error reports filed: `.claude/improvement/errors/`
- Pattern contributions: `.claude/patterns/contributions/`
- Developer feedback scores: `.claude/metrics/feedback.json`

### 3. Framework Files
- All agents: `.claude/agents/*.md`
- All skills: `.claude/skills/*/SKILL.md`
- Skill index: `.claude/skills/SKILL-INDEX.md`
- Validation results: `.claude/validation/`

---

## Analysis Tasks

### 1. YAML Format Validation (CRITICAL - Do First)

| Check | Status | Count |
|-------|--------|-------|
| Skills with valid frontmatter | ✅/❌ | X/Y |
| Skills with name matching folder | ✅/❌ | X/Y |
| Agents with valid frontmatter | ✅/❌ | X/Y |
| Agents with model field | ✅/❌ | X/Y |
| Agents with color field | ✅/❌ | X/Y |

**Any failures here are CRITICAL and must be fixed immediately.**

### 2. Broken Link Detection

For each skill, verify cited file paths still exist:

```bash
# Extract file paths from skills and verify they exist
for skill in .claude/skills/*/SKILL.md; do
  echo "Checking: $skill"
  # Find patterns like path/to/file.cs:123 or path/to/file.ts
  grep -oE '[a-zA-Z0-9_/.-]+\.(cs|ts|js|sql|md):[0-9-]+' "$skill" | while read citation; do
    filepath=$(echo "$citation" | cut -d: -f1)
    if [ ! -f "$filepath" ]; then
      echo "  BROKEN: $citation"
    fi
  done
done
```

**Report:**
| Skill | Broken Citations | Action Needed |
|-------|------------------|---------------|
| [name] | [count] | Update examples |

### 3. Skill-Agent Dependency Analysis

Map which agents reference which skills:

```bash
# Find skill references in agents
for agent in .claude/agents/*.md; do
  echo "=== $(basename $agent) ==="
  grep -oE 'skills/[a-z-]+' "$agent" | sort -u
done
```

**Report:**
| Agent | Skills Referenced | Missing Skills |
|-------|-------------------|----------------|
| requirements-analyst | [list] | [any not found] |
| solution-architect | [list] | [any not found] |
| code-engineer | [list] | [any not found] |

**Orphaned Skills** (not referenced by any agent):
- [skill-name]: Consider if still needed

### 4. Version Drift Detection

Check for skills/agents modified without version increment:

```bash
# Compare git modification date vs version footer date
for file in .claude/skills/*/SKILL.md .claude/agents/*.md; do
  git_date=$(git log -1 --format="%ci" -- "$file" 2>/dev/null | cut -d' ' -f1)
  version_date=$(grep -oE 'Last Modified: [0-9-]+' "$file" | cut -d' ' -f3)
  if [ "$git_date" != "$version_date" ]; then
    echo "DRIFT: $file (git: $git_date, footer: $version_date)"
  fi
done
```

### 5. Effectiveness Analysis

- Which agents are most/least used?
- Which skills have highest success rates?
- Which skills have most error reports?
- Are there gaps in coverage? (tasks with no applicable skill)

### 6. Pattern Drift Detection

- Compare skill examples against current codebase
- Identify outdated patterns (reference non-existent files)
- Find pattern conflicts (contradictory guidance)

### 7. Quality Issues

- Skills with <3 examples
- Skills not used in past 30 days
- Skills with >20% error rate
- Agents with bottlenecks (>1 hour average task time)
- Skills >500 lines (should be split)

### 8. Improvement Opportunities

- Frequently asked questions without skills
- Patterns that appear in multiple skills (consolidation needed)
- Missing cross-references between related skills

---

## Output Format

Create `.claude/metrics/weekly-health-report-YYYY-MM-DD.md`:

```markdown
# Framework Health Report

**Period**: [Start Date] to [End Date]
**Generated**: [Today]
**Overall Health Score**: [0-100] 🟢/🟡/🔴

## Health Score Calculation
| Category | Weight | Score | Weighted |
|----------|--------|-------|----------|
| YAML Format Valid | 30% | X/100 | X |
| No Broken Links | 20% | X/100 | X |
| Error Rate <10% | 20% | X/100 | X |
| Coverage >80% | 15% | X/100 | X |
| Usage Metrics | 15% | X/100 | X |
| **Total** | 100% | | **X** |

🟢 80-100: Healthy
🟡 60-79: Needs Attention
🔴 <60: Critical Issues

## Executive Summary
[2-3 sentences on framework health]

---

## CRITICAL: Format Validation Results

### Skills Format Check
| Metric | Result | Status |
|--------|--------|--------|
| Total Skills | X | - |
| Valid Frontmatter | X/Y | ✅/❌ |
| Name Matches Folder | X/Y | ✅/❌ |
| Has Description | X/Y | ✅/❌ |

**Skills with Format Errors:**
| Skill | Issue | Priority |
|-------|-------|----------|
| [name] | [issue] | Critical |

### Agents Format Check
| Metric | Result | Status |
|--------|--------|--------|
| Total Agents | X | - |
| Valid Frontmatter | X/Y | ✅/❌ |
| Name Matches Filename | X/Y | ✅/❌ |
| Has Description | X/Y | ✅/❌ |
| Has Valid Model | X/Y | ✅/❌ |

**Agents with Format Errors:**
| Agent | Issue | Priority |
|-------|-------|----------|
| [name] | Missing model field | Critical |

---

## Broken Link Report

| Skill | Broken Citations | Files Missing |
|-------|------------------|---------------|
| [name] | X | [list] |

**Total Broken Links**: X
**Action**: Update skill examples with current file paths

---

## Dependency Analysis

### Agent → Skill Map
| Agent | Skills Used | Status |
|-------|-------------|--------|
| requirements-analyst | [list] | ✅ All found |
| code-engineer | [list] | ❌ Missing: X |

### Orphaned Skills
Skills not referenced by any agent:
- [skill-name]: [recommendation]

### Missing Skills
Skills referenced but not found:
- [skill-name] (referenced by [agent])

---

## Version Drift

| File | Git Modified | Footer Date | Drift |
|------|--------------|-------------|-------|
| [path] | 2024-03-15 | 2024-03-01 | 14 days |

**Action**: Update version footers to match actual modification dates

---

## Usage Metrics

- Total agent invocations: X
- Most used agent: [Name] (Y invocations)
- Least used agent: [Name] (Z invocations)
- Most used skill: [Name] (W invocations)
- Least used skill: [Name] (V invocations)
- Success rate: X% (based on feedback)

## Quality Issues (Priority Order)

### 🔴 Critical (Fix Today)
1. **YAML Format Error**: [Agent/Skill] - [Issue]
   - Impact: Not discoverable by Claude Code
   - Fix: [Specific fix]

### 🟡 Warning (Fix This Week)
1. **Broken Links**: [Skill] references X deleted files
   - Impact: Examples outdated
   - Fix: Update with current file paths

2. **High Error Rate**: [Skill] has Y% error rate
   - Impact: Developers getting wrong guidance
   - Fix: Review error reports and update

### 🟢 Info (Fix This Month)
1. **Low Usage**: [Skill] not used in 30 days
   - Impact: May be obsolete
   - Fix: Verify still needed or archive

## Pattern Drift Detected

| Skill | Issue | Severity |
|-------|-------|----------|
| [name] | References deleted file [path] | Warning |
| [name] | Pattern conflicts with [other skill] | Warning |
| [name] | Code example outdated | Info |

## Coverage Gaps

| Task Type | Occurrences | Skill Needed |
|-----------|-------------|--------------|
| [type] | X times | [suggested skill] |

## Recommendations

### 1. Immediate Actions (This Week)
- [ ] Fix all YAML format errors (Critical)
- [ ] Update broken file citations
- [ ] [Other urgent items]

### 2. Short Term (This Month)
- [ ] Create skill for [coverage gap]
- [ ] Consolidate [similar skills]
- [ ] [Other items]

### 3. Strategic (This Quarter)
- [ ] [Longer-term improvements]

## Trend Comparison

| Metric | Last Week | This Week | Trend |
|--------|-----------|-----------|-------|
| Health Score | X | Y | ↑/↓ |
| Error Reports | X | Y | ↑/↓ |
| Format Errors | X | Y | ↑/↓ |
| Broken Links | X | Y | ↑/↓ |
| Skill Usage | X | Y | ↑/↓ |

## Success Stories

- [Feature name]: Completed X% faster with agent guidance
- [Developer name]: Onboarded successfully using framework

---

## Next Week Focus

1. [ ] [Top priority from recommendations]
2. [ ] [Second priority]
3. [ ] [Third priority]
```

---

## Parallel Execution Guidance

**DO (parallel - faster):**
```
Read errors/error1.md || Read errors/error2.md || Read feedback.json
Grep "pattern" skills/ || Grep "pattern" agents/
```

**DON'T (sequential - slower):**
```
Read error1.md → analyze → Read error2.md → analyze
```

---

## Automation Script

Save as `.claude/scripts/weekly-health-check.sh`:

```bash
#!/bin/bash
# Run weekly health check

echo "=== Weekly Framework Health Check ==="
echo "Date: $(date '+%Y-%m-%d')"
echo ""

# Format validation (CRITICAL)
echo "=== YAML FORMAT VALIDATION ==="
./claude/scripts/validate-framework.sh
echo ""

# Broken link detection
echo "=== BROKEN LINK DETECTION ==="
broken_count=0
for skill in .claude/skills/*/SKILL.md; do
  broken=$(grep -oE '[a-zA-Z0-9_/.-]+\.(cs|ts|js|sql):[0-9-]+' "$skill" 2>/dev/null | while read citation; do
    filepath=$(echo "$citation" | cut -d: -f1)
    [ ! -f "$filepath" ] && echo "$citation"
  done | wc -l)
  if [ "$broken" -gt 0 ]; then
    echo "$(basename $(dirname $skill)): $broken broken links"
    ((broken_count+=broken))
  fi
done
echo "Total broken links: $broken_count"
echo ""

# Skill-agent dependencies
echo "=== SKILL-AGENT DEPENDENCIES ==="
for agent in .claude/agents/*.md; do
  name=$(basename "$agent" .md)
  skills=$(grep -oE 'skills/[a-z-]+' "$agent" 2>/dev/null | sort -u | wc -l)
  echo "$name: references $skills skills"
done
echo ""

# Summary
echo "=== SUMMARY ==="
echo "Skills: $(ls .claude/skills/*/SKILL.md 2>/dev/null | wc -l)"
echo "Agents: $(ls .claude/agents/*.md 2>/dev/null | wc -l)"
echo "Error reports this week: $(ls .claude/improvement/errors/*.md 2>/dev/null | wc -l)"
```

Run every Monday morning.
