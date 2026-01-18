Consolidate and optimize the Agent-Skill framework to eliminate redundancy and improve quality.

**Mode**: Create OR Update - If `.claude/optimization/consolidation-report-*.md` files exist, read the most recent one first to track changes over time. Create new report for this month.

**First, consolidate your knowledge from these best practices:**
- https://www.anthropic.com/engineering/equipping-agents-for-the-real-world-with-agent-skills
- https://github.com/anthropics/claude-cookbooks/tree/main/skills
- https://platform.claude.com/docs/en/agents-and-tools/agent-skills/overview
- https://platform.claude.com/docs/en/agents-and-tools/agent-skills/best-practices
- https://platform.claude.com/docs/en/test-and-evaluate/strengthen-guardrails/reduce-hallucinations

**If unclear about merge thresholds, archival policies, or target skill count, ask for my preference.**

---

## CRITICAL FIRST: YAML Format Validation

**Before any consolidation, ensure all skills and agents have valid YAML format.**

### Run Full Validation

```bash
echo "=== PRE-CONSOLIDATION VALIDATION ==="
./claude/scripts/validate-framework.sh

# Or manually:
echo "=== SKILLS ==="
for skill in .claude/skills/*/SKILL.md; do
  folder=$(dirname "$skill" | xargs basename)
  errors=""
  [ "$(head -1 "$skill")" != "---" ] && errors+="frontmatter "
  grep -q "^name: $folder$" "$skill" || errors+="name "
  grep -q "^description:" "$skill" || errors+="description "
  [ -n "$errors" ] && echo "❌ $folder: $errors"
done

echo "=== AGENTS ==="
for agent in .claude/agents/*.md; do
  filename=$(basename "$agent" .md)
  errors=""
  [ "$(head -1 "$agent")" != "---" ] && errors+="frontmatter "
  grep -q "^name: $filename$" "$agent" || errors+="name "
  grep -q "^description:" "$agent" || errors+="description "
  grep -q "^model: \(inherit\|opus\|sonnet\|haiku\)$" "$agent" || errors+="model "
  grep -q "^color: \(blue\|cyan\|green\|yellow\|magenta\|red\)$" "$agent" || errors+="color "
  [ -n "$errors" ] && echo "❌ $filename: $errors"
done
```

**Fix all format errors BEFORE proceeding with consolidation.**

---

## Common Format Errors to Fix During Consolidation

| Error | Symptom | Fix |
|-------|---------|-----|
| Leading whitespace | File starts with space/tab before `---` | Remove leading whitespace |
| Missing model (agents) | Agent not discovered | Add `model: sonnet` (or inherit/opus/haiku) |
| Missing color (agents) | Agent not discovered | Add `color: blue` (or cyan/green/yellow/magenta/red) |
| Name mismatch | Skill/agent not routed correctly | Ensure `name:` matches folder/filename |
| Name too long | Validation fails | Skills: max 64 chars, Agents: 3-50 chars |
| Missing description | Poor routing decisions | Add meaningful description with trigger phrases |
| Missing `<example>` blocks | Agent triggers unreliably | Add `<example>` blocks to agent descriptions |
| No closing `---` | Content parsed as YAML | Add `---` after frontmatter fields |
| Reserved words in name | Validation fails | Remove "anthropic" or "claude" from skill names |

### Format Validation Checklist

**For Every Skill Created/Modified:**
- [ ] File starts with `---` on line 1
- [ ] `name:` uses gerund form (ending in -ing)
- [ ] `name:` matches folder name exactly
- [ ] `description:` is present and meaningful
- [ ] Closing `---` present before content
- [ ] No extra fields in frontmatter

**For Every Agent Created/Modified:**
- [ ] File starts with `---` on line 1
- [ ] `name:` is 3-50 chars, lowercase-hyphen, matches filename
- [ ] `description:` is present with `<example>` blocks
- [ ] `model:` is inherit/opus/sonnet/haiku
- [ ] `color:` is blue/cyan/green/yellow/magenta/red (REQUIRED!)
- [ ] Closing `---` present before content
- [ ] System prompt is >20 characters
- [ ] No extra fields in frontmatter

---

## Consolidation Tasks

### 1. Identify Duplicate Patterns

Search all skills for similar patterns:

```bash
# Compare all SKILL.md files for similar content
echo "=== DUPLICATE DETECTION ==="
for skill1 in .claude/skills/*/SKILL.md; do
  name1=$(dirname "$skill1" | xargs basename)
  for skill2 in .claude/skills/*/SKILL.md; do
    name2=$(dirname "$skill2" | xargs basename)
    if [[ "$name1" < "$name2" ]]; then
      # Extract "When to Use" sections and compare
      when1=$(sed -n '/## When to Use/,/## When NOT/p' "$skill1" | head -20)
      when2=$(sed -n '/## When to Use/,/## When NOT/p' "$skill2" | head -20)
      # Simple word overlap check
      overlap=$(comm -12 <(echo "$when1" | tr ' ' '\n' | sort -u) <(echo "$when2" | tr ' ' '\n' | sort -u) | wc -l)
      if [ "$overlap" -gt 10 ]; then
        echo "SIMILAR: $name1 <-> $name2 (overlap: $overlap words)"
      fi
    fi
  done
done
```

**Output**: List of skill pairs with >70% content similarity

### 2. Merge Redundant Skills

For each duplicate pair:

**Decision Criteria:**
| Situation | Action |
|-----------|--------|
| Skills cover same use case | Merge into one comprehensive skill |
| Skills have overlapping examples | Consolidate examples, cross-reference |
| Skills differ in scope | Keep separate, add "Related Patterns" section |
| One skill is subset of another | Archive subset, enhance the comprehensive one |

**Merge Process:**
1. Create new consolidated skill with best content from both
2. **Verify YAML frontmatter is valid** for new skill
3. Add deprecation notice to old skills with redirect
4. Move old skills to `.claude/skills/_archived/`
5. Update all agent references
6. Update skill-router registry
7. Update `.claude/skills/SKILL-INDEX.md`
8. **Run validation after merge**

**Deprecation Notice Template** (add to archived skills):
```markdown
---
name: old-skill-name
description: DEPRECATED - Use [new-skill-name] instead. This skill was merged on YYYY-MM-DD.
---

# DEPRECATED

This skill has been merged into **[new-skill-name]**.

**Reason**: [Why merged]
**Migration**: Use [new-skill-name] for all [use case] tasks.
**Archived**: YYYY-MM-DD
```

### 3. Identify Missing Skills

Analyze error reports and coverage gaps:

```bash
# Find common patterns in error reports
echo "=== MISSING SKILLS ANALYSIS ==="
grep -rh "Missing Skill\|No skill\|skill needed" .claude/improvement/errors/ 2>/dev/null | \
  sort | uniq -c | sort -rn | head -10

# Find tasks that had no skill match
grep -rh "fallback\|no match\|manual_guidance" .claude/logs/ 2>/dev/null | \
  sort | uniq -c | sort -rn | head -10
```

**Output**: Top 10 most requested skills that don't exist

**For each missing skill identified:**
1. Verify it's a recurring pattern (appears 3+ times)
2. Check if it should be a new skill or extension of existing skill
3. If new skill needed, use Step 2 prompt to create it
4. **Ensure new skill follows YAML format requirements**

### 4. Optimize Existing Skills

For each skill, check and improve:

**Quality Improvements:**
- [ ] Has 3-5 examples minimum
- [ ] All file references still exist (no broken links)
- [ ] Decision tree is clear and complete
- [ ] "Common Mistakes" section exists
- [ ] Cross-references to related skills exist
- [ ] Version footer is up to date

**Size Optimization:**
| Current Size | Action |
|--------------|--------|
| <100 lines | Consider merging with related skill |
| 100-200 lines | Good for simple patterns |
| 200-400 lines | Optimal range |
| 400-500 lines | Review for potential split |
| >500 lines | Must split into focused sub-skills |

**Performance Improvements:**
- Move frequently used skills higher in router priority
- Add more keywords to description for better routing
- Improve "When to Use" triggers

### 5. YAML Format Remediation

Fix any format issues found during validation:

```bash
# Auto-fix common issues (review before applying)

# Fix missing model field in agents
for agent in .claude/agents/*.md; do
  if ! grep -q "^model:" "$agent"; then
    echo "Adding model field to: $agent"
    # Add model: sonnet after description line
    sed -i '/^description:/a model: sonnet' "$agent"
  fi
done

# Check for name mismatches
for skill in .claude/skills/*/SKILL.md; do
  folder=$(dirname "$skill" | xargs basename)
  current_name=$(grep "^name:" "$skill" | cut -d: -f2 | tr -d ' ')
  if [ "$current_name" != "$folder" ]; then
    echo "Name mismatch in $skill: '$current_name' should be '$folder'"
  fi
done
```

### 6. Update Agent Workflows

Check if agent workflows reference deprecated or renamed skills:

```bash
# Scan all agent files for skill references
echo "=== AGENT SKILL REFERENCES ==="
for agent in .claude/agents/*.md; do
  echo "$(basename $agent):"
  grep -oE '[a-z]+-[a-z-]+' "$agent" | grep -E '^(creating|building|implementing|validating|testing|reviewing)-' | sort -u
done

# Check for references to archived skills
for agent in .claude/agents/*.md; do
  for archived in .claude/skills/_archived/*/SKILL.md; do
    name=$(dirname "$archived" | xargs basename)
    if grep -q "$name" "$agent"; then
      echo "WARNING: $(basename $agent) references archived skill: $name"
    fi
  done
done
```

Update references to consolidated/renamed skills.

### 7. Update Skill Index

After all changes, regenerate `.claude/skills/SKILL-INDEX.md`:

```markdown
# Skill Index

**Last Updated**: YYYY-MM-DD
**Total Active Skills**: X
**Archived Skills**: Y

## Active Skills by Category

### Backend
| Skill | Description | Lines | Last Updated |
|-------|-------------|-------|--------------|
| [name](./name/SKILL.md) | [desc] | X | YYYY-MM-DD |

### Frontend
[...]

### Database
[...]

### Testing
[...]

## Recently Merged
| Old Skills | New Skill | Date |
|------------|-----------|------|
| skill-a, skill-b | combined-skill | YYYY-MM-DD |

## Archived Skills
See `.claude/skills/_archived/` for deprecated skills.
```

---

## Post-Consolidation Validation

**CRITICAL: Run validation AFTER all changes:**

```bash
echo "=== POST-CONSOLIDATION VALIDATION ==="
./claude/scripts/validate-framework.sh

# Verify no broken references
for agent in .claude/agents/*.md; do
  echo "Checking $(basename $agent) skill references..."
  grep -oE 'skills/[a-z-]+' "$agent" | while read ref; do
    skill_path=".claude/$ref/SKILL.md"
    [ ! -f "$skill_path" ] && echo "  BROKEN: $ref"
  done
done

echo "=== VALIDATION COMPLETE ==="
```

**All validations must pass before committing changes.**

---

## Output Format

Create `.claude/optimization/consolidation-report-YYYY-MM.md`:

```markdown
# Pattern Consolidation Report

**Date**: YYYY-MM-DD
**Performed By**: [Name]

## Pre-Consolidation State
- **Skills (active)**: X
- **Skills (archived)**: Y
- **Agents**: Z
- **Format Errors**: N (fixed during consolidation)

## Post-Consolidation State
- **Skills (active)**: X
- **Skills (archived)**: Y
- **Agents**: Z
- **Format Errors**: 0 (must be zero)

## Format Issues Fixed

| File | Issue | Fix Applied |
|------|-------|-------------|
| [path] | Missing model field | Added `model: sonnet` |
| [path] | Name mismatch | Changed name to match folder |

## Skills Merged

### 1. [Skill A] + [Skill B] → [New Skill Name]
- **Reason**: [Why merged - overlapping functionality]
- **Content preserved**: [What was kept from each]
- **Impact**: Developers should use [new skill] for [use cases]
- **YAML Valid**: ✅

## Skills Created

### 1. [New Skill Name]
- **Purpose**: [Why needed - gap identified]
- **Coverage**: [What use cases it handles]
- **Priority**: High/Medium/Low
- **YAML Valid**: ✅

## Skills Optimized

### 1. [Skill Name]
- **Changes**: [What improved]
- **Examples added**: X
- **Broken links fixed**: Y
- **Size**: X lines → Y lines
- **YAML Valid**: ✅

## Skills Archived

### 1. [Skill Name]
- **Reason**: [Why deprecated]
- **Replacement**: Use [other skill] instead
- **Location**: `.claude/skills/_archived/[name]/`

## Agent Updates

| Agent | Changes | References Updated |
|-------|---------|-------------------|
| code-engineer | Updated skill refs | skill-a → skill-b |

## Quality Metrics

| Metric | Before | After | Target |
|--------|--------|-------|--------|
| Active skills | X | Y | 15-25 |
| Avg skill size | X lines | Y lines | 200-400 |
| Skills with <3 examples | X | Y | 0 |
| Skills >500 lines | X | Y | 0 |
| Format errors | X | 0 | 0 |
| Broken links | X | Y | 0 |

## Validation Results

```
Skills validated: X/X passed ✅
Agents validated: Y/Y passed ✅
Broken references: 0 ✅
```

## Next Month Focus

- [ ] [Priority area identified during consolidation]
- [ ] [Skills that need more examples]
- [ ] [Coverage gaps to address]

## Changelog

- YYYY-MM-DD: [Summary of changes]
```

---

## Parallel Execution Guidance

**DO (parallel - faster):**
```
Read skill1/SKILL.md || Read skill2/SKILL.md || Read skill3/SKILL.md
Grep "pattern" agents/*.md || Grep "pattern" skills/*/SKILL.md
```

**DON'T (sequential - slower):**
```
Read skill1 → analyze → Read skill2 → analyze
```

---

## Monthly Consolidation Checklist

```markdown
## Pre-Consolidation
- [ ] Run full validation (fix any format errors first)
- [ ] Read previous month's consolidation report
- [ ] Review weekly health reports from past month

## Analysis
- [ ] Identify duplicate skills (>70% similarity)
- [ ] Identify missing skills (from error reports)
- [ ] Identify oversized skills (>500 lines)
- [ ] Identify undersized skills (<100 lines)
- [ ] Check for broken file references

## Execution
- [ ] Merge duplicate skills
- [ ] Create high-priority missing skills
- [ ] Split oversized skills
- [ ] Merge undersized skills
- [ ] Fix all broken references
- [ ] Update agent skill references
- [ ] Update SKILL-INDEX.md

## Validation
- [ ] Run full validation (must pass 100%)
- [ ] Verify no broken skill references in agents
- [ ] Test routing for merged skills

## Documentation
- [ ] Create consolidation report
- [ ] Update changelog
- [ ] Commit all changes with descriptive message
```

Run on first Monday of each month.
