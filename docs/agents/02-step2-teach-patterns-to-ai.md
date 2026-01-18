I want you to create or update foundational Claude Code Skills by extracting patterns from our codebase.

**Mode**: Create OR Update - If skill files already exist in `.claude/skills/`, read them first and update/enhance them. If they don't exist, create them from scratch.

**First, consolidate your knowledge on Claude Code skills:**
- https://www.anthropic.com/engineering/equipping-agents-for-the-real-world-with-agent-skills
- https://github.com/anthropics/claude-cookbooks/tree/main/skills
- https://platform.claude.com/docs/en/agents-and-tools/agent-skills/overview
- https://platform.claude.com/docs/en/agents-and-tools/agent-skills/best-practices
- https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/overview
- https://platform.claude.com/docs/en/test-and-evaluate/strengthen-guardrails/reduce-hallucinations

**Ask for clarification if unsure about layer priorities, detail level, or skill naming conventions.**

**Then, analyze discovered patterns:**
- Read `.claude/discovery/codebase-patterns.md` (created in Step 1)
- For each high-frequency pattern (used in 10+ files), create a skill
- Focus on patterns used in last 3 months of commits

---

## CRITICAL: YAML Frontmatter Format (Required for Claude Code Discovery)

**Without correct YAML frontmatter, Claude Code will NOT discover your skills.**

**File Location**: `.claude/skills/[skill-name]/SKILL.md`

**YAML Frontmatter** (MUST be at very top of file, line 1):

```yaml
---
name: skill-name-in-gerund-form
description: This skill should be used when the user asks to "specific phrase 1", "specific phrase 2". Third-person description with specific trigger phrases. Max 1024 characters.
version: 1.0.0
---
```

**Format Rules (ALL MANDATORY):**
- File MUST start with `---` on line 1 (no leading whitespace, no blank lines before, no BOM)
- `name`: Max 64 chars, lowercase/numbers/hyphens only
  - MUST use gerund form (ending in -ing)
  - MUST match folder name exactly
  - Cannot contain "anthropic" or "claude" (reserved)
  - Cannot contain XML tags
  - ✅ `validating-user-inputs` (folder: `validating-user-inputs/`)
  - ❌ `validate-user-input` (wrong - not gerund, won't match)
- `description`: Max 1024 chars, third-person, with specific trigger phrases
  - Include exact phrases users would say that should trigger this skill
  - Be concrete and specific, not vague
  - Cannot contain XML tags
- `version`: Recommended (e.g., `1.0.0`) for tracking updates
- Close frontmatter with `---` on its own line
- Only `name`, `description`, and `version` fields allowed
- Everything after closing `---` is the skill's markdown content

**Directory Structure** (recommended for complex skills):
```
skill-name/
├── SKILL.md           # Required - main skill file
├── references/        # Optional - reference documentation
├── examples/          # Optional - example files and code
└── scripts/           # Optional - executable utility scripts
```

**Example (copy this exact format):**
```yaml
---
name: creating-service-handlers
description: This skill should be used when the user asks to "create a service handler", "add a new API endpoint", "implement message handling". Backend service handler patterns for message processing in the service layer.
version: 1.0.0
---

# Creating Service Handlers

[Rest of skill content...]
```

**Progressive Disclosure**: Skills load in 3 levels:
- **Level 1**: Metadata (name, description) - always loaded at startup (~100 tokens)
- **Level 2**: SKILL.md body - loaded when triggered (<5k tokens recommended)
- **Level 3+**: Referenced files - loaded as needed (unlimited)

Keep SKILL.md body under 500 lines. Split larger content into reference files.

---

## Create or Update 10-15 Skill Files

**Target Skill Categories** (2-3 skills each):

1. **Frontend Layer** (if applicable):
   - Component patterns (e.g., `implementing-ui-components`, `building-view-models`)
   - State management (e.g., `managing-application-state`, `handling-data-binding`)
   - UI patterns (e.g., `styling-with-css`, `creating-responsive-layouts`)

2. **Backend Layer**:
   - Service patterns (e.g., `creating-service-layer`, `implementing-business-services`)
   - Business logic (e.g., `processing-domain-logic`, `handling-transactions`)
   - API patterns (e.g., `designing-api-endpoints`, `versioning-apis`)

3. **Database Layer**:
   - Schema patterns (e.g., `designing-database-schemas`, `creating-migrations`)
   - Query patterns (e.g., `writing-efficient-queries`, `using-data-access-patterns`)
   - Data access (e.g., `implementing-repositories`, `handling-database-transactions`)

4. **Testing**:
   - Unit testing (e.g., `creating-unit-tests`, `mocking-dependencies`)
   - Integration testing (e.g., `writing-integration-tests`, `testing-apis`)
   - Test data (e.g., `generating-test-fixtures`, `seeding-test-data`)
   - Test planning (e.g., `planning-test-scenarios`)
   - Test data generation (e.g., `generating-test-data`)

5. **Quality & Security**:
   - Code review (e.g., `reviewing-code-quality`, `detecting-code-smells`)
   - Security (e.g., `scanning-security-vulnerabilities`, `validating-inputs`)
   - Performance (e.g., `optimizing-performance`, `profiling-bottlenecks`)

6. **Build & Deployment**:
   - Build systems (e.g., `verifying-builds`, `managing-dependencies`)
   - Deployment (e.g., `deploying-applications`, `managing-environments`)

---

## Skill Content Structure (10 Sections)

**For EACH skill file, include these sections after the YAML frontmatter:**

1. **# [Skill Name]** (heading matching the name)

2. **Quick Reference** (3-5 bullet points, concise - "Claude is already very smart")

3. **When to Use** (3-5 specific trigger conditions from OUR codebase)

4. **When NOT to Use** (2-3 anti-patterns or wrong scenarios)

5. **Implementation Guide with Decision Tree**:
```
Is [condition based on OUR patterns]?
├─ Yes → [Action using OUR utilities/methods]
│   └─ If [sub-condition] → [Specific action]
└─ No → [Alternative from OUR codebase]
```

6. **Examples** (3-5 examples from OUR codebase):
```
**Example 1: [Scenario]**

// From: [actual/file/path.ext]:[line-start]-[line-end]
[Code block with actual code from codebase]

**Explanation**: [Why we use this pattern, what it accomplishes]
```

7. **Quality Checklist**:
   - [ ] Code compiles without errors
   - [ ] Pattern exists in [actual-file:line] (cite specific usage)
   - [ ] Tests pass
   - [ ] [OUR project-specific quality check]

8. **Common Mistakes**:
   - **[Mistake we've seen]**: [Why wrong] → [Correct approach from file:line]

9. **Validation**:
   - [ ] Pattern confirmed in [actual-file:line]
   - [ ] All cited files verified to exist
   - [ ] Code snippets match actual file content

10. **Related Skills** (links to other skills in our framework)

**Version Footer** (add at end of each skill):
```markdown
<!-- Skill Metadata
Version: 1.0.0 (should match frontmatter version field)
Created: YYYY-MM-DD
Last Verified: YYYY-MM-DD
Last Modified: YYYY-MM-DD
Patterns From: .claude/discovery/codebase-patterns.md
Lines: X (target 200-400, max 500)
-->
```

---

## Skill Size Guidelines

| Complexity | Lines | Description |
|------------|-------|-------------|
| Simple | 100-200 | Single pattern, 2-3 examples, straightforward decision tree |
| Standard | 200-400 | Multiple variations, 3-5 examples, branching decision tree |
| Complex | 400-500 | Multi-step patterns, edge cases, integration considerations |
| **Split Required** | >500 | Break into sub-skills with cross-references |

**Target: 200-400 lines per skill. If >500 lines, split into focused sub-skills.**

---

## Critical Requirements for EVERY Skill

1. **Gerund Naming**: Use action forms (`validating-emails`, NOT `validate-email`)
2. **Name Matches Folder**: Folder `creating-services/` → `name: creating-services`
3. **Grounding**: EVERY code example MUST cite actual file path (`services/UserService.cs:45-67`)
4. **No Hallucinations**: If pattern doesn't exist in codebase, DON'T create that skill
5. **Progressive Disclosure**: Quick reference → detailed examples
6. **Conciseness**: Avoid over-explaining general programming concepts
7. **Security Review**: Flag any network calls, file access, or potential vulnerabilities
8. **Size Limits**: Target 200-400 lines, split if >500
9. **Validation Gate**: After creating, verify ALL cited files exist using Read tool
10. **Version Tracking**: Include version footer in every skill

---

## Process

1. Scan `.claude/discovery/codebase-patterns.md` for top 10-15 patterns
2. For each pattern, grep codebase to find 3-5 actual implementations
3. Create skill folder: `.claude/skills/[skill-name]/`
4. Create skill file: `.claude/skills/[skill-name]/SKILL.md`
5. Add YAML frontmatter with exact format (name matching folder, gerund form)
6. Add 10 content sections with actual code examples
7. **Validation**: Use Read tool to verify ALL cited files exist and code matches
8. Add version footer
9. **After all skills created**: Generate Skill Index

---

## Final Step: Create Skill Index

After creating all skills, generate `.claude/skills/SKILL-INDEX.md`:

```markdown
# Skill Index

**Generated**: [Date]
**Total Skills**: X

## By Category

### Backend
| Skill | Description | Complexity |
|-------|-------------|------------|
| [creating-service-handlers](./creating-service-handlers/SKILL.md) | Service handler patterns | Standard |

### Frontend
| Skill | Description | Complexity |
|-------|-------------|------------|
| [building-view-models](./building-view-models/SKILL.md) | Knockout.js MVVM patterns | Standard |

### Database
[...]

### Testing
[...]

## Quick Lookup

| Task | Skill |
|------|-------|
| New API endpoint | creating-service-handlers |
| New UI component | building-view-models |
| Database migration | scripting-database-changes |
```

---

## Parallel Execution Guidance

**DO (parallel - faster):**
```
Read file1.cs || Read file2.cs || Read file3.cs
```

**DON'T (sequential - slower):**
```
Read file1.cs → analyze → Read file2.cs → analyze
```

---

## Output

- **10-15 skill files** in `.claude/skills/[skill-name]/SKILL.md`
- Each with valid YAML frontmatter (name matching folder, gerund form)
- Each grounded in actual codebase code with verified file path citations
- **Skill Index** at `.claude/skills/SKILL-INDEX.md`

---

## Validation Checklist (Run After Creation)

```bash
# Verify all skills have valid YAML frontmatter
echo "=== SKILL VALIDATION ==="
for skill in .claude/skills/*/SKILL.md; do
  folder=$(dirname "$skill" | xargs basename)
  errors=""

  # Check file starts with ---
  [ "$(head -1 "$skill")" != "---" ] && errors+="no-frontmatter "

  # Check name matches folder (max 64 chars, lowercase/numbers/hyphens)
  grep -q "^name: $folder$" "$skill" || errors+="name-mismatch "

  # Check description exists
  grep -q "^description:" "$skill" || errors+="no-description "

  # Check for reserved words in name
  echo "$folder" | grep -qE "(anthropic|claude)" && errors+="reserved-word "

  # Check name length (max 64 chars)
  [ ${#folder} -gt 64 ] && errors+="name-too-long "

  # Count lines (warn if >500)
  lines=$(wc -l < "$skill")
  [ "$lines" -gt 500 ] && errors+="over-500-lines "

  if [ -n "$errors" ]; then
    echo "❌ $skill: $errors"
  else
    echo "✅ $skill: OK ($lines lines)"
  fi
done
```

**Validation Rules Summary:**
| Field | Requirement | Notes |
|-------|-------------|-------|
| `name` | Max 64 chars, lowercase/numbers/hyphens | Must match folder, gerund form |
| `name` | No reserved words | Cannot contain "anthropic" or "claude" |
| `description` | Max 1024 chars | Third-person, specific trigger phrases |
| `version` | Recommended | e.g., `1.0.0` |
| Body | <500 lines | Split into reference files if larger |
