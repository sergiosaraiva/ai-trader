I want you to create or update the 6 core Claude Code agents for this project by analyzing our codebase patterns and available skills.

**Mode**: Create OR Update - If agent files already exist in `.claude/agents/`, read them first and update/enhance them. If they don't exist, create them from scratch.

**First, consolidate your knowledge on Claude Code agents and skills:**
- https://www.anthropic.com/engineering/equipping-agents-for-the-real-world-with-agent-skills
- https://github.com/anthropics/claude-cookbooks/tree/main/skills
- https://platform.claude.com/docs/en/agents-and-tools/agent-skills/overview
- https://platform.claude.com/docs/en/agents-and-tools/agent-skills/best-practices
- https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/overview
- https://platform.claude.com/docs/en/test-and-evaluate/strengthen-guardrails/reduce-hallucinations

**If you need clarification about team structure, automation level, or agent priorities, ask me with relevant options.**

**Then, analyze the codebase and existing skills:**
- Read `.claude/discovery/codebase-patterns.md` (created in Step 1)
- Scan `.claude/skills/*/SKILL.md` (created in Step 2) to understand available skills
- Read `.claude/skills/SKILL-INDEX.md` for skill overview
- Identify technology stack, layer architecture, and development workflows
- Understand how our team structures code and implements features

---

## CRITICAL: YAML Frontmatter Format (Required for Claude Code Discovery)

**Without correct YAML frontmatter, Claude Code will NOT discover your agents.**

**File Location**: `.claude/agents/[agent-name].md`

**YAML Frontmatter** (MUST be at very top of file, line 1):

```yaml
---
name: agent-name-matching-filename
description: Use this agent when [triggering conditions]. Examples:

<example>
Context: [Situation description]
user: "[User request]"
assistant: "[How assistant should respond and use this agent]"
<commentary>
[Why this agent should be triggered]
</commentary>
</example>

model: inherit|sonnet|opus|haiku
color: blue|cyan|green|yellow|magenta|red
tools: ["Read", "Write", "Grep", "Bash"]
---
```

**Format Rules (ALL MANDATORY):**
- File MUST start with `---` on line 1 (no leading whitespace, no blank lines before, no BOM)
- `name`: 3-50 characters, lowercase-hyphen only, MUST match filename without .md
  - File: `requirements-analyst.md` → `name: requirements-analyst`
- `description`: MUST include `<example>` blocks for reliable agent triggering
- `model`: REQUIRED - must be: `inherit`, `opus`, `sonnet`, or `haiku`
- `color`: **REQUIRED** - must be: `blue`, `cyan`, `green`, `yellow`, `magenta`, or `red`
- `tools`: Optional - array of allowed tools (restricts agent capabilities)
- Close frontmatter with `---` on its own line
- System prompt after `---` must be >20 characters
- Everything after closing `---` is the agent's markdown prompt/instructions

**Example (copy this exact format):**
```yaml
---
name: requirements-analyst
description: Use this agent when analyzing work items, identifying specification gaps, or assessing cross-layer impact. Examples:

<example>
Context: Developer receives a new work item and needs analysis
user: "Analyze work item #12345 for implementation"
assistant: "I'll use the requirements-analyst agent to analyze this work item and identify any gaps."
<commentary>
Work item analysis triggers the requirements-analyst agent.
</commentary>
</example>

<example>
Context: Developer wants to understand acceptance criteria before coding
user: "What are the requirements for the new export feature?"
assistant: "Let me use the requirements-analyst agent to break down the requirements."
<commentary>
Requirements clarification triggers this agent.
</commentary>
</example>

model: sonnet
color: cyan
---

You are an elite Requirements Analyst specializing in...

[Rest of agent content...]
```

---

## Model Selection Matrix

| Model | Use For | Cost/Speed |
|-------|---------|------------|
| `inherit` | Inherits from parent context (default for subagents) | Same as parent |
| `opus` | Complex reasoning, architecture, comprehensive analysis | Highest cost, slowest |
| `sonnet` | Balanced tasks, implementation, most development work | Medium cost, balanced |
| `haiku` | Quick simple tasks, low latency requirements | Lowest cost, fastest |

**Recommended Model by Agent:**

| Agent | Model | Rationale |
|-------|-------|-----------|
| `requirements-analyst` | `sonnet` | Balanced analysis, good for structured output |
| `solution-architect` | `opus` | Complex architectural reasoning, design decisions |
| `code-engineer` | `sonnet` | Implementation work, follows established patterns |
| `quality-guardian` | `opus` | Deep analysis, security review, multi-dimensional thinking |
| `test-automator` | `sonnet` | Test generation follows patterns |
| `documentation-curator` | `sonnet` | Documentation generation, structured output |

---

## Color Selection Guide (REQUIRED)

**Every agent MUST have a `color` field. Choose based on agent function:**

| Color | Use For | Example Agents |
|-------|---------|----------------|
| `blue` | General purpose, default | code-engineer |
| `cyan` | Documentation, information, requirements | requirements-analyst, documentation-curator |
| `green` | Testing, validation, quality | test-automator, quality-guardian |
| `yellow` | Warnings, code review, analysis | - |
| `magenta` | Architecture, design, planning | solution-architect |
| `red` | Critical operations, security | - |

**Recommended Color by Agent:**

| Agent | Color | Rationale |
|-------|-------|-----------|
| `requirements-analyst` | `cyan` | Information gathering |
| `solution-architect` | `magenta` | Design/architecture |
| `code-engineer` | `blue` | General implementation |
| `quality-guardian` | `green` | Quality/validation |
| `test-automator` | `green` | Testing |
| `documentation-curator` | `cyan` | Documentation |

---

## Create or Update 6 Agent Files

**Location**: `.claude/agents/`

1. **requirements-analyst.md** - Analyzes work items, identifies specification gaps, generates clarifying questions, assesses cross-layer impact
2. **solution-architect.md** - Designs technical solutions, creates dependency-ordered implementation plans, generates Test Plan from acceptance criteria, identifies integration points
3. **code-engineer.md** - Implements code changes across layers, routes to appropriate skills, verifies builds
4. **quality-guardian.md** - Parallel code review + regression analysis + security scanning
5. **test-automator.md** - Generates and executes tests following TDD approach, verifies builds
6. **documentation-curator.md** - Generates API docs, deployment guides, release notes

---

## Agent Orchestration Flow

Agents work together in a defined sequence. Each agent should understand its place:

```
Phase 1: Requirements Analysis
  └── requirements-analyst.md
        ├── Outputs: Requirements Analysis Report
        └── Hands off to: solution-architect

Phase 2: Solution Design
  └── solution-architect.md
        ├── Inputs: Requirements Analysis Report
        ├── Outputs: Implementation Plan, Test Plan
        ├── Invokes: planning-test-scenarios skill
        └── Hands off to: code-engineer

Phase 3: Implementation
  └── code-engineer.md
        ├── Inputs: Implementation Plan
        ├── Invokes: [layer-specific skills from .claude/skills/]
        └── Hands off to: quality-guardian

Phase 4: Quality Assurance
  └── quality-guardian.md (runs checks in PARALLEL)
        ├── Code Review
        ├── Regression Analysis
        └── Security Scanning
        └── Hands off to: test-automator

Phase 5: Test Execution
  └── test-automator.md
        ├── Inputs: Test Plan from Phase 2
        └── Hands off to: documentation-curator

Phase 6: Documentation
  └── documentation-curator.md
        └── Outputs: API docs, deployment guide, release notes
```

---

## Agent Content Structure (12 Sections)

**For EACH agent file, include these sections after the YAML frontmatter:**

### 1. Mission Statement
Opening paragraph defining the agent's role, expertise, and personality.
```markdown
You are an elite [Role] specializing in [Domain]. Your expertise spans [Areas]. You have deep knowledge of [Specific Knowledge].

**Project Context**: See CLAUDE.md for complete architecture, technology stack, and coding standards.
```

### 2. Purpose Statement
What problem this agent solves, when to invoke it.

### 3. Responsibility Boundaries
What this agent DOES vs DOES NOT do. Clear scope limits.

### 4. Workflow Definition
Phase-by-phase workflow with tool usage (Read, Grep, Edit, Write, Bash, Task, etc.)

### 5. Skill Integration Points
Which specific skills this agent invokes from `.claude/skills/`:
- List skills by name (e.g., "invoke `creating-service-handlers` skill")
- How skills are selected based on task
- Fallback behavior if skill not found

### 6. Context Contract

```markdown
## Context Contract

### Required Inputs
- **From User**: [what user must provide]
- **From Previous Agent**: [artifacts from previous phase]
  - Example: `Implementation Plan from solution-architect`

### Outputs Produced
- **Artifact**: [filename and location]
  - Example: `.claude/artifacts/{work-item-id}/02-implementation-plan.md`
- **Handoff Summary**: [brief summary for next agent]

### Context Preservation
- Reference work item ID throughout
- Maintain artifact chain: 01-requirements → 02-plan → 03-implementation
```

### 7. Input/Output Contract
What context needed, what artifacts produced, success criteria.

### 8. Tool Permissions
Which Claude Code tools can be used, which require user approval.

### 9. Example Conversations
2-3 scenarios showing typical user requests and agent responses.

### 10. Failure Modes & Recovery
Common failures, recovery strategies, escalation to user.

### 11. Codebase-Specific Customizations
Based on patterns from `.claude/discovery/codebase-patterns.md`:
- Reference actual file paths from our codebase
- Mention our specific technology stack
- Include our layer organization structure
- Note our testing patterns
- Reference our build system

### 12. Anti-Hallucination Rules
```markdown
## Anti-Hallucination Rules

- If you don't have information, say "I need more context about [X]"
- If a pattern doesn't exist in codebase, say "Pattern not found - please provide guidance"
- If uncertain about architectural decision, ask user before proceeding
- NEVER invent file paths, method names, or code that doesn't exist
- When citing code, use Read tool to verify it exists first
```

**Version Footer** (add at end of each agent):
```markdown
<!-- Agent Metadata
Version: 1.0.0
Created: YYYY-MM-DD
Model: [opus|sonnet|haiku]
Skills Referenced: [list of skills this agent uses]
-->
```

---

## Critical Requirements

1. **YAML Frontmatter**: MUST include `name`, `description`, `model`, AND `color` fields
2. **Name Format**: 3-50 characters, lowercase-hyphen only
3. **Name Matches Filename**: File `code-engineer.md` → `name: code-engineer`
4. **Description with Examples**: MUST include `<example>` blocks for reliable triggering
5. **Model Required**: Must specify `inherit`, `opus`, `sonnet`, or `haiku`
6. **Color Required**: Must specify `blue`, `cyan`, `green`, `yellow`, `magenta`, or `red`
7. **System Prompt**: Must be >20 characters after the closing `---`
8. **CLAUDE.md Reference**: Each agent should reference project's CLAUDE.md for context
9. **Skill References**: Reference specific skills by exact name from `.claude/skills/`
10. **Context Contracts**: Define clear inputs/outputs for agent chaining
11. **Anti-Hallucination**: Include explicit rules about uncertainty handling
12. **No Guessing**: Agents must not invent code, paths, or patterns that don't exist

---

## Process

1. Read `.claude/discovery/codebase-patterns.md` for codebase context
2. Read `.claude/skills/SKILL-INDEX.md` for available skills
3. For each of the 6 agents:
   a. Create file: `.claude/agents/[agent-name].md`
   b. Add YAML frontmatter with name (matching filename), description, model
   c. Add 12 content sections
   d. Reference specific skills by name
   e. Add context contract for orchestration
   f. Add anti-hallucination rules
   g. Add version footer
4. After all agents created: Update `.claude/settings.json` permissions

---

## Final Step: Update Settings

After creating agents, add to `.claude/settings.json`:

```json
{
  "permissions": {
    "allow": [
      "Task(requirements-analyst)",
      "Task(solution-architect)",
      "Task(code-engineer)",
      "Task(quality-guardian)",
      "Task(test-automator)",
      "Task(documentation-curator)"
    ]
  }
}
```

---

## Parallel Execution Guidance

**DO (parallel - faster):**
```
Read patterns.md || Read SKILL-INDEX.md || Read existing-agent.md
```

**DON'T (sequential - slower):**
```
Read patterns.md → analyze → Read SKILL-INDEX.md → analyze
```

---

## Output

- **6 agent files** in `.claude/agents/[agent-name].md`
- Each with valid YAML frontmatter (name matching filename, description, model)
- Each with 12 content sections including context contracts
- Each referencing specific skills from `.claude/skills/`
- **Updated `.claude/settings.json`** with Task permissions

---

## Validation Checklist (Run After Creation)

```bash
# Verify all agents have valid YAML frontmatter with all required fields
echo "=== AGENT VALIDATION ==="
for agent in .claude/agents/*.md; do
  filename=$(basename "$agent" .md)
  errors=""

  # Check file starts with ---
  [ "$(head -1 "$agent")" != "---" ] && errors+="no-frontmatter "

  # Check name matches filename (3-50 chars, lowercase-hyphen)
  grep -q "^name: $filename$" "$agent" || errors+="name-mismatch "

  # Check description exists (should contain <example> blocks)
  grep -q "^description:" "$agent" || errors+="no-description "

  # Check model field exists with valid value
  grep -q "^model: \(inherit\|opus\|sonnet\|haiku\)$" "$agent" || errors+="invalid-model "

  # Check color field exists with valid value (REQUIRED!)
  grep -q "^color: \(blue\|cyan\|green\|yellow\|magenta\|red\)$" "$agent" || errors+="missing-color "

  if [ -n "$errors" ]; then
    echo "❌ $agent: $errors"
  else
    echo "✅ $agent: OK"
  fi
done
```

**Validation Rules Summary:**
| Field | Requirement | Check |
|-------|-------------|-------|
| `name` | 3-50 chars, lowercase-hyphen, matches filename | `^name: $filename$` |
| `description` | Non-empty, includes `<example>` blocks | `^description:` |
| `model` | inherit\|opus\|sonnet\|haiku | `^model: \(inherit\|opus\|sonnet\|haiku\)$` |
| `color` | blue\|cyan\|green\|yellow\|magenta\|red | `^color: \(blue\|cyan\|green\|yellow\|magenta\|red\)$` |
| System prompt | >20 characters after closing `---` | Content exists |
