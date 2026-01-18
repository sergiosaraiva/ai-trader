I want you to create or update a skill-router meta-skill that enables agents to dynamically discover and invoke appropriate skills.

**Mode**: Create OR Update - If `.claude/skills/routing-to-skills/SKILL.md` already exists, read it first and update/enhance it. If it doesn't exist, create it from scratch.

**First, consolidate your knowledge on Claude Code skills:**
- https://www.anthropic.com/engineering/equipping-agents-for-the-real-world-with-agent-skills
- https://github.com/anthropics/claude-cookbooks/tree/main/skills
- https://platform.claude.com/docs/en/agents-and-tools/agent-skills/overview
- https://platform.claude.com/docs/en/agents-and-tools/agent-skills/best-practices
- https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/overview
- https://platform.claude.com/docs/en/test-and-evaluate/strengthen-guardrails/reduce-hallucinations

**If ambiguous about routing logic, skill conflicts, or transparency level, ask for my preference with options.**

---

## CRITICAL: YAML Frontmatter Format

**The skill router is itself a skill - it MUST follow the correct format.**

**Create**: `.claude/skills/routing-to-skills/SKILL.md`

```yaml
---
name: routing-to-skills
description: Meta-skill that analyzes task context and routes to appropriate skills. Use when an agent needs to discover which skill applies to a given task. Scans available skills and returns ranked recommendations.
version: 1.0.0
---
```

**Format Rules:**
- Folder name: `routing-to-skills/`
- File: `routing-to-skills/SKILL.md`
- `name`: Must match folder name exactly, use gerund form (max 64 chars)
- `description`: Explains meta-routing purpose (max 1024 chars)
- `version`: Recommended for tracking updates (e.g., `1.0.0`)

---

## Capabilities

The skill router must:

### 1. Analyze Context
- Parse file paths to determine layer/technology
- Analyze task type (feature, bug fix, refactoring, migration)
- Extract domain concepts from description
- Identify phase in workflow (requirements, design, implementation, testing)

### 2. Discover Available Skills
- Read `.claude/skills/SKILL-INDEX.md` first (if exists)
- Scan `.claude/skills/*/SKILL.md` directory
- Parse each SKILL.md YAML frontmatter for name and description
- Parse "When to Use" sections for trigger conditions
- Build skill registry with metadata

### 3. Rank Skills by Relevance

**Scoring Algorithm:**

| Signal | Points | Example |
|--------|--------|---------|
| File path match | +50 | `*/services/*` → `creating-service-*` skills |
| Task type match | +30 | "add test" → `creating-unit-tests` skill |
| Keyword match | +10 each | "validation" → `validating-*` skills |
| Layer match | +20 | Frontend file → frontend skills |
| Recent success | +5 | Skill worked well on similar task |

**Return top 3 skills with confidence scores (0.0 - 1.0)**

### 4. Handle No Match (Fallback Behavior)

When no skill scores above 0.5 threshold:

```json
{
  "recommendations": [],
  "fallback": {
    "action": "manual_guidance_needed",
    "reason": "No skill matched with confidence > 0.5",
    "closest_matches": [
      {"skill": "general-backend-patterns", "confidence": 0.42}
    ],
    "suggestions": [
      "Describe the task in more detail",
      "Specify which layer (frontend/backend/database) this affects",
      "Create a new skill if this is a recurring pattern"
    ]
  }
}
```

### 5. Handle Multi-Skill Scenarios

When task requires multiple skills (confidence scores within 0.1 of each other):

```json
{
  "recommendations": [
    {"skill": "creating-service-handlers", "confidence": 0.91},
    {"skill": "scripting-database-changes", "confidence": 0.87}
  ],
  "multi_skill": true,
  "execution_order": [
    "1. scripting-database-changes (database layer first)",
    "2. creating-service-handlers (backend depends on database)"
  ],
  "reason": "Task spans multiple layers - apply skills in dependency order"
}
```

### 6. Invoke Selected Skill
- Load skill content from `.claude/skills/[skill-name]/SKILL.md`
- Provide task context to skill
- Apply skill's decision tree
- Return skill recommendations

### 7. Audit Trail (for debugging)

Log routing decisions to `.claude/logs/skill-routing.log`:

```
[2024-03-20 14:32:15] Task: "Add validation to UserService"
[2024-03-20 14:32:15] Files: ["src/services/UserService.cs"]
[2024-03-20 14:32:15] Context Analysis:
  - Layer: backend (service file)
  - Task Type: feature (add)
  - Keywords: validation, user, service
[2024-03-20 14:32:15] Skills Scored:
  - validating-user-inputs: 0.92 (file:+50, keyword:+30, type:+10)
  - creating-service-handlers: 0.65 (file:+50, type:+10)
  - scripting-database-changes: 0.32 (keyword:+10)
[2024-03-20 14:32:15] Selected: validating-user-inputs (confidence: 0.92)
```

---

## Input/Output Contract

**Input** (from agent):
```json
{
  "task": "Add validation to UserService",
  "files": ["src/services/UserService.cs"],
  "context": "Prevent users from having duplicate emails",
  "phase": "implementation",
  "agent": "code-engineer"
}
```

**Output** (to agent):
```json
{
  "recommendations": [
    {
      "skill": "validating-user-inputs",
      "confidence": 0.92,
      "reason": "File path matches backend service pattern, task mentions validation",
      "skill_path": ".claude/skills/validating-user-inputs/SKILL.md",
      "next_steps": [
        "Review existing validation patterns in skill",
        "Apply email uniqueness check pattern",
        "Add to service layer following skill examples"
      ]
    },
    {
      "skill": "scripting-database-changes",
      "confidence": 0.78,
      "reason": "Duplicate prevention often requires database unique constraint",
      "skill_path": ".claude/skills/scripting-database-changes/SKILL.md",
      "next_steps": [
        "Add unique index to email column",
        "Create migration script following skill pattern"
      ]
    }
  ],
  "multi_skill": true,
  "execution_order": ["scripting-database-changes", "validating-user-inputs"],
  "audit_logged": true
}
```

---

## Implementation

The skill router should implement these functions:

### 1. `analyzeContext(task, files, context)`
```
Input: Task description, file paths, additional context
Output: {
  layer: "backend|frontend|database|infrastructure",
  taskType: "feature|bugfix|refactor|migration|test",
  keywords: ["validation", "user", "email"],
  phase: "requirements|design|implementation|testing|documentation"
}
```

### 2. `discoverSkills()`
```
Input: None (reads from .claude/skills/)
Output: [
  {
    name: "validating-user-inputs",
    path: ".claude/skills/validating-user-inputs/SKILL.md",
    description: "...",
    whenToUse: ["...", "..."],
    layer: "backend",
    keywords: ["validation", "input", "form"]
  },
  ...
]
```

### 3. `scoreSkills(context, availableSkills)`
```
Input: Analyzed context, list of available skills
Output: [
  {skill: "validating-user-inputs", score: 0.92, reasons: [...]},
  {skill: "creating-service-handlers", score: 0.65, reasons: [...]},
  ...
] // Sorted by score descending
```

### 4. `handleNoMatch(closestMatches)`
```
Input: Skills that scored below threshold
Output: Fallback response with suggestions
```

### 5. `invokeSkill(skillName, context)`
```
Input: Selected skill name, task context
Output: Skill content loaded and ready to apply
```

---

## Example Usage

<example>
**Agent**: code-engineer analyzing new task

Task: "Fix login button alignment on mobile"
Files: ["src/views/LoginView.cshtml"]

Router analysis:
- Layer: frontend (view file)
- Task type: bugfix (keyword "fix")
- Domain: UI/styling (keywords "button", "alignment", "mobile")

Scored skills:
1. building-view-models (0.95) - View file + UI component work
2. styling-responsive-layouts (0.88) - Mobile-specific issue
3. applying-css-conventions (0.72) - Alignment is styling concern

Selected: building-view-models
Multi-skill: No (clear winner)
Next: Load skill and apply responsive design guidelines
</example>

<example>
**Agent**: solution-architect after completing technical design

Task: "Design Account Export PDF feature"
Context: Technical design complete, acceptance criteria available
Phase: design → testing transition

Router analysis:
- Phase: Post-design (test planning needed)
- Task type: test planning
- Input: Acceptance criteria from requirements

Scored skills:
1. planning-test-scenarios (0.98) - Must invoke after technical design
2. generating-test-data (0.85) - Will be invoked by planning-test-scenarios

Selected: planning-test-scenarios
Next: Generate Test Plan with scenarios, data requirements, AC coverage
</example>

<example>
**Agent**: code-engineer with multi-layer task

Task: "Add CustomerStatus field to customer management"
Files: ["src/services/CustomerService.cs", "src/views/Customer.cshtml"]
Context: New field needs database, backend, and frontend changes

Router analysis:
- Layer: MULTIPLE (backend + frontend)
- Task type: feature
- Keywords: customer, status, field

Scored skills:
1. scripting-database-changes (0.91) - New field needs migration
2. creating-service-handlers (0.89) - Backend service changes
3. building-view-models (0.87) - Frontend changes

Multi-skill: Yes (3 skills within 0.1 confidence)
Execution order:
1. scripting-database-changes (database first)
2. creating-service-handlers (backend depends on DB)
3. building-view-models (frontend depends on backend)
</example>

<example>
**Agent**: code-engineer with no matching skill

Task: "Implement WebSocket real-time notifications"
Files: ["src/services/NotificationService.cs"]
Context: New real-time feature, no existing pattern

Router analysis:
- Layer: backend
- Task type: feature
- Keywords: websocket, real-time, notifications

Scored skills:
1. creating-service-handlers (0.45) - Service file but not WebSocket specific
2. (no other matches above 0.3)

Fallback triggered:
```json
{
  "recommendations": [],
  "fallback": {
    "action": "manual_guidance_needed",
    "reason": "No skill for WebSocket patterns exists",
    "suggestions": [
      "Search codebase for existing WebSocket usage",
      "If pattern found, consider creating 'implementing-websocket-handlers' skill",
      "Proceed with manual implementation following general service patterns"
    ]
  }
}
```
</example>

---

## Implementation Guidelines

1. **Keep routing logic simple and transparent** - Easy to debug
2. **Log all routing decisions** - Audit trail for troubleshooting
3. **Allow manual skill override** - User can specify skill directly
4. **Handle edge cases gracefully** - No match, multiple matches, conflicts
5. **Update skill registry on changes** - Re-scan when skills added/modified
6. **Respect execution order** - Database before backend before frontend

---

## Skill Content Structure

The router skill itself should follow the standard 10-section structure:

1. **# Routing to Skills** (heading)
2. **Quick Reference** - How routing works in 3-5 bullets
3. **When to Use** - Agent needs to find applicable skill
4. **When NOT to Use** - User explicitly specifies skill
5. **Implementation Guide** - The scoring algorithm and logic
6. **Examples** - The examples above
7. **Quality Checklist** - Routing accuracy checks
8. **Common Mistakes** - Wrong skill selected scenarios
9. **Validation** - How to verify routing is correct
10. **Related Skills** - Links to all routable skills

---

## Version Footer

```markdown
<!-- Skill Metadata
Version: 1.0.0
Created: YYYY-MM-DD
Skills Indexed: [count]
Last Registry Update: YYYY-MM-DD
-->
```

---

## Output

- **Skill router file**: `.claude/skills/routing-to-skills/SKILL.md`
- With valid YAML frontmatter (name: routing-to-skills, description)
- With all routing logic documented
- With fallback and multi-skill handling
- With audit trail specification

---

## Validation Checklist

After creating the skill router:

```bash
# Verify YAML frontmatter
head -4 .claude/skills/routing-to-skills/SKILL.md

# Verify name matches folder
grep "name: routing-to-skills" .claude/skills/routing-to-skills/SKILL.md

# Verify it can find other skills
ls .claude/skills/*/SKILL.md | wc -l
```
