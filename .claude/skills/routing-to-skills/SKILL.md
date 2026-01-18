---
name: routing-to-skills
description: Meta-skill that analyzes task context and routes to appropriate skills. Use when an agent needs to discover which skill applies to a given task. Scans available skills in .claude/skills/ and returns ranked recommendations with confidence scores. Enables cross-project portability without hardcoded skill names.
version: 1.2.0
---

# Routing to Skills

A meta-skill that enables agents to dynamically discover and invoke appropriate skills based on task context. This eliminates hardcoded skill references and enables cross-project portability.

## Quick Reference

```
1. analyzeContext(task, files)     → Extract layer, task type, keywords
2. discoverSkills()                → Scan .claude/skills/ for available skills
3. scoreSkills(context, skills)    → Rank by relevance (0-100 points)
4. handleNoMatch(closestMatches)   → Generate fallback when confidence < 0.5
5. invokeSkill(skill, context)     → Load and apply selected skill
6. logRouting(decision)            → Record decision for audit trail
```

## When to Use

- Starting any implementation task (auto-route to best skill)
- Agent workflow step needs skill-based guidance
- Multiple skills might apply (need ranking)
- Task description is ambiguous about which pattern to follow
- Post-design phase (Solution Architect → Test Automator handoff)

## When NOT to Use

- Task explicitly names which skill to use ("use creating-fastapi-endpoints")
- Simple tasks requiring no specialized patterns
- Research/exploration tasks (not implementation)
- User provides explicit manual override

---

## Implementation Guide

### Function 1: `analyzeContext(task, files, context)`

Extracts layer, task type, and keywords from the input.

**Input:**
```json
{
  "task": "Add validation to UserService",
  "files": ["src/api/services/user_service.py"],
  "context": "Prevent users from having duplicate emails",
  "phase": "implementation",
  "agent": "code-engineer"
}
```

**Output:**
```json
{
  "layer": "api-services",
  "taskType": "feature",
  "keywords": ["validation", "user", "email", "service"],
  "phase": "implementation"
}
```

**Layer Detection from File Paths:**

| Path Pattern | Layer | Primary Skills |
|-------------|-------|----------------|
| `src/api/routes/**` | api-routes | creating-fastapi-endpoints |
| `src/api/services/**` | api-services | creating-python-services |
| `src/api/schemas/**` | api-schemas | creating-pydantic-schemas |
| `src/api/database/**` | database | creating-sqlalchemy-models |
| `frontend/src/components/**` | frontend-components | creating-react-components |
| `frontend/src/api/**` | frontend-api | creating-api-clients |
| `src/models/**` | model | implementing-prediction-models |
| `src/features/technical/**` | feature-engineering | creating-technical-indicators |
| `src/data/sources/**` | data-sources | adding-data-sources |
| `src/data/processors/**` | data-processors | creating-data-processors |
| `src/trading/**` | trading | implementing-risk-management |
| `src/simulation/**` | simulation | running-backtests |
| `tests/**` or `*.test.*` | testing | writing-pytest-tests, writing-vitest-tests |
| `scripts/**` | cli | creating-cli-scripts |
| `configs/**` | config | configuring-indicator-yaml |

**Task Type Detection:**

| Keywords | Task Type | Score Boost |
|----------|-----------|-------------|
| fix, bug, error, broken, issue | bug_fix | +10 to domain skill |
| add, create, new, implement, build | feature | +20 to pattern skill |
| refactor, improve, clean, optimize | refactor | +15 to pattern skill |
| test, validate, verify, spec | testing | +30 to testing skills |
| config, configure, setup, env | config | +25 to config skills |
| migrate, upgrade, update, move | migration | +10 to affected layer |

### Function 2: `discoverSkills()`

Scans the skills directory and builds a registry.

**Steps:**
1. Read `.claude/skills/SKILL-INDEX.md` first if exists (authoritative registry)
2. Scan `.claude/skills/**/SKILL.md` and `.claude/skills/**/*.md`
3. Parse each file's YAML frontmatter for name and description
4. Extract "When to Use" sections for trigger conditions
5. Build skill registry with metadata

**Registry Entry:**
```json
{
  "name": "creating-python-services",
  "description": "Thread-safe singleton services with lazy initialization",
  "path": ".claude/skills/backend/creating-python-services.md",
  "pathPatterns": ["src/api/services/**"],
  "taskTypes": ["feature", "refactor"],
  "keywords": ["service", "singleton", "cache", "Lock", "initialize"],
  "whenToUse": ["New service needed", "Adding caching to service"]
}
```

### Function 3: `scoreSkills(context, availableSkills)`

Ranks skills by relevance using the scoring algorithm.

**Scoring Algorithm (0-100 scale):**

| Signal | Points | Example |
|--------|--------|---------|
| File path match | +50 | `*/services/*` → `creating-python-services` |
| Task type match | +30 | "add test" → `writing-pytest-tests` |
| Keyword match | +10 each (max 30) | "validation" → `creating-pydantic-schemas` |
| Layer match | +20 | Frontend file → frontend skills |
| Recent success | +5 | Skill worked well on similar task |

**Confidence Thresholds:**

| Score | Confidence | Action |
|-------|------------|--------|
| 80-100 | 0.80-1.00 | High confidence, auto-select |
| 50-79 | 0.50-0.79 | Medium confidence, review and select |
| 0-49 | 0.00-0.49 | Low confidence, trigger fallback |

**Return top 3 skills with confidence scores.**

### Function 4: `handleNoMatch(closestMatches)`

When no skill scores above 0.5 threshold:

**Fallback Response:**
```json
{
  "recommendations": [],
  "fallback": {
    "action": "manual_guidance_needed",
    "reason": "No skill matched with confidence > 0.5",
    "closest_matches": [
      {"skill": "creating-python-services", "confidence": 0.42}
    ],
    "suggestions": [
      "Describe the task in more detail",
      "Specify which layer (frontend/backend/database) this affects",
      "Check if this is a new pattern that needs a skill created"
    ]
  }
}
```

**Fallback Behavior:**
1. Present closest matches (even if low confidence)
2. Suggest task refinement
3. Offer to proceed with manual implementation
4. Flag for potential new skill creation

### Function 5: `invokeSkill(skillName, context)`

Loads the selected skill and applies its decision tree.

**Steps:**
1. Read full skill content from SKILL.md
2. Extract decision tree and examples
3. Apply decision tree to context
4. Return structured recommendations

**Output:**
```json
{
  "skill_applied": "creating-python-services",
  "recommendations": ["Use singleton pattern", "Add Lock for thread-safety"],
  "patterns_to_follow": ["src/api/services/model_service.py:15"],
  "examples": ["Similar: trading_service.py uses same pattern"],
  "quality_checklist": ["Thread-safe with Lock", "is_loaded property"]
}
```

### Function 6: `logRouting(decision)`

Records routing decisions for audit trail.

**Log Format:**
```
[2026-01-16 14:32:15] Task: "Add validation to UserService"
[2026-01-16 14:32:15] Files: ["src/api/services/user_service.py"]
[2026-01-16 14:32:15] Context Analysis:
  - Layer: api-services (from src/api/services/)
  - Task Type: feature (keywords: add, validation)
  - Keywords: validation, user, email, service
[2026-01-16 14:32:15] Skills Scored:
  - creating-python-services: 0.90 (path:+50, type:+30, keyword:+10)
  - creating-pydantic-schemas: 0.78 (type:+30, keyword:+20, related:+28)
  - creating-sqlalchemy-models: 0.65 (keyword:+20, related:+45)
[2026-01-16 14:32:15] Selected: creating-python-services (confidence: 0.90)
```

---

## Input/Output Contract

### Input Format (from Agent)

```json
{
  "task": "Add validation to UserService",
  "files": ["src/api/services/user_service.py"],
  "context": "Prevent users from having duplicate emails",
  "phase": "implementation",
  "agent": "code-engineer"
}
```

### Output Format (to Agent)

```json
{
  "analysis": {
    "layer": "api-services",
    "taskType": "feature",
    "keywords": ["validation", "user", "email", "service"]
  },
  "recommendations": [
    {
      "skill": "creating-python-services",
      "confidence": 0.90,
      "reason": "File path matches backend service pattern, task mentions validation",
      "skill_path": ".claude/skills/backend/creating-python-services.md",
      "next_steps": [
        "Review existing validation patterns in service",
        "Add validation method with proper error handling",
        "Ensure thread-safety for shared state"
      ]
    },
    {
      "skill": "creating-pydantic-schemas",
      "confidence": 0.78,
      "reason": "Validation often requires Pydantic schema constraints",
      "skill_path": ".claude/skills/backend/creating-pydantic-schemas.md",
      "next_steps": [
        "Add unique constraint validation in schema",
        "Use Field() with constraints"
      ]
    },
    {
      "skill": "creating-sqlalchemy-models",
      "confidence": 0.65,
      "reason": "Duplicate prevention often requires database unique constraint",
      "skill_path": ".claude/skills/database/SKILL.md",
      "next_steps": [
        "Add unique index to email column",
        "Add migration for schema change"
      ]
    }
  ],
  "multi_skill": false,
  "execution_order": null,
  "audit_logged": true,
  "routing_log": [
    "Input: task='Add validation to UserService', files=['src/api/services/user_service.py']",
    "Detected layer: api-services (from src/api/services/)",
    "Detected task type: feature (keywords: add, validation)",
    "Discovered 24 skills in .claude/skills/",
    "Scored all skills, top 3:",
    "  1. creating-python-services: 90 pts (path:+50, type:+30, keyword:+10)",
    "  2. creating-pydantic-schemas: 78 pts (type:+30, keyword:+20, related:+28)",
    "  3. creating-sqlalchemy-models: 65 pts (keyword:+20, related:+45)",
    "Selected: creating-python-services (highest confidence)"
  ]
}
```

---

## Multi-Skill Scenarios

When task requires multiple skills (confidence scores within 0.1 of each other):

**Detection:**
```
Score[skill_1] - Score[skill_2] <= 10 points (0.10 confidence)
```

**Multi-Skill Response:**
```json
{
  "recommendations": [
    {"skill": "creating-sqlalchemy-models", "confidence": 0.91},
    {"skill": "creating-python-services", "confidence": 0.89},
    {"skill": "creating-pydantic-schemas", "confidence": 0.87}
  ],
  "multi_skill": true,
  "execution_order": [
    "1. creating-sqlalchemy-models (database layer first)",
    "2. creating-pydantic-schemas (schemas depend on models)",
    "3. creating-python-services (service depends on schemas)"
  ],
  "reason": "Task spans multiple layers - apply skills in dependency order"
}
```

**Dependency Order:**
```
1. Database (creating-sqlalchemy-models)
2. Schemas (creating-pydantic-schemas)
3. Services (creating-python-services)
4. Routes (creating-fastapi-endpoints)
5. Tests (writing-pytest-tests)
6. Frontend (creating-react-components, if applicable)
```

---

## Examples

### Example 1: Frontend UI Bug Fix

**Input:**
```json
{
  "task": "Fix login button alignment on mobile",
  "files": ["frontend/src/components/LoginForm.jsx"]
}
```

**Router Analysis:**
```
Layer:     frontend-components (from frontend/src/components/)
Task Type: bug_fix (keyword "fix")
Keywords:  button, alignment, mobile, login

Skill Scores:
1. creating-react-components: 95 pts
   - File path: +50 (frontend/src/components/)
   - Task type: +30 (bug_fix on UI component)
   - Keywords: +15 (component, mobile)

2. writing-vitest-tests: 50 pts
   - Task type: +30 (testing related)
   - Keywords: +20 (component, test)

Selected: creating-react-components
Multi-skill: No (clear winner)
```

**Output:**
```json
{
  "recommendations": [
    {
      "skill": "creating-react-components",
      "confidence": 0.95,
      "reason": "File path matches frontend component, UI fix task",
      "next_steps": [
        "Check responsive breakpoints in TailwindCSS classes",
        "Test on mobile viewport (320px, 375px, 414px)",
        "Use flexbox/grid for alignment instead of margins"
      ]
    }
  ],
  "multi_skill": false
}
```

### Example 2: Post-Design Test Planning

**Input:**
```json
{
  "task": "Generate test plan for Account Export PDF feature",
  "files": ["src/api/routes/export.py"],
  "context": {
    "phase": "post-design",
    "acceptance_criteria": ["PDF contains account summary", "Includes transaction history"]
  }
}
```

**Router Analysis:**
```
Phase:     post-design (triggers test planning priority)
Layer:     api-routes
Task Type: testing (post-design phase implies test planning)
Keywords:  export, pdf, account, test, plan

Skill Scores:
1. planning-test-scenarios: 95 pts
   - Phase: +40 (post-design trigger)
   - Task type: +30 (testing)
   - Keywords: +25 (acceptance criteria present)

2. creating-fastapi-endpoints: 60 pts
   - File path: +50 (src/api/routes/)
   - Keywords: +10 (export)

Selected: planning-test-scenarios (phase-based priority)
```

**Output:**
```json
{
  "recommendations": [
    {
      "skill": "planning-test-scenarios",
      "confidence": 0.95,
      "reason": "Post-design phase requires test planning, acceptance criteria available",
      "next_steps": [
        "Generate test scenarios from acceptance criteria",
        "Define test data requirements for PDF export",
        "Chain to generating-test-data for fixtures"
      ]
    }
  ],
  "multi_skill": false
}
```

### Example 3: Multi-Layer API Feature

**Input:**
```json
{
  "task": "Add CustomerStatus field to customer management",
  "files": [
    "src/api/database/models.py",
    "src/api/schemas/customer.py",
    "src/api/services/customer_service.py",
    "frontend/src/components/CustomerCard.jsx"
  ],
  "context": "New field needs database, backend, and frontend changes"
}
```

**Router Analysis:**
```
Layers:    database, api-schemas, api-services, frontend-components (MULTI-LAYER)
Task Type: feature (keyword "add")
Keywords:  customer, status, field

Skill Scores:
1. creating-sqlalchemy-models: 91 pts
   - File path: +50 (src/api/database/)
   - Task type: +30 (feature)
   - Keywords: +11 (field)

2. creating-python-services: 89 pts
   - File path: +50 (src/api/services/)
   - Task type: +30 (feature)
   - Keywords: +9 (customer, service)

3. creating-pydantic-schemas: 87 pts
   - File path: +50 (src/api/schemas/)
   - Task type: +30 (feature)
   - Keywords: +7 (customer)

Multi-skill: Yes (3 skills within 0.1 confidence)
```

**Output:**
```json
{
  "recommendations": [
    {"skill": "creating-sqlalchemy-models", "confidence": 0.91},
    {"skill": "creating-python-services", "confidence": 0.89},
    {"skill": "creating-pydantic-schemas", "confidence": 0.87}
  ],
  "multi_skill": true,
  "execution_order": [
    "1. creating-sqlalchemy-models (database first - add CustomerStatus enum/column)",
    "2. creating-pydantic-schemas (schema depends on model)",
    "3. creating-python-services (service depends on schema)",
    "4. creating-fastapi-endpoints (route uses service, if needed)",
    "5. creating-react-components (frontend displays data)"
  ],
  "reason": "Task spans 4 layers - apply skills in dependency order"
}
```

### Example 4: No Match - WebSocket Feature

**Input:**
```json
{
  "task": "Implement WebSocket real-time notifications",
  "files": ["src/api/services/notification_service.py"],
  "context": "New real-time feature, no existing pattern"
}
```

**Router Analysis:**
```
Layer:     api-services
Task Type: feature
Keywords:  websocket, real-time, notifications

Skill Scores:
1. creating-python-services: 45 pts
   - File path: +50 (src/api/services/)
   - Task type: +30 (feature)
   - Penalty: -35 (no WebSocket-specific guidance)

2. (no other matches above 0.3)

Fallback triggered: No skill scores above 0.5 threshold
```

**Output:**
```json
{
  "recommendations": [],
  "fallback": {
    "action": "manual_guidance_needed",
    "reason": "No skill for WebSocket patterns exists",
    "closest_matches": [
      {"skill": "creating-python-services", "confidence": 0.45}
    ],
    "suggestions": [
      "Search codebase for existing WebSocket usage: grep -r 'websocket' src/",
      "If pattern found, consider creating 'implementing-websocket-handlers' skill",
      "Proceed with manual implementation following general service patterns",
      "Reference FastAPI WebSocket documentation"
    ]
  },
  "audit_logged": true
}
```

---

## Available Skills Registry

### Backend Layer

| Skill | Path Triggers | Keyword Triggers |
|-------|---------------|------------------|
| `creating-fastapi-endpoints` | `src/api/routes/**` | endpoint, route, FastAPI, APIRouter, GET, POST |
| `creating-python-services` | `src/api/services/**` | service, singleton, cache, Lock, initialize |
| `creating-pydantic-schemas` | `src/api/schemas/**` | schema, BaseModel, Field, response, request |
| `implementing-prediction-models` | `src/models/**` | model, predict, train, ensemble, MTF |
| `creating-data-processors` | `src/data/processors/**` | processor, transform, validate, clean |

### Frontend Layer

| Skill | Path Triggers | Keyword Triggers |
|-------|---------------|------------------|
| `creating-react-components` | `frontend/src/components/**` | component, jsx, useState, loading, error |
| `creating-api-clients` | `frontend/src/api/**` | fetch, client, API, hook |

### Database Layer

| Skill | Path Triggers | Keyword Triggers |
|-------|---------------|------------------|
| `creating-sqlalchemy-models` | `src/api/database/**` | SQLAlchemy, Column, Table, Index, relationship |

### Feature Engineering

| Skill | Path Triggers | Keyword Triggers |
|-------|---------------|------------------|
| `creating-technical-indicators` | `src/features/technical/**` | indicator, RSI, MACD, EMA, bollinger, ATR |
| `configuring-indicator-yaml` | `configs/indicators/**` | config, yaml, priority, enabled |

### Data Layer

| Skill | Path Triggers | Keyword Triggers |
|-------|---------------|------------------|
| `adding-data-sources` | `src/data/sources/**` | source, connector, fetch, provider |

### Trading Domain

| Skill | Path Triggers | Keyword Triggers |
|-------|---------------|------------------|
| `running-backtests` | `src/simulation/**` | backtest, simulate, walk-forward, WFO |
| `analyzing-trading-performance` | - | sharpe, sortino, drawdown, metrics, performance |
| `implementing-risk-management` | `src/trading/**` | risk, position, circuit, drawdown |

### Testing

| Skill | Path Triggers | Keyword Triggers |
|-------|---------------|------------------|
| `writing-pytest-tests` | `tests/**` | pytest, TestClient, Mock, fixture, assert |
| `writing-vitest-tests` | `*.test.jsx`, `*.test.tsx` | vitest, render, screen, Testing Library |

### Quality & Testing

| Skill | Path Triggers | Keyword Triggers |
|-------|---------------|------------------|
| `planning-test-scenarios` | - | test plan, acceptance criteria, coverage |
| `generating-test-data` | - | fixture, mock, test data, builder, factory |
| `validating-time-series-data` | - | time series, leakage, chronological, shift |
| `creating-dataclasses` | - | dataclass, DTO, @dataclass, frozen |

### Build & Deployment

| Skill | Path Triggers | Keyword Triggers |
|-------|---------------|------------------|
| `creating-cli-scripts` | `scripts/**` | argparse, CLI, command-line, main |

### Meta-Skills

| Skill | Purpose |
|-------|---------|
| `routing-to-skills` | This skill - dynamic skill discovery |
| `improving-framework-continuously` | Process errors to improve agents/skills |

---

## Conflict Resolution

When multiple skills score equally or within 5 points:

### Priority Rules

```
1. Phase-specific skills first
   - post-design → planning-test-scenarios
   - implementation → domain skill

2. More specific path match wins
   - src/api/routes/ > src/api/ > src/

3. Explicit skill reference in task wins
   - "using the creating-pydantic-schemas pattern" → that skill

4. Multi-file: apply skills in dependency order
   - database → schemas → services → routes → frontend

5. Session history bonus
   - Previously successful skill gets +5 points

6. Ask for preference if still tied
   - Present options with tradeoffs
```

### Manual Override

Agents can bypass routing with explicit skill selection:

```
"Skip routing, use skill: implementing-prediction-models"
```

The router respects explicit skill requests and logs the override.

---

## Integration with Agents

### Agent-Specific Routing Behavior

| Agent | Phase | Routing Behavior |
|-------|-------|------------------|
| Requirements Analyst | pre-design | Reference skills for capabilities |
| Solution Architect | design | Reference skills for patterns, invoke test planning post-design |
| Code Engineer | implementation | Primary skill consumer - invoke implementation skills |
| Quality Guardian | review | Reference skills for pattern verification |
| Test Automator | testing | Invoke test planning and data generation skills |
| Documentation Curator | documentation | Reference skills for accuracy |

### Workflow Integration

```
Agent receives task
│
├─ Invoke skill-router with {task, files, context}
│
├─ Router returns recommendations:
│   ├─ Top 3 skills with confidence scores
│   ├─ Reasons for each recommendation
│   └─ Suggested next steps
│
├─ Agent reviews recommendations:
│   ├─ High confidence (≥0.80): Auto-select
│   ├─ Medium confidence (0.50-0.79): Review and select
│   └─ Low confidence (<0.50): Fallback triggered
│
├─ Agent loads selected skill content
│
├─ Agent follows skill's decision tree
│
└─ If skill references another → Router handles chaining
```

---

## Quality Checklist

- [ ] Task analyzed for layer, type, and keywords
- [ ] All skills discovered from `.claude/skills/`
- [ ] Top 3 recommendations returned with confidence scores
- [ ] Each recommendation includes reasons and next_steps
- [ ] Confidence scores calibrated (0-1 range)
- [ ] Routing log captures decision process
- [ ] Manual override respected if provided
- [ ] Multi-file tasks get sequenced skill recommendations
- [ ] Fallback triggered when confidence < 0.5
- [ ] Audit trail recorded for debugging

## Common Mistakes

| Mistake | Impact | Fix |
|---------|--------|-----|
| Hardcoding skill names | Breaks when skills renamed | Always discover dynamically |
| Ignoring context.phase | Misses test planning triggers | Check for post-design phase |
| Single skill bias | User can't see alternatives | Always return top 3 |
| Opaque routing | User can't debug selections | Include routing_log |
| Ignoring multi-file | Wrong skill order | Sequence by dependency |
| Low confidence auto-select | Wrong skill applied | Require review <0.80 |
| No fallback handling | Silent failure on no match | Always provide suggestions |
| Missing audit trail | Can't debug wrong routing | Log all decisions |

## Anti-Hallucination Rules

1. **Skill Validation**: Use Glob to verify skill files exist before recommending
2. **No Invented Skills**: Only recommend skills discovered in `.claude/skills/`
3. **Path Citation**: When referencing patterns, cite actual skill_path from registry
4. **Confidence Honesty**: If unsure about match, return lower confidence
5. **Fallback Over Guess**: When no match, trigger fallback instead of guessing
6. **Log Everything**: All decisions must be auditable in routing_log

---

## Related Skills

All discoverable skills are listed in [SKILL-INDEX.md](../SKILL-INDEX.md).

This meta-skill enables agents to work across different projects without hardcoded skill references.

---

## Verification & Grounding

### Citation Requirements (Anti-Hallucination)

When the router recommends skills, it MUST:

1. **Verify skill exists**: Use Glob to confirm `.claude/skills/[path]` exists before recommending
2. **Cite exact paths**: Provide `skill_path` from actual filesystem, not invented paths
3. **Ground in evidence**: Each recommendation must cite why (path match, keyword match, etc.)
4. **Allow uncertainty**: If confidence < 0.5, use fallback instead of forcing a match

### Verification Steps

Before returning recommendations:

```
1. For each recommended skill:
   □ Glob confirms skill file exists at skill_path
   □ Skill name matches YAML frontmatter
   □ Confidence score is justified by scoring algorithm
   □ Reasons cite specific matches (file:+50, keyword:+10)

2. For multi-skill scenarios:
   □ All skills exist and are valid
   □ Execution order respects dependencies
   □ No circular dependencies

3. For fallback scenarios:
   □ No invented skills in suggestions
   □ Suggestions are actionable
   □ Closest matches are real skills
```

### What to Say When Uncertain

If the router cannot confidently match a skill:

**DO say:**
- "No skill matched with confidence > 0.5"
- "Closest match: [skill] at 0.42 confidence - may not be appropriate"
- "Consider creating a new skill for [pattern]"

**DO NOT say:**
- "Use [invented-skill-name] for this task"
- Recommend skills that don't exist in `.claude/skills/`
- Force high confidence when evidence is weak

---

<!-- Skill Metadata
Version: 1.2.0
Created: 2026-01-07
Updated: 2026-01-18
Skills Indexed: 24
Last Registry Update: 2026-01-18

Changes in 1.2.0:
- Added Verification & Grounding section (anti-hallucination)
- Added Citation Requirements for skill recommendations
- Added "What to Say When Uncertain" guidance
- Fixed SKILL-INDEX.md reference (was README.md)
- Updated skills count to 24
- Enhanced from Anthropic best practices documentation

Changes in 1.1.0:
- Added no-match fallback handling (confidence < 0.5)
- Added multi-skill scenario detection (within 0.1 confidence)
- Added audit trail specification with log format
- Enhanced examples with multi-layer and fallback cases
- Added anti-hallucination rules
- Improved conflict resolution documentation
-->
