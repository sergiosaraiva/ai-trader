---
name: routing-to-skills
description: Routes tasks to appropriate skills by analyzing file paths, task types, and domain keywords. Returns ranked skill recommendations with confidence scores. Use when starting any implementation task to discover relevant skills, or when agents need to dynamically select skills for a workflow step.
---

# Routing to Skills

A meta-skill that enables agents to dynamically discover and invoke appropriate skills based on task context. This enables cross-project portability without hardcoding skill names.

## Quick Reference

```
1. analyzeContext(task, files) → Extract layer, task type, keywords
2. discoverSkills()            → Scan .claude/skills/ for available skills
3. scoreSkills(context, skills)→ Rank by relevance (0-100 points)
4. invokeSkill(skill, context) → Load and apply selected skill
```

## When to Use

- Starting any implementation task (auto-route to best skill)
- Agent workflow step needs skill-based guidance
- Multiple skills might apply (need ranking)
- Task description is ambiguous about which pattern to follow
- Post-design phase (Solution Architect → Test Automator handoff)

## When NOT to Use

- Task explicitly names which skill to use
- Simple tasks requiring no specialized patterns
- Research/exploration tasks (not implementation)

---

## Implementation Functions

### Function 1: `analyzeContext(task, files)`

Extracts layer, task type, and keywords from the input.

```python
def analyzeContext(task: str, files: list[str]) -> dict:
    """
    Parse file paths to determine layer/technology.
    Analyze task type (feature, bug fix, refactoring, migration).
    Extract domain concepts from description.
    """
    context = {
        "layer": None,
        "task_type": None,
        "keywords": set(),
        "technologies": set()
    }

    # Layer detection from file paths
    for file in files:
        if "src/api/routes/" in file:
            context["layer"] = "api-routes"
            context["technologies"].add("FastAPI")
        elif "src/api/services/" in file:
            context["layer"] = "api-services"
        elif "src/api/schemas/" in file:
            context["layer"] = "api-schemas"
            context["technologies"].add("Pydantic")
        elif "src/api/database/" in file:
            context["layer"] = "database"
            context["technologies"].add("SQLAlchemy")
        elif "frontend/src/components/" in file:
            context["layer"] = "frontend-components"
            context["technologies"].add("React")
        elif "frontend/src/api/" in file:
            context["layer"] = "frontend-api"
        elif "src/models/" in file:
            context["layer"] = "model"
        elif "src/features/technical/" in file:
            context["layer"] = "feature-engineering"
        elif "src/data/sources/" in file:
            context["layer"] = "data-sources"
        elif "src/data/processors/" in file:
            context["layer"] = "data-processors"
        elif "src/trading/" in file:
            context["layer"] = "trading"
        elif "tests/" in file or ".test." in file:
            context["layer"] = "testing"
        elif "scripts/" in file:
            context["layer"] = "cli"

    # Task type detection from keywords
    task_lower = task.lower()
    if any(kw in task_lower for kw in ["fix", "bug", "error", "broken", "issue"]):
        context["task_type"] = "bug_fix"
    elif any(kw in task_lower for kw in ["add", "create", "new", "implement", "build"]):
        context["task_type"] = "feature"
    elif any(kw in task_lower for kw in ["refactor", "improve", "clean", "optimize"]):
        context["task_type"] = "refactor"
    elif any(kw in task_lower for kw in ["test", "validate", "verify", "spec"]):
        context["task_type"] = "testing"
    elif any(kw in task_lower for kw in ["migrate", "upgrade", "update", "move"]):
        context["task_type"] = "migration"
    elif any(kw in task_lower for kw in ["config", "configure", "setup", "env"]):
        context["task_type"] = "config"
    else:
        context["task_type"] = "feature"  # Default

    # Keyword extraction
    domain_keywords = {
        "model", "train", "predict", "ml", "ensemble",
        "indicator", "rsi", "macd", "ema", "bollinger", "atr",
        "risk", "position", "drawdown", "circuit",
        "backtest", "simulate", "walk-forward",
        "source", "fetch", "ohlcv", "candle",
        "endpoint", "route", "api", "request", "response",
        "component", "state", "loading", "error",
        "sharpe", "sortino", "performance", "metrics",
        "test", "mock", "fixture", "assert",
        "database", "table", "index", "query",
        "schema", "validation", "field",
        "service", "singleton", "cache"
    }
    for kw in domain_keywords:
        if kw in task_lower:
            context["keywords"].add(kw)

    return context
```

### Function 2: `discoverSkills()`

Scans the skills directory and builds a registry.

```python
def discoverSkills() -> list[dict]:
    """
    Scan .claude/skills/ directory.
    Parse each SKILL.md for 'When to Use' conditions.
    Build skill registry with metadata.
    """
    skills = []
    skill_dirs = glob(".claude/skills/**/SKILL.md") + glob(".claude/skills/**/*.md")

    for skill_path in skill_dirs:
        # Parse YAML frontmatter
        content = read(skill_path)
        metadata = parse_yaml_frontmatter(content)

        skill = {
            "name": metadata.get("name"),
            "description": metadata.get("description"),
            "path": skill_path,
            "path_patterns": extract_path_patterns(content),
            "task_types": extract_task_types(content),
            "keywords": extract_keywords(content),
            "when_to_use": extract_when_to_use(content)
        }
        skills.append(skill)

    return skills
```

### Function 3: `scoreSkills(context, availableSkills)`

Ranks skills by relevance using the scoring algorithm.

```python
def scoreSkills(context: dict, availableSkills: list[dict]) -> list[dict]:
    """
    Score each skill based on context match.
    Return top 3 skills with confidence scores.
    """
    scored = []

    for skill in availableSkills:
        score = 0
        reasons = []
        next_steps = []

        # File path match: +50 points (strongest signal)
        if context["layer"] and skill_matches_layer(skill, context["layer"]):
            score += 50
            reasons.append(f"File path matches {context['layer']} layer")

        # Task type match: +30 points
        if context["task_type"] in skill["task_types"]:
            score += 30
            reasons.append(f"Task type '{context['task_type']}' matches skill")

        # Keyword match: +10 points per keyword (max 30)
        matched_keywords = context["keywords"] & skill["keywords"]
        keyword_score = min(len(matched_keywords) * 10, 30)
        score += keyword_score
        if matched_keywords:
            reasons.append(f"Keywords: {', '.join(matched_keywords)}")

        # Recent success bonus: +5 points
        if skill["name"] in session_successes:
            score += 5
            reasons.append("Previously successful in session")

        # Generate next steps from skill's decision tree
        next_steps = generate_next_steps(skill, context)

        scored.append({
            "skill": skill["name"],
            "confidence": round(score / 100, 2),
            "score": score,
            "reasons": reasons,
            "next_steps": next_steps
        })

    # Sort by score descending, return top 3
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:3]
```

### Function 4: `invokeSkill(skill, context)`

Loads the selected skill and applies its decision tree.

```python
def invokeSkill(skill: str, context: dict) -> dict:
    """
    Load skill content from SKILL.md.
    Provide task context to skill.
    Apply skill's decision tree.
    Return skill recommendations.
    """
    # Read full skill content
    skill_content = read(f".claude/skills/{skill_path}/SKILL.md")

    # Extract decision tree
    decision_tree = extract_decision_tree(skill_content)

    # Apply decision tree to context
    recommendations = apply_decision_tree(decision_tree, context)

    # Return structured recommendations
    return {
        "skill_applied": skill,
        "recommendations": recommendations,
        "patterns_to_follow": extract_patterns(skill_content),
        "examples": extract_relevant_examples(skill_content, context),
        "quality_checklist": extract_checklist(skill_content)
    }
```

---

## Scoring Algorithm

### Point System (0-100 scale)

```
SCORING WEIGHTS:
  file_path_match:    +50 points  # Strongest signal
  task_type_match:    +30 points  # Strong signal
  keyword_match:      +10 points  # Per keyword (max 30)
  recent_success:     +5 points   # Session history bonus

CONFIDENCE THRESHOLDS:
  High:   >= 0.80  # Strong recommendation, auto-select
  Medium: 0.50-0.79  # Good match, consider this skill
  Low:    < 0.50  # Weak match, verify fit manually
```

### Confidence Calibration

| Score | Confidence | Interpretation |
|-------|------------|----------------|
| 90-100 | 0.90-1.00 | Perfect match, proceed confidently |
| 80-89 | 0.80-0.89 | Strong match, high confidence |
| 60-79 | 0.60-0.79 | Good match, review recommendations |
| 40-59 | 0.40-0.59 | Partial match, verify applicability |
| 0-39 | 0.00-0.39 | Weak match, consider alternatives |

---

## Input/Output Contract

### Input Format

```json
{
  "task": "Add validation to UserService",
  "files": ["src/services/UserService.py"],
  "context": "Prevent users from having duplicate emails"
}
```

### Output Format

```json
{
  "analysis": {
    "layer": "api-services",
    "task_type": "feature",
    "keywords": ["validation", "service"]
  },
  "recommendations": [
    {
      "skill": "creating-python-services",
      "confidence": 0.92,
      "reason": "File path matches backend service pattern, task mentions validation",
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
      "next_steps": [
        "Add unique constraint validation in schema",
        "Use Field() with constraints"
      ]
    },
    {
      "skill": "creating-sqlalchemy-models",
      "confidence": 0.65,
      "reason": "Duplicate prevention often requires database unique constraint",
      "next_steps": [
        "Add unique index to email column",
        "Add migration for schema change"
      ]
    }
  ],
  "selected": "creating-python-services",
  "routing_log": [
    "Input: task='Add validation to UserService', files=['src/services/UserService.py']",
    "Detected layer: api-services (from src/services/)",
    "Detected task type: feature (keywords: add, validation)",
    "Scored 23 skills",
    "Top 3: creating-python-services(92), creating-pydantic-schemas(78), creating-sqlalchemy-models(65)",
    "Selected: creating-python-services (highest confidence)"
  ]
}
```

---

## Layer Detection

### File Path → Layer Mapping

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

### Task Type Detection

| Keywords | Task Type | Score Boost |
|----------|-----------|-------------|
| fix, bug, error, broken, issue | bug_fix | +10 to domain skill |
| add, create, new, implement, build | feature | +20 to pattern skill |
| refactor, improve, clean, optimize | refactor | +15 to pattern skill |
| test, validate, verify, spec | testing | +30 to testing skills |
| config, configure, setup, env | config | +25 to config skills |
| migrate, upgrade, update, move | migration | +10 to affected layer |

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
  ]
}
```

### Example 2: Backend Service Validation

**Input:**
```json
{
  "task": "Add validation to UserService",
  "files": ["src/api/services/user_service.py"],
  "context": "Prevent users from having duplicate emails"
}
```

**Router Analysis:**
```
Layer:     api-services (from src/api/services/)
Task Type: feature (keyword "add")
Keywords:  validation, service, duplicate, email

Skill Scores:
1. creating-python-services: 90 pts
   - File path: +50 (src/api/services/)
   - Task type: +30 (feature)
   - Keywords: +10 (service)

2. creating-pydantic-schemas: 70 pts
   - Task type: +30 (feature)
   - Keywords: +20 (validation)
   - Related: +20 (validation context)

3. creating-sqlalchemy-models: 60 pts
   - Keywords: +20 (duplicate prevention)
   - Related: +20 (unique constraint)
   - Database layer: +20

Selected: creating-python-services
Note: Consider chaining to creating-sqlalchemy-models for unique constraint
```

### Example 3: Post-Design Test Planning

**Input:**
```json
{
  "task": "Design Account Export PDF feature",
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
Keywords:  export, pdf, account

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
  ]
}
```

### Example 4: Multi-File API Feature

**Input:**
```json
{
  "task": "Add new portfolio analytics endpoint",
  "files": [
    "src/api/routes/analytics.py",
    "src/api/schemas/analytics.py",
    "src/api/services/analytics_service.py"
  ]
}
```

**Router Analysis:**
```
Layers:    api-routes, api-schemas, api-services (multi-layer)
Task Type: feature (keyword "add", "new")
Keywords:  analytics, endpoint, portfolio

Skill Scores:
1. creating-fastapi-endpoints: 90 pts
   - File path: +50 (src/api/routes/)
   - Task type: +30 (feature)
   - Keywords: +10 (endpoint)

2. creating-python-services: 85 pts
   - File path: +50 (src/api/services/)
   - Task type: +30 (feature)
   - Keywords: +5 (analytics)

3. creating-pydantic-schemas: 80 pts
   - File path: +50 (src/api/schemas/)
   - Task type: +30 (feature)

Recommendation: Apply skills in sequence
1. creating-pydantic-schemas → Define request/response models first
2. creating-python-services → Implement business logic
3. creating-fastapi-endpoints → Wire up the route
```

### Example 5: Database Model Change

**Input:**
```json
{
  "task": "Add trade positions table with foreign key to accounts",
  "files": ["src/api/database/models.py"]
}
```

**Router Analysis:**
```
Layer:     database (from src/api/database/)
Task Type: feature (keyword "add")
Keywords:  table, foreign key, positions, accounts

Skill Scores:
1. creating-sqlalchemy-models: 95 pts
   - File path: +50 (src/api/database/)
   - Task type: +30 (feature)
   - Keywords: +15 (table, foreign key)

2. creating-data-processors: 45 pts
   - Related: +30 (data layer)
   - Keywords: +15 (positions)

Selected: creating-sqlalchemy-models
```

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
   - schemas → services → routes

5. Session history bonus
   - Previously successful skill gets +5 points

6. Ask for preference if still tied
   - Present options with tradeoffs
```

### Multi-Skill Workflows

For complex tasks touching multiple layers:

```
Recommended Order:
1. Database (creating-sqlalchemy-models)
2. Schemas (creating-pydantic-schemas)
3. Services (creating-python-services)
4. Routes (creating-fastapi-endpoints)
5. Tests (writing-pytest-tests)
6. Frontend (creating-react-components, if applicable)
```

### Manual Override

Agents can bypass routing with explicit skill selection:

```
"Skip routing, use skill: implementing-prediction-models"
```

The router respects explicit skill requests and logs the override.

---

## Transparency & Logging

### Routing Log Format

Every routing decision includes a transparent log:

```
routing_log: [
  "Input: task='...', files=[...]",
  "Detected layer: {layer} (from {path_evidence})",
  "Detected task type: {type} (keywords: {keywords})",
  "Discovered {n} skills in .claude/skills/",
  "Scored all skills, top 3:",
  "  1. {skill1}: {score1} pts ({reasons})",
  "  2. {skill2}: {score2} pts ({reasons})",
  "  3. {skill3}: {score3} pts ({reasons})",
  "Selected: {skill} (reason: {selection_reason})",
  "Override: {none|manual|phase-based}"
]
```

### Why Logging Matters

- **Debugging**: Understand unexpected routing decisions
- **Improvement**: Identify patterns where routing fails
- **Trust**: Users can verify routing logic
- **Audit**: Track which skills were applied to tasks

---

## Available Skills Registry

### Backend Layer (`backend/`)

| Skill | Path Triggers | Keyword Triggers |
|-------|---------------|------------------|
| `creating-fastapi-endpoints` | `src/api/routes/**` | endpoint, route, FastAPI, APIRouter, GET, POST |
| `creating-python-services` | `src/api/services/**` | service, singleton, cache, Lock, initialize |
| `creating-pydantic-schemas` | `src/api/schemas/**` | schema, BaseModel, Field, response, request |
| `implementing-prediction-models` | `src/models/**` | model, predict, train, ensemble, MTF |
| ~~`creating-api-endpoints`~~ | `src/api/**` | *DEPRECATED → use creating-fastapi-endpoints* |
| `creating-data-processors` | `src/data/processors/**` | processor, transform, validate, clean |

### Frontend Layer (`frontend/`)

| Skill | Path Triggers | Keyword Triggers |
|-------|---------------|------------------|
| `creating-react-components` | `frontend/src/components/**` | component, jsx, useState, loading, error |
| `creating-api-clients` | `frontend/src/api/**` | fetch, client, API, hook |

### Database Layer (`database/`)

| Skill | Path Triggers | Keyword Triggers |
|-------|---------------|------------------|
| `creating-sqlalchemy-models` | `src/api/database/**` | SQLAlchemy, Column, Table, Index, relationship |

### Feature Engineering (`feature-engineering/`)

| Skill | Path Triggers | Keyword Triggers |
|-------|---------------|------------------|
| `creating-technical-indicators` | `src/features/technical/**` | indicator, RSI, MACD, EMA, bollinger, ATR |
| `configuring-indicator-yaml` | `configs/indicators/**` | config, yaml, priority, enabled |

### Data Layer (`data-layer/`)

| Skill | Path Triggers | Keyword Triggers |
|-------|---------------|------------------|
| `adding-data-sources` | `src/data/sources/**` | source, connector, fetch, provider |
| `creating-data-processors` | `src/data/processors/**` | OHLCV, candle, sequence, processor |

### Trading Domain (`trading-domain/`)

| Skill | Path Triggers | Keyword Triggers |
|-------|---------------|------------------|
| `running-backtests` | `src/simulation/**` | backtest, simulate, walk-forward, WFO |
| `analyzing-trading-performance` | - | sharpe, sortino, drawdown, metrics, performance |
| `implementing-risk-management` | `src/trading/**` | risk, position, circuit, drawdown |

### Testing (`testing/`)

| Skill | Path Triggers | Keyword Triggers |
|-------|---------------|------------------|
| `writing-pytest-tests` | `tests/**` | pytest, TestClient, Mock, fixture, assert |
| `writing-vitest-tests` | `*.test.jsx`, `*.test.tsx` | vitest, render, screen, Testing Library |

### Quality & Testing (`quality-testing/`)

| Skill | Path Triggers | Keyword Triggers |
|-------|---------------|------------------|
| `creating-dataclasses` | - | dataclass, DTO, @dataclass, frozen |
| `validating-time-series-data` | - | time series, leakage, chronological, shift |
| `planning-test-scenarios` | - | test plan, acceptance criteria, coverage |
| `generating-test-data` | - | fixture, mock, test data, builder, factory |

### Build & Deployment (`build-deployment/`)

| Skill | Path Triggers | Keyword Triggers |
|-------|---------------|------------------|
| `creating-cli-scripts` | `scripts/**` | argparse, CLI, command-line, main |

### Meta-Skills

| Skill | Purpose |
|-------|---------|
| `routing-to-skills` | This skill - dynamic skill discovery |
| `improving-framework-continuously` | Process errors to improve agents/skills |

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
├─ Agent reviews recommendations
│   ├─ High confidence (≥0.80): Auto-select
│   ├─ Medium confidence (0.50-0.79): Review and select
│   └─ Low confidence (<0.50): Manual selection or ask
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

## Common Mistakes

| Mistake | Impact | Fix |
|---------|--------|-----|
| Hardcoding skill names | Breaks when skills renamed | Always discover dynamically |
| Ignoring context.phase | Misses test planning triggers | Check for post-design phase |
| Single skill bias | User can't see alternatives | Always return top 3 |
| Opaque routing | User can't debug selections | Include routing_log |
| Ignoring multi-file | Wrong skill order | Sequence by dependency |
| Low confidence auto-select | Wrong skill applied | Require review <0.80 |

## Related Skills

All discoverable skills are listed in [README.md](../README.md).

This meta-skill enables agents to work across different projects without hardcoded skill references.
