# Shared Agent Documentation

This document contains common patterns and rules for all agents. Individual agent files reference this to avoid duplication.

## Project Context

See `CLAUDE.md` for complete architecture, technology stack (Python 3.12+, FastAPI, React 19, XGBoost), and coding standards.

## Skill Routing Protocol

All agents use `routing-to-skills` meta-skill for dynamic skill discovery.

### Invocation

```
Input: { "task": "[description]", "files": [paths], "context": "[details]", "phase": "[phase]", "agent": "[agent-name]" }
Output: { "recommendations": [{"skill": "name", "confidence": 0.XX, "skill_path": "path"}], "multi_skill": bool }
```

### Confidence Thresholds

| Confidence | Action |
|------------|--------|
| >= 0.80 | Auto-select top skill |
| 0.50-0.79 | Review top 3, select best |
| < 0.50 | Use fallback table |

### Fallback Table

| Path Pattern | Default Skill |
|--------------|---------------|
| `backend/src/api/routes/**` | `backend` |
| `backend/src/api/services/**` | `creating-python-services` |
| `backend/src/api/schemas/**` | `creating-pydantic-schemas` |
| `backend/src/api/database/**` | `database` |
| `frontend/src/components/**` | `frontend` |
| `frontend/src/components/*Chart*` | `creating-chart-components` |
| `frontend/src/api/**` | `creating-api-clients` |
| `backend/tests/**` | `testing` |
| `frontend/**/*.test.jsx` | `writing-vitest-tests` |
| `backend/scripts/**` | `build-deployment` |
| `backend/src/features/**` | `creating-technical-indicators` |
| `backend/src/models/**/enhanced_*` | `creating-ml-features` |

### Multi-Skill Execution Order

```
database -> schemas -> services -> routes -> frontend -> tests
```

## Anti-Hallucination Rules (All Agents)

### Verification Requirements
1. **Read Before Act**: Always read files before editing/reviewing
2. **Verify Imports**: Check imported modules exist using Grep/Glob
3. **Pattern Matching**: Copy patterns from actual files, don't invent
4. **Cite Sources**: Reference file:line for claims

### Skill Guardrails
5. **Verify skill exists**: Use Glob to confirm `.claude/skills/[path]` before loading
6. **Trust router confidence**: Use fallback if confidence < 0.50
7. **Cite skill source**: Reference skill file and section when applying patterns
8. **No skill mixing**: Complete one skill's patterns before switching
9. **Don't invent skills**: Only use skills in `.claude/skills/`

### Prohibited Actions
10. **No placeholders**: All code must be functional
11. **No time estimates**: Never estimate duration
12. **No invented patterns**: Only use documented patterns

## Common Tool Permissions

| Tool | All Agents |
|------|------------|
| `Read` | Yes - read files, skills |
| `Grep` | Yes - search patterns |
| `Glob` | Yes - find files |
| `Bash` | Limited - static analysis only |
| `Task` | Yes - delegate exploration |

### Bash Allowed
- `python -m py_compile` - Syntax check
- `black --check` - Format check
- `mypy` - Type check
- `cd frontend && npm run lint` - Frontend lint
- `pytest -x` - Quick test
- `bandit -r` - Security scan

### Bash Prohibited
- `rm`, `mv` for files (use tools)
- `git` operations (user handles)
- Package install without approval

## Verification Commands

```bash
# Backend
python -m py_compile [file]
cd backend && black --check [file]
mypy [file] --ignore-missing-imports

# Frontend
cd frontend && npm run lint
cd frontend && npm test
```

## Skills Reference

See `.claude/skills/SKILL-INDEX.md` for complete skill catalog.

**Backend**: `backend`, `creating-python-services`, `creating-pydantic-schemas`, `implementing-prediction-models`, `creating-data-processors`

**Frontend**: `frontend`, `creating-api-clients`, `creating-chart-components`

**Database**: `database`

**Feature Engineering**: `creating-technical-indicators`, `configuring-indicator-yaml`, `creating-ml-features`

**Trading**: `running-backtests`, `analyzing-trading-performance`, `implementing-risk-management`

**Testing**: `testing`, `writing-vitest-tests`

**Caching**: `implementing-caching-strategies`

**Build**: `build-deployment`
