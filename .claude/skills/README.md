# AI-Trader Claude Code Skills

This directory contains Claude Code Skills extracted from the ai-trader codebase patterns.

## Skills Index

### Meta-Skills

| Skill | Description |
|-------|-------------|
| [routing-to-skills](skill-router/SKILL.md) | Routes tasks to appropriate skills by analyzing file paths, task types, and domain keywords |
| [improving-framework-continuously](continuous-improvement/SKILL.md) | Processes error reports to evolve agents and skills, enabling self-healing framework |

### Backend Layer (`backend/`)

| Skill | Description |
|-------|-------------|
| [creating-fastapi-endpoints](backend/SKILL.md) | FastAPI REST endpoints with error handling, Pydantic schemas, and service integration |
| [creating-python-services](backend/creating-python-services.md) | Thread-safe singleton services with lazy initialization and caching |
| [creating-pydantic-schemas](backend/creating-pydantic-schemas.md) | Request/response schemas with Field descriptions and validation |
| [implementing-prediction-models](backend/implementing-prediction-models.md) | BaseModel pattern with registry for ML models |
| ~~[creating-api-endpoints](backend/creating-api-endpoints.md)~~ | *DEPRECATED - Merged into creating-fastapi-endpoints* |
| [creating-data-processors](backend/creating-data-processors.md) | Validate/Clean/Transform pipeline pattern |

### Frontend Layer (`frontend/`)

| Skill | Description |
|-------|-------------|
| [creating-react-components](frontend/SKILL.md) | React components with loading, error, and data states using TailwindCSS |
| [creating-api-clients](frontend/creating-api-clients.md) | Centralized API client with error handling and typed endpoints |

### Database Layer (`database/`)

| Skill | Description |
|-------|-------------|
| [creating-sqlalchemy-models](database/SKILL.md) | SQLAlchemy ORM models with indexes and relationships for SQLite |

### Feature Engineering (`feature-engineering/`)

| Skill | Description |
|-------|-------------|
| [creating-technical-indicators](feature-engineering/creating-technical-indicators.md) | Indicator calculator class pattern |
| [configuring-indicator-yaml](feature-engineering/configuring-indicator-yaml.md) | YAML configuration for indicators |

### Data Layer (`data-layer/`)

| Skill | Description |
|-------|-------------|
| [adding-data-sources](data-layer/adding-data-sources.md) | BaseDataSource + Factory pattern |
| ~~processing-ohlcv-data~~ | *ARCHIVED - Was merged into creating-data-processors* |

### Trading Domain (`trading-domain/`)

| Skill | Description |
|-------|-------------|
| [running-backtests](trading-domain/running-backtests.md) | Backtester usage and walk-forward validation |
| [analyzing-trading-performance](trading-domain/analyzing-trading-performance.md) | Performance metrics (Sharpe, Sortino, drawdown) |
| [implementing-risk-management](trading-domain/implementing-risk-management.md) | RiskManager with circuit breakers |

### Testing (`testing/`)

| Skill | Description |
|-------|-------------|
| [writing-pytest-tests](testing/SKILL.md) | pytest tests with TestClient and mocked services |
| [writing-vitest-tests](testing/writing-vitest-tests.md) | Vitest tests for React components with Testing Library |

### Quality & Testing (`quality-testing/`)

| Skill | Description |
|-------|-------------|
| [creating-dataclasses](quality-testing/creating-dataclasses.md) | Python dataclass patterns for DTOs |
| [validating-time-series-data](quality-testing/validating-time-series-data.md) | Time series validation and leakage prevention |
| [planning-test-scenarios](quality-testing/planning-test-scenarios.md) | Generate test plans from acceptance criteria |
| [generating-test-data](quality-testing/generating-test-data.md) | Create test fixtures, mocks, and synthetic data |

### Build & Deployment (`build-deployment/`)

| Skill | Description |
|-------|-------------|
| [creating-cli-scripts](build-deployment/SKILL.md) | Python CLI scripts with argparse, logging, and progress output |

## Usage

Skills are automatically loaded by Claude Code when relevant to the current task. Each skill follows the structure:

```
skill-category/
└── skill-name.md
    ├── YAML frontmatter (name, description)
    ├── Quick Reference
    ├── When to Use / When NOT to Use
    ├── Implementation Guide with Decision Tree
    ├── Examples (with file:line citations)
    ├── Quality Checklist
    ├── Common Mistakes
    ├── Validation
    └── Related Skills
```

### Skill Router (Meta-Skill)

The `routing-to-skills` meta-skill enables dynamic skill discovery and selection:

```json
// Input
{
  "task": "Add validation to UserService",
  "files": ["src/services/UserService.py"],
  "context": "Prevent duplicate emails"
}

// Output
{
  "recommendations": [
    {
      "skill": "creating-data-processors",
      "confidence": 0.85,
      "reasons": ["File path matches data layer", "Task mentions validation"]
    }
  ]
}
```

**Scoring Algorithm:**
- File path match: +50 points
- Task type match: +30 points
- Keyword match: +10 points per keyword (max 30)
- Recent success: +5 points

**Integration with Agents:**
Agents invoke the skill router at workflow start to discover relevant skills, enabling cross-project portability without hardcoded skill references.

## Key Patterns

All skills are grounded in actual codebase patterns:

| Pattern | Files | Frequency |
|---------|-------|-----------|
| Abstract Base + Registry | `src/models/base.py`, `src/data/sources/base.py` | High |
| Dataclass DTOs | All modules | Very High |
| DEFAULT_CONFIG + Merge | `src/models/technical/*.py` | High |
| Indicator Calculator | `src/features/technical/*.py` | High |
| FastAPI Router | `src/api/routes/*.py` | Standard |
| Risk Manager | `src/trading/risk.py` | Core |

## Maintenance

### Continuous Improvement System

The framework includes a self-healing improvement system:

| Component | Location | Purpose |
|-----------|----------|---------|
| Error Template | `.claude/improvement/error-template.md` | Structured error capture |
| Error Reports | `.claude/improvement/errors/*.md` | Individual error records |
| Improvement Skill | `.claude/skills/continuous-improvement/SKILL.md` | Error processing workflow |
| Maintenance Checklist | `.claude/improvement/maintenance-checklist.md` | Quarterly review guide |

### When to Report Framework Errors

**Report when:**
- Agent gives wrong guidance
- Skill references outdated pattern
- Skill-router selects wrong skill
- Agent misses important consideration
- Quality checks don't catch real issue

**Don't report:**
- User error (misunderstood the tool)
- Legitimate design trade-offs
- Known limitations (documented)

### How to Report

```bash
# 1. Copy template
cp .claude/improvement/error-template.md \
   .claude/improvement/errors/$(date +%Y-%m-%d)-[description].md

# 2. Fill out all sections
# 3. Continue work (don't block on report)
```

### Maintenance Schedule

| Frequency | Task | Reference |
|-----------|------|-----------|
| Real-time | Capture errors | error-template.md |
| Weekly | Triage and fix errors | continuous-improvement skill |
| Quarterly | Full framework review | maintenance-checklist.md |

### Target Metrics

| Metric | Target |
|--------|--------|
| Error recurrence rate | <5% |
| Resolution time (avg) | <7 days |
| Backlog size | <10 errors |
| Critical errors | 0 |

### Standard Maintenance Tasks

- Update skills when codebase patterns change
- Verify file:line citations are still accurate
- Test with multiple Claude models (Haiku, Sonnet, Opus)
- Keep SKILL.md files under 500 lines

## Created

- Date: 2026-01-07
- Based on: `.claude/discovery/codebase-patterns.md`
- Skills: 23 total (21 domain + 2 meta-skills)
- Last updated: 2026-01-12
  - Added: creating-fastapi-endpoints, creating-python-services, creating-pydantic-schemas
  - Added: creating-react-components, creating-api-clients
  - Added: creating-sqlalchemy-models
  - Added: writing-pytest-tests, writing-vitest-tests
  - Added: creating-cli-scripts
