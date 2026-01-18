# Skill Index

**Last Updated**: 2026-01-18
**Total Active Skills**: 24
**Archived Skills**: 1
**Deprecated Stubs**: 1
**Version**: 1.1.0 (Enhanced trigger phrases)

---

## Active Skills by Layer

### Meta-Skills (2)

| Skill | Description | Lines | Version |
|-------|-------------|-------|---------|
| [routing-to-skills](./routing-to-skills/SKILL.md) | Routes tasks to appropriate skills with scoring algorithm | 817 | 1.2.0 |
| [improving-framework-continuously](./improving-framework-continuously/SKILL.md) | Processes error reports to evolve agents and skills | 551 | 1.2.0 |

### Backend Layer (6)

| Skill | Description | Lines | Version |
|-------|-------------|-------|---------|
| [backend](./backend/SKILL.md) | FastAPI REST endpoints with error handling | 254 | 1.1.0 |
| [creating-python-services](./backend/creating-python-services.md) | Thread-safe singleton services | 262 | 1.1.0 |
| [creating-pydantic-schemas](./backend/creating-pydantic-schemas.md) | Request/response schemas with validation | 229 | 1.1.0 |
| [implementing-prediction-models](./backend/implementing-prediction-models.md) | BaseModel pattern with registry for ML | 296 | 1.0.0 |
| [creating-data-processors](./backend/creating-data-processors.md) | Validate/Clean/Transform pipeline pattern | 338 | 1.0.0 |
| ~~[creating-api-endpoints](./backend/creating-api-endpoints.md)~~ | *DEPRECATED - Use backend* | 37 | 1.0.0 |

### Frontend Layer (2)

| Skill | Description | Lines | Version |
|-------|-------------|-------|---------|
| [frontend](./frontend/SKILL.md) | React components with loading/error/data states | 251 | 1.1.0 |
| [creating-api-clients](./frontend/creating-api-clients.md) | Centralized API client with error handling | 220 | 1.1.0 |

### Database Layer (1)

| Skill | Description | Lines | Version |
|-------|-------------|-------|---------|
| [database](./database/SKILL.md) | SQLAlchemy ORM models with indexes | 291 | 1.1.0 |

### Feature Engineering (2)

| Skill | Description | Lines | Version |
|-------|-------------|-------|---------|
| [creating-technical-indicators](./feature-engineering/creating-technical-indicators.md) | Indicator calculator class pattern | 321 | 1.0.0 |
| [configuring-indicator-yaml](./feature-engineering/configuring-indicator-yaml.md) | YAML configuration for indicators | 338 | 1.0.0 |

### Data Layer (1)

| Skill | Description | Lines | Version |
|-------|-------------|-------|---------|
| [adding-data-sources](./data-layer/adding-data-sources.md) | BaseDataSource + Factory pattern | 383 | 1.0.0 |

### Trading Domain (3)

| Skill | Description | Lines | Version |
|-------|-------------|-------|---------|
| [running-backtests](./trading-domain/running-backtests.md) | Backtester usage and walk-forward validation | 409 | 1.0.0 |
| [analyzing-trading-performance](./trading-domain/analyzing-trading-performance.md) | Performance metrics (Sharpe, Sortino, drawdown) | 382 | 1.0.0 |
| [implementing-risk-management](./trading-domain/implementing-risk-management.md) | RiskManager with circuit breakers | 410 | 1.0.0 |

### Testing (2)

| Skill | Description | Lines | Version |
|-------|-------------|-------|---------|
| [testing](./testing/SKILL.md) | pytest tests with TestClient and mocked services | 260 | 1.1.0 |
| [writing-vitest-tests](./testing/writing-vitest-tests.md) | Vitest tests for React components | 223 | 1.1.0 |

### Quality & Testing (4)

| Skill | Description | Lines | Version |
|-------|-------------|-------|---------|
| [creating-dataclasses](./quality-testing/creating-dataclasses.md) | Python dataclass patterns for DTOs | 319 | 1.0.0 |
| [validating-time-series-data](./quality-testing/validating-time-series-data.md) | Time series validation and leakage prevention | 412 | 1.0.0 |
| [planning-test-scenarios](./quality-testing/planning-test-scenarios.md) | Generate test plans from acceptance criteria | 500 | 1.0.0 |
| [generating-test-data](./quality-testing/generating-test-data.md) | Create test fixtures and synthetic data | 650 | 1.0.0 |

### Build & Deployment (1)

| Skill | Description | Lines | Version |
|-------|-------------|-------|---------|
| [build-deployment](./build-deployment/SKILL.md) | Python CLI scripts with argparse and logging | 331 | 1.1.0 |

---

## Size Distribution

| Size Range | Count | Percentage |
|------------|-------|------------|
| <200 lines | 1 | 4% |
| 200-300 lines | 10 | 42% |
| 300-400 lines | 8 | 33% |
| 400-500 lines | 1 | 4% |
| >500 lines | 4 | 17% |

**Note**: Skills >500 lines are meta-skills or comprehensive testing skills - documented as acceptable exceptions.

---

## Archived Skills

| Skill | Archive Date | Merged Into |
|-------|--------------|-------------|
| [processing-ohlcv-data](./_archived/processing-ohlcv-data.md) | 2026-01-12 | creating-data-processors |

---

## Recently Updated

| Skill | Version | Last Update | Change |
|-------|---------|-------------|--------|
| backend | 1.1.0 | 2026-01-18 | Enhanced trigger phrases |
| frontend | 1.1.0 | 2026-01-18 | Enhanced trigger phrases |
| testing | 1.1.0 | 2026-01-18 | Enhanced trigger phrases |
| database | 1.1.0 | 2026-01-18 | Enhanced trigger phrases |
| build-deployment | 1.1.0 | 2026-01-18 | Enhanced trigger phrases |
| creating-python-services | 1.1.0 | 2026-01-18 | Enhanced trigger phrases |
| creating-pydantic-schemas | 1.1.0 | 2026-01-18 | Enhanced trigger phrases |
| creating-api-clients | 1.1.0 | 2026-01-18 | Enhanced trigger phrases |
| writing-vitest-tests | 1.1.0 | 2026-01-18 | Enhanced trigger phrases |
| routing-to-skills | 1.2.0 | 2026-01-18 | Added verification & grounding (anti-hallucination) |
| improving-framework-continuously | 1.2.0 | 2026-01-18 | Synced with v1.2.0 framework |

---

## Skill Dependencies

```
routing-to-skills (meta)
└── References all skills for routing

improving-framework-continuously (meta)
└── References routing-to-skills

creating-pydantic-schemas ↔ creating-dataclasses
└── Complementary patterns (validation vs DTOs)

planning-test-scenarios → generating-test-data
└── Test plans identify data requirements

writing-pytest-tests ↔ writing-vitest-tests
└── Python/JavaScript testing counterparts
```

---

## Quick Reference: When to Use

| Task | Primary Skill |
|------|---------------|
| Add API endpoint | backend |
| Create service class | creating-python-services |
| Define API schema | creating-pydantic-schemas |
| Internal DTO | creating-dataclasses |
| React component | frontend |
| Database model | database |
| Technical indicator | creating-technical-indicators |
| New data source | adding-data-sources |
| Backtest strategy | running-backtests |
| Analyze performance | analyzing-trading-performance |
| Position sizing | implementing-risk-management |
| Python tests | testing |
| Frontend tests | writing-vitest-tests |
| Test data | generating-test-data |
| Time series validation | validating-time-series-data |
| CLI script | build-deployment |

---

*Generated: 2026-01-18*
*Next update: 2026-02-18 (monthly consolidation)*
