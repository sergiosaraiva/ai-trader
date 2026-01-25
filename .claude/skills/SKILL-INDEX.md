# Skill Index

**Version**: 1.2.0 | **Total Skills**: 27 | **Updated**: 2026-01-24

## Skills by Layer

### Meta (2)
| Skill | Description |
|-------|-------------|
| `routing-to-skills` | Routes tasks to skills with scoring algorithm |
| `improving-framework-continuously` | Processes error reports to evolve agents/skills |

### Backend (5 + 1 deprecated)
| Skill | Description |
|-------|-------------|
| `backend` | FastAPI REST endpoints with error handling |
| `creating-python-services` | Thread-safe singleton services |
| `creating-pydantic-schemas` | Request/response schemas with validation |
| `implementing-prediction-models` | BaseModel pattern with registry |
| `creating-data-processors` | Validate/Clean/Transform pipeline |

### Frontend (3)
| Skill | Description |
|-------|-------------|
| `frontend` | React components with loading/error/data states |
| `creating-api-clients` | Centralized API client with error handling |
| `creating-chart-components` | Recharts with useMemo and memoized tooltips |

### Database (1)
| Skill | Description |
|-------|-------------|
| `database` | SQLAlchemy ORM models with indexes |

### Feature Engineering (3)
| Skill | Description |
|-------|-------------|
| `creating-technical-indicators` | Indicator calculator class pattern |
| `configuring-indicator-yaml` | YAML configuration for indicators |
| `creating-ml-features` | ML features with data leakage prevention |

### Data Layer (1)
| Skill | Description |
|-------|-------------|
| `adding-data-sources` | BaseDataSource + Factory pattern |

### Trading Domain (3)
| Skill | Description |
|-------|-------------|
| `running-backtests` | Backtester usage and WFO validation |
| `analyzing-trading-performance` | Sharpe, Sortino, drawdown metrics |
| `implementing-risk-management` | RiskManager with circuit breakers |

### Testing (2)
| Skill | Description |
|-------|-------------|
| `testing` | pytest with TestClient and mocked services |
| `writing-vitest-tests` | Vitest tests for React components |

### Quality & Testing (4)
| Skill | Description |
|-------|-------------|
| `creating-dataclasses` | Python dataclass patterns for DTOs |
| `validating-time-series-data` | Time series validation, leakage prevention |
| `planning-test-scenarios` | Generate test plans from acceptance criteria |
| `generating-test-data` | Create test fixtures and synthetic data |

### Caching (1)
| Skill | Description |
|-------|-------------|
| `implementing-caching-strategies` | Hash-based and TTL caching patterns |

### Build (1)
| Skill | Description |
|-------|-------------|
| `build-deployment` | Python CLI scripts with argparse/logging |

---

## Quick Reference

| Task | Skill |
|------|-------|
| API endpoint | `backend` |
| Service class | `creating-python-services` |
| API schema | `creating-pydantic-schemas` |
| Internal DTO | `creating-dataclasses` |
| React component | `frontend` |
| Chart/graph | `creating-chart-components` |
| Database model | `database` |
| Technical indicator | `creating-technical-indicators` |
| ML feature (rolling) | `creating-ml-features` |
| Add caching | `implementing-caching-strategies` |
| New data source | `adding-data-sources` |
| Backtest | `running-backtests` |
| Performance analysis | `analyzing-trading-performance` |
| Position sizing | `implementing-risk-management` |
| Python tests | `testing` |
| Frontend tests | `writing-vitest-tests` |
| Test fixtures | `generating-test-data` |
| CLI script | `build-deployment` |

## Trigger Phrases

| Phrase | Skill |
|--------|-------|
| "add API endpoint", "create REST route" | `backend` |
| "create React component", "add PropTypes" | `frontend` |
| "create a chart", "add visualization" | `creating-chart-components` |
| "add database table", "create model" | `database` |
| "create ML features", "prevent data leakage" | `creating-ml-features` |
| "add caching", "cache invalidation" | `implementing-caching-strategies` |
| "write tests", "test the API" | `testing` |

## Skill Dependencies

```
routing-to-skills → References all skills
frontend → creating-chart-components
creating-ml-features → validating-time-series-data
implementing-caching-strategies → creating-python-services
planning-test-scenarios → generating-test-data
testing ↔ writing-vitest-tests
```

---
*Next update: 2026-02-24 (monthly consolidation)*
