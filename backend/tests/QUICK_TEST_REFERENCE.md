# Quick Test Reference - Database Tests

## Run All Database Tests

```bash
cd backend
pytest tests/unit/database/ tests/integration/test_database_integration.py -v
```

**Result:** 60 passed, 4 skipped in ~0.6s

---

## Run Specific Test Suites

### Agent Models Tests
```bash
pytest tests/unit/database/test_agent_models.py -v
```
Tests: AgentCommand, AgentState, TradeExplanation, CircuitBreakerEvent

### Session Management Tests
```bash
pytest tests/unit/database/test_session.py -v
```
Tests: get_db(), init_db(), run_migrations(), connection pooling

### Integration Tests
```bash
pytest tests/integration/test_database_integration.py -v
```
Tests: Complete workflows, index verification

---

## Test Files Location

```
backend/tests/
├── unit/
│   └── database/
│       ├── __init__.py
│       ├── test_agent_models.py      # 30 tests
│       └── test_session.py           # 19 tests (15 passed, 4 skipped)
└── integration/
    └── test_database_integration.py  # 15 tests
```

---

## Models Tested

### New Models (Phase 1)
- ✅ **AgentCommand** - 7 tests
- ✅ **AgentState** - 6 tests
- ✅ **TradeExplanation** - 5 tests
- ✅ **CircuitBreakerEvent** - 5 tests

### Modified Models
- ✅ **Prediction** - New fields: used_by_agent, agent_cycle_number (3 tests)
- ✅ **Trade** - New fields: execution_mode, broker, mt5_ticket, explanation_id (4 tests)

---

## Key Test Scenarios

1. **Command Workflow** - Backend → Agent communication
2. **State Persistence** - Agent crash recovery
3. **Trade Explanation** - LLM integration
4. **Circuit Breaker** - Safety system audit trail
5. **Complete Agent Cycle** - End-to-end workflow

---

## Quick Verification

```bash
# Quick smoke test (should take ~0.6s)
pytest tests/unit/database/ tests/integration/test_database_integration.py -q

# Expected output:
# 60 passed, 4 skipped in 0.61s
```

---

## Troubleshooting

### Import Errors
- Tests use direct module loading to avoid full API initialization
- No pandas/yfinance dependencies required for database tests

### Skipped Tests
4 tests skipped due to SQLite limitations:
- Migration column additions (works with PostgreSQL)
- Environment configuration (tested in integration)

### Test Isolation
- Each test uses in-memory SQLite
- No shared state between tests
- Clean database for every test

---

For detailed test report, see: `DATABASE_TESTS_REPORT.md`
