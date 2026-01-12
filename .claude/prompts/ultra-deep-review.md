# Ultra-Deep Project Review Prompt

Use this prompt to perform a comprehensive review, fix issues, and establish test automation for the ai-trader project.

---

## Prompt

```
Perform an ultra-deep review of this ai-trader project using a systematic multi-agent approach.

## Phase 1: Quality Guardian - Code Review & Issue Detection

Act as the **Quality Guardian** agent (see `.claude/agents/quality-guardian.md`).

### 1.1 Pattern Compliance Audit

Scan the entire codebase for pattern violations:

**Backend Layer** (`src/api/`):
- [ ] All routes follow `creating-fastapi-endpoints` patterns
- [ ] All services follow `creating-python-services` singleton pattern
- [ ] All schemas follow `creating-pydantic-schemas` validation patterns
- [ ] All database models follow `creating-sqlalchemy-models` patterns
- [ ] Error handling: HTTPException re-raised, others wrapped

**Frontend Layer** (`frontend/src/`):
- [ ] All components follow `creating-react-components` state patterns
- [ ] All API calls follow `creating-api-clients` patterns
- [ ] Loading/error/empty/data states handled in all components

**Trading Domain** (`src/models/`, `src/features/`, `src/trading/`):
- [ ] Time series validation per `validating-time-series-data`
- [ ] No future data leakage (shift(-n) in features)
- [ ] No shuffling before chronological split
- [ ] Risk controls in place per `implementing-risk-management`

### 1.2 Security Scan

Check for vulnerabilities:
- [ ] No hardcoded credentials or API keys
- [ ] All secrets from environment variables
- [ ] No SQL injection (parameterized queries only)
- [ ] Input validation on all endpoints
- [ ] CORS configured appropriately for production
- [ ] Rate limiting on prediction endpoints

### 1.3 Code Quality Issues

Identify and document:
- Missing type hints
- Missing docstrings on public methods
- Functions exceeding 50 lines
- Duplicated code blocks
- Unused imports
- Dead code paths
- Magic numbers without constants

### 1.4 Architecture Issues

Check for:
- Circular imports
- Tight coupling between layers
- Missing dependency injection
- Hardcoded configuration
- Missing error boundaries

**Output**: Create `.claude/reviews/quality-review-YYYY-MM-DD.md` with all findings categorized by severity (Critical/High/Medium/Low).

---

## Phase 2: Code Engineer - Fix All Issues

Act as the **Code Engineer** agent (see `.claude/agents/code-engineer.md`).

### 2.1 Critical Fixes (Security & Data Integrity)

Fix immediately:
1. Any security vulnerabilities found
2. Any data leakage issues in time series
3. Any risk control bypasses
4. Any hardcoded credentials

### 2.2 High Priority Fixes (Functionality)

Fix:
1. Pattern violations in routes/services/schemas
2. Missing error handling
3. Missing service availability checks
4. Incorrect async/await usage

### 2.3 Medium Priority Fixes (Quality)

Fix:
1. Add missing type hints to all functions
2. Add docstrings to public methods (Google style)
3. Extract magic numbers to constants
4. Remove unused imports and dead code

### 2.4 Refactoring

Apply these refactorings where needed:
1. Split functions >50 lines
2. Extract duplicated code to shared utilities
3. Add proper dependency injection where missing
4. Consolidate repeated patterns

**Constraint**: Follow skills guidance for each file type. Verify builds pass after each change.

---

## Phase 3: Test Automator - Comprehensive Test Suite

Act as the **Test Automator** agent (see `.claude/agents/test-automator.md`).

### 3.1 Test Coverage Analysis

First, analyze current coverage:
```bash
# Backend
pytest tests/ --cov=src --cov-report=term-missing --cov-report=html

# Frontend
cd frontend && npm run test:coverage
```

Identify untested:
- Routes without integration tests
- Services without unit tests
- Components without render tests
- Edge cases not covered

### 3.2 Backend Test Suite (`tests/`)

Create comprehensive tests following `writing-pytest-tests` skill:

**API Integration Tests** (`tests/api/`):
```python
# For each endpoint:
# - Happy path with valid data
# - Validation errors (422)
# - Service unavailable (503)
# - Internal errors (500)
# - Edge cases (empty data, max limits)
```

Required test files:
- [ ] `test_predictions.py` - All prediction endpoints
- [ ] `test_trading.py` - All trading endpoints
- [ ] `test_market.py` - All market data endpoints
- [ ] `test_pipeline.py` - Pipeline management endpoints
- [ ] `test_health.py` - Health checks

**Service Unit Tests** (`tests/services/`):
- [ ] `test_model_service.py` - Model loading, prediction, caching
- [ ] `test_trading_service.py` - Order execution, position management
- [ ] `test_data_service.py` - Data fetching, caching, validation
- [ ] `test_pipeline_service.py` - Pipeline scheduling, refresh

**Model Tests** (`tests/models/`):
- [ ] `test_mtf_ensemble.py` - Ensemble prediction, confidence
- [ ] `test_improved_model.py` - Individual timeframe models

### 3.3 Frontend Test Suite (`frontend/src/`)

Create comprehensive tests following `writing-vitest-tests` skill:

**Component Tests**:
- [ ] `Dashboard.test.jsx` - Layout, data flow, polling
- [ ] `PredictionCard.test.jsx` - All states, direction colors
- [ ] `AccountStatus.test.jsx` - Balance formatting, P&L display
- [ ] `PerformanceStats.test.jsx` - Metrics calculation display
- [ ] `TradeHistory.test.jsx` - Table rendering, pagination
- [ ] `PriceChart.test.jsx` - Chart rendering, data updates

**API Client Tests**:
- [ ] `client.test.js` - All endpoints, error handling, retries

### 3.4 Test Data Generation

Follow `generating-test-data` skill to create:

**Fixtures** (`tests/fixtures/`):
- `sample_ohlcv.csv` - 1000 rows of valid OHLCV data
- `sample_predictions.json` - Various prediction scenarios
- `sample_trades.json` - Trade history samples

**Factories** (`tests/factories/`):
- `prediction_factory.py` - Generate prediction objects
- `trade_factory.py` - Generate trade objects
- `ohlcv_factory.py` - Generate OHLCV DataFrames

### 3.5 Test Scenarios

Follow `planning-test-scenarios` skill:

**Critical Paths**:
1. Model loads → Prediction generated → Trade executed → Balance updated
2. Market data fetched → Cached → Served to frontend
3. Pipeline refreshes → Data updated → New prediction available

**Edge Cases**:
1. Model not loaded - graceful 503 response
2. Insufficient data - clear error message
3. Trading disabled - orders rejected
4. Stale cache - automatic refresh
5. Network timeout - retry with backoff

**Failure Scenarios**:
1. Database connection lost
2. External API unavailable (yfinance)
3. Model prediction fails
4. Concurrent trade requests

### 3.6 Coverage Targets

Achieve these coverage levels:
- Backend overall: >80%
- API routes: >90%
- Services: >85%
- Models: >75%
- Frontend components: >80%

---

## Phase 4: Validation & Documentation

### 4.1 Run Full Test Suite

```bash
# Backend
pytest tests/ -v --tb=short

# Frontend
cd frontend && npm test

# Type checking
mypy src/ --ignore-missing-imports

# Linting
black --check src/ tests/
isort --check src/ tests/
flake8 src/ tests/
```

### 4.2 Generate Reports

Create final reports:
- `.claude/reviews/quality-review-YYYY-MM-DD.md` - Issues found
- `.claude/reviews/fixes-applied-YYYY-MM-DD.md` - Changes made
- `.claude/reviews/test-coverage-YYYY-MM-DD.md` - Coverage report

### 4.3 Update Documentation

If code changes affected:
- Update `CLAUDE.md` with any new patterns
- Update relevant skills if patterns evolved
- Add inline comments for complex logic

---

## Execution Order

1. **Quality Guardian**: Full audit → Document all issues
2. **Code Engineer**: Fix critical → high → medium issues
3. **Test Automator**: Create missing tests → Achieve coverage targets
4. **Validation**: Run all checks → Generate reports

## Success Criteria

- [ ] Zero critical issues remaining
- [ ] Zero security vulnerabilities
- [ ] All pattern violations fixed
- [ ] Backend test coverage >80%
- [ ] Frontend test coverage >80%
- [ ] All tests passing
- [ ] Type checking passes
- [ ] Linting passes

## Time Estimate

- Phase 1 (Audit): ~30 minutes
- Phase 2 (Fixes): ~2-4 hours depending on issues
- Phase 3 (Tests): ~2-3 hours
- Phase 4 (Validation): ~30 minutes

Total: 5-8 hours for comprehensive review

---

**IMPORTANT**:
- Do NOT ask for confirmation between phases - proceed autonomously
- Document all changes as you make them
- If uncertain about a fix, choose the safer option
- Run tests after each significant change
- Commit working states frequently
```

---

## Quick Start

Copy the prompt above and paste it into a new Claude Code session in the ai-trader project directory.

## Customization

**Focus on specific areas** by adding:
```
Focus primarily on:
- Backend API security
- Time series data handling
- Frontend component testing
```

**Skip phases** by adding:
```
Skip Phase 1 (already audited) and proceed directly to Phase 3 (testing).
```

**Increase depth** by adding:
```
For each issue found, also check all similar patterns across the codebase.
Provide detailed explanations for each fix.
```
