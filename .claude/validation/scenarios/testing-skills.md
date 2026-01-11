# Testing Skills Validation Scenarios

## planning-test-scenarios

### Scenario 1: Post-Design Test Planning
**Task**: "Create test plan for the new squeeze detection feature"
**Context**: Technical design complete, acceptance criteria available
**Files**: `src/features/technical/volatility.py`

**Expected**:
- Skill selected: `planning-test-scenarios`
- Generates: Unit test scenarios for squeeze_detection()
- Includes: Edge cases (NaN, zero ATR, insufficient data)
- Maps: Acceptance criteria to test scenarios
- References: `generating-test-data` for fixtures

**Validation**:
- [ ] Correct skill selected
- [ ] Unit tests for each method
- [ ] Edge cases identified
- [ ] Acceptance criteria mapped
- [ ] Test data requirements listed

---

### Scenario 2: API Test Planning
**Task**: "Generate test scenarios for the prediction API endpoint"
**Files**: `src/api/routes/predictions.py`

**Expected**:
- Skill selected: `planning-test-scenarios`
- Includes: Request validation tests
- Includes: Error response tests (422, 500)
- Includes: Rate limiting tests
- Format: Structured test scenario YAML

**Validation**:
- [ ] Valid request test
- [ ] Invalid request tests
- [ ] Error code coverage
- [ ] Integration test for model connection

---

### Scenario 3: Backtest Test Planning
**Task**: "Plan tests for walk-forward validation"
**Files**: `src/simulation/backtester.py`

**Expected**:
- Skill selected: `planning-test-scenarios`
- Includes: Chronological split tests
- Includes: No data leakage verification
- Includes: Edge cases (insufficient data)
- References: `validating-time-series-data`

**Validation**:
- [ ] Time series considerations
- [ ] Data leakage checks
- [ ] Fold generation tests

---

## generating-test-data

### Scenario 4: OHLCV Test Fixtures
**Task**: "Create test data for technical indicator testing"
**Requirements**: 100 candles, known patterns, edge cases

**Expected**:
- Skill selected: `generating-test-data`
- Recommends: Builder pattern or fixtures
- Includes: Deterministic seeding
- Includes: Edge case data (NaN, zeros)

**Validation**:
- [ ] Correct skill selected
- [ ] Builder pattern shown
- [ ] Seed for reproducibility
- [ ] Edge case factory

---

### Scenario 5: Mock Data Source
**Task**: "Create mock data source for integration testing"
**Files**: `tests/mocks/mock_data_source.py`

**Expected**:
- Skill selected: `generating-test-data`
- Implements: BaseDataSource interface
- Includes: Configurable behavior (fail mode, latency)
- Includes: Pre-loaded data support

**Validation**:
- [ ] Implements base interface
- [ ] set_should_fail() method
- [ ] set_data() method

---

## validating-time-series-data

### Scenario 6: Leakage Detection
**Task**: "Review this code for time series data leakage"
**Code**: Contains `df['feature'] = df['close'].shift(-1)`

**Expected**:
- Skill selected: `validating-time-series-data`
- Identifies: Future data leakage (negative shift)
- Explains: Why this causes overfitting
- Suggests: Use positive shift or mark as target

**Validation**:
- [ ] Correct skill selected
- [ ] Leakage identified
- [ ] Clear explanation
- [ ] Correct fix suggested

---

### Scenario 7: Train/Test Split Review
**Task**: "Verify this train/test split is correct for time series"
**Code**: Contains random shuffle before split

**Expected**:
- Skill selected: `validating-time-series-data`
- Identifies: Shuffle as leakage source
- Recommends: Chronological split only
- References: Walk-forward validation

**Validation**:
- [ ] Shuffle identified as problem
- [ ] Chronological split recommended
- [ ] Time-based indexing suggested

---

## Results Template

```markdown
## Validation Run: YYYY-MM-DD

### planning-test-scenarios
| Scenario | Pass/Fail | Notes |
|----------|-----------|-------|
| 1: Feature Test Plan | | |
| 2: API Test Plan | | |
| 3: Backtest Test Plan | | |

### generating-test-data
| Scenario | Pass/Fail | Notes |
|----------|-----------|-------|
| 4: OHLCV Fixtures | | |
| 5: Mock Data Source | | |

### validating-time-series-data
| Scenario | Pass/Fail | Notes |
|----------|-----------|-------|
| 6: Leakage Detection | | |
| 7: Split Review | | |

**Total**: X/7 passed
**Issues Found**: [List any issues]
```
