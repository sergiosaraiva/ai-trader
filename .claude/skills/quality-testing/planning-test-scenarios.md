---
name: planning-test-scenarios
description: Generates comprehensive test plans from acceptance criteria and technical designs. Creates unit, integration, and edge case scenarios with data requirements. Use after Solution Architect completes technical design, or when Test Automator needs test coverage planning.
version: 1.0.0
---

# Planning Test Scenarios

Generate structured test plans that ensure complete coverage of acceptance criteria and edge cases.

## Quick Reference

```
1. Map acceptance criteria → test scenarios
2. Identify unit tests per component
3. Plan integration tests for workflows
4. Define edge cases and error paths
5. Specify test data requirements
6. Output structured test plan
```

## When to Use

- Solution Architect completed technical design
- Test Automator starting test implementation
- Code review identifies missing test coverage
- New feature needs test planning
- Regression test expansion needed

## When NOT to Use

- Writing actual test code (use test framework directly)
- Simple bug fixes (single unit test sufficient)
- Exploratory testing (manual process)

---

## Test Planning Workflow

### Phase 1: Acceptance Criteria Analysis

```
For each acceptance criterion:
├─ Extract testable conditions
├─ Identify success/failure states
├─ Define boundary values
└─ Note required preconditions
```

### Phase 2: Component Test Mapping

```
For each component in technical design:
├─ Unit tests for public methods
├─ Edge cases for each parameter
├─ Error handling scenarios
└─ State transition tests (if stateful)
```

### Phase 3: Integration Test Planning

```
For each workflow/data flow:
├─ Happy path end-to-end
├─ Error propagation
├─ Timeout/retry scenarios
└─ Concurrent access (if applicable)
```

### Phase 4: Test Data Requirements

```
For each scenario:
├─ Input data needed
├─ Expected output format
├─ Mock dependencies
└─ Invoke generating-test-data if complex
```

---

## Test Scenario Structure

### Unit Test Template

```yaml
scenario:
  name: "[Method]_[Condition]_[Expected]"
  component: "[Class/Function name]"
  type: unit

given:
  - "[Precondition 1]"
  - "[Precondition 2]"

when:
  action: "[Method call or operation]"
  input:
    param1: "[Value]"
    param2: "[Value]"

then:
  - "[Expected outcome 1]"
  - "[Expected outcome 2]"

edge_cases:
  - "[Edge case 1]"
  - "[Edge case 2]"
```

### Integration Test Template

```yaml
scenario:
  name: "[Workflow]_[Condition]_[Expected]"
  components: ["Component1", "Component2"]
  type: integration

workflow:
  1: "[Step 1 description]"
  2: "[Step 2 description]"
  3: "[Step 3 description]"

assertions:
  - step: 1
    check: "[What to verify]"
  - step: 3
    check: "[Final state verification]"

error_scenarios:
  - trigger: "[What causes error]"
    expected: "[Expected error handling]"
```

---

## Examples

### Example 1: Model Prediction Test Plan

**Input**: Technical design for ShortTermModel prediction

**Test Plan**:

```yaml
test_plan:
  component: ShortTermModel
  acceptance_criteria:
    - "Model returns prediction within 100ms"
    - "Prediction includes direction and confidence"
    - "Handles missing data gracefully"

unit_tests:
  - name: "predict_ValidInput_ReturnsPrediction"
    given: ["Trained model", "Valid 168-candle sequence"]
    when: "predict(X) called"
    then:
      - "Returns Prediction dataclass"
      - "direction in ['bullish', 'bearish', 'neutral']"
      - "confidence between 0 and 1"
      - "price_predictions_multi has 4 horizons"

  - name: "predict_UntrainedModel_RaisesError"
    given: ["Model not trained (is_trained=False)"]
    when: "predict(X) called"
    then: ["Raises RuntimeError with message"]

  - name: "predict_WrongInputShape_RaisesError"
    given: ["Trained model", "Input with wrong sequence length"]
    when: "predict(X) called"
    then: ["Raises ValueError with shape info"]

  - name: "predict_NaNInInput_HandlesGracefully"
    given: ["Trained model", "Input containing NaN values"]
    when: "predict(X) called"
    then: ["Returns prediction with reduced confidence OR raises clear error"]

  - name: "predict_PerformanceWithin100ms"
    given: ["Trained model", "Standard input"]
    when: "predict(X) called 100 times"
    then: ["95th percentile latency < 100ms"]

integration_tests:
  - name: "EndToEnd_DataToPrediction"
    workflow:
      1: "Load OHLCV data from source"
      2: "Calculate technical indicators"
      3: "Create sequences"
      4: "Run prediction"
    assertions:
      - "Data flows without errors"
      - "Prediction matches expected format"

test_data_requirements:
  - "168 candles of EURUSD hourly data"
  - "Data with known NaN positions"
  - "Edge case: all prices identical"
  - "Edge case: extreme volatility"
```

### Example 2: API Endpoint Test Plan

**Input**: Technical design for /api/v1/predictions endpoint

**Test Plan**:

```yaml
test_plan:
  component: PredictionRouter
  endpoint: POST /api/v1/predictions
  acceptance_criteria:
    - "Returns valid prediction JSON"
    - "Validates request body"
    - "Returns appropriate error codes"
    - "Rate limited to 60/min"

unit_tests:
  - name: "post_ValidRequest_Returns200"
    given: ["Valid PredictionRequest body"]
    when: "POST /api/v1/predictions"
    then:
      - "Status code 200"
      - "Response matches PredictionResponse schema"
      - "Contains symbol, direction, confidence"

  - name: "post_MissingSymbol_Returns422"
    given: ["Request body without symbol field"]
    when: "POST /api/v1/predictions"
    then:
      - "Status code 422"
      - "Error mentions 'symbol' field"

  - name: "post_InvalidTimeframe_Returns422"
    given: ["Request with timeframe='invalid'"]
    when: "POST /api/v1/predictions"
    then:
      - "Status code 422"
      - "Error lists valid timeframes"

  - name: "post_UnknownSymbol_Returns404"
    given: ["Request with symbol='UNKNOWN'"]
    when: "POST /api/v1/predictions"
    then:
      - "Status code 404"
      - "Error message helpful"

  - name: "post_RateLimit_Returns429"
    given: ["61 requests in 1 minute"]
    when: "POST /api/v1/predictions"
    then:
      - "61st request returns 429"
      - "Retry-After header present"

integration_tests:
  - name: "API_ModelIntegration_ReturnsRealPrediction"
    workflow:
      1: "Send valid request"
      2: "API loads model"
      3: "Model generates prediction"
      4: "API formats response"
    assertions:
      - "Response contains real prediction values"
      - "Latency < 500ms total"

test_data_requirements:
  - "Valid symbols list from data source"
  - "Request body variations for validation testing"
```

### Example 3: Technical Indicator Test Plan

**Input**: Technical design for squeeze detection indicator

**Test Plan**:

```yaml
test_plan:
  component: VolatilityIndicators.squeeze_detection
  acceptance_criteria:
    - "Detects squeeze when BB inside KC"
    - "Calculates momentum correctly"
    - "Handles edge cases"

unit_tests:
  - name: "squeeze_BBInsideKC_ReturnsSqueezeTRue"
    given: ["OHLCV data where BB width < KC width"]
    when: "squeeze_detection(df) called"
    then:
      - "squeeze_on column contains True"
      - "squeeze_momentum calculated"

  - name: "squeeze_BBOutsideKC_ReturnsSqueezeFalse"
    given: ["OHLCV data where BB width > KC width"]
    when: "squeeze_detection(df) called"
    then: ["squeeze_on column contains False"]

  - name: "squeeze_InsufficientData_ReturnsNaN"
    given: ["OHLCV data with < 20 rows"]
    when: "squeeze_detection(df) called"
    then: ["First N rows are NaN (warmup period)"]

  - name: "squeeze_ZeroATR_HandlesGracefully"
    given: ["OHLCV data with zero price movement"]
    when: "squeeze_detection(df) called"
    then: ["No division by zero error", "squeeze_on is False or NaN"]

  - name: "squeeze_AddsToFeatureNames"
    given: ["VolatilityIndicators instance"]
    when: "squeeze_detection(df) called"
    then:
      - "get_feature_names() includes squeeze_on"
      - "get_feature_names() includes squeeze_momentum"
      - "get_feature_names() includes squeeze_histogram"

test_data_requirements:
  - "Historical data with known squeeze periods (consolidation)"
  - "Historical data with known breakouts (no squeeze)"
  - "Edge case: exactly at BB=KC threshold"
  - "Invoke generating-test-data for synthetic patterns"
```

### Example 4: Backtester Test Plan

**Input**: Technical design for walk-forward validation

**Test Plan**:

```yaml
test_plan:
  component: Backtester.walk_forward_validation
  acceptance_criteria:
    - "Splits data chronologically"
    - "No future data leakage"
    - "Aggregates results across folds"

unit_tests:
  - name: "walkForward_SplitsChronologically"
    given: ["100 days of data", "5 folds"]
    when: "walk_forward_validation() called"
    then:
      - "Each fold's test data comes after train data"
      - "No overlap between train and test"

  - name: "walkForward_NoFutureLeakage"
    given: ["Data with timestamps"]
    when: "walk_forward_validation() called"
    then:
      - "Train max_date < Test min_date for all folds"
      - "Model retrained fresh each fold"

  - name: "walkForward_AggregatesMetrics"
    given: ["5 fold results"]
    when: "Results aggregated"
    then:
      - "Mean Sharpe across folds"
      - "Std dev for confidence"
      - "Per-fold breakdown available"

  - name: "walkForward_InsufficientData_RaisesError"
    given: ["10 data points", "5 folds requested"]
    when: "walk_forward_validation() called"
    then: ["Raises ValueError with minimum requirement"]

integration_tests:
  - name: "WalkForward_FullPipeline"
    workflow:
      1: "Load historical data"
      2: "Configure backtester"
      3: "Run walk-forward with real model"
      4: "Generate performance report"
    assertions:
      - "All folds complete without error"
      - "Results match expected format"
      - "No data leakage detected"

test_data_requirements:
  - "1+ year of daily OHLCV data"
  - "Known profitable strategy for validation"
  - "Known losing strategy for comparison"
```

### Example 5: Risk Management Test Plan

**Input**: Technical design for position sizing with circuit breakers

**Test Plan**:

```yaml
test_plan:
  component: RiskManager
  acceptance_criteria:
    - "Enforces max position size"
    - "Triggers circuit breaker on drawdown"
    - "Respects daily loss limits"

unit_tests:
  - name: "calculatePositionSize_RespectsMaximum"
    given: ["Account balance $10,000", "Max position 2%"]
    when: "calculate_position_size(signal) called"
    then: ["Position size <= $200"]

  - name: "circuitBreaker_TriggersOnDrawdown"
    given: ["Daily loss > 5% threshold"]
    when: "check_circuit_breaker() called"
    then:
      - "Returns should_halt=True"
      - "Logs circuit breaker activation"

  - name: "circuitBreaker_ResetsNextDay"
    given: ["Circuit breaker triggered yesterday"]
    when: "New trading day starts"
    then: ["Circuit breaker reset", "Trading allowed"]

  - name: "validateTrade_RejectsTooLarge"
    given: ["Proposed trade > max position"]
    when: "validate_trade(trade) called"
    then: ["Returns False with reason"]

  - name: "validateTrade_RejectsInCircuitBreaker"
    given: ["Circuit breaker active"]
    when: "validate_trade(any_trade) called"
    then: ["Returns False", "Reason: circuit breaker"]

integration_tests:
  - name: "RiskManager_TradingEngineIntegration"
    workflow:
      1: "Generate trade signal"
      2: "RiskManager validates"
      3: "Position sized appropriately"
      4: "Trade executed or rejected"
    assertions:
      - "No trade exceeds limits"
      - "Circuit breaker respected"

test_data_requirements:
  - "Trade signals with various sizes"
  - "Scenario: gradual drawdown to trigger breaker"
  - "Scenario: rapid loss (flash crash)"
```

---

## Output Format

### Test Plan Document

```markdown
# Test Plan: [Feature Name]

## Overview
- Component: [Name]
- Acceptance Criteria: [List]
- Test Types: Unit, Integration, E2E

## Unit Tests
[List of scenarios]

## Integration Tests
[List of scenarios]

## Edge Cases
[List of edge cases]

## Test Data Requirements
[List of data needs]
- [ ] Invoke generating-test-data skill if needed

## Coverage Matrix
| Acceptance Criterion | Test Scenarios |
|---------------------|----------------|
| AC1 | Test1, Test2 |
| AC2 | Test3, Test4 |
```

---

## Quality Checklist

- [ ] Every acceptance criterion has at least one test
- [ ] Happy path covered for each component
- [ ] Error cases covered (invalid input, missing data)
- [ ] Edge cases identified (boundaries, empty, null)
- [ ] Performance tests included if latency matters
- [ ] Integration points tested
- [ ] Test data requirements specified
- [ ] No future data leakage in time series tests

## Common Mistakes

- **Testing implementation not behavior**: Tests break on refactor → Test public interface only
- **Missing edge cases**: Bugs in production → Systematically identify boundaries
- **Insufficient error testing**: Unhelpful errors → Test every error path
- **Data leakage in time series**: Overfitting → Always verify chronological splits
- **Flaky tests**: Random failures → Use deterministic test data

## Related Skills

- [generating-test-data](generating-test-data.md) - Create test fixtures
- [validating-time-series-data](validating-time-series-data.md) - Time series test considerations
- [creating-dataclasses](creating-dataclasses.md) - Test data structures
