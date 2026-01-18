---
name: testing
description: This skill should be used when the user asks to "write tests", "add unit tests", "test the API endpoint", "create test fixtures". Writes pytest test classes with fixtures, mocked services, and FastAPI TestClient for API endpoint testing.
version: 1.1.0
---

# Writing Pytest Tests

## Quick Reference

- Use `TestClient` from FastAPI for endpoint testing
- Mock services with `unittest.mock.Mock`
- Restore original services in `finally` block
- Use `@pytest.fixture(autouse=True)` for common setup
- Test both success and error paths

## When to Use

- Testing API endpoints
- Mocking external services
- Testing business logic
- Verifying error handling

## When NOT to Use

- Frontend component tests (use Vitest)
- Integration tests with real database (use separate fixtures)
- E2E tests (use different framework)

## Implementation Guide

```
Is this an API endpoint test?
├─ Yes → Use TestClient with router
│   └─ Mock services before creating client
│   └─ Restore in finally block
└─ No → Test functions/classes directly

Does test need mock data?
├─ Yes → Create mock in fixture
│   └─ Use realistic sample data
└─ No → Test with minimal input

Testing error conditions?
├─ Yes → Test status codes and error messages
│   └─ Mock service to return errors
└─ No → Test happy path responses
```

## Examples

**Example 1: Test Class with Setup Fixture**

```python
# From: tests/api/test_predictions.py:1-48
"""Tests for prediction endpoints using FastAPI TestClient."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock
from fastapi.testclient import TestClient
from fastapi import FastAPI


class TestPredictionEndpoints:
    """Test prediction endpoints."""

    @pytest.fixture(autouse=True)
    def setup_mocks(self):
        """Set up service mocks before each test."""
        self.mock_model_service = Mock()
        self.mock_model_service.is_loaded = True
        self.mock_model_service.get_model_info.return_value = {
            "loaded": True,
            "model_dir": "models/mtf_ensemble",
            "weights": {"1H": 0.6, "4H": 0.3, "D": 0.1},
            "agreement_bonus": 0.05,
            "sentiment_enabled": True,
            "sentiment_by_timeframe": {"1H": False, "4H": False, "D": True},
            "models": {
                "1H": {"trained": True, "val_accuracy": 0.67},
                "4H": {"trained": True, "val_accuracy": 0.65},
                "D": {"trained": True, "val_accuracy": 0.62},
            },
            "initialized_at": "2025-01-12T10:00:00Z",
        }

        self.mock_data_service = Mock()
        self.mock_data_service._initialized = True
        self.mock_data_service.get_current_price.return_value = 1.08543
        self.mock_data_service.get_latest_vix.return_value = 15.5

        # Create mock DataFrame with enough data
        self.mock_df = pd.DataFrame({
            "open": np.random.rand(200) + 1.08,
            "high": np.random.rand(200) + 1.085,
            "low": np.random.rand(200) + 1.075,
            "close": np.random.rand(200) + 1.08,
            "volume": np.random.randint(1000, 10000, 200),
        })
        self.mock_data_service.get_data_for_prediction.return_value = self.mock_df
```

**Explanation**: Class groups related tests. `autouse=True` runs fixture before each test. Mock services with realistic return values. Create sample DataFrame matching expected structure.

**Example 2: Basic Endpoint Test**

```python
# From: tests/api/test_predictions.py:49-68
def test_model_status_endpoint(self):
    """Test model status endpoint returns correct information."""
    from src.api.routes import predictions

    original_model = predictions.model_service
    predictions.model_service = self.mock_model_service

    try:
        app = FastAPI()
        app.include_router(predictions.router)
        client = TestClient(app)

        response = client.get("/models/status")

        assert response.status_code == 200
        data = response.json()
        assert data["loaded"] is True
        assert data["weights"] == {"1H": 0.6, "4H": 0.3, "D": 0.1}
    finally:
        predictions.model_service = original_model
```

**Explanation**: Import route module. Save original service. Replace with mock. Create TestClient. Make request. Assert response. Restore original in finally.

**Example 3: Error Condition Test**

```python
# From: tests/api/test_predictions.py:70-88
def test_latest_prediction_model_not_loaded(self):
    """Test latest prediction returns 503 when model not loaded."""
    from src.api.routes import predictions

    original_model = predictions.model_service
    self.mock_model_service.is_loaded = False
    predictions.model_service = self.mock_model_service

    try:
        app = FastAPI()
        app.include_router(predictions.router)
        client = TestClient(app)

        response = client.get("/predictions/latest")

        assert response.status_code == 503
        assert "Model not loaded" in response.json()["detail"]
    finally:
        predictions.model_service = original_model
```

**Explanation**: Set mock to error state. Verify 503 status code. Check error message in response detail.

**Example 4: Test with Multiple Mocked Services**

```python
# From: tests/api/test_predictions.py:90-114
def test_latest_prediction_insufficient_data(self):
    """Test latest prediction returns 503 with insufficient data."""
    from src.api.routes import predictions

    original_model = predictions.model_service
    original_data = predictions.data_service

    # Return only 10 rows - insufficient for prediction
    self.mock_data_service.get_data_for_prediction.return_value = self.mock_df.head(10)

    predictions.model_service = self.mock_model_service
    predictions.data_service = self.mock_data_service

    try:
        app = FastAPI()
        app.include_router(predictions.router)
        client = TestClient(app)

        response = client.get("/predictions/latest")

        assert response.status_code == 503
        assert "Insufficient market data" in response.json()["detail"]
    finally:
        predictions.model_service = original_model
        predictions.data_service = original_data
```

**Explanation**: Mock multiple services. Modify mock return to trigger edge case. Restore all services in finally.

**Example 5: POST Endpoint Test**

```python
# From: tests/api/test_predictions.py:116-133
def test_generate_prediction_model_not_loaded(self):
    """Test manual prediction generation with model not loaded."""
    from src.api.routes import predictions

    original_model = predictions.model_service
    self.mock_model_service.is_loaded = False
    predictions.model_service = self.mock_model_service

    try:
        app = FastAPI()
        app.include_router(predictions.router)
        client = TestClient(app)

        response = client.post("/predictions/generate")

        assert response.status_code == 503
    finally:
        predictions.model_service = original_model
```

**Explanation**: Use `client.post()` for POST endpoints. Test precondition failures return appropriate errors.

## Quality Checklist

- [ ] Test class with descriptive name
- [ ] `@pytest.fixture(autouse=True)` for setup
- [ ] Pattern matches `tests/api/test_predictions.py:11-48`
- [ ] Services restored in `finally` block
- [ ] Both success and error paths tested
- [ ] Mock data matches expected structure
- [ ] Descriptive docstrings on test methods

## Common Mistakes

- **Not restoring mocks**: Affects other tests
  - Wrong: Replace service without finally block
  - Correct: Save original, restore in finally

- **Missing error path tests**: Only test happy path
  - Wrong: Only test 200 responses
  - Correct: Test 400, 404, 500, 503 cases

- **Insufficient mock data**: Tests fail on data validation
  - Wrong: `mock_df = pd.DataFrame({"x": [1]})`
  - Correct: Create DataFrame with realistic structure and size

## Validation

- [ ] Pattern confirmed in `tests/api/test_predictions.py:11-68`
- [ ] Tests pass with `pytest tests/api/test_predictions.py -v`
- [ ] Coverage includes error handling paths

## Related Skills

- `creating-fastapi-endpoints` - Endpoints being tested
- `creating-python-services` - Services being mocked
- `writing-vitest-tests` - Frontend component tests

---

*Version 1.0.0 | Last verified: 2026-01-16 | Source: tests/api/*
