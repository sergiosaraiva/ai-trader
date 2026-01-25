---
name: testing
description: Writes pytest tests with fixtures, mocked services, and FastAPI TestClient for API endpoint testing.
version: 1.3.0
---

# Writing Pytest Tests

## Quick Reference

- Use `TestClient` from FastAPI for endpoint testing
- Mock services with `unittest.mock.Mock`
- **Always restore original services in `finally` block**
- Use `@pytest.fixture(autouse=True)` for common setup
- Test both success and error paths

## Decision Tree

```
API endpoint test? → Use TestClient with router, mock services, restore in finally
Service test? → Create instance directly, use tmp_path for file isolation
Error test? → Mock service to error state, verify status code + message
```

## Pattern: Test Class with Setup

```python
# Reference: backend/tests/api/test_predictions.py
class TestPredictionEndpoints:
    @pytest.fixture(autouse=True)
    def setup_mocks(self):
        self.mock_model_service = Mock()
        self.mock_model_service.is_loaded = True
        self.mock_model_service.get_model_info.return_value = {
            "loaded": True, "weights": {"1H": 0.6, "4H": 0.3, "D": 0.1}
        }
        self.mock_df = pd.DataFrame({
            "open": np.random.rand(200) + 1.08,
            "high": np.random.rand(200) + 1.085,
            "low": np.random.rand(200) + 1.075,
            "close": np.random.rand(200) + 1.08,
            "volume": np.random.randint(1000, 10000, 200),
        })
```

## Pattern: Basic Endpoint Test

```python
def test_model_status_endpoint(self):
    from backend.src.api.routes import predictions
    original_model = predictions.model_service
    predictions.model_service = self.mock_model_service
    try:
        app = FastAPI()
        app.include_router(predictions.router)
        client = TestClient(app)
        response = client.get("/models/status")
        assert response.status_code == 200
        assert response.json()["loaded"] is True
    finally:
        predictions.model_service = original_model  # CRITICAL: Always restore
```

## Pattern: Error Test

```python
def test_model_not_loaded(self):
    from backend.src.api.routes import predictions
    original = predictions.model_service
    self.mock_model_service.is_loaded = False
    predictions.model_service = self.mock_model_service
    try:
        client = TestClient(FastAPI().include_router(predictions.router))
        response = client.get("/predictions/latest")
        assert response.status_code == 503
        assert "Model not loaded" in response.json()["detail"]
    finally:
        predictions.model_service = original
```

## Pattern: Service Unit Test

```python
class TestPerformanceService:
    def test_default_metrics_when_files_missing(self, tmp_path):
        service = PerformanceService()
        service.DEFAULT_MODEL_DIR = tmp_path / "nonexistent"
        service.initialize()
        assert service.is_loaded
        assert service._metrics["win_rate"] == DEFAULT_BASELINE_METRICS["WIN_RATE"]
```

## Quality Checklist

- [ ] `@pytest.fixture(autouse=True)` for setup
- [ ] Services restored in `finally` block
- [ ] Both success (200) and error (400/500/503) paths tested
- [ ] Mock data has realistic structure (200+ rows for DataFrames)

## Common Mistakes

| Wrong | Correct |
|-------|---------|
| No `finally` block | Always restore mocks in `finally` |
| Only test 200 responses | Test 400, 404, 500, 503 cases |
| `mock_df = pd.DataFrame({"x": [1]})` | Create DataFrame with 200+ rows |
| Assert internal method called | Assert response data |

## Related Skills

- `backend` - Endpoints being tested
- `writing-vitest-tests` - Frontend tests
- `generating-test-data` - Creating fixtures

---
<!-- v1.3.0 | 2026-01-24 -->
