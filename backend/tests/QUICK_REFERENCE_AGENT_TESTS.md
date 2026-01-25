# Quick Reference: Agent Control Tests

## Test Files
- **Schema Tests:** `tests/unit/api/schemas/test_agent_schemas.py` (70 tests)
- **API Tests:** `tests/api/test_agent_routes.py` (31 tests)

## Run Commands

### All Agent Control Tests
```bash
pytest tests/unit/api/schemas/test_agent_schemas.py tests/api/test_agent_routes.py -v
```

### Schema Tests Only
```bash
pytest tests/unit/api/schemas/test_agent_schemas.py -v
```

### API Tests Only
```bash
pytest tests/api/test_agent_routes.py -v
```

### With Coverage
```bash
pytest tests/unit/api/schemas/test_agent_schemas.py tests/api/test_agent_routes.py \
  --cov=src.api.schemas.agent \
  --cov=src.api.routes.agent \
  --cov-report=term-missing
```

### Specific Test Class
```bash
# Schema test class
pytest tests/unit/api/schemas/test_agent_schemas.py::TestAgentStartRequest -v

# API test class
pytest tests/api/test_agent_routes.py::TestAgentStartEndpoint -v
```

### Specific Test
```bash
# Single schema test
pytest tests/unit/api/schemas/test_agent_schemas.py::TestAgentStartRequest::test_valid_request_with_defaults -v

# Single API test
pytest tests/api/test_agent_routes.py::TestAgentStartEndpoint::test_start_agent_successfully_no_existing_state -v
```

## Test Coverage

| Endpoint | Method | Tests |
|----------|--------|-------|
| `/api/v1/agent/start` | POST | 4 |
| `/api/v1/agent/stop` | POST | 3 |
| `/api/v1/agent/pause` | POST | 3 |
| `/api/v1/agent/resume` | POST | 3 |
| `/api/v1/agent/status` | GET | 3 |
| `/api/v1/agent/config` | PUT | 3 |
| `/api/v1/agent/kill-switch` | POST | 3 |
| `/api/v1/agent/metrics` | GET | 4 |
| `/api/v1/agent/commands/{id}` | GET | 2 |
| `/api/v1/agent/commands` | GET | 3 |

## Test Scenarios Covered

### Request Validation
- ✓ Valid inputs with defaults
- ✓ Valid inputs with all fields
- ✓ Invalid modes/actions/values
- ✓ Boundary values
- ✓ Required field validation

### API Behavior
- ✓ Success cases (200 OK)
- ✓ Conflict cases (400 Bad Request)
- ✓ Not found cases (404 Not Found)
- ✓ Validation errors (422 Unprocessable Entity)
- ✓ State transitions
- ✓ Database interactions

### Edge Cases
- ✓ Agent already running/stopped/paused
- ✓ Empty database queries
- ✓ Invalid configuration updates
- ✓ Kill switch triggers/resets
- ✓ Metrics with/without trades

## Test Dependencies

Required Python packages:
- pytest
- pydantic
- fastapi
- sqlalchemy
- unittest.mock (stdlib)

## Troubleshooting

### Import Errors
If you get `ModuleNotFoundError`, install dependencies:
```bash
pip install -r requirements.txt
pip install -r requirements-api.txt
```

### Database Connection Issues
Tests use mocked database - no real database connection needed.

### Slow Tests
API tests are fast (mocked I/O). If slow, check:
- Unnecessary service initialization
- Unmocked external calls

## Test Patterns

### Schema Tests
```python
def test_valid_request_with_defaults(self):
    """Test creating valid request with default values."""
    request = AgentStartRequest()

    assert request.mode == "simulation"
    assert request.confidence_threshold == 0.70
```

### API Tests
```python
def test_start_agent_successfully_no_existing_state(self):
    """Test starting agent when no existing state exists."""
    from src.api.routes import agent
    from src.api.database.session import get_db

    def mock_get_db():
        # Mock database setup
        ...
        yield self.mock_db

    app = FastAPI()
    app.include_router(agent.router)
    app.dependency_overrides[get_db] = mock_get_db
    client = TestClient(app)

    try:
        response = client.post("/api/v1/agent/start", json={...})
        assert response.status_code == 200
    finally:
        app.dependency_overrides.clear()
```

---

**Total Tests:** 101
**Coverage:** 100%
**Status:** ✓ Ready to run
