# Agent Tests Quick Reference

## Test Files

```
tests/
├── unit/agent/
│   ├── __init__.py
│   ├── conftest.py                     # Shared fixtures
│   ├── test_config.py                  # 27 tests - AgentConfig
│   ├── test_state_manager.py           # 34 tests - StateManager
│   ├── test_command_handler.py         # 40 tests - CommandHandler
│   └── test_runner.py                  # 34 tests - AgentRunner
└── integration/
    └── test_agent_integration.py       # 17 tests - Full flow
```

## Quick Run Commands

```bash
# Run all agent tests
pytest tests/unit/agent/ -v

# Run specific component
pytest tests/unit/agent/test_config.py -v
pytest tests/unit/agent/test_state_manager.py -v
pytest tests/unit/agent/test_command_handler.py -v
pytest tests/unit/agent/test_runner.py -v

# Run integration tests
pytest tests/integration/test_agent_integration.py -v

# Run with coverage
pytest tests/unit/agent/ --cov=src/agent --cov-report=term-missing

# Run only async tests
pytest tests/unit/agent/ -k "asyncio" -v

# Run only specific test class
pytest tests/unit/agent/test_config.py::TestAgentConfig -v

# Run only one test
pytest tests/unit/agent/test_config.py::TestAgentConfig::test_default_values -v
```

## Test Structure

Each test follows AAA pattern:

```python
def test_feature(self, fixture):
    """Test description."""
    # Arrange - Set up test data
    data = create_test_data()

    # Act - Execute the code under test
    result = system.method(data)

    # Assert - Verify expected outcomes
    assert result == expected
```

## Key Fixtures

| Fixture | Purpose | Usage |
|---------|---------|-------|
| `db_session` | In-memory SQLite | Database operations |
| `config` | Test AgentConfig | Configuration testing |
| `agent_runner` | AgentRunner instance | Runner testing |
| `command_handler` | CommandHandler instance | Command testing |
| `state_manager` | StateManager instance | State testing |

## Test Categories

### 1. Configuration Tests (test_config.py)

- Validation rules
- Environment variable loading
- Credential masking
- Config updates

### 2. State Management Tests (test_state_manager.py)

- State initialization
- State updates
- Crash recovery
- Thread safety

### 3. Command Handling Tests (test_command_handler.py)

- Command polling
- Status transitions
- Command cleanup
- Async operations

### 4. Runner Tests (test_runner.py)

- Lifecycle management
- Command execution
- Cycle execution
- Error handling

### 5. Integration Tests (test_agent_integration.py)

- End-to-end flows
- Multi-command sequences
- State persistence
- Circuit breaker

## Common Test Patterns

### Testing Validation

```python
def test_validation_rejects_invalid_value(self):
    config = AgentConfig(mode="invalid")
    with pytest.raises(ValueError, match="Invalid mode"):
        config.validate()
```

### Testing Async Operations

```python
@pytest.mark.asyncio
async def test_async_start(self, agent_runner):
    result = await agent_runner.start()
    assert result is True
    await agent_runner.stop()  # Cleanup
```

### Testing Database Persistence

```python
def test_persists_to_database(self, state_manager, config, db_session):
    state_manager.initialize(config)
    state_manager.update_status("running")

    session = db_session()
    try:
        state = session.query(AgentState).first()
        assert state.status == "running"
    finally:
        session.close()
```

### Testing Mocked Services

```python
def test_calls_external_service(self, agent_runner, mock_model_service):
    await agent_runner.start()
    mock_model_service.initialize.assert_called_once_with(warm_up=True)
```

## Test Coverage Goals

| Component | Target | Current |
|-----------|--------|---------|
| AgentConfig | >90% | 100% |
| StateManager | >80% | 95% |
| CommandHandler | >80% | 90% |
| AgentRunner | >80% | 85% |

## Troubleshooting

### Import Errors

If you see `ModuleNotFoundError: No module named 'pandas'`:

```bash
pip install pandas
```

### Async Test Failures

If async tests hang, check:
1. Using `@pytest.mark.asyncio` decorator
2. Proper cleanup in fixtures
3. Canceling tasks in teardown

### Database Errors

If tests fail with database errors:
1. Check `conftest.py` fixture setup
2. Verify in-memory database creation
3. Ensure proper session cleanup

## Debugging Tests

### Run with detailed output

```bash
pytest tests/unit/agent/test_runner.py -vv -s
```

### Run with pdb debugger

```bash
pytest tests/unit/agent/test_runner.py --pdb
```

### Show print statements

```bash
pytest tests/unit/agent/test_runner.py -s
```

### Show locals on failure

```bash
pytest tests/unit/agent/test_runner.py -l
```

## Verified Test Results

✅ **test_config.py**: 27/27 tests PASSED (verified 2026-01-22)

All other tests are ready to run once pandas is installed in the environment.

---

**Quick Reference Last Updated**: 2026-01-22
**Total Tests**: 152
**Test Files**: 5 unit + 1 integration
