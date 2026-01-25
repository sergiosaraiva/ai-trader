# Quick Reference: Running RFECV Tests

## Prerequisites

Install test dependencies:

```bash
cd backend
pip install pytest pytest-asyncio pytest-cov
```

Or if using requirements file:

```bash
pip install -r requirements.txt  # Should include xgboost, scikit-learn
pip install pytest pytest-asyncio pytest-cov  # Test framework
```

## Quick Commands

### Run All RFECV Tests (68 tests)

```bash
pytest tests/unit/models/test_rfecv_*.py tests/unit/models/test_feature_selection_manager.py tests/integration/test_rfecv_integration.py -v
```

### Run Specific Test Files

```bash
# Config tests (16 tests)
pytest tests/unit/models/test_rfecv_config.py -v

# Selector tests (21 tests)
pytest tests/unit/models/test_rfecv_selector.py -v

# Manager tests (20 tests)
pytest tests/unit/models/test_feature_selection_manager.py -v

# Integration tests (11 tests)
pytest tests/integration/test_rfecv_integration.py -v
```

### Run Critical Data Leakage Tests Only

```bash
pytest tests/ -v -k "timeseriessplit or chronological_order"
```

### Run with Coverage Report

```bash
pytest tests/unit/models/test_rfecv_*.py tests/unit/models/test_feature_selection_manager.py \
    --cov=src/models/feature_selection \
    --cov-report=term-missing \
    --cov-report=html
```

View coverage report:

```bash
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
```

## Expected Output

### Successful Run

```
tests/unit/models/test_rfecv_config.py::TestRFECVConfig::test_default_config PASSED
tests/unit/models/test_rfecv_config.py::TestRFECVConfig::test_custom_config PASSED
...
tests/integration/test_rfecv_integration.py::TestRFECVIntegration::test_model_with_rfecv_enabled PASSED

========================= 68 passed in 2.34s ==========================
```

### With Coverage

```
---------- coverage: platform linux, python 3.12.3 -----------
Name                                           Stmts   Miss  Cover   Missing
----------------------------------------------------------------------------
src/models/feature_selection/rfecv_config.py      12      0   100%
src/models/feature_selection/rfecv_selector.py   123      6    95%   82-84, 112
src/models/feature_selection/manager.py          156      8    95%   45-47, 201-203
----------------------------------------------------------------------------
TOTAL                                            291     14    95%
```

## Troubleshooting

### ModuleNotFoundError: No module named 'xgboost'

```bash
pip install xgboost scikit-learn
```

### ModuleNotFoundError: No module named 'pytest'

```bash
pip install pytest pytest-asyncio
```

### Tests Run Very Slowly (>30 seconds)

**Problem:** Mocks may not be applied correctly, causing real RFECV execution.

**Solution:** Verify you're running the correct test files. Tests use mocking to run fast.

**Check:** Inspect test output - if you see RFECV fitting messages, mocks aren't working.

### Import Error: Cannot import 'TimeSeriesSplit'

```bash
pip install --upgrade scikit-learn>=1.3.0
```

### Cache Directory Permission Errors

Tests use `tempfile.TemporaryDirectory()` which should handle permissions automatically. If issues persist:

```bash
# Linux/macOS
export TMPDIR=/tmp

# Verify temp directory is writable
python3 -c "import tempfile; print(tempfile.gettempdir())"
```

## Test File Locations

```
backend/tests/
├── unit/models/
│   ├── test_rfecv_config.py                # RFECVConfig tests (16)
│   ├── test_rfecv_selector.py              # RFECVSelector tests (21)
│   └── test_feature_selection_manager.py   # Manager tests (20)
└── integration/
    └── test_rfecv_integration.py           # Integration tests (11)
```

## Running in Docker

### Option 1: Install pytest in container

```bash
docker exec ai-trader-backend pip install pytest pytest-asyncio
docker exec ai-trader-backend python -m pytest tests/unit/models/test_rfecv_*.py -v
```

### Option 2: Run tests during build

Add to `Dockerfile`:

```dockerfile
RUN pip install pytest pytest-asyncio
RUN python -m pytest tests/unit/models/test_rfecv_*.py
```

### Option 3: docker-compose test service

Add to `docker-compose.yml`:

```yaml
services:
  backend-test:
    build: ./backend
    command: pytest tests/unit/models/test_rfecv_*.py -v
    volumes:
      - ./backend:/app
```

Run:

```bash
docker-compose run backend-test
```

## CI/CD Integration

### GitHub Actions

Create `.github/workflows/test-rfecv.yml`:

```yaml
name: RFECV Tests
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - uses: actions/setup-python@v4
        with:
          python-version: '3.12'

      - name: Install dependencies
        run: |
          cd backend
          pip install -r requirements.txt
          pip install pytest pytest-asyncio pytest-cov

      - name: Run RFECV tests
        run: |
          cd backend
          pytest tests/unit/models/test_rfecv_*.py \
                 tests/unit/models/test_feature_selection_manager.py \
                 tests/integration/test_rfecv_integration.py \
                 -v --cov=src/models/feature_selection

      - name: Upload coverage
        uses: codecov/codecov-action@v3
```

### Pre-commit Hook

Create `.git/hooks/pre-commit`:

```bash
#!/bin/bash
cd backend
pytest tests/unit/models/test_rfecv_*.py --quiet
if [ $? -ne 0 ]; then
    echo "❌ RFECV tests failed. Commit aborted."
    exit 1
fi
echo "✅ RFECV tests passed"
```

Make executable:

```bash
chmod +x .git/hooks/pre-commit
```

## Performance Benchmarks

Expected test execution times (with mocking):

| Test File | Tests | Time | Speed |
|-----------|-------|------|-------|
| `test_rfecv_config.py` | 16 | ~0.2s | Very Fast |
| `test_rfecv_selector.py` | 21 | ~0.5s | Fast |
| `test_feature_selection_manager.py` | 20 | ~0.8s | Fast |
| `test_rfecv_integration.py` | 11 | ~1.0s | Fast |
| **Total** | **68** | **~2.5s** | **Fast** |

If tests take >30 seconds, mocking is not working correctly.

## Syntax Validation (No pytest required)

If pytest is not available, validate syntax only:

```bash
python3 -m py_compile tests/unit/models/test_rfecv_config.py
python3 -m py_compile tests/unit/models/test_rfecv_selector.py
python3 -m py_compile tests/unit/models/test_feature_selection_manager.py
python3 -m py_compile tests/integration/test_rfecv_integration.py
```

No output = syntax is valid ✓

## Next Steps

1. **Install pytest**: `pip install pytest pytest-asyncio`
2. **Run all tests**: `pytest tests/unit/models/test_rfecv_*.py tests/integration/test_rfecv_integration.py -v`
3. **Check coverage**: Add `--cov=src/models/feature_selection --cov-report=html`
4. **Fix failures**: If any tests fail, investigate source code issues
5. **Add to CI/CD**: Integrate into your automated pipeline

---

**Quick Help:**
- See `RFECV_TEST_SUMMARY.md` for detailed test documentation
- See test files for inline documentation
- Run `pytest --help` for more options
