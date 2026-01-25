# Unit Tests for Backend Scripts

This directory contains unit tests for backend Python scripts.

## Test Files

### test_walk_forward_optimization.py

Comprehensive tests for the Tier 1 drawdown mitigation functionality.

**Function Tested**: `get_drawdown_position_multiplier()`

**Test Count**: 67 tests

**Coverage**:
- Core functionality (5 drawdown levels)
- Edge cases
- Custom parameters
- Compounding behavior
- Documentation verification
- Performance tests

**Quick Run**:
```bash
cd backend
source ../.venv/bin/activate
python -m pytest tests/unit/scripts/test_walk_forward_optimization.py -v
```

**Documentation**: See `TEST_SUMMARY_DRAWDOWN_MITIGATION.md` for detailed information.

## Running Tests

### All Script Tests
```bash
cd backend
source ../.venv/bin/activate
python -m pytest tests/unit/scripts/ -v
```

### Specific Test File
```bash
python -m pytest tests/unit/scripts/test_walk_forward_optimization.py -v
```

### Specific Test Class
```bash
python -m pytest tests/unit/scripts/test_walk_forward_optimization.py::TestGetDrawdownPositionMultiplier -v
```

### Specific Test Method
```bash
python -m pytest tests/unit/scripts/test_walk_forward_optimization.py::TestGetDrawdownPositionMultiplier::test_exact_5_percent_drawdown -v
```

### With Coverage
```bash
python -m pytest tests/unit/scripts/ --cov=scripts --cov-report=term-missing
```

### Quiet Mode (Summary Only)
```bash
python -m pytest tests/unit/scripts/ -q
```

### Stop on First Failure
```bash
python -m pytest tests/unit/scripts/ -x
```

## Test Patterns

All tests in this directory follow these conventions:

1. **AAA Pattern**: Arrange, Act, Assert
2. **Clear Naming**: Test names describe what they validate
3. **Comprehensive Coverage**: Core functionality, edge cases, boundaries
4. **Documentation**: Each test includes docstrings explaining purpose
5. **Fixtures**: Common test data in pytest fixtures
6. **Parametrization**: Multiple similar cases tested efficiently

## Adding New Tests

When adding tests for new scripts:

1. Create test file: `test_<script_name>.py`
2. Import the script functions using sys.path manipulation
3. Organize tests into logical classes
4. Follow existing test patterns
5. Include docstrings for all test classes and methods
6. Add parametrized tests where applicable
7. Test edge cases and boundary conditions
8. Update this README with new test file information

Example structure:
```python
"""Unit tests for <script_name>.py"""

import pytest
import sys
from pathlib import Path

# Add scripts directory to path
scripts_dir = Path(__file__).parent.parent.parent.parent / "scripts"
sys.path.insert(0, str(scripts_dir))

from <script_name> import <function_name>

class Test<FunctionName>:
    """Tests for <function_name>."""

    def test_<scenario>(self):
        """Test description."""
        # Arrange
        # Act
        # Assert
```

---

**Last Updated**: 2026-01-25
