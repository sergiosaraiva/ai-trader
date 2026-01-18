"""
Pytest configuration and fixtures for AI Trader tests.
"""

import sys
from pathlib import Path

# Add src to Python path BEFORE any other imports
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

# Also add the project root
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import pytest
from unittest.mock import Mock, MagicMock


@pytest.fixture
def sample_prediction():
    """Sample prediction data."""
    return {
        "signal": "BUY",
        "confidence": 0.72,
        "current_price": 1.08543,
        "symbol": "EURUSD",
        "timestamp": "2025-01-12T10:00:00Z",
        "timeframe_signals": {
            "1H": {"signal": "BUY", "confidence": 0.75},
            "4H": {"signal": "BUY", "confidence": 0.70},
            "D": {"signal": "HOLD", "confidence": 0.55},
        }
    }


@pytest.fixture
def sample_candles():
    """Sample candle data."""
    return [
        {"open": 1.0850, "high": 1.0860, "low": 1.0845, "close": 1.0855, "volume": 1000},
        {"open": 1.0855, "high": 1.0865, "low": 1.0850, "close": 1.0862, "volume": 1200},
        {"open": 1.0862, "high": 1.0870, "low": 1.0858, "close": 1.0868, "volume": 1100},
    ]
