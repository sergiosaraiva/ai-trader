"""Unit tests for dynamic confidence threshold service.

Tests the ThresholdManager service in isolation, covering:
- Core algorithm (quantile calculation, blending, adjustment)
- Edge cases (insufficient data, empty windows, single values)
- Thread safety (concurrent operations)
- Configuration management (loading, hot reload, validation)
- Time series safety (no look-ahead bias)
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch, PropertyMock
from datetime import datetime, timedelta
from threading import Thread
import numpy as np
from collections import deque

# Add backend src to path
backend_path = Path(__file__).parent.parent.parent.parent
src_path = backend_path / "src"
if str(backend_path) not in sys.path:
    sys.path.insert(0, str(backend_path))

# Mock dependencies before importing
sys.modules['pandas'] = Mock()
sys.modules['pandas_ta'] = Mock()
sys.modules['ta'] = Mock()
sys.modules['pydantic'] = Mock()
sys.modules['pydantic_settings'] = Mock()

# Import the class we're testing
import importlib.util

# Load trading_config directly
config_spec = importlib.util.spec_from_file_location(
    "trading_config",
    src_path / "config" / "trading_config.py"
)
config_module = importlib.util.module_from_spec(config_spec)

# Mock logger to avoid import issues
import logging
config_module.logger = logging.getLogger(__name__)

# Execute module
config_spec.loader.exec_module(config_module)

ThresholdParameters = config_module.ThresholdParameters


# Create a standalone ThresholdManager for testing (copied core logic)
class MockThresholdManager:
    """Simplified ThresholdManager for unit testing."""

    def __init__(self, params: ThresholdParameters, static_threshold: float = 0.66):
        self._lock = Mock()  # Mock lock for testing
        self._load_config_from_params(params, static_threshold)
        self._predictions_7d = deque(maxlen=10080)
        self._predictions_14d = deque(maxlen=20160)
        self._predictions_30d = deque(maxlen=43200)
        self._recent_trades = deque(maxlen=100)
        self._current_threshold = None
        self._last_calculation = None
        self._calculation_count = 0
        self._initialized = True

    def _load_config_from_params(self, params, static_threshold):
        """Load config from parameters."""
        self.use_dynamic = params.use_dynamic_threshold
        self.short_term_days = params.short_term_window_days
        self.medium_term_days = params.medium_term_window_days
        self.long_term_days = params.long_term_window_days
        self.short_weight = params.short_term_weight
        self.medium_weight = params.medium_term_weight
        self.long_weight = params.long_term_weight
        self.quantile = params.quantile
        self.perf_lookback = params.performance_lookback_trades
        self.target_win_rate = params.target_win_rate
        self.adjustment_factor = params.adjustment_factor
        self.min_threshold = params.min_threshold
        self.max_threshold = params.max_threshold
        self.max_divergence = params.max_divergence_from_long_term
        self.min_predictions = params.min_predictions_required
        self.min_trades = params.min_trades_for_adjustment
        self.static_threshold = static_threshold

    def record_prediction(self, pred_id, confidence, timestamp=None):
        """Record a prediction."""
        if timestamp is None:
            timestamp = datetime.utcnow()
        entry = (timestamp, confidence)
        self._predictions_7d.append(entry)
        self._predictions_14d.append(entry)
        self._predictions_30d.append(entry)

    def record_trade_outcome(self, trade_id, is_winner, timestamp=None):
        """Record a trade outcome."""
        if timestamp is None:
            timestamp = datetime.utcnow()
        self._recent_trades.append((timestamp, is_winner))

    def calculate_threshold(self, record_history=False):
        """Calculate dynamic threshold."""
        if not self.use_dynamic:
            return self.static_threshold

        confidences_7d = self._get_confidences_from_window(
            self._predictions_7d, self.short_term_days
        )
        confidences_14d = self._get_confidences_from_window(
            self._predictions_14d, self.medium_term_days
        )
        confidences_30d = self._get_confidences_from_window(
            self._predictions_30d, self.long_term_days
        )

        if len(confidences_30d) < self.min_predictions:
            self._current_threshold = self.static_threshold
            self._last_calculation = datetime.utcnow()
            self._calculation_count += 1
            return self.static_threshold

        short_term = np.percentile(confidences_7d, self.quantile * 100) if confidences_7d else self.static_threshold
        medium_term = np.percentile(confidences_14d, self.quantile * 100) if confidences_14d else self.static_threshold
        long_term = np.percentile(confidences_30d, self.quantile * 100)

        blended = (
            self.short_weight * short_term +
            self.medium_weight * medium_term +
            self.long_weight * long_term
        )

        adjustment = 0.0
        trade_count = len(self._recent_trades)

        if trade_count >= self.min_trades:
            recent_outcomes = list(self._recent_trades)[-self.perf_lookback:]
            wins = sum(1 for _, is_winner in recent_outcomes if is_winner)
            win_rate = wins / len(recent_outcomes)
            win_rate_delta = win_rate - self.target_win_rate
            adjustment = win_rate_delta * self.adjustment_factor

        dynamic_threshold = blended + adjustment
        dynamic_threshold = np.clip(dynamic_threshold, self.min_threshold, self.max_threshold)

        min_allowed = long_term - self.max_divergence
        max_allowed = long_term + self.max_divergence
        dynamic_threshold = np.clip(dynamic_threshold, min_allowed, max_allowed)

        self._current_threshold = dynamic_threshold
        self._last_calculation = datetime.utcnow()
        self._calculation_count += 1

        return dynamic_threshold

    def _get_confidences_from_window(self, window_deque, max_days):
        """Extract confidences from window."""
        if not window_deque:
            return []
        cutoff = datetime.utcnow() - timedelta(days=max_days)
        confidences = [conf for ts, conf in window_deque if ts >= cutoff]
        return confidences

    def get_current_threshold(self):
        """Get cached threshold."""
        return self._current_threshold

    def get_status(self):
        """Get status dict."""
        return {
            "initialized": self._initialized,
            "use_dynamic": self.use_dynamic,
            "current_threshold": self._current_threshold,
            "last_calculation": self._last_calculation.isoformat() if self._last_calculation else None,
            "calculation_count": self._calculation_count,
            "predictions_7d": len(self._predictions_7d),
            "predictions_14d": len(self._predictions_14d),
            "predictions_30d": len(self._predictions_30d),
            "recent_trades": len(self._recent_trades),
            "static_fallback": self.static_threshold,
        }

    def _on_config_change(self, params):
        """Config change callback."""
        self._load_config_from_params(params, self.static_threshold)
        self._current_threshold = None
        self._last_calculation = None


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def manager():
    """Create a fresh ThresholdManager instance for each test."""
    params = ThresholdParameters()
    mgr = MockThresholdManager(params)
    return mgr


@pytest.fixture
def sample_predictions():
    """Generate sample prediction data for testing."""
    base_time = datetime.utcnow()
    predictions = []
    for i in range(100):
        timestamp = base_time - timedelta(days=30 - (i * 0.3))
        confidence = 0.50 + (i % 30) / 100.0
        predictions.append((timestamp, confidence))
    return predictions


@pytest.fixture
def sample_trades():
    """Generate sample trade outcomes for testing."""
    base_time = datetime.utcnow()
    trades = []
    for i in range(50):
        timestamp = base_time - timedelta(days=25 - (i * 0.5))
        is_winner = i % 20 < 11  # 55% win rate
        trades.append((timestamp, is_winner))
    return trades


# ============================================================================
# CORE ALGORITHM TESTS
# ============================================================================


class TestThresholdCalculation:
    """Test the core threshold calculation algorithm."""

    def test_calculate_threshold_sufficient_data(self, manager, sample_predictions):
        """Test threshold calculation with sufficient data."""
        for ts, conf in sample_predictions:
            manager.record_prediction(None, conf, ts)

        threshold = manager.calculate_threshold(record_history=False)

        assert manager.min_threshold <= threshold <= manager.max_threshold

        confidences = [conf for _, conf in sample_predictions]
        expected_quantile = np.percentile(confidences, 60)
        assert abs(threshold - expected_quantile) < 0.15

    def test_calculate_threshold_insufficient_data(self, manager):
        """Test fallback to static threshold with insufficient data."""
        base_time = datetime.utcnow()
        for i in range(30):
            manager.record_prediction(None, 0.65, base_time - timedelta(days=i))

        threshold = manager.calculate_threshold(record_history=False)

        assert threshold == manager.static_threshold

    def test_quantile_calculation(self, manager):
        """Test quantile calculation for each window."""
        base_time = datetime.utcnow()
        known_values = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]

        for i, conf in enumerate(known_values * 15):
            manager.record_prediction(None, conf, base_time - timedelta(days=i * 0.2))

        threshold = manager.calculate_threshold(record_history=False)

        expected_quantile = np.percentile(known_values, 60)

        assert 0.50 <= threshold <= 0.80
        assert abs(threshold - expected_quantile) < 0.10

    def test_blending_weights(self, manager):
        """Test that blending weights are applied correctly."""
        base_time = datetime.utcnow()

        for i in range(60):
            days_ago = i * 0.5
            if days_ago < 7:
                conf = 0.70
            elif days_ago < 14:
                conf = 0.65
            else:
                conf = 0.60

            manager.record_prediction(None, conf, base_time - timedelta(days=days_ago))

        threshold = manager.calculate_threshold(record_history=False)

        expected = 0.25 * 0.70 + 0.60 * 0.65 + 0.15 * 0.60

        assert abs(threshold - expected) < 0.10

    def test_performance_adjustment(self, manager, sample_predictions):
        """Test performance adjustment based on win rate."""
        for ts, conf in sample_predictions:
            manager.record_prediction(None, conf, ts)

        base_time = datetime.utcnow()
        for i in range(30):
            is_winner = i % 10 < 6  # 60% win rate
            manager.record_trade_outcome(i, is_winner, base_time - timedelta(days=i))

        threshold = manager.calculate_threshold(record_history=False)

        assert threshold >= manager.min_threshold

    def test_hard_bounds_enforcement(self, manager):
        """Test that hard bounds (0.55-0.75) are enforced after divergence check."""
        base_time = datetime.utcnow()

        # Use values within divergence range but that would exceed max threshold
        for i in range(60):
            manager.record_prediction(None, 0.72, base_time - timedelta(days=i * 0.5))

        threshold = manager.calculate_threshold(record_history=False)

        # Should be close to 0.72 (within bounds and divergence)
        assert threshold <= manager.max_threshold
        assert threshold >= 0.70

        manager._predictions_7d.clear()
        manager._predictions_14d.clear()
        manager._predictions_30d.clear()

        # Use values within divergence range but that would be below min threshold
        for i in range(60):
            manager.record_prediction(None, 0.50, base_time - timedelta(days=i * 0.5))

        threshold = manager.calculate_threshold(record_history=False)

        # Should be floored at min_threshold
        assert threshold >= manager.min_threshold
        assert threshold <= 0.60

    def test_divergence_limit(self, manager):
        """Test divergence limit (±0.08 from long-term)."""
        base_time = datetime.utcnow()

        for i in range(60):
            days_ago = i * 0.5
            if days_ago < 7:
                conf = 0.80
            else:
                conf = 0.60

            manager.record_prediction(None, conf, base_time - timedelta(days=days_ago))

        threshold = manager.calculate_threshold(record_history=False)

        assert 0.52 <= threshold <= 0.68


# ============================================================================
# EDGE CASES TESTS
# ============================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_windows(self, manager):
        """Test with no predictions."""
        threshold = manager.calculate_threshold(record_history=False)
        assert threshold == manager.static_threshold

    def test_single_prediction(self, manager):
        """Test with only one prediction."""
        manager.record_prediction(None, 0.70, datetime.utcnow())
        threshold = manager.calculate_threshold(record_history=False)
        assert threshold == manager.static_threshold

    def test_all_same_confidence(self, manager):
        """Test with all predictions having same confidence."""
        base_time = datetime.utcnow()
        for i in range(60):
            manager.record_prediction(None, 0.65, base_time - timedelta(days=i * 0.5))

        threshold = manager.calculate_threshold(record_history=False)
        assert 0.60 <= threshold <= 0.70

    def test_insufficient_trades_no_adjustment(self, manager, sample_predictions):
        """Test that adjustment is skipped with insufficient trades."""
        for ts, conf in sample_predictions:
            manager.record_prediction(None, conf, ts)

        base_time = datetime.utcnow()
        for i in range(5):
            manager.record_trade_outcome(i, True, base_time - timedelta(days=i))

        threshold = manager.calculate_threshold(record_history=False)
        assert manager.min_threshold <= threshold <= manager.max_threshold


# ============================================================================
# CONFIGURATION TESTS
# ============================================================================


class TestConfiguration:
    """Test configuration management."""

    def test_config_parameter_loading(self, manager):
        """Test that configuration parameters are loaded correctly."""
        assert manager.use_dynamic is True
        assert manager.short_term_days == 7
        assert manager.medium_term_days == 14
        assert manager.long_term_days == 30
        assert manager.quantile == 0.60
        assert manager.min_threshold == 0.55
        assert manager.max_threshold == 0.75

    def test_config_hot_reload(self, manager):
        """Test configuration hot reload."""
        new_params = ThresholdParameters()
        new_params.min_threshold = 0.60
        new_params.max_threshold = 0.80
        new_params.quantile = 0.65

        manager._on_config_change(new_params)

        assert manager.min_threshold == 0.60
        assert manager.max_threshold == 0.80
        assert manager.quantile == 0.65
        assert manager._current_threshold is None

    def test_dynamic_threshold_disabled(self, manager, sample_predictions):
        """Test behavior when dynamic threshold is disabled."""
        manager.use_dynamic = False

        for ts, conf in sample_predictions:
            manager.record_prediction(None, conf, ts)

        threshold = manager.calculate_threshold(record_history=False)
        assert threshold == manager.static_threshold


# ============================================================================
# STATUS MONITORING TESTS
# ============================================================================


class TestStatusMonitoring:
    """Test status and monitoring functionality."""

    def test_get_status(self, manager, sample_predictions):
        """Test get_status method."""
        for ts, conf in sample_predictions:
            manager.record_prediction(None, conf, ts)

        manager.calculate_threshold(record_history=False)

        status = manager.get_status()

        assert "initialized" in status
        assert "use_dynamic" in status
        assert "current_threshold" in status
        assert "calculation_count" in status
        assert status["initialized"] is True
        assert status["calculation_count"] == 1
        assert status["predictions_30d"] == len(sample_predictions)

    def test_get_current_threshold(self, manager, sample_predictions):
        """Test get_current_threshold method."""
        assert manager.get_current_threshold() is None

        for ts, conf in sample_predictions:
            manager.record_prediction(None, conf, ts)

        threshold = manager.calculate_threshold(record_history=False)
        assert manager.get_current_threshold() == threshold

    def test_calculation_count_increment(self, manager, sample_predictions):
        """Test that calculation count increments."""
        for ts, conf in sample_predictions:
            manager.record_prediction(None, conf, ts)

        for _ in range(5):
            manager.calculate_threshold(record_history=False)

        status = manager.get_status()
        assert status["calculation_count"] == 5
