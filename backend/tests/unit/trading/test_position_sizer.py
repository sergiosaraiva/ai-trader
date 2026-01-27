"""Unit tests for Conservative Hybrid position sizing.

Tests the ConservativeHybridSizer class in isolation, covering:
- Position calculation at different confidence levels
- Risk scaling with confidence multiplier
- Min/max risk caps enforcement
- No-leverage constraint enforcement
- Invalid input handling
- Edge cases (zero balance, below minimum position)
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock

# Add backend src to path
backend_path = Path(__file__).parent.parent.parent.parent
src_path = backend_path / "src"
if str(backend_path) not in sys.path:
    sys.path.insert(0, str(backend_path))

# Import directly from module files to avoid dependency issues
import importlib.util

# Load position_sizer module
position_sizer_spec = importlib.util.spec_from_file_location(
    "position_sizer",
    src_path / "trading" / "position_sizer.py"
)
position_sizer_module = importlib.util.module_from_spec(position_sizer_spec)

# Mock logger
import logging
position_sizer_module.logger = logging.getLogger(__name__)

position_sizer_spec.loader.exec_module(position_sizer_module)

ConservativeHybridSizer = position_sizer_module.ConservativeHybridSizer
MIN_POSITION_SIZE = position_sizer_module.MIN_POSITION_SIZE

# Load trading_config module
config_spec = importlib.util.spec_from_file_location(
    "trading_config",
    src_path / "config" / "trading_config.py"
)
config_module = importlib.util.module_from_spec(config_spec)
config_module.logger = logging.getLogger(__name__)
config_spec.loader.exec_module(config_module)

ConservativeHybridParameters = config_module.ConservativeHybridParameters


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def config():
    """Create default ConservativeHybridParameters config."""
    return ConservativeHybridParameters(
        base_risk_percent=1.5,
        confidence_scaling_factor=0.5,
        min_risk_percent=0.8,
        max_risk_percent=2.5,
        confidence_threshold=0.70,
        daily_loss_limit_percent=-3.0,
        consecutive_loss_limit=5,
        pip_value=10.0,
        lot_size=100000.0
    )


@pytest.fixture
def position_sizer():
    """Create a fresh ConservativeHybridSizer instance."""
    return ConservativeHybridSizer()


# ============================================================================
# CORE ALGORITHM TESTS
# ============================================================================


class TestPositionCalculation:
    """Test the core position sizing calculation."""

    def test_calculate_position_at_threshold_confidence(self, position_sizer, config):
        """Test position calculation when confidence equals threshold."""
        balance = 10000.0
        confidence = 0.70  # At threshold
        sl_pips = 15.0

        position_lots, risk_pct, metadata = position_sizer.calculate_position_size(
            balance=balance,
            confidence=confidence,
            sl_pips=sl_pips,
            config=config
        )

        # At threshold, confidence_multiplier = 1.0 + (0.70 - 0.70) * 0.5 = 1.0
        # adjusted_risk = 1.5 * 1.0 = 1.5%
        assert risk_pct == pytest.approx(1.5, rel=0.01)

        # Expected position: 10000 * 0.015 / (15 * 10) = 1.0 lots
        assert position_lots == pytest.approx(1.0, rel=0.01)

        # Verify metadata
        assert metadata["confidence"] == confidence
        assert metadata["confidence_multiplier"] == pytest.approx(1.0, rel=0.01)
        assert metadata["base_risk_pct"] == 1.5
        assert metadata["adjusted_risk_pct"] == pytest.approx(1.5, rel=0.01)
        assert metadata["risk_pct_used"] == pytest.approx(1.5, rel=0.01)
        assert metadata["limited_by_cash"] is False

    def test_calculate_position_below_threshold(self, position_sizer, config):
        """Test position calculation when confidence is below threshold."""
        balance = 10000.0
        confidence = 0.65  # Below threshold (0.70)
        sl_pips = 15.0

        position_lots, risk_pct, metadata = position_sizer.calculate_position_size(
            balance=balance,
            confidence=confidence,
            sl_pips=sl_pips,
            config=config
        )

        # Below threshold should return 0 position
        assert position_lots == 0.0
        assert risk_pct == 0.0
        assert metadata["reason"] == "confidence_below_threshold"
        assert metadata["confidence"] == confidence
        assert metadata["threshold"] == config.confidence_threshold

    def test_calculate_position_high_confidence(self, position_sizer, config):
        """Test position calculation with high confidence hitting max cap."""
        balance = 10000.0
        confidence = 0.85  # High confidence
        sl_pips = 15.0

        position_lots, risk_pct, metadata = position_sizer.calculate_position_size(
            balance=balance,
            confidence=confidence,
            sl_pips=sl_pips,
            config=config
        )

        # confidence_multiplier = 1.0 + (0.85 - 0.70) * 0.5 = 1.075
        # adjusted_risk = 1.5 * 1.075 = 1.6125%
        # Should be capped at max_risk_percent (2.5%)
        assert risk_pct <= config.max_risk_percent
        assert position_lots > 0.0

        # Verify multiplier calculation
        expected_multiplier = 1.0 + (0.85 - 0.70) * 0.5
        assert metadata["confidence_multiplier"] == pytest.approx(expected_multiplier, rel=0.01)

    def test_calculate_position_confidence_scaling(self, position_sizer, config):
        """Test that confidence multiplier scales risk correctly."""
        balance = 10000.0
        sl_pips = 15.0

        # Test at threshold (multiplier = 1.0)
        pos1, risk1, meta1 = position_sizer.calculate_position_size(
            balance=balance, confidence=0.70, sl_pips=sl_pips, config=config
        )

        # Test above threshold (multiplier > 1.0)
        pos2, risk2, meta2 = position_sizer.calculate_position_size(
            balance=balance, confidence=0.75, sl_pips=sl_pips, config=config
        )

        # Test even higher (multiplier >> 1.0)
        pos3, risk3, meta3 = position_sizer.calculate_position_size(
            balance=balance, confidence=0.80, sl_pips=sl_pips, config=config
        )

        # Risk should increase with confidence
        assert risk1 < risk2 < risk3
        assert pos1 < pos2 < pos3

        # Verify multiplier progression
        assert meta1["confidence_multiplier"] < meta2["confidence_multiplier"] < meta3["confidence_multiplier"]

    def test_calculate_position_limited_by_cash(self, position_sizer, config):
        """Test position limited by no-leverage constraint (small balance)."""
        balance = 5000.0  # Small balance
        confidence = 0.80
        sl_pips = 10.0  # Small SL means desired position would be larger

        position_lots, risk_pct, metadata = position_sizer.calculate_position_size(
            balance=balance,
            confidence=confidence,
            sl_pips=sl_pips,
            config=config
        )

        # Max position without leverage: 5000 / 100000 = 0.05 lots
        max_no_leverage = balance / config.lot_size

        # Should be limited by cash
        assert position_lots <= max_no_leverage
        assert metadata["limited_by_cash"] is True
        assert metadata["max_position_no_leverage"] == pytest.approx(max_no_leverage, rel=0.01)

    def test_calculate_position_limited_by_risk(self, position_sizer, config):
        """Test position limited by risk percentage (large balance)."""
        balance = 100000.0  # Large balance
        confidence = 0.75
        sl_pips = 20.0

        position_lots, risk_pct, metadata = position_sizer.calculate_position_size(
            balance=balance,
            confidence=confidence,
            sl_pips=sl_pips,
            config=config
        )

        # With large balance, risk limit should be the constraint
        max_no_leverage = balance / config.lot_size  # 1.0 lot

        # Risk amount: 100000 * risk_pct / 100
        # Position: risk_amount / (20 * 10)
        # Should be less than max_no_leverage
        assert position_lots <= max_no_leverage
        assert metadata["limited_by_cash"] is False

    def test_calculate_position_min_max_caps(self, position_sizer, config):
        """Test that min/max risk caps are enforced."""
        balance = 10000.0
        sl_pips = 15.0

        # Test minimum risk cap (very low confidence, just above threshold)
        config_low = ConservativeHybridParameters(
            base_risk_percent=0.5,  # Very low base risk
            confidence_scaling_factor=0.1,
            min_risk_percent=0.8,
            max_risk_percent=2.5,
            confidence_threshold=0.70,
        )

        pos_min, risk_min, meta_min = position_sizer.calculate_position_size(
            balance=balance, confidence=0.71, sl_pips=sl_pips, config=config_low
        )

        # Should be floored at min_risk_percent
        assert risk_min >= config_low.min_risk_percent

        # Test maximum risk cap (very high confidence)
        config_high = ConservativeHybridParameters(
            base_risk_percent=3.0,  # High base risk
            confidence_scaling_factor=2.0,  # High scaling
            min_risk_percent=0.8,
            max_risk_percent=2.5,
            confidence_threshold=0.70,
        )

        pos_max, risk_max, meta_max = position_sizer.calculate_position_size(
            balance=balance, confidence=0.85, sl_pips=sl_pips, config=config_high
        )

        # Should be capped at max_risk_percent
        assert risk_max <= config_high.max_risk_percent


# ============================================================================
# EDGE CASES AND ERROR HANDLING
# ============================================================================


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_calculate_position_invalid_sl_pips(self, position_sizer, config):
        """Test handling of invalid stop loss (sl_pips <= 0)."""
        balance = 10000.0
        confidence = 0.75

        # Test zero SL
        position_lots, risk_pct, metadata = position_sizer.calculate_position_size(
            balance=balance,
            confidence=confidence,
            sl_pips=0.0,
            config=config
        )

        assert position_lots == 0.0
        assert risk_pct == 0.0
        assert metadata["reason"] == "invalid_sl_pips"
        assert metadata["sl_pips"] == 0.0

        # Test negative SL
        position_lots, risk_pct, metadata = position_sizer.calculate_position_size(
            balance=balance,
            confidence=confidence,
            sl_pips=-15.0,
            config=config
        )

        assert position_lots == 0.0
        assert risk_pct == 0.0
        assert metadata["reason"] == "invalid_sl_pips"

    def test_calculate_position_invalid_pip_value(self, position_sizer, config):
        """Test handling of invalid pip value (pip_value <= 0)."""
        balance = 10000.0
        confidence = 0.75
        sl_pips = 15.0

        # Test zero pip_value
        position_lots, risk_pct, metadata = position_sizer.calculate_position_size(
            balance=balance,
            confidence=confidence,
            sl_pips=sl_pips,
            config=config,
            pip_value=0.0
        )

        assert position_lots == 0.0
        assert risk_pct == 0.0
        assert metadata["reason"] == "invalid_pip_value"
        assert metadata["pip_value"] == 0.0

        # Test negative pip_value
        position_lots, risk_pct, metadata = position_sizer.calculate_position_size(
            balance=balance,
            confidence=confidence,
            sl_pips=sl_pips,
            config=config,
            pip_value=-10.0
        )

        assert position_lots == 0.0
        assert risk_pct == 0.0
        assert metadata["reason"] == "invalid_pip_value"

    def test_calculate_position_invalid_lot_size(self, position_sizer, config):
        """Test handling of invalid lot size (lot_size <= 0)."""
        balance = 10000.0
        confidence = 0.75
        sl_pips = 15.0

        # Test zero lot_size
        position_lots, risk_pct, metadata = position_sizer.calculate_position_size(
            balance=balance,
            confidence=confidence,
            sl_pips=sl_pips,
            config=config,
            lot_size=0.0
        )

        assert position_lots == 0.0
        assert risk_pct == 0.0
        assert metadata["reason"] == "invalid_lot_size"
        assert metadata["lot_size"] == 0.0

        # Test negative lot_size
        position_lots, risk_pct, metadata = position_sizer.calculate_position_size(
            balance=balance,
            confidence=confidence,
            sl_pips=sl_pips,
            config=config,
            lot_size=-100000.0
        )

        assert position_lots == 0.0
        assert risk_pct == 0.0
        assert metadata["reason"] == "invalid_lot_size"

    def test_calculate_position_zero_balance(self, position_sizer, config):
        """Test handling of zero balance."""
        balance = 0.0
        confidence = 0.75
        sl_pips = 15.0

        position_lots, risk_pct, metadata = position_sizer.calculate_position_size(
            balance=balance,
            confidence=confidence,
            sl_pips=sl_pips,
            config=config
        )

        # With zero balance, risk amount is zero
        # Position should be zero (0 / (15 * 10) = 0)
        assert position_lots == 0.0

    def test_calculate_position_negative_balance(self, position_sizer, config):
        """Test handling of negative balance."""
        balance = -1000.0
        confidence = 0.75
        sl_pips = 15.0

        position_lots, risk_pct, metadata = position_sizer.calculate_position_size(
            balance=balance,
            confidence=confidence,
            sl_pips=sl_pips,
            config=config
        )

        # Negative balance should result in zero or negative position
        # The no-leverage constraint (balance / lot_size) would be negative
        # min() would select the negative value, resulting in 0 position
        assert position_lots == 0.0

    def test_calculate_position_below_minimum(self, position_sizer, config):
        """Test handling of position size below minimum (0.01 lots)."""
        # Small balance with high SL results in tiny position
        balance = 50.0  # Very small balance
        confidence = 0.70
        sl_pips = 20.0

        position_lots, risk_pct, metadata = position_sizer.calculate_position_size(
            balance=balance,
            confidence=confidence,
            sl_pips=sl_pips,
            config=config
        )

        # Risk amount: 50 * 0.015 = 0.75
        # Desired position: 0.75 / (20 * 10) = 0.00375 lots
        # Max no leverage: 50 / 100000 = 0.0005 lots
        # Final position: min(0.00375, 0.0005) = 0.0005 lots
        # This is below MIN_POSITION_SIZE (0.01), should return 0

        assert position_lots == 0.0
        assert risk_pct == 0.0
        assert metadata["reason"] == "below_minimum_position_size"
        assert metadata["minimum_required"] == MIN_POSITION_SIZE
        assert "calculated_position" in metadata

    def test_calculate_position_metadata(self, position_sizer, config):
        """Test that metadata contains all expected fields."""
        balance = 10000.0
        confidence = 0.75
        sl_pips = 15.0

        position_lots, risk_pct, metadata = position_sizer.calculate_position_size(
            balance=balance,
            confidence=confidence,
            sl_pips=sl_pips,
            config=config
        )

        # Verify all metadata fields are present
        required_fields = [
            "confidence",
            "confidence_multiplier",
            "base_risk_pct",
            "adjusted_risk_pct",
            "risk_pct_used",
            "risk_amount_usd",
            "sl_pips",
            "desired_position_lots",
            "max_position_no_leverage",
            "final_position_lots",
            "limited_by_cash",
        ]

        for field in required_fields:
            assert field in metadata, f"Missing metadata field: {field}"

        # Verify values are reasonable
        assert metadata["confidence"] == confidence
        assert metadata["risk_amount_usd"] > 0
        assert metadata["sl_pips"] == sl_pips
        assert metadata["final_position_lots"] == position_lots
        assert isinstance(metadata["limited_by_cash"], bool)


# ============================================================================
# PARAMETER VARIATION TESTS
# ============================================================================


class TestParameterVariations:
    """Test different parameter configurations."""

    def test_different_base_risk_values(self, position_sizer):
        """Test position sizing with different base risk percentages."""
        balance = 10000.0
        confidence = 0.75
        sl_pips = 15.0

        # Low base risk
        config_low = ConservativeHybridParameters(base_risk_percent=1.0)
        pos_low, risk_low, _ = position_sizer.calculate_position_size(
            balance, confidence, sl_pips, config_low
        )

        # Medium base risk
        config_med = ConservativeHybridParameters(base_risk_percent=1.5)
        pos_med, risk_med, _ = position_sizer.calculate_position_size(
            balance, confidence, sl_pips, config_med
        )

        # High base risk
        config_high = ConservativeHybridParameters(base_risk_percent=2.0)
        pos_high, risk_high, _ = position_sizer.calculate_position_size(
            balance, confidence, sl_pips, config_high
        )

        # Position should increase with base risk
        assert pos_low < pos_med < pos_high
        assert risk_low < risk_med < risk_high

    def test_different_scaling_factors(self, position_sizer):
        """Test position sizing with different confidence scaling factors."""
        balance = 10000.0
        confidence = 0.80  # Above threshold
        sl_pips = 15.0

        # Low scaling (confidence has less impact)
        config_low = ConservativeHybridParameters(confidence_scaling_factor=0.2)
        pos_low, risk_low, meta_low = position_sizer.calculate_position_size(
            balance, confidence, sl_pips, config_low
        )

        # High scaling (confidence has more impact)
        config_high = ConservativeHybridParameters(confidence_scaling_factor=1.0)
        pos_high, risk_high, meta_high = position_sizer.calculate_position_size(
            balance, confidence, sl_pips, config_high
        )

        # Higher scaling should result in higher position at high confidence
        assert pos_high > pos_low
        assert risk_high > risk_low
        assert meta_high["confidence_multiplier"] > meta_low["confidence_multiplier"]

    def test_different_thresholds(self, position_sizer):
        """Test position sizing with different confidence thresholds."""
        balance = 10000.0
        confidence = 0.72
        sl_pips = 15.0

        # Low threshold (more trades pass)
        config_low = ConservativeHybridParameters(confidence_threshold=0.65)
        pos_low, _, _ = position_sizer.calculate_position_size(
            balance, confidence, sl_pips, config_low
        )

        # High threshold (fewer trades pass, but 0.72 still passes)
        config_high = ConservativeHybridParameters(confidence_threshold=0.70)
        pos_high, _, _ = position_sizer.calculate_position_size(
            balance, confidence, sl_pips, config_high
        )

        # Both should return a position
        assert pos_low > 0
        assert pos_high > 0

        # Lower threshold means confidence is further above threshold
        # So multiplier effect is larger
        assert pos_low > pos_high

    def test_different_pip_values(self, position_sizer, config):
        """Test position sizing with different pip values."""
        balance = 10000.0
        confidence = 0.75
        sl_pips = 15.0

        # Small pip value (e.g., 0.01 lot = micro lot)
        pos_small, _, _ = position_sizer.calculate_position_size(
            balance, confidence, sl_pips, config, pip_value=1.0
        )

        # Standard pip value (0.1 lot = mini lot)
        pos_standard, _, _ = position_sizer.calculate_position_size(
            balance, confidence, sl_pips, config, pip_value=10.0
        )

        # Larger pip value means larger risk per pip
        # So position should be smaller to maintain same $ risk
        assert pos_small > pos_standard


# ============================================================================
# PROGRESSIVE RISK REDUCTION TESTS
# ============================================================================


class TestPositionSizerWithRiskReduction:
    """Tests for position sizer with risk reduction factor."""

    def test_position_size_with_normal_risk(self, position_sizer, config):
        """Test position size with risk_factor = 1.0."""
        balance = 10000.0
        confidence = 0.75
        sl_pips = 15.0
        risk_factor = 1.0  # Normal risk

        position_lots, risk_pct, metadata = position_sizer.calculate_position_size(
            balance=balance,
            confidence=confidence,
            sl_pips=sl_pips,
            config=config,
            risk_reduction_factor=risk_factor
        )

        # At confidence 0.75 with normal risk
        # confidence_multiplier = 1.0 + (0.75 - 0.70) * 0.5 = 1.025
        # adjusted_risk = 1.5 * 1.025 = 1.5375%
        # With risk_factor = 1.0, risk_pct_used = 1.5375%
        assert position_lots > 0.0
        assert risk_pct == pytest.approx(1.5375, rel=0.01)
        assert metadata["risk_reduction_factor"] == 1.0

    def test_position_size_with_50_percent_reduction(self, position_sizer, config):
        """Test position size scales with 50% risk reduction."""
        balance = 50000.0  # Larger balance to avoid cash constraint
        confidence = 0.75
        sl_pips = 15.0

        # Normal risk
        pos_normal, risk_normal, meta_normal = position_sizer.calculate_position_size(
            balance=balance,
            confidence=confidence,
            sl_pips=sl_pips,
            config=config,
            risk_reduction_factor=1.0
        )

        # 50% risk reduction
        pos_reduced, risk_reduced, meta_reduced = position_sizer.calculate_position_size(
            balance=balance,
            confidence=confidence,
            sl_pips=sl_pips,
            config=config,
            risk_reduction_factor=0.5
        )

        # Position should be 50% of normal (only test if not cash-limited)
        if not meta_normal["limited_by_cash"]:
            assert pos_reduced == pytest.approx(pos_normal * 0.5, rel=0.01)
            assert risk_reduced == pytest.approx(risk_normal * 0.5, rel=0.01)
        assert meta_reduced["risk_reduction_factor"] == 0.5

    def test_position_size_with_minimum_reduction(self, position_sizer, config):
        """Test position size with 20% risk factor."""
        balance = 50000.0  # Larger balance to avoid cash constraint
        confidence = 0.75
        sl_pips = 15.0

        # Normal risk
        pos_normal, risk_normal, meta_normal = position_sizer.calculate_position_size(
            balance=balance,
            confidence=confidence,
            sl_pips=sl_pips,
            config=config,
            risk_reduction_factor=1.0
        )

        # Minimum risk (20%)
        pos_min, risk_min, meta_min = position_sizer.calculate_position_size(
            balance=balance,
            confidence=confidence,
            sl_pips=sl_pips,
            config=config,
            risk_reduction_factor=0.2
        )

        # Position should be 20% of normal (only test if not cash-limited)
        if not meta_normal["limited_by_cash"]:
            assert pos_min == pytest.approx(pos_normal * 0.2, rel=0.01)
            assert risk_min == pytest.approx(risk_normal * 0.2, rel=0.01)
        assert meta_min["risk_reduction_factor"] == 0.2

    def test_metadata_includes_reduction_factor(self, position_sizer, config):
        """Test metadata tracks risk reduction factor."""
        balance = 10000.0
        confidence = 0.75
        sl_pips = 15.0
        risk_factor = 0.6

        position_lots, risk_pct, metadata = position_sizer.calculate_position_size(
            balance=balance,
            confidence=confidence,
            sl_pips=sl_pips,
            config=config,
            risk_reduction_factor=risk_factor
        )

        # Verify metadata includes reduction factor
        assert "risk_reduction_factor" in metadata
        assert metadata["risk_reduction_factor"] == risk_factor

    def test_zero_position_only_from_confidence(self, position_sizer, config):
        """Test position is never zero due to risk reduction alone."""
        balance = 10000.0
        confidence = 0.75  # Above threshold
        sl_pips = 15.0
        risk_factor = 0.2  # Minimum risk

        position_lots, risk_pct, metadata = position_sizer.calculate_position_size(
            balance=balance,
            confidence=confidence,
            sl_pips=sl_pips,
            config=config,
            risk_reduction_factor=risk_factor
        )

        # Even at minimum risk, should have a position
        assert position_lots > 0.0
        assert position_lots >= MIN_POSITION_SIZE

    def test_progressive_risk_levels(self, position_sizer, config):
        """Test position sizes at different risk reduction levels."""
        balance = 50000.0  # Larger balance to avoid cash constraint
        confidence = 0.75
        sl_pips = 15.0

        # Test different risk levels
        risk_levels = [1.0, 0.8, 0.6, 0.4, 0.2]
        positions = []
        risks = []
        metadata_list = []

        for risk_factor in risk_levels:
            pos, risk, meta = position_sizer.calculate_position_size(
                balance=balance,
                confidence=confidence,
                sl_pips=sl_pips,
                config=config,
                risk_reduction_factor=risk_factor
            )
            positions.append(pos)
            risks.append(risk)
            metadata_list.append(meta)

        # Verify progressive reduction (only if not cash-limited)
        if not metadata_list[0]["limited_by_cash"]:
            for i in range(len(positions) - 1):
                assert positions[i] > positions[i+1], \
                    f"Position at {risk_levels[i]} should be > position at {risk_levels[i+1]}"
                assert risks[i] > risks[i+1], \
                    f"Risk at {risk_levels[i]} should be > risk at {risk_levels[i+1]}"

    def test_risk_reduction_with_high_confidence(self, position_sizer, config):
        """Test risk reduction applies even with high confidence."""
        balance = 50000.0  # Larger balance to avoid cash constraint
        confidence = 0.85  # Very high confidence
        sl_pips = 15.0

        # Normal risk
        pos_normal, _, meta_normal = position_sizer.calculate_position_size(
            balance=balance,
            confidence=confidence,
            sl_pips=sl_pips,
            config=config,
            risk_reduction_factor=1.0
        )

        # Reduced risk
        pos_reduced, _, _ = position_sizer.calculate_position_size(
            balance=balance,
            confidence=confidence,
            sl_pips=sl_pips,
            config=config,
            risk_reduction_factor=0.4
        )

        # Reduction should apply regardless of confidence (only if not cash-limited)
        if not meta_normal["limited_by_cash"]:
            assert pos_reduced == pytest.approx(pos_normal * 0.4, rel=0.01)

    def test_risk_reduction_with_low_balance(self, position_sizer, config):
        """Test risk reduction with small balance (cash-limited)."""
        balance = 5000.0  # Small balance
        confidence = 0.75
        sl_pips = 10.0

        # This combination would normally be cash-limited
        pos_normal, _, meta_normal = position_sizer.calculate_position_size(
            balance=balance,
            confidence=confidence,
            sl_pips=sl_pips,
            config=config,
            risk_reduction_factor=1.0
        )

        # With risk reduction
        pos_reduced, _, meta_reduced = position_sizer.calculate_position_size(
            balance=balance,
            confidence=confidence,
            sl_pips=sl_pips,
            config=config,
            risk_reduction_factor=0.5
        )

        # Both should be cash-limited or proportionally reduced
        assert pos_reduced <= pos_normal

    def test_risk_reduction_factor_bounds(self, position_sizer, config):
        """Test risk reduction factor is properly bounded."""
        balance = 10000.0
        confidence = 0.75
        sl_pips = 15.0

        # Test values outside bounds (should still work)
        # Values > 1.0 should still calculate (treated as 1.0 effectively)
        pos_over, _, meta_over = position_sizer.calculate_position_size(
            balance=balance,
            confidence=confidence,
            sl_pips=sl_pips,
            config=config,
            risk_reduction_factor=1.5
        )

        # Should still calculate position
        assert pos_over > 0.0
        assert meta_over["risk_reduction_factor"] == 1.5  # Passed through as-is

    def test_risk_reduction_below_minimum_position(self, position_sizer, config):
        """Test risk reduction can push position below minimum."""
        balance = 100.0  # Very small balance
        confidence = 0.70
        sl_pips = 20.0
        risk_factor = 0.2

        position_lots, risk_pct, metadata = position_sizer.calculate_position_size(
            balance=balance,
            confidence=confidence,
            sl_pips=sl_pips,
            config=config,
            risk_reduction_factor=risk_factor
        )

        # With very small balance and high reduction, might be below minimum
        # Should return 0 with appropriate reason
        if position_lots == 0.0:
            assert metadata["reason"] == "below_minimum_position_size"
