"""Unit tests for Walk-Forward Optimization risk management.

This test suite validates both Tier 1 and Tier 2 risk management functionality:

TIER 1 - Drawdown Mitigation:
- get_drawdown_position_multiplier: Progressive position reduction (0-15% drawdown)
- Normal operation across all drawdown levels (0-5%, 5-7.5%, 7.5-10%, 10-15%, 15%+)
- Boundary conditions, edge cases, custom max_allowed parameter
- Circuit breaker integration (halts trading at 15% drawdown)

TIER 2 - Advanced Risk Controls:
- calculate_volatility_adjusted_risk: Inverse volatility scaling (high vol → smaller size)
- equity_curve_filter: Equity MA filter (reduce size when below MA)
- get_regime_position_multiplier: Market regime adjustment (skip/reduce/full size)

INTEGRATION TESTS:
- Tier 2 multipliers compounding with each other
- Tier 1 + Tier 2 full compounding (worst case scenarios)
- Regime skip override (ranging_high_vol blocks all trading)
"""

import pytest
import sys
from pathlib import Path

# Add scripts directory to path for import
scripts_dir = Path(__file__).parent.parent.parent.parent / "scripts"
sys.path.insert(0, str(scripts_dir))

from walk_forward_optimization import (
    get_drawdown_position_multiplier,
    calculate_volatility_adjusted_risk,
    equity_curve_filter,
    get_regime_position_multiplier,
)


class TestGetDrawdownPositionMultiplier:
    """Tests for the drawdown position multiplier function.

    This function implements Tier 1 drawdown mitigation by progressively
    reducing position size as drawdown increases, ultimately halting trading
    at the maximum allowed drawdown (default 15%).
    """

    # ========================================================================
    # LEVEL 1: No Drawdown / Minimal Drawdown (0-5%) - Full Position Size
    # ========================================================================

    def test_zero_drawdown_returns_full_size(self):
        """Test that 0% drawdown returns 1.0x multiplier (full position)."""
        result = get_drawdown_position_multiplier(0.0)
        assert result == 1.0

    def test_small_drawdown_returns_full_size(self):
        """Test that drawdown < 5% returns 1.0x multiplier."""
        test_cases = [0.01, 0.02, 0.03, 0.04, 0.049]  # 1%, 2%, 3%, 4%, 4.9%

        for dd in test_cases:
            result = get_drawdown_position_multiplier(dd)
            assert result == 1.0, f"Expected 1.0 for {dd:.1%} drawdown, got {result}"

    def test_boundary_below_5_percent(self):
        """Test value just below 5% threshold returns full size."""
        result = get_drawdown_position_multiplier(0.0499)  # 4.99%
        assert result == 1.0

    # ========================================================================
    # LEVEL 2: Moderate Drawdown (5-7.5%) - Reduced Position (75%)
    # ========================================================================

    def test_exact_5_percent_drawdown(self):
        """Test that exactly 5% drawdown returns 0.75x multiplier."""
        result = get_drawdown_position_multiplier(0.05)
        assert result == 0.75

    def test_moderate_drawdown_returns_75_percent(self):
        """Test that 5-7.5% drawdown returns 0.75x multiplier."""
        test_cases = [0.05, 0.06, 0.07, 0.074]  # 5%, 6%, 7%, 7.4%

        for dd in test_cases:
            result = get_drawdown_position_multiplier(dd)
            assert result == 0.75, f"Expected 0.75 for {dd:.1%} drawdown, got {result}"

    def test_boundary_below_7_5_percent(self):
        """Test value just below 7.5% threshold returns 0.75x."""
        result = get_drawdown_position_multiplier(0.0749)  # 7.49%
        assert result == 0.75

    # ========================================================================
    # LEVEL 3: High Drawdown (7.5-10%) - Reduced Position (50%)
    # ========================================================================

    def test_exact_7_5_percent_drawdown(self):
        """Test that exactly 7.5% drawdown returns 0.50x multiplier."""
        result = get_drawdown_position_multiplier(0.075)
        assert result == 0.50

    def test_high_drawdown_returns_50_percent(self):
        """Test that 7.5-10% drawdown returns 0.50x multiplier."""
        test_cases = [0.075, 0.08, 0.09, 0.099]  # 7.5%, 8%, 9%, 9.9%

        for dd in test_cases:
            result = get_drawdown_position_multiplier(dd)
            assert result == 0.50, f"Expected 0.50 for {dd:.1%} drawdown, got {result}"

    def test_boundary_below_10_percent(self):
        """Test value just below 10% threshold returns 0.50x."""
        result = get_drawdown_position_multiplier(0.0999)  # 9.99%
        assert result == 0.50

    # ========================================================================
    # LEVEL 4: Critical Drawdown (10-15%) - Minimal Position (25%)
    # ========================================================================

    def test_exact_10_percent_drawdown(self):
        """Test that exactly 10% drawdown returns 0.25x multiplier."""
        result = get_drawdown_position_multiplier(0.10)
        assert result == 0.25

    def test_critical_drawdown_returns_25_percent(self):
        """Test that 10-15% drawdown returns 0.25x multiplier."""
        test_cases = [0.10, 0.11, 0.12, 0.13, 0.14, 0.149]

        for dd in test_cases:
            result = get_drawdown_position_multiplier(dd)
            assert result == 0.25, f"Expected 0.25 for {dd:.1%} drawdown, got {result}"

    def test_boundary_below_15_percent(self):
        """Test value just below 15% threshold returns 0.25x."""
        result = get_drawdown_position_multiplier(0.1499)  # 14.99%
        assert result == 0.25

    # ========================================================================
    # LEVEL 5: Maximum Drawdown (15%+) - Circuit Breaker (0%)
    # ========================================================================

    def test_exact_15_percent_drawdown_halts_trading(self):
        """Test that exactly 15% drawdown returns 0.0x (circuit breaker)."""
        result = get_drawdown_position_multiplier(0.15)
        assert result == 0.0

    def test_extreme_drawdown_halts_trading(self):
        """Test that drawdown >= 15% returns 0.0x (circuit breaker)."""
        test_cases = [0.15, 0.16, 0.20, 0.30, 0.50, 0.99]

        for dd in test_cases:
            result = get_drawdown_position_multiplier(dd)
            assert result == 0.0, f"Expected 0.0 for {dd:.1%} drawdown, got {result}"

    def test_boundary_at_15_percent(self):
        """Test value just at 15% threshold returns 0.0x."""
        result = get_drawdown_position_multiplier(0.1500)
        assert result == 0.0

    # ========================================================================
    # EDGE CASES
    # ========================================================================

    def test_negative_drawdown_returns_full_size(self):
        """Test that negative drawdown (gain) returns 1.0x multiplier.

        Negative drawdown means the account is above the previous peak.
        This should allow full position sizing.
        """
        result = get_drawdown_position_multiplier(-0.05)
        assert result == 1.0

    def test_very_large_drawdown(self):
        """Test that extremely large drawdown returns 0.0x."""
        result = get_drawdown_position_multiplier(1.0)  # 100% drawdown
        assert result == 0.0

    def test_small_positive_drawdown(self):
        """Test that very small positive drawdown returns full size."""
        result = get_drawdown_position_multiplier(0.001)  # 0.1%
        assert result == 1.0

    # ========================================================================
    # CUSTOM MAX_ALLOWED PARAMETER
    # ========================================================================

    def test_custom_max_allowed_10_percent(self):
        """Test custom max_allowed parameter (10% instead of 15%).

        Note: The function uses hardcoded thresholds (5%, 7.5%, 10%) for the
        intermediate levels. The max_allowed parameter only controls when the
        circuit breaker (0.0x multiplier) activates.
        """
        # At 9% drawdown with 10% max: still returns 0.50x (7.5-10% range)
        # because the hardcoded 10% threshold takes precedence
        result = get_drawdown_position_multiplier(0.09, max_allowed=0.10)
        assert result == 0.50

        # At 10% drawdown with 10% max: should return 0.0x (circuit breaker)
        result = get_drawdown_position_multiplier(0.10, max_allowed=0.10)
        assert result == 0.0

    def test_custom_max_allowed_20_percent(self):
        """Test custom max_allowed parameter (20% instead of 15%)."""
        # At 15% drawdown with 20% max: should return 0.25x (still in 10-max range)
        result = get_drawdown_position_multiplier(0.15, max_allowed=0.20)
        assert result == 0.25

        # At 20% drawdown with 20% max: should return 0.0x (circuit breaker)
        result = get_drawdown_position_multiplier(0.20, max_allowed=0.20)
        assert result == 0.0

    def test_custom_max_allowed_5_percent(self):
        """Test very conservative max_allowed parameter (5%).

        Note: The function uses hardcoded thresholds (5%, 7.5%, 10%) for the
        intermediate levels. When max_allowed=5%, drawdown of exactly 5% still
        hits the 5-7.5% bucket (0.75x) before checking against max_allowed.
        """
        # At 3% drawdown with 5% max: should return 1.0x (below 5%)
        result = get_drawdown_position_multiplier(0.03, max_allowed=0.05)
        assert result == 1.0

        # At 5% drawdown with 5% max: hits the 5-7.5% bucket first (0.75x)
        # The circuit breaker check (< max_allowed) comes later in the if-elif chain
        result = get_drawdown_position_multiplier(0.05, max_allowed=0.05)
        assert result == 0.75

        # At 6% drawdown with 5% max: still returns 0.75x (5-7.5% bucket)
        # The hardcoded thresholds take precedence over max_allowed
        # Only when DD is >= 10% does max_allowed matter (10-max_allowed range → 0.25x)
        result = get_drawdown_position_multiplier(0.06, max_allowed=0.05)
        assert result == 0.75

        # At 10% drawdown with 5% max: now in the 10-max range check, but
        # since 10% >= 5% max_allowed, circuit breaker activates (0.0x)
        result = get_drawdown_position_multiplier(0.10, max_allowed=0.05)
        assert result == 0.0

    # ========================================================================
    # BOUNDARY VALUE ANALYSIS
    # ========================================================================

    def test_all_exact_boundary_values(self):
        """Test all exact boundary values for default max_allowed=0.15."""
        boundaries = {
            0.0: 1.0,      # 0% drawdown
            0.04999: 1.0,  # Just below 5%
            0.05: 0.75,    # Exactly 5%
            0.07499: 0.75, # Just below 7.5%
            0.075: 0.50,   # Exactly 7.5%
            0.09999: 0.50, # Just below 10%
            0.10: 0.25,    # Exactly 10%
            0.14999: 0.25, # Just below 15%
            0.15: 0.0,     # Exactly 15% (circuit breaker)
        }

        for dd, expected in boundaries.items():
            result = get_drawdown_position_multiplier(dd)
            assert result == expected, (
                f"Expected {expected} for {dd:.5f} drawdown, got {result}"
            )

    # ========================================================================
    # INTEGRATION WITH CIRCUIT BREAKER
    # ========================================================================

    def test_circuit_breaker_activates_at_max_drawdown(self):
        """Test that circuit breaker (0.0x multiplier) activates at max_allowed."""
        # Default max_allowed = 0.15
        result = get_drawdown_position_multiplier(0.15)
        assert result == 0.0, "Circuit breaker should activate at 15% drawdown"

    def test_circuit_breaker_stays_active_above_max(self):
        """Test that circuit breaker stays active for any drawdown >= max_allowed."""
        test_cases = [0.15, 0.16, 0.20, 0.25, 0.30]

        for dd in test_cases:
            result = get_drawdown_position_multiplier(dd)
            assert result == 0.0, (
                f"Circuit breaker should remain active at {dd:.1%} drawdown"
            )

    def test_circuit_breaker_not_active_below_max(self):
        """Test that circuit breaker is not active below max_allowed."""
        # Just below 15%
        result = get_drawdown_position_multiplier(0.1499)
        assert result > 0.0, "Circuit breaker should not be active below 15%"
        assert result == 0.25, "Should return minimal position (0.25x) just below max"

    # ========================================================================
    # RETURN TYPE VALIDATION
    # ========================================================================

    def test_return_type_is_float(self):
        """Test that function always returns float type."""
        test_cases = [0.0, 0.05, 0.10, 0.15, 0.20]

        for dd in test_cases:
            result = get_drawdown_position_multiplier(dd)
            assert isinstance(result, float), (
                f"Expected float return type, got {type(result)}"
            )

    def test_return_value_range(self):
        """Test that return value is always between 0.0 and 1.0."""
        test_cases = [-0.1, 0.0, 0.03, 0.06, 0.08, 0.12, 0.15, 0.20, 0.50]

        for dd in test_cases:
            result = get_drawdown_position_multiplier(dd)
            assert 0.0 <= result <= 1.0, (
                f"Expected multiplier in [0.0, 1.0] range, got {result} for {dd:.1%} drawdown"
            )

    def test_return_values_are_expected_levels(self):
        """Test that return values are only the expected discrete levels."""
        valid_levels = {0.0, 0.25, 0.50, 0.75, 1.0}

        # Test a wide range of drawdown values
        test_cases = [i/100 for i in range(0, 30)]  # 0% to 29%

        for dd in test_cases:
            result = get_drawdown_position_multiplier(dd)
            assert result in valid_levels, (
                f"Expected one of {valid_levels}, got {result} for {dd:.1%} drawdown"
            )


class TestDrawdownPositionMultiplierCompounding:
    """Tests for compounding behavior with other risk reduction mechanisms.

    The drawdown multiplier is designed to compound with consecutive loss
    reduction. For example:
    - 2 consecutive losses: 50% risk
    - 8% drawdown: 50% multiplier
    - Effective risk: 50% × 50% = 25% of base risk
    """

    def test_compounding_with_consecutive_loss_reduction(self):
        """Test example compounding calculation from docstring.

        Example: 2 consecutive losses (50% risk) + 8% drawdown (50% multiplier)
        = 25% of base risk (0.5 × 0.5 = 0.25)
        """
        # Get drawdown multiplier for 8% drawdown
        dd_multiplier = get_drawdown_position_multiplier(0.08)
        assert dd_multiplier == 0.50

        # Simulate consecutive loss reduction (2 losses = 50% risk)
        consecutive_loss_multiplier = 0.50

        # Compound the two multipliers
        effective_risk = dd_multiplier * consecutive_loss_multiplier
        assert effective_risk == 0.25, (
            f"Expected 25% effective risk, got {effective_risk:.1%}"
        )

    def test_no_compounding_at_zero_drawdown(self):
        """Test that compounding has no effect at 0% drawdown (1.0x multiplier)."""
        dd_multiplier = get_drawdown_position_multiplier(0.0)
        assert dd_multiplier == 1.0

        # With any consecutive loss multiplier
        consecutive_loss_multiplier = 0.50
        effective_risk = dd_multiplier * consecutive_loss_multiplier

        # Effective risk equals consecutive loss multiplier (no reduction from DD)
        assert effective_risk == consecutive_loss_multiplier

    def test_compounding_at_circuit_breaker(self):
        """Test that circuit breaker (0.0x) makes effective risk zero regardless of other factors."""
        dd_multiplier = get_drawdown_position_multiplier(0.15)
        assert dd_multiplier == 0.0

        # Even with full consecutive loss multiplier
        consecutive_loss_multiplier = 1.0
        effective_risk = dd_multiplier * consecutive_loss_multiplier

        # Effective risk is zero (no trading)
        assert effective_risk == 0.0

    def test_aggressive_compounding_scenario(self):
        """Test aggressive capital preservation in worst-case scenario.

        Scenario: 3 consecutive losses (25% risk) + 12% drawdown (25% multiplier)
        = 6.25% of base risk (0.25 × 0.25 = 0.0625)
        """
        dd_multiplier = get_drawdown_position_multiplier(0.12)
        assert dd_multiplier == 0.25

        # Simulate 3 consecutive losses (25% risk)
        consecutive_loss_multiplier = 0.25

        # Compound the two multipliers
        effective_risk = dd_multiplier * consecutive_loss_multiplier
        assert effective_risk == 0.0625, (
            f"Expected 6.25% effective risk, got {effective_risk:.1%}"
        )


class TestDrawdownPositionMultiplierDocumentation:
    """Tests that verify the function behaves as documented in its docstring."""

    def test_docstring_example_3_percent(self):
        """Test example from docstring: 3% drawdown returns 1.0."""
        result = get_drawdown_position_multiplier(0.03)
        assert result == 1.0

    def test_docstring_example_8_percent(self):
        """Test example from docstring: 8% drawdown returns 0.5."""
        result = get_drawdown_position_multiplier(0.08)
        assert result == 0.5

    def test_docstring_example_16_percent(self):
        """Test example from docstring: 16% drawdown returns 0.0."""
        result = get_drawdown_position_multiplier(0.16)
        assert result == 0.0

    def test_docstring_level_descriptions(self):
        """Test all levels as described in docstring."""
        levels = {
            # 0-5% DD: Full size (1.0x)
            0.03: 1.0,
            # 5-7.5% DD: 75% size (0.75x)
            0.06: 0.75,
            # 7.5-10% DD: 50% size (0.50x)
            0.085: 0.50,
            # 10-15% DD: 25% size (0.25x)
            0.12: 0.25,
            # 15%+ DD: No trading (0x)
            0.20: 0.0,
        }

        for dd, expected in levels.items():
            result = get_drawdown_position_multiplier(dd)
            assert result == expected, (
                f"Expected {expected} for {dd:.1%} drawdown, got {result}"
            )


class TestDrawdownPositionMultiplierParameterValidation:
    """Tests for parameter validation and type handling."""

    def test_accepts_float_input(self):
        """Test that function accepts float input."""
        result = get_drawdown_position_multiplier(0.05)
        assert isinstance(result, float)

    def test_accepts_int_input(self):
        """Test that function accepts int input (gets converted to float)."""
        result = get_drawdown_position_multiplier(0)  # int 0
        assert isinstance(result, float)
        assert result == 1.0

    def test_max_allowed_parameter_default(self):
        """Test that default max_allowed is 0.15 (15%)."""
        # At 15% with default max_allowed, should trigger circuit breaker
        result = get_drawdown_position_multiplier(0.15)
        assert result == 0.0

    def test_max_allowed_parameter_custom(self):
        """Test that custom max_allowed overrides default."""
        # At 15% with custom max_allowed=0.20, should return 0.25x
        result = get_drawdown_position_multiplier(0.15, max_allowed=0.20)
        assert result == 0.25


# ========================================================================
# PYTEST CONFIGURATION
# ========================================================================

@pytest.fixture
def sample_drawdowns():
    """Fixture providing sample drawdown values for testing."""
    return {
        "no_drawdown": 0.0,
        "minimal": 0.02,  # 2%
        "moderate": 0.06,  # 6%
        "high": 0.08,  # 8%
        "critical": 0.12,  # 12%
        "circuit_breaker": 0.15,  # 15%
        "extreme": 0.30,  # 30%
    }


@pytest.fixture
def expected_multipliers():
    """Fixture providing expected multipliers for each drawdown level."""
    return {
        "no_drawdown": 1.0,
        "minimal": 1.0,
        "moderate": 0.75,
        "high": 0.50,
        "critical": 0.25,
        "circuit_breaker": 0.0,
        "extreme": 0.0,
    }


class TestDrawdownPositionMultiplierWithFixtures:
    """Tests using pytest fixtures for common test data."""

    def test_all_levels_with_fixtures(self, sample_drawdowns, expected_multipliers):
        """Test all drawdown levels using fixtures."""
        for level, dd in sample_drawdowns.items():
            result = get_drawdown_position_multiplier(dd)
            expected = expected_multipliers[level]
            assert result == expected, (
                f"Level '{level}' ({dd:.1%}): expected {expected}, got {result}"
            )


# ========================================================================
# PARAMETRIZED TESTS
# ========================================================================

@pytest.mark.parametrize("drawdown,expected", [
    (0.0, 1.0),      # No drawdown
    (0.01, 1.0),     # 1% drawdown
    (0.04, 1.0),     # 4% drawdown
    (0.05, 0.75),    # 5% drawdown
    (0.06, 0.75),    # 6% drawdown
    (0.07, 0.75),    # 7% drawdown
    (0.075, 0.50),   # 7.5% drawdown
    (0.08, 0.50),    # 8% drawdown
    (0.09, 0.50),    # 9% drawdown
    (0.10, 0.25),    # 10% drawdown
    (0.12, 0.25),    # 12% drawdown
    (0.14, 0.25),    # 14% drawdown
    (0.15, 0.0),     # 15% drawdown (circuit breaker)
    (0.20, 0.0),     # 20% drawdown
    (0.50, 0.0),     # 50% drawdown
])
def test_drawdown_multiplier_parametrized(drawdown, expected):
    """Parametrized test for all drawdown levels."""
    result = get_drawdown_position_multiplier(drawdown)
    assert result == expected, (
        f"Drawdown {drawdown:.1%}: expected {expected}, got {result}"
    )


@pytest.mark.parametrize("drawdown,max_allowed,expected", [
    (0.05, 0.10, 0.75),   # 5% DD, 10% max → 0.75x (hardcoded 5-7.5% bucket)
    (0.10, 0.10, 0.0),    # 10% DD, 10% max → 0.0x (circuit breaker, DD >= max_allowed)
    (0.05, 0.20, 0.75),   # 5% DD, 20% max → 0.75x (hardcoded 5-7.5% bucket)
    (0.15, 0.20, 0.25),   # 15% DD, 20% max → 0.25x (10-max range)
    (0.20, 0.20, 0.0),    # 20% DD, 20% max → 0.0x (circuit breaker, DD >= max_allowed)
    (0.03, 0.05, 1.0),    # 3% DD, 5% max → 1.0x (below 5%)
    (0.05, 0.05, 0.75),   # 5% DD, 5% max → 0.75x (hardcoded 5-7.5% bucket takes precedence)
    (0.06, 0.05, 0.75),   # 6% DD, 5% max → 0.75x (hardcoded 5-7.5% bucket takes precedence)
    (0.10, 0.05, 0.0),    # 10% DD, 5% max → 0.0x (circuit breaker, DD >= max_allowed in final check)
])
def test_custom_max_allowed_parametrized(drawdown, max_allowed, expected):
    """Parametrized test for custom max_allowed values.

    Note: The function uses hardcoded thresholds (5%, 7.5%, 10%) for
    intermediate levels. The max_allowed parameter only affects the
    final check: if DD < max_allowed return 0.25x, else return 0.0x.
    This means max_allowed only matters for drawdowns >= 10%.
    """
    result = get_drawdown_position_multiplier(drawdown, max_allowed)
    assert result == expected, (
        f"Drawdown {drawdown:.1%}, max {max_allowed:.1%}: expected {expected}, got {result}"
    )


# ========================================================================
# PERFORMANCE TESTS (Optional)
# ========================================================================

class TestDrawdownPositionMultiplierPerformance:
    """Optional performance tests to ensure function efficiency."""

    def test_function_is_fast(self):
        """Test that function executes quickly (< 1ms for 10000 calls)."""
        import time

        start = time.perf_counter()
        for _ in range(10000):
            get_drawdown_position_multiplier(0.08)
        duration = time.perf_counter() - start

        # Should complete 10000 calls in less than 100ms
        assert duration < 0.1, f"Function too slow: {duration:.3f}s for 10000 calls"

    def test_function_has_no_side_effects(self):
        """Test that function has no side effects (pure function)."""
        drawdown = 0.08

        # Call multiple times with same input
        result1 = get_drawdown_position_multiplier(drawdown)
        result2 = get_drawdown_position_multiplier(drawdown)
        result3 = get_drawdown_position_multiplier(drawdown)

        # Results should be identical (no state changes)
        assert result1 == result2 == result3


# ========================================================================
# TIER 2 FUNCTION TESTS
# ========================================================================


class TestCalculateVolatilityAdjustedRisk:
    """Tests for volatility-adjusted risk calculation.

    This function implements Tier 2 risk management by scaling position size
    inversely with volatility:
    - High volatility → reduce position size (maintain constant dollar risk)
    - Low volatility → increase position size (capture opportunity)
    """

    # ========================================================================
    # NORMAL VOLATILITY CASES
    # ========================================================================

    def test_normal_volatility_returns_base_risk(self):
        """Test that equal current and average ATR returns base risk."""
        base_risk = 0.02  # 2%
        current_atr = 50.0
        avg_atr = 50.0

        result = calculate_volatility_adjusted_risk(base_risk, current_atr, avg_atr)
        assert result == base_risk

    def test_multiple_base_risk_values_at_normal_vol(self):
        """Test different base risk values at normal volatility."""
        test_cases = [0.01, 0.015, 0.02, 0.025, 0.03]

        for base_risk in test_cases:
            result = calculate_volatility_adjusted_risk(base_risk, 50.0, 50.0)
            assert result == base_risk, f"Expected {base_risk} at normal vol, got {result}"

    # ========================================================================
    # HIGH VOLATILITY CASES
    # ========================================================================

    def test_high_volatility_reduces_risk(self):
        """Test that high volatility reduces position size."""
        base_risk = 0.02
        current_atr = 100.0  # Double average
        avg_atr = 50.0

        result = calculate_volatility_adjusted_risk(base_risk, current_atr, avg_atr)
        # vol_ratio = 50/100 = 0.5, so result = 0.02 * 0.5 = 0.01
        assert result == 0.01

    def test_very_high_volatility_hits_minimum(self):
        """Test that very high volatility is clamped at minimum multiplier."""
        base_risk = 0.02
        current_atr = 200.0  # 4x average
        avg_atr = 50.0
        min_multiplier = 0.25

        result = calculate_volatility_adjusted_risk(base_risk, current_atr, avg_atr, min_multiplier=min_multiplier)
        # vol_ratio = 50/200 = 0.25, exactly at min
        assert result == base_risk * min_multiplier
        assert result == 0.005

    def test_extreme_volatility_clamped_at_minimum(self):
        """Test that extreme volatility is clamped at minimum."""
        base_risk = 0.02
        current_atr = 1000.0  # 10x average
        avg_atr = 100.0
        min_multiplier = 0.25

        result = calculate_volatility_adjusted_risk(base_risk, current_atr, avg_atr, min_multiplier=min_multiplier)
        # vol_ratio = 100/1000 = 0.1, but clamped to 0.25
        assert result == base_risk * min_multiplier
        assert result == 0.005

    # ========================================================================
    # LOW VOLATILITY CASES
    # ========================================================================

    def test_low_volatility_increases_risk(self):
        """Test that low volatility increases position size."""
        base_risk = 0.02
        current_atr = 50.0
        avg_atr = 100.0  # Half of average

        result = calculate_volatility_adjusted_risk(base_risk, current_atr, avg_atr)
        # vol_ratio = 100/50 = 2.0, but capped at max_multiplier (1.5 default)
        assert result == base_risk * 1.5  # Capped at max
        assert result == 0.03

    def test_very_low_volatility_hits_maximum(self):
        """Test that very low volatility is clamped at maximum multiplier."""
        base_risk = 0.02
        current_atr = 25.0
        avg_atr = 100.0  # 4x current
        max_multiplier = 1.5

        result = calculate_volatility_adjusted_risk(base_risk, current_atr, avg_atr, max_multiplier=max_multiplier)
        # vol_ratio = 100/25 = 4.0, but clamped to 1.5
        assert result == base_risk * max_multiplier
        assert result == 0.03

    def test_low_volatility_within_max(self):
        """Test low volatility that doesn't hit maximum."""
        base_risk = 0.02
        current_atr = 80.0
        avg_atr = 100.0

        result = calculate_volatility_adjusted_risk(base_risk, current_atr, avg_atr)
        # vol_ratio = 100/80 = 1.25 (below max of 1.5)
        assert result == 0.025

    # ========================================================================
    # EDGE CASES
    # ========================================================================

    def test_zero_current_atr_returns_base_risk(self):
        """Test that zero current ATR returns base risk (safety check)."""
        base_risk = 0.02
        current_atr = 0.0
        avg_atr = 50.0

        result = calculate_volatility_adjusted_risk(base_risk, current_atr, avg_atr)
        assert result == base_risk

    def test_zero_avg_atr_returns_base_risk(self):
        """Test that zero average ATR returns base risk (safety check)."""
        base_risk = 0.02
        current_atr = 50.0
        avg_atr = 0.0

        result = calculate_volatility_adjusted_risk(base_risk, current_atr, avg_atr)
        assert result == base_risk

    def test_both_zero_atr_returns_base_risk(self):
        """Test that zero for both ATR values returns base risk."""
        base_risk = 0.02

        result = calculate_volatility_adjusted_risk(base_risk, 0.0, 0.0)
        assert result == base_risk

    def test_negative_current_atr_returns_base_risk(self):
        """Test that negative current ATR returns base risk."""
        base_risk = 0.02
        current_atr = -50.0
        avg_atr = 50.0

        result = calculate_volatility_adjusted_risk(base_risk, current_atr, avg_atr)
        assert result == base_risk

    def test_negative_avg_atr_returns_base_risk(self):
        """Test that negative average ATR returns base risk."""
        base_risk = 0.02
        current_atr = 50.0
        avg_atr = -50.0

        result = calculate_volatility_adjusted_risk(base_risk, current_atr, avg_atr)
        assert result == base_risk

    # ========================================================================
    # CUSTOM MIN/MAX MULTIPLIER PARAMETERS
    # ========================================================================

    def test_custom_min_multiplier(self):
        """Test custom minimum multiplier."""
        base_risk = 0.02
        current_atr = 400.0  # 4x average
        avg_atr = 100.0
        min_multiplier = 0.5  # Higher floor than default 0.25

        result = calculate_volatility_adjusted_risk(base_risk, current_atr, avg_atr, min_multiplier=min_multiplier)
        # vol_ratio = 100/400 = 0.25, but clamped to custom min 0.5
        assert result == base_risk * min_multiplier
        assert result == 0.01

    def test_custom_max_multiplier(self):
        """Test custom maximum multiplier."""
        base_risk = 0.02
        current_atr = 50.0
        avg_atr = 200.0  # 4x current
        max_multiplier = 2.0  # Higher ceiling than default 1.5

        result = calculate_volatility_adjusted_risk(base_risk, current_atr, avg_atr, max_multiplier=max_multiplier)
        # vol_ratio = 200/50 = 4.0, but clamped to custom max 2.0
        assert result == base_risk * max_multiplier
        assert result == 0.04

    def test_custom_min_and_max_multipliers(self):
        """Test custom both min and max multipliers."""
        base_risk = 0.02
        min_multiplier = 0.5
        max_multiplier = 2.0

        # Test at minimum
        result_min = calculate_volatility_adjusted_risk(
            base_risk, 400.0, 100.0, min_multiplier=min_multiplier, max_multiplier=max_multiplier
        )
        assert result_min == base_risk * min_multiplier

        # Test at maximum
        result_max = calculate_volatility_adjusted_risk(
            base_risk, 25.0, 100.0, min_multiplier=min_multiplier, max_multiplier=max_multiplier
        )
        assert result_max == base_risk * max_multiplier

    # ========================================================================
    # MIN/MAX CLAMPING VERIFICATION
    # ========================================================================

    def test_min_multiplier_clamping_works_correctly(self):
        """Test that minimum multiplier clamping works correctly."""
        base_risk = 0.02
        min_multiplier = 0.25

        # Test cases that should hit the minimum
        test_cases = [
            (400.0, 100.0),   # vol_ratio = 0.25 (exactly at min)
            (500.0, 100.0),   # vol_ratio = 0.20 (below min)
            (1000.0, 100.0),  # vol_ratio = 0.10 (well below min)
        ]

        for current_atr, avg_atr in test_cases:
            result = calculate_volatility_adjusted_risk(base_risk, current_atr, avg_atr, min_multiplier=min_multiplier)
            assert result == base_risk * min_multiplier, (
                f"Expected {base_risk * min_multiplier} for ATR {current_atr}/{avg_atr}, got {result}"
            )

    def test_max_multiplier_clamping_works_correctly(self):
        """Test that maximum multiplier clamping works correctly."""
        base_risk = 0.02
        max_multiplier = 1.5

        # Test cases that should hit the maximum
        test_cases = [
            (50.0, 75.0),    # vol_ratio = 1.5 (exactly at max)
            (50.0, 100.0),   # vol_ratio = 2.0 (above max)
            (25.0, 100.0),   # vol_ratio = 4.0 (well above max)
        ]

        for current_atr, avg_atr in test_cases:
            result = calculate_volatility_adjusted_risk(base_risk, current_atr, avg_atr, max_multiplier=max_multiplier)
            assert result == base_risk * max_multiplier, (
                f"Expected {base_risk * max_multiplier} for ATR {current_atr}/{avg_atr}, got {result}"
            )

    # ========================================================================
    # REALISTIC SCENARIOS
    # ========================================================================

    def test_realistic_high_volatility_scenario(self):
        """Test realistic high volatility scenario (2x normal)."""
        base_risk = 0.02  # 2% base risk
        current_atr = 100.0  # 100 pips (2x normal)
        avg_atr = 50.0  # 50 pips normal

        result = calculate_volatility_adjusted_risk(base_risk, current_atr, avg_atr)
        # Should reduce to 1% risk (vol_ratio = 0.5)
        assert result == 0.01

    def test_realistic_low_volatility_scenario(self):
        """Test realistic low volatility scenario (0.7x normal)."""
        base_risk = 0.02  # 2% base risk
        current_atr = 35.0  # 35 pips (0.7x normal)
        avg_atr = 50.0  # 50 pips normal

        result = calculate_volatility_adjusted_risk(base_risk, current_atr, avg_atr)
        # vol_ratio = 50/35 = 1.428 (below max 1.5)
        expected = 0.02 * (50.0 / 35.0)
        assert abs(result - expected) < 1e-6

    # ========================================================================
    # RETURN TYPE VALIDATION
    # ========================================================================

    def test_return_type_is_float(self):
        """Test that function always returns float type."""
        test_cases = [
            (0.02, 50.0, 50.0),
            (0.02, 100.0, 50.0),
            (0.02, 25.0, 50.0),
            (0.02, 0.0, 50.0),
        ]

        for base_risk, current_atr, avg_atr in test_cases:
            result = calculate_volatility_adjusted_risk(base_risk, current_atr, avg_atr)
            assert isinstance(result, float), f"Expected float, got {type(result)}"

    def test_return_value_is_positive(self):
        """Test that return value is always positive."""
        test_cases = [
            (0.02, 50.0, 50.0),
            (0.02, 100.0, 50.0),
            (0.02, 25.0, 50.0),
        ]

        for base_risk, current_atr, avg_atr in test_cases:
            result = calculate_volatility_adjusted_risk(base_risk, current_atr, avg_atr)
            assert result > 0, f"Expected positive result, got {result}"


class TestEquityCurveFilter:
    """Tests for equity curve filter.

    This function implements Tier 2 risk management by filtering trades
    based on equity curve health:
    - Equity above MA → trade normally (1.0x)
    - Equity below MA → reduce position size based on distance below
    """

    # ========================================================================
    # NORMAL CASES - EQUITY ABOVE MA
    # ========================================================================

    def test_equity_above_ma_returns_full_multiplier(self):
        """Test that equity above MA returns (True, 1.0)."""
        equity_history = [100, 102, 104, 106, 108, 110, 112, 114, 116, 118,
                          120, 122, 124, 126, 128, 130, 132, 134, 136, 138, 140]

        should_trade, multiplier = equity_curve_filter(equity_history, ma_period=20)
        assert should_trade is True
        assert multiplier == 1.0

    def test_equity_exactly_at_ma_returns_full_multiplier(self):
        """Test that equity exactly at MA returns (True, 1.0)."""
        # Create flat equity curve
        equity_history = [100] * 25

        should_trade, multiplier = equity_curve_filter(equity_history, ma_period=20)
        assert should_trade is True
        assert multiplier == 1.0

    def test_equity_slightly_above_ma_returns_full_multiplier(self):
        """Test that equity slightly above MA returns (True, 1.0)."""
        equity_history = [100] * 20 + [100.01]  # Just above

        should_trade, multiplier = equity_curve_filter(equity_history, ma_period=20)
        assert should_trade is True
        assert multiplier == 1.0

    # ========================================================================
    # EQUITY BELOW MA - REDUCED MULTIPLIERS
    # ========================================================================

    def test_equity_5_percent_below_ma_reduces_multiplier(self):
        """Test that equity 5% below MA returns reduced multiplier."""
        # MA of 100, current equity 95 (5% below)
        equity_history = [100] * 20 + [95]

        should_trade, multiplier = equity_curve_filter(equity_history, ma_period=20)
        assert should_trade is True
        # Multiplier reduced when below MA (approximate due to MA calculation)
        assert 0.70 < multiplier < 0.85  # Approximately 0.75

    def test_equity_10_percent_below_ma_reduces_multiplier(self):
        """Test that equity 10% below MA returns reduced multiplier."""
        # MA of 100, current equity 90 (10% below)
        equity_history = [100] * 20 + [90]

        should_trade, multiplier = equity_curve_filter(equity_history, ma_period=20)
        assert should_trade is True
        # Multiplier significantly reduced when 10% below MA
        assert 0.45 < multiplier < 0.60  # Approximately 0.5

    def test_equity_15_percent_below_ma_hits_floor(self):
        """Test that equity 15%+ below MA hits minimum multiplier."""
        # MA of 100, current equity 85 (15% below)
        equity_history = [100] * 20 + [85]

        should_trade, multiplier = equity_curve_filter(equity_history, ma_period=20)
        assert should_trade is True
        # Multiplier near floor when significantly below MA
        assert 0.25 <= multiplier < 0.35  # At or near floor of 0.25

    def test_equity_20_percent_below_ma_stays_at_floor(self):
        """Test that equity 20%+ below MA stays at minimum multiplier."""
        # MA of 100, current equity 80 (20% below)
        equity_history = [100] * 20 + [80]

        should_trade, multiplier = equity_curve_filter(equity_history, ma_period=20)
        assert should_trade is True
        # pct_below = 0.20, multiplier = max(0.25, 1.0 - 1.0) = 0.25
        assert multiplier == 0.25

    # ========================================================================
    # EDGE CASES
    # ========================================================================

    def test_empty_history_returns_full_multiplier(self):
        """Test that empty history returns (True, 1.0)."""
        equity_history = []

        should_trade, multiplier = equity_curve_filter(equity_history, ma_period=20)
        assert should_trade is True
        assert multiplier == 1.0

    def test_history_shorter_than_ma_period_returns_full_multiplier(self):
        """Test that history shorter than MA period returns (True, 1.0)."""
        equity_history = [100, 105, 110]  # Only 3 values, ma_period=20

        should_trade, multiplier = equity_curve_filter(equity_history, ma_period=20)
        assert should_trade is True
        assert multiplier == 1.0

    def test_history_exactly_ma_period_length(self):
        """Test that history exactly equal to MA period works correctly."""
        equity_history = [100] * 20  # Exactly 20 values

        should_trade, multiplier = equity_curve_filter(equity_history, ma_period=20)
        assert should_trade is True
        assert multiplier == 1.0

    def test_history_one_more_than_ma_period(self):
        """Test that history one more than MA period works correctly."""
        equity_history = [100] * 20 + [95]  # 21 values

        should_trade, multiplier = equity_curve_filter(equity_history, ma_period=20)
        assert should_trade is True
        assert 0.70 < multiplier < 0.85  # Approximately 0.75

    # ========================================================================
    # INCREASING EQUITY CURVE
    # ========================================================================

    def test_increasing_equity_curve_returns_full_multiplier(self):
        """Test that steadily increasing equity returns full multiplier."""
        equity_history = list(range(100, 125))  # 100, 101, 102, ..., 124

        should_trade, multiplier = equity_curve_filter(equity_history, ma_period=20)
        assert should_trade is True
        assert multiplier == 1.0

    def test_sharp_increase_in_equity_returns_full_multiplier(self):
        """Test that sharp increase in equity returns full multiplier."""
        equity_history = [100] * 20 + [150]  # Sharp jump

        should_trade, multiplier = equity_curve_filter(equity_history, ma_period=20)
        assert should_trade is True
        assert multiplier == 1.0

    # ========================================================================
    # DECREASING EQUITY CURVE
    # ========================================================================

    def test_decreasing_equity_curve_reduces_multiplier(self):
        """Test that decreasing equity curve reduces multiplier."""
        # Start at 120, decrease to 95 (below initial MA of ~110)
        equity_history = list(range(120, 95, -1))  # 120, 119, 118, ..., 96, 95

        should_trade, multiplier = equity_curve_filter(equity_history[-21:], ma_period=20)
        assert should_trade is True
        assert multiplier < 1.0  # Should be reduced

    def test_sharp_decrease_in_equity_reduces_multiplier(self):
        """Test that sharp decrease in equity reduces multiplier."""
        equity_history = [100] * 20 + [80]  # Sharp drop (20% below MA)

        should_trade, multiplier = equity_curve_filter(equity_history, ma_period=20)
        assert should_trade is True
        assert multiplier == 0.25  # At floor

    # ========================================================================
    # CUSTOM MA PERIOD
    # ========================================================================

    def test_custom_ma_period_5(self):
        """Test with custom MA period of 5."""
        equity_history = [100] * 5 + [95]

        should_trade, multiplier = equity_curve_filter(equity_history, ma_period=5)
        assert should_trade is True
        assert 0.70 < multiplier < 0.85  # Approximately 0.75

    def test_custom_ma_period_10(self):
        """Test with custom MA period of 10."""
        equity_history = [100] * 10 + [90]

        should_trade, multiplier = equity_curve_filter(equity_history, ma_period=10)
        assert should_trade is True
        assert 0.45 < multiplier < 0.60  # Approximately 0.5

    def test_custom_ma_period_50(self):
        """Test with custom MA period of 50."""
        equity_history = [100] * 50 + [85]

        should_trade, multiplier = equity_curve_filter(equity_history, ma_period=50)
        assert should_trade is True
        assert 0.25 <= multiplier < 0.35  # Near floor

    # ========================================================================
    # REALISTIC SCENARIOS
    # ========================================================================

    def test_realistic_drawdown_scenario(self):
        """Test realistic drawdown scenario."""
        # Start at 10000, grow to 11000, then drawdown to 10500
        equity_history = (
            list(range(10000, 11000, 50)) +  # Growth phase
            list(range(11000, 10500, -25))    # Drawdown phase
        )

        should_trade, multiplier = equity_curve_filter(equity_history[-21:], ma_period=20)
        assert should_trade is True
        # Exact multiplier depends on current equity vs MA
        assert 0.25 <= multiplier <= 1.0

    def test_realistic_recovery_scenario(self):
        """Test realistic recovery scenario."""
        # Drawdown then recovery
        equity_history = (
            [10000] * 10 +
            [9500] * 5 +  # Drawdown
            [9700, 9900, 10100, 10300, 10500]  # Recovery
        )

        should_trade, multiplier = equity_curve_filter(equity_history, ma_period=20)
        assert should_trade is True
        # Should be back to higher multiplier during recovery
        assert multiplier > 0.5

    # ========================================================================
    # RETURN TYPE VALIDATION
    # ========================================================================

    def test_return_type_is_tuple(self):
        """Test that function returns tuple of (bool, float)."""
        equity_history = [100] * 25

        result = equity_curve_filter(equity_history, ma_period=20)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], bool)
        assert isinstance(result[1], float)

    def test_should_trade_is_always_true(self):
        """Test that should_trade is always True (never skips trades)."""
        test_cases = [
            [],  # Empty
            [100] * 10,  # Short history
            [100] * 20 + [95],  # Below MA
            [100] * 20 + [105],  # Above MA
        ]

        for equity_history in test_cases:
            should_trade, _ = equity_curve_filter(equity_history, ma_period=20)
            assert should_trade is True

    def test_multiplier_range_is_valid(self):
        """Test that multiplier is always between 0.25 and 1.0."""
        test_cases = [
            [100] * 20 + [95],   # 5% below
            [100] * 20 + [90],   # 10% below
            [100] * 20 + [85],   # 15% below
            [100] * 20 + [50],   # 50% below (extreme)
            [100] * 20 + [110],  # 10% above
        ]

        for equity_history in test_cases:
            _, multiplier = equity_curve_filter(equity_history, ma_period=20)
            assert 0.25 <= multiplier <= 1.0, f"Multiplier {multiplier} out of range"


class TestGetRegimePositionMultiplier:
    """Tests for regime-based position multiplier.

    This function implements Tier 2 risk management by adjusting position
    size based on market regime:
    - Dangerous regimes (ranging_high_vol) → skip trading (0.0x)
    - Suboptimal regimes (ranging_normal, trending_high_vol) → reduce (0.75x)
    - Good regimes (trending_normal, trending_low_vol, ranging_low_vol) → full (1.0x)
    """

    # ========================================================================
    # DANGEROUS REGIMES - SKIP TRADING
    # ========================================================================

    def test_ranging_high_vol_skips_trading(self):
        """Test that ranging_high_vol returns (False, 0.0) - worst regime."""
        should_trade, multiplier = get_regime_position_multiplier("ranging_high_vol")
        assert should_trade is False
        assert multiplier == 0.0

    # ========================================================================
    # SUBOPTIMAL REGIMES - REDUCED POSITION SIZE
    # ========================================================================

    def test_ranging_normal_reduces_position(self):
        """Test that ranging_normal returns (True, 0.75) - reduced size."""
        should_trade, multiplier = get_regime_position_multiplier("ranging_normal")
        assert should_trade is True
        assert multiplier == 0.75

    def test_trending_high_vol_reduces_position(self):
        """Test that trending_high_vol returns (True, 0.75) - reduced size."""
        should_trade, multiplier = get_regime_position_multiplier("trending_high_vol")
        assert should_trade is True
        assert multiplier == 0.75

    # ========================================================================
    # GOOD REGIMES - FULL POSITION SIZE
    # ========================================================================

    def test_trending_normal_full_position(self):
        """Test that trending_normal returns (True, 1.0) - ideal regime."""
        should_trade, multiplier = get_regime_position_multiplier("trending_normal")
        assert should_trade is True
        assert multiplier == 1.0

    def test_trending_low_vol_full_position(self):
        """Test that trending_low_vol returns (True, 1.0) - good regime."""
        should_trade, multiplier = get_regime_position_multiplier("trending_low_vol")
        assert should_trade is True
        assert multiplier == 1.0

    def test_ranging_low_vol_full_position(self):
        """Test that ranging_low_vol returns (True, 1.0) - acceptable regime."""
        should_trade, multiplier = get_regime_position_multiplier("ranging_low_vol")
        assert should_trade is True
        assert multiplier == 1.0

    # ========================================================================
    # UNKNOWN/DEFAULT REGIME
    # ========================================================================

    def test_unknown_regime_returns_default(self):
        """Test that unknown regime returns (True, 1.0) - default behavior."""
        should_trade, multiplier = get_regime_position_multiplier("unknown_regime")
        assert should_trade is True
        assert multiplier == 1.0

    def test_empty_string_regime_returns_default(self):
        """Test that empty string returns (True, 1.0) - default behavior."""
        should_trade, multiplier = get_regime_position_multiplier("")
        assert should_trade is True
        assert multiplier == 1.0

    def test_random_string_regime_returns_default(self):
        """Test that random string returns (True, 1.0) - default behavior."""
        should_trade, multiplier = get_regime_position_multiplier("random_regime_xyz")
        assert should_trade is True
        assert multiplier == 1.0

    # ========================================================================
    # CASE SENSITIVITY
    # ========================================================================

    def test_case_sensitivity_uppercase(self):
        """Test that uppercase regime name returns default (case-sensitive)."""
        should_trade, multiplier = get_regime_position_multiplier("RANGING_HIGH_VOL")
        assert should_trade is True
        assert multiplier == 1.0  # Not recognized, returns default

    def test_case_sensitivity_mixed_case(self):
        """Test that mixed case regime name returns default (case-sensitive)."""
        should_trade, multiplier = get_regime_position_multiplier("Trending_Normal")
        assert should_trade is True
        assert multiplier == 1.0  # Not recognized, returns default

    # ========================================================================
    # RETURN TYPE VALIDATION
    # ========================================================================

    def test_return_type_is_tuple(self):
        """Test that function returns tuple of (bool, float)."""
        result = get_regime_position_multiplier("trending_normal")
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], bool)
        assert isinstance(result[1], (int, float))

    def test_multiplier_values_are_expected_levels(self):
        """Test that multiplier values are only expected discrete levels."""
        valid_multipliers = {0.0, 0.75, 1.0}

        test_regimes = [
            "ranging_high_vol",
            "ranging_normal",
            "ranging_low_vol",
            "trending_high_vol",
            "trending_normal",
            "trending_low_vol",
            "unknown_regime",
        ]

        for regime in test_regimes:
            _, multiplier = get_regime_position_multiplier(regime)
            assert multiplier in valid_multipliers, (
                f"Regime '{regime}' returned unexpected multiplier {multiplier}"
            )

    # ========================================================================
    # ALL REGIME COMBINATIONS
    # ========================================================================

    def test_all_defined_regimes(self):
        """Test all explicitly defined regime combinations."""
        expected_results = {
            # Skip regimes
            "ranging_high_vol": (False, 0.0),
            # Reduced regimes
            "ranging_normal": (True, 0.75),
            "trending_high_vol": (True, 0.75),
            # Full regimes
            "ranging_low_vol": (True, 1.0),
            "trending_normal": (True, 1.0),
            "trending_low_vol": (True, 1.0),
        }

        for regime, expected in expected_results.items():
            result = get_regime_position_multiplier(regime)
            assert result == expected, (
                f"Regime '{regime}': expected {expected}, got {result}"
            )


class TestTier2Integration:
    """Integration tests for Tier 2 multipliers compounding correctly.

    Tests that all three Tier 2 multipliers (volatility, equity curve, regime)
    compound correctly with each other and with Tier 1 (drawdown, consecutive losses).
    """

    # ========================================================================
    # TIER 2 ONLY - NO TIER 1
    # ========================================================================

    def test_all_tier2_at_normal_returns_full_size(self):
        """Test that all Tier 2 at normal conditions returns 1.0x."""
        # Volatility: normal
        vol_multiplier = calculate_volatility_adjusted_risk(0.02, 50.0, 50.0) / 0.02

        # Equity: above MA
        _, equity_multiplier = equity_curve_filter([100] * 20 + [110], ma_period=20)

        # Regime: trending_normal
        _, regime_multiplier = get_regime_position_multiplier("trending_normal")

        combined = vol_multiplier * equity_multiplier * regime_multiplier
        assert combined == 1.0

    def test_tier2_compounding_volatile_below_equity(self):
        """Test Tier 2 compounding: high volatility + equity below MA."""
        # Volatility: 2x normal → 0.5x risk
        vol_multiplier = calculate_volatility_adjusted_risk(0.02, 100.0, 50.0) / 0.02
        assert vol_multiplier == 0.5

        # Equity: below MA → reduced multiplier
        _, equity_multiplier = equity_curve_filter([100] * 20 + [95], ma_period=20)
        assert 0.70 < equity_multiplier < 0.85  # Approximately 0.75

        # Regime: trending_normal → 1.0x
        _, regime_multiplier = get_regime_position_multiplier("trending_normal")
        assert regime_multiplier == 1.0

        # Combined: 0.5 * ~0.76 * 1.0 ≈ 0.38
        combined = vol_multiplier * equity_multiplier * regime_multiplier
        assert 0.30 < combined < 0.45  # Approximately 0.375

    def test_tier2_compounding_suboptimal_regime(self):
        """Test Tier 2 compounding: normal vol + equity OK + suboptimal regime."""
        # Volatility: normal → 1.0x
        vol_multiplier = calculate_volatility_adjusted_risk(0.02, 50.0, 50.0) / 0.02
        assert vol_multiplier == 1.0

        # Equity: above MA → 1.0x
        _, equity_multiplier = equity_curve_filter([100] * 20 + [105], ma_period=20)
        assert equity_multiplier == 1.0

        # Regime: ranging_normal → 0.75x
        _, regime_multiplier = get_regime_position_multiplier("ranging_normal")
        assert regime_multiplier == 0.75

        # Combined: 1.0 * 1.0 * 0.75 = 0.75
        combined = vol_multiplier * equity_multiplier * regime_multiplier
        assert combined == 0.75

    def test_tier2_worst_case_compounding(self):
        """Test Tier 2 worst case: high vol + equity below + suboptimal regime."""
        # Volatility: 2x normal → 0.5x
        vol_multiplier = calculate_volatility_adjusted_risk(0.02, 100.0, 50.0) / 0.02
        assert vol_multiplier == 0.5

        # Equity: significantly below MA → reduced
        _, equity_multiplier = equity_curve_filter([100] * 20 + [90], ma_period=20)
        assert 0.45 < equity_multiplier < 0.60  # Approximately 0.5

        # Regime: trending_high_vol → 0.75x
        _, regime_multiplier = get_regime_position_multiplier("trending_high_vol")
        assert regime_multiplier == 0.75

        # Combined: 0.5 * ~0.52 * 0.75 ≈ 0.19
        combined = vol_multiplier * equity_multiplier * regime_multiplier
        assert 0.15 < combined < 0.25  # Approximately 0.19

    # ========================================================================
    # TIER 2 + TIER 1 COMPOUNDING
    # ========================================================================

    def test_tier2_with_tier1_drawdown(self):
        """Test Tier 2 compounding with Tier 1 drawdown multiplier."""
        # Tier 2: all normal → 1.0x
        vol_multiplier = 1.0
        equity_multiplier = 1.0
        regime_multiplier = 1.0

        # Tier 1: 8% drawdown → 0.5x
        dd_multiplier = get_drawdown_position_multiplier(0.08)
        assert dd_multiplier == 0.5

        # Combined: 1.0 * 0.5 = 0.5
        combined = vol_multiplier * equity_multiplier * regime_multiplier * dd_multiplier
        assert combined == 0.5

    def test_tier2_with_tier1_consecutive_losses(self):
        """Test Tier 2 compounding with Tier 1 consecutive loss reduction."""
        # Tier 2: suboptimal regime → 0.75x
        _, regime_multiplier = get_regime_position_multiplier("ranging_normal")
        assert regime_multiplier == 0.75

        # Tier 1: 2 consecutive losses → 0.5x (simulated)
        consecutive_loss_multiplier = 0.5

        # Combined: 0.75 * 0.5 = 0.375
        combined = regime_multiplier * consecutive_loss_multiplier
        assert combined == 0.375

    def test_full_tier1_and_tier2_compounding(self):
        """Test full Tier 1 + Tier 2 compounding (worst case scenario)."""
        # Tier 1: 12% drawdown → 0.25x
        dd_multiplier = get_drawdown_position_multiplier(0.12)
        assert dd_multiplier == 0.25

        # Tier 1: 3 consecutive losses → 0.25x (simulated)
        consecutive_loss_multiplier = 0.25

        # Tier 2: high vol → 0.5x
        vol_multiplier = 0.5

        # Tier 2: equity 10% below → 0.5x
        equity_multiplier = 0.5

        # Tier 2: suboptimal regime → 0.75x
        regime_multiplier = 0.75

        # Combined: 0.25 * 0.25 * 0.5 * 0.5 * 0.75 = 0.00234375 (~0.23% of base risk!)
        combined = (dd_multiplier * consecutive_loss_multiplier *
                    vol_multiplier * equity_multiplier * regime_multiplier)
        expected = 0.25 * 0.25 * 0.5 * 0.5 * 0.75
        assert abs(combined - expected) < 1e-10

    # ========================================================================
    # REGIME SKIP OVERRIDES ALL
    # ========================================================================

    def test_regime_skip_overrides_all_multipliers(self):
        """Test that regime skip (ranging_high_vol) overrides all other multipliers."""
        # All other factors favorable
        vol_multiplier = 1.0
        equity_multiplier = 1.0
        dd_multiplier = 1.0

        # But regime says skip
        should_trade, regime_multiplier = get_regime_position_multiplier("ranging_high_vol")
        assert should_trade is False
        assert regime_multiplier == 0.0

        # Combined is zero (no trade)
        combined = vol_multiplier * equity_multiplier * dd_multiplier * regime_multiplier
        assert combined == 0.0

    # ========================================================================
    # REALISTIC COMPOUNDING SCENARIOS
    # ========================================================================

    def test_realistic_moderate_risk_reduction(self):
        """Test realistic moderate risk reduction scenario."""
        # Tier 1: 6% drawdown → 0.75x
        dd_multiplier = get_drawdown_position_multiplier(0.06)

        # Tier 2: slightly high vol (1.3x) → ~0.77x
        vol_multiplier = calculate_volatility_adjusted_risk(0.02, 65.0, 50.0) / 0.02

        # Tier 2: equity slightly below MA → ~0.9x
        _, equity_multiplier = equity_curve_filter([100] * 20 + [98], ma_period=20)

        # Tier 2: good regime → 1.0x
        _, regime_multiplier = get_regime_position_multiplier("trending_normal")

        combined = dd_multiplier * vol_multiplier * equity_multiplier * regime_multiplier
        # Should be around 0.52 (roughly 52% of base risk)
        assert 0.45 < combined < 0.60

    def test_realistic_aggressive_preservation(self):
        """Test realistic aggressive capital preservation scenario."""
        # Tier 1: 12% drawdown → 0.25x
        dd_multiplier = get_drawdown_position_multiplier(0.12)

        # Tier 2: high vol (2x) → 0.5x
        vol_multiplier = 0.5

        # Tier 2: equity significantly below MA → near floor
        _, equity_multiplier = equity_curve_filter([100] * 20 + [85], ma_period=20)
        assert 0.25 <= equity_multiplier < 0.35  # Near floor

        # Tier 2: suboptimal regime → 0.75x
        _, regime_multiplier = get_regime_position_multiplier("ranging_normal")

        combined = dd_multiplier * vol_multiplier * equity_multiplier * regime_multiplier
        # Approximately: 0.25 * 0.5 * ~0.28 * 0.75 ≈ 0.026 (~2.6% of base risk)
        assert 0.02 < combined < 0.04  # Very aggressive capital preservation


# ========================================================================
# PARAMETRIZED TESTS FOR TIER 2 FUNCTIONS
# ========================================================================

@pytest.mark.parametrize("base_risk,current_atr,avg_atr,expected_multiplier", [
    (0.02, 50.0, 50.0, 1.0),      # Normal volatility
    (0.02, 100.0, 50.0, 0.5),     # High volatility (2x)
    (0.02, 200.0, 50.0, 0.25),    # Very high volatility (4x, hits min)
    (0.02, 50.0, 100.0, 1.5),     # Low volatility (hits max)
    (0.02, 25.0, 100.0, 1.5),     # Very low volatility (hits max)
    (0.02, 0.0, 50.0, 1.0),       # Zero current ATR (edge case)
    (0.02, 50.0, 0.0, 1.0),       # Zero avg ATR (edge case)
])
def test_volatility_adjusted_risk_parametrized(base_risk, current_atr, avg_atr, expected_multiplier):
    """Parametrized test for volatility-adjusted risk calculation."""
    result = calculate_volatility_adjusted_risk(base_risk, current_atr, avg_atr)
    expected = base_risk * expected_multiplier
    assert abs(result - expected) < 1e-6, (
        f"ATR {current_atr}/{avg_atr}: expected {expected}, got {result}"
    )


@pytest.mark.parametrize("regime,expected_should_trade,expected_multiplier", [
    ("ranging_high_vol", False, 0.0),      # Skip
    ("ranging_normal", True, 0.75),        # Reduced
    ("ranging_low_vol", True, 1.0),        # Full
    ("trending_high_vol", True, 0.75),     # Reduced
    ("trending_normal", True, 1.0),        # Full
    ("trending_low_vol", True, 1.0),       # Full
    ("unknown_regime", True, 1.0),         # Default
    ("", True, 1.0),                       # Empty string default
])
def test_regime_position_multiplier_parametrized(regime, expected_should_trade, expected_multiplier):
    """Parametrized test for regime position multiplier."""
    should_trade, multiplier = get_regime_position_multiplier(regime)
    assert should_trade == expected_should_trade, (
        f"Regime '{regime}': expected should_trade={expected_should_trade}, got {should_trade}"
    )
    assert multiplier == expected_multiplier, (
        f"Regime '{regime}': expected multiplier={expected_multiplier}, got {multiplier}"
    )
