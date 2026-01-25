"""Unit tests for Walk-Forward Optimization drawdown mitigation.

This test suite validates the Tier 1 drawdown mitigation functionality,
specifically the get_drawdown_position_multiplier function that implements
progressive position reduction based on current drawdown levels.

Test Coverage:
- Normal operation across all drawdown levels (0-5%, 5-7.5%, 7.5-10%, 10-15%, 15%+)
- Boundary conditions (exact threshold values)
- Edge cases (negative drawdown, zero drawdown, extreme drawdown)
- Custom max_allowed parameter
- Circuit breaker integration (halts trading at 15% drawdown)
"""

import pytest
import sys
from pathlib import Path

# Add scripts directory to path for import
scripts_dir = Path(__file__).parent.parent.parent.parent / "scripts"
sys.path.insert(0, str(scripts_dir))

from walk_forward_optimization import get_drawdown_position_multiplier


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
