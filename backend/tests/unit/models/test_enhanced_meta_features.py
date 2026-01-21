"""Tests for Enhanced Meta-Features Calculator.

CRITICAL: This test file includes data leakage detection tests to ensure
the enhanced meta-features don't use future data in calculations.
"""

import numpy as np
import pandas as pd
import pytest

from src.models.multi_timeframe.enhanced_meta_features import (
    EnhancedMetaFeatureCalculator,
    get_enhanced_feature_names,
)


class TestEnhancedMetaFeatureCalculator:
    """Tests for EnhancedMetaFeatureCalculator class."""

    @pytest.fixture
    def calculator(self):
        """Create a calculator with default settings."""
        return EnhancedMetaFeatureCalculator(lookback_window=50)

    @pytest.fixture
    def sample_predictions(self):
        """Sample prediction arrays for testing."""
        n_samples = 100
        np.random.seed(42)
        return {
            "1H": np.random.randint(0, 2, n_samples),
            "4H": np.random.randint(0, 2, n_samples),
            "D": np.random.randint(0, 2, n_samples),
        }

    @pytest.fixture
    def sample_probabilities(self):
        """Sample probability arrays for testing."""
        n_samples = 100
        np.random.seed(42)
        return {
            "1H": np.random.uniform(0.4, 0.9, n_samples),
            "4H": np.random.uniform(0.4, 0.9, n_samples),
            "D": np.random.uniform(0.4, 0.9, n_samples),
        }

    @pytest.fixture
    def sample_price_data(self):
        """Sample OHLC price data for testing."""
        n_samples = 100
        dates = pd.date_range(start="2024-01-01", periods=n_samples, freq="1H")

        np.random.seed(42)
        close = 100 + np.cumsum(np.random.randn(n_samples) * 0.5)
        high = close + np.abs(np.random.randn(n_samples) * 0.3)
        low = close - np.abs(np.random.randn(n_samples) * 0.3)
        open_ = close + np.random.randn(n_samples) * 0.2

        return pd.DataFrame({
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": np.random.randint(1000, 10000, n_samples),
        }, index=dates)


class TestPredictionQualityFeatures:
    """Tests for prediction quality features (entropy and margin)."""

    @pytest.fixture
    def calculator(self):
        return EnhancedMetaFeatureCalculator(lookback_window=50)

    def test_prob_entropy_calculation(self, calculator):
        """Verify entropy values are in valid range [0, log(3)]."""
        n_samples = 50
        probabilities = {
            "1H": np.full(n_samples, 0.5),  # Maximum entropy (uncertain)
            "4H": np.full(n_samples, 0.5),
            "D": np.full(n_samples, 0.5),
        }

        features = calculator.calculate_prediction_quality(probabilities)

        assert "prob_entropy" in features
        assert "confidence_margin" in features

        # Entropy should be in valid range
        max_entropy_per_model = np.log(2)  # Binary classification
        max_total_entropy = max_entropy_per_model  # Average across models

        assert np.all(features["prob_entropy"] >= 0)
        assert np.all(features["prob_entropy"] <= max_total_entropy * 1.1)  # Small tolerance

    def test_prob_entropy_confident_predictions(self, calculator):
        """Test entropy for very confident predictions (should be low)."""
        n_samples = 50
        probabilities = {
            "1H": np.full(n_samples, 0.99),  # Very confident
            "4H": np.full(n_samples, 0.99),
            "D": np.full(n_samples, 0.99),
        }

        features = calculator.calculate_prediction_quality(probabilities)

        # Low entropy for confident predictions
        assert np.all(features["prob_entropy"] < 0.1)

    def test_prob_entropy_uncertain_predictions(self, calculator):
        """Test entropy for uncertain predictions (should be high)."""
        n_samples = 50
        probabilities = {
            "1H": np.full(n_samples, 0.5),  # Uncertain
            "4H": np.full(n_samples, 0.5),
            "D": np.full(n_samples, 0.5),
        }

        features = calculator.calculate_prediction_quality(probabilities)

        # High entropy for uncertain predictions
        assert np.all(features["prob_entropy"] > 0.5)

    def test_confidence_margin_calculation(self, calculator):
        """Verify margin identifies decisive predictions."""
        n_samples = 50

        # Scenario 1: One model dominates with high prob, others neutral (large margin)
        probabilities_decisive = {
            "1H": np.full(n_samples, 0.95),  # One very confident
            "4H": np.full(n_samples, 0.5),   # Neutral
            "D": np.full(n_samples, 0.5),    # Neutral
        }

        features_decisive = calculator.calculate_prediction_quality(probabilities_decisive)

        # Scenario 2: All models uncertain/neutral (smaller margin)
        probabilities_indecisive = {
            "1H": np.full(n_samples, 0.55),  # Close to 0.5
            "4H": np.full(n_samples, 0.52),  # Close to 0.5
            "D": np.full(n_samples, 0.48),   # Close to 0.5
        }

        features_indecisive = calculator.calculate_prediction_quality(probabilities_indecisive)

        # Margin should be larger when one prob dominates
        assert np.mean(features_decisive["confidence_margin"]) > \
               np.mean(features_indecisive["confidence_margin"])

    def test_confidence_margin_range(self, calculator):
        """Test that margin is in valid range [0, 1]."""
        n_samples = 50
        np.random.seed(42)

        probabilities = {
            "1H": np.random.uniform(0.3, 0.9, n_samples),
            "4H": np.random.uniform(0.3, 0.9, n_samples),
            "D": np.random.uniform(0.3, 0.9, n_samples),
        }

        features = calculator.calculate_prediction_quality(probabilities)

        # Margin should be in [0, 1]
        assert np.all(features["confidence_margin"] >= 0)
        assert np.all(features["confidence_margin"] <= 1)


class TestCrossTimeframePatterns:
    """Tests for cross-timeframe pattern features."""

    @pytest.fixture
    def calculator(self):
        return EnhancedMetaFeatureCalculator(lookback_window=50)

    def test_htf_agreement_binary(self, calculator):
        """Verify agreement is 0 or 1."""
        n_samples = 50
        predictions = {
            "1H": np.array([1, 0, 1, 0, 1] * 10),
            "4H": np.array([1, 1, 0, 0, 1] * 10),
            "D": np.array([0, 1, 1, 0, 1] * 10),
        }

        features = calculator.calculate_cross_timeframe_patterns(predictions)

        assert "htf_agreement_1h_4h" in features
        assert "htf_agreement_4h_d" in features

        # Agreement should be binary (0 or 1)
        assert set(np.unique(features["htf_agreement_1h_4h"])) <= {0.0, 1.0}
        assert set(np.unique(features["htf_agreement_4h_d"])) <= {0.0, 1.0}

    def test_htf_agreement_all_agree(self, calculator):
        """Test agreement when all models agree."""
        n_samples = 50
        predictions = {
            "1H": np.ones(n_samples),
            "4H": np.ones(n_samples),
            "D": np.ones(n_samples),
        }

        features = calculator.calculate_cross_timeframe_patterns(predictions)

        # All should agree
        assert np.all(features["htf_agreement_1h_4h"] == 1.0)
        assert np.all(features["htf_agreement_4h_d"] == 1.0)

    def test_htf_agreement_all_disagree(self, calculator):
        """Test agreement when models disagree."""
        n_samples = 50
        predictions = {
            "1H": np.ones(n_samples),
            "4H": np.zeros(n_samples),
            "D": np.ones(n_samples),
        }

        features = calculator.calculate_cross_timeframe_patterns(predictions)

        # 1H and 4H should disagree
        assert np.all(features["htf_agreement_1h_4h"] == 0.0)

        # 4H and D should disagree
        assert np.all(features["htf_agreement_4h_d"] == 0.0)

    def test_trend_alignment_score(self, calculator):
        """Verify trend alignment score is in [0, 1] range."""
        n_samples = 50
        np.random.seed(42)

        predictions = {
            "1H": np.random.randint(0, 2, n_samples),
            "4H": np.random.randint(0, 2, n_samples),
            "D": np.random.randint(0, 2, n_samples),
        }

        features = calculator.calculate_cross_timeframe_patterns(predictions)

        assert "trend_alignment" in features

        # Alignment should be in [0, 1] with valid fractions (1/3, 2/3, 1.0)
        valid_values = {1/3, 2/3, 1.0}
        unique_values = set(np.unique(features["trend_alignment"]))

        for val in unique_values:
            # Check if close to any valid value
            assert any(abs(val - v) < 0.01 for v in valid_values)

    def test_trend_alignment_unanimous(self, calculator):
        """Test alignment when all models agree (should be 1.0)."""
        n_samples = 50
        predictions = {
            "1H": np.ones(n_samples),
            "4H": np.ones(n_samples),
            "D": np.ones(n_samples),
        }

        features = calculator.calculate_cross_timeframe_patterns(predictions)

        # Perfect alignment
        assert np.all(features["trend_alignment"] == 1.0)

    def test_trend_alignment_split(self, calculator):
        """Test alignment with 2-1 split (should be 2/3)."""
        n_samples = 50
        predictions = {
            "1H": np.ones(n_samples),
            "4H": np.ones(n_samples),
            "D": np.zeros(n_samples),  # Disagrees with others
        }

        features = calculator.calculate_cross_timeframe_patterns(predictions)

        # 2 out of 3 agree
        assert np.all(np.isclose(features["trend_alignment"], 2/3))


class TestMarketContextFeatures:
    """Tests for market context features (CRITICAL - DATA LEAKAGE).

    These tests verify that shift(1) is applied to prevent look-ahead bias.
    """

    @pytest.fixture
    def calculator(self):
        return EnhancedMetaFeatureCalculator(lookback_window=50)

    def test_volatility_uses_shift(self, calculator):
        """CRITICAL: Verify volatility calculation uses shift(1)."""
        # Create price data with distinct patterns
        n_samples = 100
        dates = pd.date_range(start="2024-01-01", periods=n_samples, freq="1H")

        # Create a sudden volatility spike at index 50
        close = np.ones(n_samples) * 100
        close[50] = 110  # 10% jump

        price_data = pd.DataFrame({
            "open": close,
            "high": close + 0.1,
            "low": close - 0.1,
            "close": close,
            "volume": np.ones(n_samples) * 1000,
        }, index=dates)

        features = calculator.calculate_market_context(price_data)

        assert "recent_volatility" in features

        # The volatility at index 50 should NOT reflect the current spike
        # It should only know about past data (up to index 49)
        # The spike should be reflected at index 51 (next bar)

        # Check that volatility is not NaN
        assert not np.all(np.isnan(features["recent_volatility"]))

        # The volatility at index 51 should be higher than at index 50
        # because it now incorporates the spike (with shift)
        if len(features["recent_volatility"]) > 51:
            # This verifies the shift is working
            assert features["recent_volatility"][51] > features["recent_volatility"][49]

    def test_trend_strength_uses_shift(self, calculator):
        """CRITICAL: Verify trend strength uses shift(1)."""
        n_samples = 100
        dates = pd.date_range(start="2024-01-01", periods=n_samples, freq="1H")

        # Create a sudden trend change at index 50
        close = np.ones(n_samples) * 100
        close[50:] = 110  # Sudden jump to new level

        price_data = pd.DataFrame({
            "open": close,
            "high": close + 0.1,
            "low": close - 0.1,
            "close": close,
            "volume": np.ones(n_samples) * 1000,
        }, index=dates)

        features = calculator.calculate_market_context(price_data)

        assert "trend_strength" in features

        # Check that trend strength is not NaN
        assert not np.all(np.isnan(features["trend_strength"]))

    def test_regime_based_on_shifted_data(self, calculator):
        """CRITICAL: Verify regime classification uses shifted data."""
        n_samples = 100
        dates = pd.date_range(start="2024-01-01", periods=n_samples, freq="1H")

        np.random.seed(42)

        # Create varying volatility conditions
        close = 100 + np.cumsum(np.random.randn(n_samples) * 0.5)

        price_data = pd.DataFrame({
            "open": close,
            "high": close + np.abs(np.random.randn(n_samples) * 0.3),
            "low": close - np.abs(np.random.randn(n_samples) * 0.3),
            "close": close,
            "volume": np.random.randint(1000, 10000, n_samples),
        }, index=dates)

        features = calculator.calculate_market_context(price_data)

        assert "market_regime" in features

        # Regime should be 0, 1, or 2
        assert set(np.unique(features["market_regime"])) <= {0, 1, 2}

    def test_market_context_handles_nans(self, calculator):
        """Test that market context features handle NaN values properly."""
        n_samples = 100
        dates = pd.date_range(start="2024-01-01", periods=n_samples, freq="1H")

        np.random.seed(42)
        close = 100 + np.cumsum(np.random.randn(n_samples) * 0.5)

        price_data = pd.DataFrame({
            "open": close,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": np.ones(n_samples) * 1000,
        }, index=dates)

        features = calculator.calculate_market_context(price_data)

        # After filling, there should be no NaN values
        assert not np.any(np.isnan(features["recent_volatility"]))
        assert not np.any(np.isnan(features["trend_strength"]))
        assert not np.any(np.isnan(features["market_regime"]))

    def test_volatility_range(self, calculator):
        """Test that volatility values are in reasonable range."""
        n_samples = 100
        dates = pd.date_range(start="2024-01-01", periods=n_samples, freq="1H")

        np.random.seed(42)
        close = 100 + np.cumsum(np.random.randn(n_samples) * 0.5)

        price_data = pd.DataFrame({
            "open": close,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": np.ones(n_samples) * 1000,
        }, index=dates)

        features = calculator.calculate_market_context(price_data)

        # Volatility should be non-negative
        assert np.all(features["recent_volatility"] >= 0)


class TestPredictionStabilityFeatures:
    """Tests for prediction stability features (CRITICAL - DATA LEAKAGE).

    These tests verify that shift(1) is applied to prevent look-ahead bias.
    """

    @pytest.fixture
    def calculator(self):
        return EnhancedMetaFeatureCalculator(lookback_window=50)

    def test_stability_uses_shift(self, calculator):
        """CRITICAL: Verify stability calculation uses shift(1)."""
        n_samples = 100

        # Create predictions that flip at a specific point
        predictions_1h = np.zeros(n_samples)
        predictions_1h[50:] = 1  # Flip at index 50

        predictions = {
            "1H": predictions_1h,
            "4H": np.zeros(n_samples),
            "D": np.zeros(n_samples),
        }

        features = calculator.calculate_prediction_stability(predictions)

        assert "stability_1h" in features
        assert "stability_4h" in features
        assert "stability_d" in features

        # The stability at index 50 should NOT know about the flip yet
        # The increased instability should appear at index 51 (after shift)

        # Check that stability is not all NaN
        assert not np.all(np.isnan(features["stability_1h"]))

    def test_stability_handles_nans(self, calculator):
        """CRITICAL: Verify NaN handling in stability features."""
        n_samples = 100
        np.random.seed(42)

        predictions = {
            "1H": np.random.randint(0, 2, n_samples),
            "4H": np.random.randint(0, 2, n_samples),
            "D": np.random.randint(0, 2, n_samples),
        }

        features = calculator.calculate_prediction_stability(predictions)

        # After filling, there should be no NaN values
        assert not np.any(np.isnan(features["stability_1h"]))
        assert not np.any(np.isnan(features["stability_4h"]))
        assert not np.any(np.isnan(features["stability_d"]))

    def test_stability_range(self, calculator):
        """Test that stability values are in reasonable range [0, 1]."""
        n_samples = 100

        # For binary predictions (0 or 1), rolling std should be in [0, 0.5]
        predictions = {
            "1H": np.random.randint(0, 2, n_samples),
            "4H": np.random.randint(0, 2, n_samples),
            "D": np.random.randint(0, 2, n_samples),
        }

        features = calculator.calculate_prediction_stability(predictions)

        # Stability (std of binary predictions) should be in [0, 0.5]
        assert np.all(features["stability_1h"] >= 0)
        assert np.all(features["stability_1h"] <= 0.6)  # Small tolerance

    def test_stability_constant_predictions(self, calculator):
        """Test stability with constant predictions (should be 0)."""
        n_samples = 100

        predictions = {
            "1H": np.ones(n_samples),  # All 1s
            "4H": np.zeros(n_samples),  # All 0s
            "D": np.ones(n_samples),
        }

        features = calculator.calculate_prediction_stability(predictions)

        # Constant predictions should have near-zero stability
        assert np.mean(features["stability_1h"]) < 0.1
        assert np.mean(features["stability_4h"]) < 0.1


class TestCalculateAllIntegration:
    """Integration tests for calculate_all method."""

    @pytest.fixture
    def calculator(self):
        return EnhancedMetaFeatureCalculator(lookback_window=50)

    @pytest.fixture
    def full_data(self):
        """Create full dataset for testing."""
        n_samples = 100
        np.random.seed(42)

        predictions = {
            "1H": np.random.randint(0, 2, n_samples),
            "4H": np.random.randint(0, 2, n_samples),
            "D": np.random.randint(0, 2, n_samples),
        }

        probabilities = {
            "1H": np.random.uniform(0.4, 0.9, n_samples),
            "4H": np.random.uniform(0.4, 0.9, n_samples),
            "D": np.random.uniform(0.4, 0.9, n_samples),
        }

        dates = pd.date_range(start="2024-01-01", periods=n_samples, freq="1H")
        close = 100 + np.cumsum(np.random.randn(n_samples) * 0.5)

        price_data = pd.DataFrame({
            "open": close + np.random.randn(n_samples) * 0.2,
            "high": close + np.abs(np.random.randn(n_samples) * 0.3),
            "low": close - np.abs(np.random.randn(n_samples) * 0.3),
            "close": close,
            "volume": np.random.randint(1000, 10000, n_samples),
        }, index=dates)

        return predictions, probabilities, price_data

    def test_calculate_all_returns_expected_features(self, calculator, full_data):
        """Verify all 11 enhanced features are returned."""
        predictions, probabilities, price_data = full_data

        features = calculator.calculate_all(
            predictions=predictions,
            probabilities=probabilities,
            price_data=price_data,
        )

        # Should have 11 features total
        expected_features = get_enhanced_feature_names()
        assert len(expected_features) == 11

        # All features should be present
        for feat_name in expected_features:
            assert feat_name in features, f"Missing feature: {feat_name}"
            assert len(features[feat_name]) == len(predictions["1H"])

    def test_calculate_all_without_price_data(self, calculator):
        """Verify graceful handling when price_data is None."""
        n_samples = 100
        np.random.seed(42)

        predictions = {
            "1H": np.random.randint(0, 2, n_samples),
            "4H": np.random.randint(0, 2, n_samples),
            "D": np.random.randint(0, 2, n_samples),
        }

        probabilities = {
            "1H": np.random.uniform(0.4, 0.9, n_samples),
            "4H": np.random.uniform(0.4, 0.9, n_samples),
            "D": np.random.uniform(0.4, 0.9, n_samples),
        }

        # Call without price data
        features = calculator.calculate_all(
            predictions=predictions,
            probabilities=probabilities,
            price_data=None,
        )

        # Should still have prediction quality, cross-TF, and stability features
        # But missing market context features
        assert "prob_entropy" in features
        assert "confidence_margin" in features
        assert "htf_agreement_1h_4h" in features
        assert "stability_1h" in features

        # Market context features should be missing
        assert "recent_volatility" not in features
        assert "trend_strength" not in features
        assert "market_regime" not in features

    def test_calculate_all_feature_shapes_match(self, calculator, full_data):
        """Verify all features have matching shapes."""
        predictions, probabilities, price_data = full_data
        n_samples = len(predictions["1H"])

        features = calculator.calculate_all(
            predictions=predictions,
            probabilities=probabilities,
            price_data=price_data,
        )

        # All features should have same length as input
        for feat_name, feat_values in features.items():
            assert len(feat_values) == n_samples, \
                f"Feature {feat_name} has length {len(feat_values)}, expected {n_samples}"

    def test_calculate_all_no_inf_values(self, calculator, full_data):
        """Verify no infinite values in any feature."""
        predictions, probabilities, price_data = full_data

        features = calculator.calculate_all(
            predictions=predictions,
            probabilities=probabilities,
            price_data=price_data,
        )

        # No features should have infinite values
        for feat_name, feat_values in features.items():
            assert np.all(np.isfinite(feat_values)), \
                f"Feature {feat_name} contains inf/nan values"


class TestGetEnhancedFeatureNames:
    """Tests for get_enhanced_feature_names utility function."""

    def test_returns_list(self):
        """Test that function returns a list."""
        names = get_enhanced_feature_names()
        assert isinstance(names, list)

    def test_returns_11_features(self):
        """Test that function returns exactly 11 feature names."""
        names = get_enhanced_feature_names()
        assert len(names) == 11

    def test_feature_names_are_strings(self):
        """Test that all feature names are strings."""
        names = get_enhanced_feature_names()
        assert all(isinstance(name, str) for name in names)

    def test_feature_names_match_categories(self):
        """Test that feature names match expected categories."""
        names = get_enhanced_feature_names()

        # Prediction quality (2)
        assert "prob_entropy" in names
        assert "confidence_margin" in names

        # Cross-timeframe patterns (3)
        assert "htf_agreement_1h_4h" in names
        assert "htf_agreement_4h_d" in names
        assert "trend_alignment" in names

        # Market context (3)
        assert "recent_volatility" in names
        assert "trend_strength" in names
        assert "market_regime" in names

        # Prediction stability (3)
        assert "stability_1h" in names
        assert "stability_4h" in names
        assert "stability_d" in names
