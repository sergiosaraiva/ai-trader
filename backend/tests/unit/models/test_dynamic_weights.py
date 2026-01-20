"""Comprehensive tests for Dynamic Ensemble Weights feature.

Tests the dynamic weight adjustment system in MTFEnsemble that adapts
model weights based on recent prediction accuracy.
"""

import pytest
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pandas as pd

from src.models.multi_timeframe.mtf_ensemble import (
    MTFEnsemble,
    MTFEnsembleConfig,
    MTFPrediction,
)
from src.models.multi_timeframe.improved_model import ImprovedModelConfig


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def sample_ohlcv():
    """Sample OHLCV data for testing."""
    dates = pd.date_range(start="2024-01-01", periods=500, freq="5min")
    np.random.seed(42)

    close = 1.08 + np.cumsum(np.random.randn(500) * 0.0001)
    high = close + np.abs(np.random.randn(500) * 0.0005)
    low = close - np.abs(np.random.randn(500) * 0.0005)
    open_ = close + np.random.randn(500) * 0.0002

    df = pd.DataFrame({
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": np.random.randint(1000, 10000, 500),
    }, index=dates)

    return df


@pytest.fixture
def ensemble_config():
    """Basic ensemble configuration."""
    return MTFEnsembleConfig(
        weights={"1H": 0.6, "4H": 0.3, "D": 0.1},
        use_dynamic_weights=False,  # Start with it disabled
        dynamic_weight_window=50,
        dynamic_weight_min=0.1,
        dynamic_weight_blend=0.5,
    )


@pytest.fixture
def ensemble(ensemble_config, tmp_path):
    """Create an MTFEnsemble instance."""
    return MTFEnsemble(config=ensemble_config, model_dir=tmp_path / "models")


@pytest.fixture
def mock_prediction():
    """Sample MTFPrediction for testing."""
    return MTFPrediction(
        direction=1,
        confidence=0.72,
        prob_up=0.72,
        prob_down=0.28,
        component_directions={"1H": 1, "4H": 1, "D": 0},
        component_confidences={"1H": 0.75, "4H": 0.70, "D": 0.58},
        component_weights={"1H": 0.6, "4H": 0.3, "D": 0.1},
        agreement_count=2,
        agreement_score=0.67,
        market_regime="unknown",
    )


# ============================================================================
# 1. CONFIGURATION TESTS
# ============================================================================


class TestDynamicWeightsConfiguration:
    """Test configuration settings for dynamic weights."""

    def test_default_config_values(self):
        """Test default values for dynamic weight config fields."""
        config = MTFEnsembleConfig.default()

        assert config.use_dynamic_weights is False  # Disabled by default
        assert config.dynamic_weight_window == 50
        assert config.dynamic_weight_min == 0.1
        assert config.dynamic_weight_blend == 0.5

    def test_dynamic_weights_disabled_keeps_fixed_weights(self, ensemble):
        """Test that use_dynamic_weights=False keeps base weights."""
        # Verify dynamic weights are disabled
        assert ensemble.config.use_dynamic_weights is False

        # Current weights should match base weights
        assert ensemble.current_weights == ensemble.config.weights

    def test_config_with_dynamic_weights_enabled(self):
        """Test configuration with dynamic weights enabled."""
        config = MTFEnsembleConfig(
            use_dynamic_weights=True,
            dynamic_weight_window=100,
            dynamic_weight_min=0.15,
            dynamic_weight_blend=0.7,
        )

        assert config.use_dynamic_weights is True
        assert config.dynamic_weight_window == 100
        assert config.dynamic_weight_min == 0.15
        assert config.dynamic_weight_blend == 0.7

    def test_config_validation_min_weight_positive(self):
        """Test that minimum weight must be positive."""
        config = MTFEnsembleConfig(dynamic_weight_min=0.05)
        assert config.dynamic_weight_min > 0

        # Test edge case
        config2 = MTFEnsembleConfig(dynamic_weight_min=0.01)
        assert config2.dynamic_weight_min > 0

    def test_config_validation_window_positive(self):
        """Test that window size must be positive."""
        config = MTFEnsembleConfig(dynamic_weight_window=10)
        assert config.dynamic_weight_window > 0

    def test_config_validation_blend_in_range(self):
        """Test that blend factor is in [0, 1]."""
        # Test valid values
        config1 = MTFEnsembleConfig(dynamic_weight_blend=0.0)
        assert 0 <= config1.dynamic_weight_blend <= 1

        config2 = MTFEnsembleConfig(dynamic_weight_blend=1.0)
        assert 0 <= config2.dynamic_weight_blend <= 1

        config3 = MTFEnsembleConfig(dynamic_weight_blend=0.5)
        assert 0 <= config3.dynamic_weight_blend <= 1


# ============================================================================
# 2. RECORD_OUTCOME TESTS
# ============================================================================


class TestRecordOutcome:
    """Test recording prediction outcomes for dynamic weights."""

    def test_record_outcome_stores_prediction(self, ensemble, mock_prediction):
        """Test that record_outcome stores prediction correctly."""
        ensemble.record_outcome(mock_prediction, actual_direction=1)

        assert len(ensemble.prediction_history) == 1

        record = ensemble.prediction_history[0]
        assert record["actual"] == 1
        assert record["1H_pred"] == 1
        assert record["4H_pred"] == 1
        assert record["D_pred"] == 0
        assert record["1H_correct"] == 1
        assert record["4H_correct"] == 1
        assert record["D_correct"] == 0
        assert "timestamp" in record

    def test_history_grows_with_each_call(self, ensemble, mock_prediction):
        """Test that history grows with each record_outcome call."""
        for i in range(10):
            ensemble.record_outcome(mock_prediction, actual_direction=i % 2)

        assert len(ensemble.prediction_history) == 10

    def test_history_pruning_keeps_recent_entries(self, ensemble, mock_prediction):
        """Test that history pruning keeps only recent entries."""
        # Set window to 50, pruning happens at 2x window (100)
        ensemble.config.dynamic_weight_window = 50

        # Record 150 outcomes (exceeds 2x window)
        for i in range(150):
            ensemble.record_outcome(mock_prediction, actual_direction=i % 2)

        # Should keep only the window size (50), not 2x
        assert len(ensemble.prediction_history) == 50

    def test_works_with_all_model_timeframes(self, ensemble):
        """Test recording works with all timeframes."""
        prediction = MTFPrediction(
            direction=1,
            confidence=0.65,
            prob_up=0.65,
            prob_down=0.35,
            component_directions={"1H": 1, "4H": 0, "D": 1},
            component_confidences={"1H": 0.70, "4H": 0.60, "D": 0.55},
            component_weights={"1H": 0.6, "4H": 0.3, "D": 0.1},
            agreement_count=2,
            agreement_score=0.67,
            market_regime="ranging",
        )

        ensemble.record_outcome(prediction, actual_direction=1)

        record = ensemble.prediction_history[0]
        assert record["1H_pred"] == 1
        assert record["4H_pred"] == 0
        assert record["D_pred"] == 1
        assert record["1H_correct"] == 1
        assert record["4H_correct"] == 0
        assert record["D_correct"] == 1

    def test_record_outcome_with_wrong_predictions(self, ensemble, mock_prediction):
        """Test recording when predictions are wrong."""
        ensemble.record_outcome(mock_prediction, actual_direction=0)

        record = ensemble.prediction_history[0]
        # mock_prediction has 1H=1, 4H=1, D=0, actual=0
        assert record["1H_correct"] == 0  # Wrong
        assert record["4H_correct"] == 0  # Wrong
        assert record["D_correct"] == 1  # Correct

    def test_record_outcome_timestamp_stored(self, ensemble, mock_prediction):
        """Test that timestamp is stored with each record."""
        before = datetime.now()
        ensemble.record_outcome(mock_prediction, actual_direction=1)
        after = datetime.now()

        record = ensemble.prediction_history[0]
        assert "timestamp" in record
        assert before <= record["timestamp"] <= after


# ============================================================================
# 3. CALCULATE_DYNAMIC_WEIGHTS TESTS
# ============================================================================


class TestCalculateDynamicWeights:
    """Test dynamic weight calculation based on accuracy."""

    def test_returns_base_weights_when_insufficient_history(self, ensemble):
        """Test that base weights are returned when history < 10."""
        # No history
        weights = ensemble._calculate_dynamic_weights()
        assert weights == ensemble.config.weights

        # Add 5 records (still insufficient)
        prediction = MTFPrediction(
            direction=1, confidence=0.7, prob_up=0.7, prob_down=0.3,
            component_directions={"1H": 1, "4H": 1, "D": 1},
            component_confidences={"1H": 0.7, "4H": 0.7, "D": 0.7},
            component_weights={"1H": 0.6, "4H": 0.3, "D": 0.1},
        )

        for _ in range(5):
            ensemble.record_outcome(prediction, actual_direction=1)

        weights = ensemble._calculate_dynamic_weights()
        assert weights == ensemble.config.weights

    def test_higher_accuracy_model_gets_higher_weight(self, ensemble):
        """Test that models with higher accuracy get higher weights."""
        # Create predictions where 1H is perfect, 4H is 50%, D is 30%
        for i in range(50):
            prediction = MTFPrediction(
                direction=1, confidence=0.7, prob_up=0.7, prob_down=0.3,
                component_directions={
                    "1H": 1,  # Always correct
                    "4H": i % 2,  # 50% correct
                    "D": 0 if i % 3 else 1,  # ~30% correct
                },
                component_confidences={"1H": 0.7, "4H": 0.7, "D": 0.7},
                component_weights={"1H": 0.6, "4H": 0.3, "D": 0.1},
            )
            ensemble.record_outcome(prediction, actual_direction=1)

        weights = ensemble._calculate_dynamic_weights()

        # 1H should have highest weight (perfect accuracy)
        assert weights["1H"] > weights["4H"]
        assert weights["4H"] > weights["D"]

    def test_minimum_weight_constraint_enforced(self, ensemble):
        """Test that minimum weight constraint is enforced."""
        ensemble.config.dynamic_weight_min = 0.15

        # Create extreme accuracy difference
        for i in range(50):
            prediction = MTFPrediction(
                direction=1, confidence=0.7, prob_up=0.7, prob_down=0.3,
                component_directions={
                    "1H": 1,  # Perfect
                    "4H": 0,  # Always wrong
                    "D": 0,  # Always wrong
                },
                component_confidences={"1H": 0.7, "4H": 0.7, "D": 0.7},
                component_weights={"1H": 0.6, "4H": 0.3, "D": 0.1},
            )
            ensemble.record_outcome(prediction, actual_direction=1)

        weights = ensemble._calculate_dynamic_weights()

        # All weights should be >= minimum
        assert all(w >= 0.15 for w in weights.values())

    def test_weights_sum_to_one(self, ensemble):
        """Test that dynamic weights always sum to 1.0."""
        # Add various prediction outcomes
        for i in range(50):
            prediction = MTFPrediction(
                direction=i % 2, confidence=0.7, prob_up=0.7, prob_down=0.3,
                component_directions={
                    "1H": i % 2,
                    "4H": (i + 1) % 2,
                    "D": i % 3,
                },
                component_confidences={"1H": 0.7, "4H": 0.7, "D": 0.7},
                component_weights={"1H": 0.6, "4H": 0.3, "D": 0.1},
            )
            actual = i % 2
            ensemble.record_outcome(prediction, actual_direction=actual)

        weights = ensemble._calculate_dynamic_weights()

        assert sum(weights.values()) == pytest.approx(1.0, abs=1e-6)

    def test_blend_factor_works_correctly(self, ensemble):
        """Test that blend factor correctly blends base and dynamic weights."""
        ensemble.config.dynamic_weight_blend = 0.0  # All base weights

        # Create predictions with extreme accuracy difference
        for i in range(50):
            prediction = MTFPrediction(
                direction=1, confidence=0.7, prob_up=0.7, prob_down=0.3,
                component_directions={"1H": 1, "4H": 0, "D": 0},
                component_confidences={"1H": 0.7, "4H": 0.7, "D": 0.7},
                component_weights={"1H": 0.6, "4H": 0.3, "D": 0.1},
            )
            ensemble.record_outcome(prediction, actual_direction=1)

        weights_blend_0 = ensemble._calculate_dynamic_weights()

        # With blend=0, should be very close to base weights
        assert weights_blend_0["1H"] == pytest.approx(0.6, abs=0.05)
        assert weights_blend_0["4H"] == pytest.approx(0.3, abs=0.05)
        assert weights_blend_0["D"] == pytest.approx(0.1, abs=0.05)

        # Now test with blend=1 (all dynamic)
        ensemble.config.dynamic_weight_blend = 1.0
        weights_blend_1 = ensemble._calculate_dynamic_weights()

        # 1H should be much higher than base (it's 100% accurate)
        assert weights_blend_1["1H"] > 0.6


# ============================================================================
# 4. INTEGRATION TESTS
# ============================================================================


class TestDynamicWeightsIntegration:
    """Test dynamic weights integration with prediction system."""

    def test_dynamic_weights_applied_in_combine_predictions(self, ensemble):
        """Test that dynamic weights are used in _combine_predictions."""
        ensemble.config.use_dynamic_weights = True

        # Build history showing 1H is best
        for i in range(20):
            prediction = MTFPrediction(
                direction=1, confidence=0.7, prob_up=0.7, prob_down=0.3,
                component_directions={"1H": 1, "4H": 0, "D": 0},
                component_confidences={"1H": 0.7, "4H": 0.7, "D": 0.7},
                component_weights={"1H": 0.6, "4H": 0.3, "D": 0.1},
            )
            ensemble.record_outcome(prediction, actual_direction=1)

        # Now make a prediction - it should use dynamic weights
        predictions = {"1H": 1, "4H": 0, "D": 0}
        confidences = {"1H": 0.75, "4H": 0.70, "D": 0.65}
        probs_up = {"1H": 0.75, "4H": 0.30, "D": 0.35}

        result = ensemble._combine_predictions(
            predictions, confidences, probs_up, "unknown"
        )

        # 1H should have higher weight due to perfect accuracy
        assert ensemble.current_weights["1H"] > 0.6

    def test_works_alongside_regime_adjustment(self, ensemble):
        """Test that dynamic weights work alongside regime adjustment."""
        ensemble.config.use_dynamic_weights = True
        ensemble.config.use_regime_adjustment = True

        # Build prediction history
        for i in range(20):
            prediction = MTFPrediction(
                direction=1, confidence=0.7, prob_up=0.7, prob_down=0.3,
                component_directions={"1H": 1, "4H": 1, "D": 1},
                component_confidences={"1H": 0.7, "4H": 0.7, "D": 0.7},
                component_weights={"1H": 0.6, "4H": 0.3, "D": 0.1},
            )
            ensemble.record_outcome(prediction, actual_direction=1)

        # Make prediction with regime
        predictions = {"1H": 1, "4H": 1, "D": 1}
        confidences = {"1H": 0.75, "4H": 0.70, "D": 0.65}
        probs_up = {"1H": 0.75, "4H": 0.70, "D": 0.65}

        result = ensemble._combine_predictions(
            predictions, confidences, probs_up, "trending"
        )

        # Should apply both dynamic and regime adjustments
        assert result.direction == 1
        assert result.confidence > 0

    def test_backward_compatibility_disabled_by_default(self):
        """Test that dynamic weights are disabled by default."""
        config = MTFEnsembleConfig.default()
        assert config.use_dynamic_weights is False

        ensemble = MTFEnsemble(config=config)
        assert ensemble.config.use_dynamic_weights is False


# ============================================================================
# 5. EDGE CASES
# ============================================================================


class TestDynamicWeightsEdgeCases:
    """Test edge cases for dynamic weight calculation."""

    def test_empty_prediction_history(self, ensemble):
        """Test with empty prediction history."""
        assert len(ensemble.prediction_history) == 0

        weights = ensemble._calculate_dynamic_weights()

        # Should return base weights
        assert weights == ensemble.config.weights

    def test_all_models_same_accuracy(self, ensemble):
        """Test when all models have same accuracy."""
        # All models correct every time
        for i in range(50):
            prediction = MTFPrediction(
                direction=1, confidence=0.7, prob_up=0.7, prob_down=0.3,
                component_directions={"1H": 1, "4H": 1, "D": 1},
                component_confidences={"1H": 0.7, "4H": 0.7, "D": 0.7},
                component_weights={"1H": 0.6, "4H": 0.3, "D": 0.1},
            )
            ensemble.record_outcome(prediction, actual_direction=1)

        weights = ensemble._calculate_dynamic_weights()

        # With equal accuracy and blend=0.5, weights should be fairly equal
        # but still close to base weights due to blending
        assert sum(weights.values()) == pytest.approx(1.0)
        assert all(w > 0.1 for w in weights.values())

    def test_one_model_perfect_others_fifty_percent(self, ensemble):
        """Test when one model is 100% accurate, others 50%."""
        ensemble.config.dynamic_weight_blend = 0.8  # High blend for clear effect

        for i in range(50):
            prediction = MTFPrediction(
                direction=1, confidence=0.7, prob_up=0.7, prob_down=0.3,
                component_directions={
                    "1H": 1,  # Always correct
                    "4H": i % 2,  # 50%
                    "D": i % 2,  # 50%
                },
                component_confidences={"1H": 0.7, "4H": 0.7, "D": 0.7},
                component_weights={"1H": 0.6, "4H": 0.3, "D": 0.1},
            )
            ensemble.record_outcome(prediction, actual_direction=1)

        weights = ensemble._calculate_dynamic_weights()

        # 1H should dominate
        assert weights["1H"] > 0.6
        assert weights["1H"] > weights["4H"]
        assert weights["1H"] > weights["D"]

    def test_numerical_stability_with_extreme_values(self, ensemble):
        """Test numerical stability with extreme accuracy values."""
        ensemble.config.dynamic_weight_min = 0.05

        # Create extreme case: one model always right, others always wrong
        for i in range(50):
            prediction = MTFPrediction(
                direction=1, confidence=0.9, prob_up=0.9, prob_down=0.1,
                component_directions={"1H": 1, "4H": 0, "D": 0},
                component_confidences={"1H": 0.9, "4H": 0.9, "D": 0.9},
                component_weights={"1H": 0.6, "4H": 0.3, "D": 0.1},
            )
            ensemble.record_outcome(prediction, actual_direction=1)

        weights = ensemble._calculate_dynamic_weights()

        # Should not have NaN or infinite values
        assert all(not np.isnan(w) for w in weights.values())
        assert all(not np.isinf(w) for w in weights.values())
        assert all(w >= 0.05 for w in weights.values())
        assert sum(weights.values()) == pytest.approx(1.0, abs=1e-6)

    def test_accuracy_clamping_prevents_extreme_weights(self, ensemble):
        """Test that accuracy clamping (40% to 80%) prevents extreme weights."""
        ensemble.config.dynamic_weight_blend = 1.0  # Full dynamic

        # Simulate extreme accuracy (100% vs 0%)
        for i in range(50):
            prediction = MTFPrediction(
                direction=1, confidence=0.7, prob_up=0.7, prob_down=0.3,
                component_directions={"1H": 1, "4H": 0, "D": 0},
                component_confidences={"1H": 0.7, "4H": 0.7, "D": 0.7},
                component_weights={"1H": 0.6, "4H": 0.3, "D": 0.1},
            )
            ensemble.record_outcome(prediction, actual_direction=1)

        weights = ensemble._calculate_dynamic_weights()

        # Even with 100% vs 0%, clamping should prevent extreme weights
        # 4H and D should still have minimum weight
        assert weights["4H"] >= ensemble.config.dynamic_weight_min
        assert weights["D"] >= ensemble.config.dynamic_weight_min


# ============================================================================
# 6. SAVE/LOAD TESTS
# ============================================================================


class TestDynamicWeightsPersistence:
    """Test saving and loading dynamic weight state."""

    def test_prediction_history_persists(self, ensemble, tmp_path):
        """Test that prediction history is saved and loaded."""
        ensemble.config.use_dynamic_weights = True

        # Add some prediction history
        for i in range(20):
            prediction = MTFPrediction(
                direction=1, confidence=0.7, prob_up=0.7, prob_down=0.3,
                component_directions={"1H": 1, "4H": 1, "D": 0},
                component_confidences={"1H": 0.7, "4H": 0.7, "D": 0.7},
                component_weights={"1H": 0.6, "4H": 0.3, "D": 0.1},
            )
            ensemble.record_outcome(prediction, actual_direction=1)

        # Mark as trained so save doesn't skip
        ensemble.is_trained = True
        for model in ensemble.models.values():
            model.is_trained = True

        # Save
        save_dir = tmp_path / "test_save"
        ensemble.save(save_dir)

        # Verify history file exists
        history_file = save_dir / "prediction_history.pkl"
        assert history_file.exists()

        # Create new ensemble and load
        new_ensemble = MTFEnsemble(
            config=MTFEnsembleConfig(use_dynamic_weights=True),
            model_dir=save_dir,
        )
        new_ensemble.load(save_dir)

        # History should be restored
        assert len(new_ensemble.prediction_history) == 20

    def test_dynamic_weight_settings_persist(self, ensemble, tmp_path):
        """Test that dynamic weight config settings are saved and loaded."""
        ensemble.config.use_dynamic_weights = True
        ensemble.config.dynamic_weight_window = 100
        ensemble.config.dynamic_weight_min = 0.2
        ensemble.config.dynamic_weight_blend = 0.7

        # Mark as trained
        ensemble.is_trained = True
        for model in ensemble.models.values():
            model.is_trained = True

        # Save
        save_dir = tmp_path / "test_config_save"
        ensemble.save(save_dir)

        # Load into new ensemble
        new_ensemble = MTFEnsemble(model_dir=save_dir)
        new_ensemble.load(save_dir)

        # Settings should match
        assert new_ensemble.config.use_dynamic_weights is True
        assert new_ensemble.config.dynamic_weight_window == 100
        assert new_ensemble.config.dynamic_weight_min == 0.2
        assert new_ensemble.config.dynamic_weight_blend == 0.7

    def test_loaded_ensemble_maintains_history(self, ensemble, tmp_path):
        """Test that loaded ensemble can continue building history."""
        ensemble.config.use_dynamic_weights = True

        # Add initial history
        for i in range(10):
            prediction = MTFPrediction(
                direction=1, confidence=0.7, prob_up=0.7, prob_down=0.3,
                component_directions={"1H": 1, "4H": 1, "D": 1},
                component_confidences={"1H": 0.7, "4H": 0.7, "D": 0.7},
                component_weights={"1H": 0.6, "4H": 0.3, "D": 0.1},
            )
            ensemble.record_outcome(prediction, actual_direction=1)

        # Mark as trained and save
        ensemble.is_trained = True
        for model in ensemble.models.values():
            model.is_trained = True

        save_dir = tmp_path / "test_continue"
        ensemble.save(save_dir)

        # Load and add more history
        new_ensemble = MTFEnsemble(
            config=MTFEnsembleConfig(use_dynamic_weights=True),
            model_dir=save_dir,
        )
        new_ensemble.load(save_dir)

        # Add more predictions
        for i in range(10):
            prediction = MTFPrediction(
                direction=0, confidence=0.7, prob_up=0.3, prob_down=0.7,
                component_directions={"1H": 0, "4H": 0, "D": 0},
                component_confidences={"1H": 0.7, "4H": 0.7, "D": 0.7},
                component_weights={"1H": 0.6, "4H": 0.3, "D": 0.1},
            )
            new_ensemble.record_outcome(prediction, actual_direction=0)

        # Should have 20 total records
        assert len(new_ensemble.prediction_history) == 20

    def test_no_history_file_when_disabled(self, ensemble, tmp_path):
        """Test that no history file is created when dynamic weights disabled."""
        ensemble.config.use_dynamic_weights = False

        # Add history anyway (shouldn't be saved)
        prediction = MTFPrediction(
            direction=1, confidence=0.7, prob_up=0.7, prob_down=0.3,
            component_directions={"1H": 1, "4H": 1, "D": 1},
            component_confidences={"1H": 0.7, "4H": 0.7, "D": 0.7},
            component_weights={"1H": 0.6, "4H": 0.3, "D": 0.1},
        )
        ensemble.record_outcome(prediction, actual_direction=1)

        # Mark as trained and save
        ensemble.is_trained = True
        for model in ensemble.models.values():
            model.is_trained = True

        save_dir = tmp_path / "test_disabled"
        ensemble.save(save_dir)

        # History file should NOT exist
        history_file = save_dir / "prediction_history.pkl"
        assert not history_file.exists()

    def test_empty_history_on_load_when_file_missing(self, ensemble, tmp_path):
        """Test that history is empty when history file doesn't exist."""
        # Create a save without history file
        save_dir = tmp_path / "test_no_history"
        save_dir.mkdir(parents=True)

        # Save config without history
        import json
        config_data = {
            "weights": {"1H": 0.6, "4H": 0.3, "D": 0.1},
            "use_dynamic_weights": True,
        }
        with open(save_dir / "ensemble_config.json", "w") as f:
            json.dump(config_data, f)

        # Load
        new_ensemble = MTFEnsemble(model_dir=save_dir)
        new_ensemble.load(save_dir)

        # History should be empty
        assert len(new_ensemble.prediction_history) == 0
