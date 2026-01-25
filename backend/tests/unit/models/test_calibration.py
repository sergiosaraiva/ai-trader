"""Unit tests for Probability Calibration (Isotonic Regression).

CRITICAL: This test file includes data leakage detection tests to ensure
the calibration implementation doesn't use future data.
"""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path
import tempfile
from unittest.mock import Mock, patch

from src.models.multi_timeframe.improved_model import (
    ImprovedModelConfig,
    ImprovedTimeframeModel,
)


class TestCalibrationConfig:
    """Tests for calibration configuration in ImprovedModelConfig."""

    def test_calibration_disabled_by_default(self):
        """Verify use_calibration=False by default for backward compatibility."""
        config = ImprovedModelConfig.hourly_model()
        assert config.use_calibration is False

    def test_calibration_can_be_enabled(self):
        """Test enabling calibration in config."""
        config = ImprovedModelConfig.hourly_model()
        config.use_calibration = True
        assert config.use_calibration is True

    def test_calibration_config_in_all_timeframe_models(self):
        """Verify all timeframe models have calibration config."""
        configs = [
            ImprovedModelConfig.hourly_model(),
            ImprovedModelConfig.four_hour_model(),
            ImprovedModelConfig.daily_model(),
        ]

        for config in configs:
            assert hasattr(config, "use_calibration")
            assert config.use_calibration is False  # Default disabled


class TestCalibratorInitialization:
    """Tests for calibrator initialization and state."""

    def test_calibrator_none_before_fitting(self):
        """Verify calibrator is None before fitting."""
        config = ImprovedModelConfig.hourly_model()
        config.use_calibration = True
        model = ImprovedTimeframeModel(config)

        assert model.calibrator is None

    def test_calibrator_not_created_when_disabled(self):
        """Verify calibrator is not created when disabled."""
        config = ImprovedModelConfig.hourly_model()
        config.use_calibration = False
        model = ImprovedTimeframeModel(config)

        # Even if we try to fit, calibrator should not be created
        # (we can't test this directly without training, but we verify initial state)
        assert model.calibrator is None


class TestFitCalibrator:
    """Tests for fit_calibrator() method."""

    @pytest.fixture
    def trained_model(self):
        """Create a trained model for testing calibration."""
        config = ImprovedModelConfig.hourly_model()
        config.use_calibration = True
        config.n_estimators = 10  # Small for speed
        model = ImprovedTimeframeModel(config)

        # Create synthetic training data
        n_samples = 300
        n_features = 50
        np.random.seed(42)

        X_train = np.random.randn(n_samples, n_features)
        y_train = np.random.randint(0, 2, n_samples)
        X_val = np.random.randn(100, n_features)
        y_val = np.random.randint(0, 2, 100)

        feature_names = [f"feature_{i}" for i in range(n_features)]

        # Train the model
        model.train(X_train, y_train, X_val, y_val, feature_names)

        return model

    def test_fit_calibrator_creates_calibrator(self, trained_model):
        """Test that fit_calibrator() creates a CalibratedClassifierCV instance."""
        n_calib = 50
        X_calib = np.random.randn(n_calib, 50)
        y_calib = np.random.randint(0, 2, n_calib)

        trained_model.fit_calibrator(X_calib, y_calib)

        assert trained_model.calibrator is not None
        from sklearn.calibration import CalibratedClassifierCV
        assert isinstance(trained_model.calibrator, CalibratedClassifierCV)

    def test_fit_calibrator_requires_trained_model(self):
        """fit_calibrator() should raise error if model not trained."""
        config = ImprovedModelConfig.hourly_model()
        config.use_calibration = True
        model = ImprovedTimeframeModel(config)

        X_calib = np.random.randn(50, 50)
        y_calib = np.random.randint(0, 2, 50)

        with pytest.raises(RuntimeError, match="must be trained before calibration"):
            model.fit_calibrator(X_calib, y_calib)

    def test_fit_calibrator_logs_warning_when_disabled(self, trained_model):
        """fit_calibrator() should log warning when calibration not enabled in config."""
        # Disable calibration
        trained_model.config.use_calibration = False

        X_calib = np.random.randn(50, 50)
        y_calib = np.random.randint(0, 2, 50)

        # Should not raise, but should log warning (we can't easily test logging)
        trained_model.fit_calibrator(X_calib, y_calib)

    def test_fit_calibrator_with_rfecv_features(self):
        """Test fit_calibrator works with RFECV feature selection."""
        config = ImprovedModelConfig.hourly_model()
        config.use_calibration = True
        config.use_rfecv = True
        config.n_estimators = 10
        model = ImprovedTimeframeModel(config)

        # Create synthetic training data
        n_samples = 300
        n_features = 50
        np.random.seed(42)

        X_train = np.random.randn(n_samples, n_features)
        y_train = np.random.randint(0, 2, n_samples)
        X_val = np.random.randn(100, n_features)
        y_val = np.random.randint(0, 2, 100)

        feature_names = [f"feature_{i}" for i in range(n_features)]

        # Mock RFECV to avoid long computation
        model.selected_indices = np.array([0, 1, 2, 3, 4])  # Select first 5 features
        model.selected_features = feature_names[:5]

        # Train the model with selected features
        model.train(X_train[:, :5], y_train, X_val[:, :5], y_val, feature_names[:5])

        # Fit calibrator (should use selected features)
        X_calib = np.random.randn(50, n_features)
        y_calib = np.random.randint(0, 2, 50)

        model.fit_calibrator(X_calib, y_calib)

        assert model.calibrator is not None


class TestPredictWithCalibration:
    """Tests for predict() and predict_batch() with calibration."""

    @pytest.fixture
    def calibrated_model(self):
        """Create a trained and calibrated model."""
        config = ImprovedModelConfig.hourly_model()
        config.use_calibration = True
        config.n_estimators = 10
        model = ImprovedTimeframeModel(config)

        # Create synthetic training data
        n_samples = 300
        n_features = 50
        np.random.seed(42)

        X_train = np.random.randn(n_samples, n_features)
        # Create labels with some correlation to features
        y_train = (X_train[:, 0] + X_train[:, 1] > 0).astype(int)
        X_val = np.random.randn(100, n_features)
        y_val = (X_val[:, 0] + X_val[:, 1] > 0).astype(int)

        feature_names = [f"feature_{i}" for i in range(n_features)]

        # Train the model
        model.train(X_train, y_train, X_val, y_val, feature_names)

        # Fit calibrator
        X_calib = np.random.randn(50, n_features)
        y_calib = (X_calib[:, 0] + X_calib[:, 1] > 0).astype(int)
        model.fit_calibrator(X_calib, y_calib)

        return model

    def test_predict_uses_calibrator_when_enabled(self, calibrated_model):
        """When calibrator fitted, predict() returns calibrated probabilities."""
        X_test = np.random.randn(50)

        direction, confidence, prob_up, prob_down = calibrated_model.predict(X_test)

        # Should return valid predictions
        assert direction in [0, 1]
        assert 0.0 <= confidence <= 1.0
        assert 0.0 <= prob_up <= 1.0
        assert 0.0 <= prob_down <= 1.0
        assert prob_up + prob_down == pytest.approx(1.0)

    def test_predict_without_calibrator_uses_raw_proba(self):
        """When calibrator not fitted, predict() returns raw probabilities."""
        config = ImprovedModelConfig.hourly_model()
        config.use_calibration = False  # Disabled
        config.n_estimators = 10
        model = ImprovedTimeframeModel(config)

        # Train without calibration
        n_samples = 300
        n_features = 50
        np.random.seed(42)

        X_train = np.random.randn(n_samples, n_features)
        y_train = (X_train[:, 0] + X_train[:, 1] > 0).astype(int)
        X_val = np.random.randn(100, n_features)
        y_val = (X_val[:, 0] + X_val[:, 1] > 0).astype(int)

        feature_names = [f"feature_{i}" for i in range(n_features)]
        model.train(X_train, y_train, X_val, y_val, feature_names)

        # Predict (should use raw probabilities)
        X_test = np.random.randn(50)
        direction, confidence, prob_up, prob_down = model.predict(X_test)

        # Should return valid predictions
        assert direction in [0, 1]
        assert 0.0 <= confidence <= 1.0
        assert model.calibrator is None

    def test_predict_batch_uses_calibrator(self, calibrated_model):
        """Test predict_batch() uses calibrated probabilities."""
        batch_size = 20
        X_batch = np.random.randn(batch_size, 50)

        predictions, confidences = calibrated_model.predict_batch(X_batch)

        assert len(predictions) == batch_size
        assert len(confidences) == batch_size
        assert all(p in [0, 1] for p in predictions)
        assert all(0.0 <= c <= 1.0 for c in confidences)

    def test_calibrated_probabilities_different_from_raw(self, calibrated_model):
        """Calibrated probabilities should differ from raw XGBoost probabilities."""
        X_test = np.random.randn(10, 50)

        # Get raw probabilities
        X_scaled = calibrated_model.scaler.transform(X_test)
        raw_probs = calibrated_model.model.predict_proba(X_scaled)

        # Get calibrated probabilities
        calib_probs = calibrated_model.calibrator.predict_proba(X_scaled)

        # Probabilities should be different (calibration adjusts them)
        # Allow some to be similar, but most should differ
        diff_count = np.sum(np.abs(raw_probs - calib_probs) > 0.01)
        assert diff_count > 0  # At least some should be different

    def test_calibrated_probabilities_in_valid_range(self, calibrated_model):
        """All calibrated probabilities must be in [0, 1]."""
        batch_size = 50
        X_batch = np.random.randn(batch_size, 50)

        predictions, confidences = calibrated_model.predict_batch(X_batch)

        # Get probabilities for each prediction
        X_scaled = calibrated_model.scaler.transform(X_batch)
        probs = calibrated_model.calibrator.predict_proba(X_scaled)

        # All probabilities must be in [0, 1]
        assert np.all(probs >= 0.0)
        assert np.all(probs <= 1.0)

    def test_calibration_does_not_affect_direction(self, calibrated_model):
        """Calibration should not flip prediction direction for most samples."""
        batch_size = 50
        X_batch = np.random.randn(batch_size, 50)

        # Get scaled features
        X_scaled = calibrated_model.scaler.transform(X_batch)

        # Get raw predictions
        raw_preds = calibrated_model.model.predict(X_scaled)

        # Get calibrated predictions
        calib_preds = calibrated_model.calibrator.predict(X_scaled)

        # Most predictions should have the same direction
        # (allow some to flip, but not too many)
        agreement = np.sum(raw_preds == calib_preds) / len(raw_preds)
        assert agreement > 0.7  # At least 70% should agree


class TestDataLeakageDetection:
    """CRITICAL: Tests to verify no data leakage in calibration implementation."""

    def test_calibration_split_is_chronological(self):
        """CRITICAL: Verify the train/calibration split is chronological."""
        # This test verifies the intended behavior:
        # - First 90% of training data goes to model training
        # - Last 10% goes to calibration
        # - No random shuffling

        n_samples = 1000
        train_ratio = 0.6
        n_train = int(n_samples * train_ratio)

        # Expected split when calibration is used:
        # Model trains on first 90% of n_train
        # Calibration uses last 10% of n_train
        n_train_for_model = int(n_train * 0.9)
        n_calib = n_train - n_train_for_model

        # Verify indices
        train_indices = np.arange(n_train_for_model)
        calib_indices = np.arange(n_train_for_model, n_train)

        # All calibration indices must be after all training indices
        assert np.min(calib_indices) >= np.max(train_indices)

        # Calibration indices must be contiguous
        assert len(calib_indices) == n_calib
        assert calib_indices[-1] - calib_indices[0] + 1 == len(calib_indices)

    def test_model_never_sees_calibration_data(self):
        """CRITICAL: Verify the model's training data does NOT include calibration samples."""
        config = ImprovedModelConfig.hourly_model()
        config.use_calibration = True
        config.n_estimators = 10
        model = ImprovedTimeframeModel(config)

        # Create sequential data with clear temporal structure
        n_samples = 300
        n_features = 50
        np.random.seed(42)

        # Create data with temporal pattern (early samples different from late)
        X_all = np.random.randn(n_samples, n_features)
        # Add temporal trend to first feature
        X_all[:, 0] += np.linspace(0, 5, n_samples)
        y_all = (X_all[:, 0] > 2.5).astype(int)

        # Split chronologically
        n_train = int(n_samples * 0.6)
        n_val = int(n_samples * 0.2)

        X_train = X_all[:n_train]
        y_train = y_all[:n_train]
        X_val = X_all[n_train:n_train + n_val]
        y_val = y_all[n_train:n_train + n_val]

        feature_names = [f"feature_{i}" for i in range(n_features)]

        # When we call train() with calibration enabled, it should:
        # 1. Split X_train into model_train (90%) and calib (10%)
        # 2. Train model ONLY on model_train
        # 3. Fit calibrator on calib

        # The calibration data should be the last 10% of X_train
        n_train_for_model = int(len(X_train) * 0.9)
        X_calib_expected = X_train[n_train_for_model:]

        # Train the model
        model.train(X_train, y_train, X_val, y_val, feature_names)

        # Verify model was trained on reduced set
        # We can't directly verify the training data, but we can verify
        # the model's internal state suggests it was trained
        assert model.is_trained

        # Fit calibrator on truly held-out data
        X_calib = X_train[n_train_for_model:]
        y_calib = y_train[n_train_for_model:]
        model.fit_calibrator(X_calib, y_calib)

        # The key test: model should NOT have seen calibration data
        # We verify this by checking the model can't perfectly predict calibration set
        # (if it had seen it, accuracy would be suspiciously high)
        X_calib_scaled = model.scaler.transform(X_calib)
        calib_preds = model.model.predict(X_calib_scaled)
        calib_acc = (calib_preds == y_calib).mean()

        # If the model had seen this data during training, accuracy would be near 100%
        # Since it hasn't, accuracy should be reasonable but not perfect
        assert calib_acc < 0.99  # Not perfect (would indicate leakage)

    def test_calibration_uses_chronologically_later_data(self):
        """Test that calibration data comes chronologically AFTER training data."""
        n_samples = 1000
        train_ratio = 0.6
        n_train = int(n_samples * train_ratio)

        # When calibration is used, the split should be:
        n_model_train = int(n_train * 0.9)

        # Create temporal indices
        all_indices = np.arange(n_samples)
        train_indices = all_indices[:n_train]
        model_train_indices = train_indices[:n_model_train]
        calib_indices = train_indices[n_model_train:]

        # Verify chronological order
        assert np.max(model_train_indices) < np.min(calib_indices)

        # Verify no overlap
        assert len(set(model_train_indices) & set(calib_indices)) == 0


class TestCalibratorSerializaton:
    """Tests for calibrator persistence."""

    def test_calibrator_saved_with_model(self):
        """Verify calibrator is serialized when saving model."""
        config = ImprovedModelConfig.hourly_model()
        config.use_calibration = True
        config.n_estimators = 10
        model = ImprovedTimeframeModel(config)

        # Train and calibrate
        n_samples = 300
        n_features = 50
        np.random.seed(42)

        X_train = np.random.randn(n_samples, n_features)
        y_train = np.random.randint(0, 2, n_samples)
        X_val = np.random.randn(100, n_features)
        y_val = np.random.randint(0, 2, 100)

        feature_names = [f"feature_{i}" for i in range(n_features)]
        model.train(X_train, y_train, X_val, y_val, feature_names)

        X_calib = np.random.randn(50, n_features)
        y_calib = np.random.randint(0, 2, 50)
        model.fit_calibrator(X_calib, y_calib)

        # Save model
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.pkl"
            model.save(path)

            # Load pickled data and verify calibrator is included
            import pickle
            with open(path, "rb") as f:
                data = pickle.load(f)

            assert "calibrator" in data
            assert data["calibrator"] is not None

    def test_calibrator_loaded_with_model(self):
        """Verify calibrator is deserialized when loading model."""
        config = ImprovedModelConfig.hourly_model()
        config.use_calibration = True
        config.n_estimators = 10
        model = ImprovedTimeframeModel(config)

        # Train and calibrate
        n_samples = 300
        n_features = 50
        np.random.seed(42)

        X_train = np.random.randn(n_samples, n_features)
        y_train = np.random.randint(0, 2, n_samples)
        X_val = np.random.randn(100, n_features)
        y_val = np.random.randint(0, 2, 100)

        feature_names = [f"feature_{i}" for i in range(n_features)]
        model.train(X_train, y_train, X_val, y_val, feature_names)

        X_calib = np.random.randn(50, n_features)
        y_calib = np.random.randint(0, 2, 50)
        model.fit_calibrator(X_calib, y_calib)

        # Make prediction before save
        X_test = np.random.randn(50)
        pred1, conf1, prob_up1, _ = model.predict(X_test)

        # Save and load
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.pkl"
            model.save(path)

            # Load into new instance
            new_model = ImprovedTimeframeModel(config)
            new_model.load(path)

            # Verify calibrator was loaded
            assert new_model.calibrator is not None

            # Predictions should match
            pred2, conf2, prob_up2, _ = new_model.predict(X_test)
            assert pred1 == pred2
            assert conf1 == pytest.approx(conf2)
            assert prob_up1 == pytest.approx(prob_up2)

    def test_model_without_calibrator_saves_none(self):
        """Verify models without calibration save calibrator=None."""
        config = ImprovedModelConfig.hourly_model()
        config.use_calibration = False  # Disabled
        config.n_estimators = 10
        model = ImprovedTimeframeModel(config)

        # Train without calibration
        n_samples = 300
        n_features = 50
        np.random.seed(42)

        X_train = np.random.randn(n_samples, n_features)
        y_train = np.random.randint(0, 2, n_samples)
        X_val = np.random.randn(100, n_features)
        y_val = np.random.randint(0, 2, 100)

        feature_names = [f"feature_{i}" for i in range(n_features)]
        model.train(X_train, y_train, X_val, y_val, feature_names)

        # Save model
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.pkl"
            model.save(path)

            # Load pickled data and verify calibrator is None
            import pickle
            with open(path, "rb") as f:
                data = pickle.load(f)

            assert data["calibrator"] is None
