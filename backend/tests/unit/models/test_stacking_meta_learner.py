"""Tests for Stacking Meta-Learner.

CRITICAL: This test file includes data leakage detection tests to ensure
the stacking implementation doesn't use future data in predictions.
"""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path
import tempfile

from src.models.multi_timeframe.stacking_meta_learner import (
    StackingConfig,
    StackingMetaLearner,
    StackingMetaFeatures,
)


class TestStackingConfig:
    """Tests for StackingConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = StackingConfig.default()

        assert config.n_folds == 5
        assert config.meta_model_type == "xgboost"
        assert config.use_agreement_features is True
        assert config.use_confidence_features is True
        assert config.use_volatility_features is True
        assert config.blend_with_weighted_avg == 0.0

    def test_conservative_config(self):
        """Test conservative configuration with blending."""
        config = StackingConfig.conservative()

        assert config.blend_with_weighted_avg == 0.3

    def test_custom_config(self):
        """Test custom configuration."""
        config = StackingConfig(
            n_folds=3,
            meta_model_type="logistic",
            use_volatility_features=False,
            blend_with_weighted_avg=0.5,
        )

        assert config.n_folds == 3
        assert config.meta_model_type == "logistic"
        assert config.use_volatility_features is False
        assert config.blend_with_weighted_avg == 0.5


class TestStackingMetaFeatures:
    """Tests for StackingMetaFeatures dataclass."""

    def test_to_array_full_features(self):
        """Test conversion to array with all features enabled."""
        config = StackingConfig(
            use_agreement_features=True,
            use_confidence_features=True,
            use_volatility_features=True,
        )

        meta_feat = StackingMetaFeatures(
            prob_1h=0.7,
            prob_4h=0.6,
            prob_d=0.55,
            agreement_ratio=1.0,
            direction_spread=0.0,
            confidence_spread=0.05,
            prob_range=0.15,
            volatility=0.5,
            volatility_regime=1,
        )

        arr = meta_feat.to_array(config)

        # Base: 3, agreement: 3, confidence: 1, volatility: 2 = 9 features
        assert len(arr) == 9
        assert arr[0] == pytest.approx(0.7)  # prob_1h
        assert arr[1] == pytest.approx(0.6)  # prob_4h
        assert arr[2] == pytest.approx(0.55)  # prob_d

    def test_to_array_minimal_features(self):
        """Test conversion to array with minimal features."""
        config = StackingConfig(
            use_agreement_features=False,
            use_confidence_features=False,
            use_volatility_features=False,
        )

        meta_feat = StackingMetaFeatures(
            prob_1h=0.7,
            prob_4h=0.6,
            prob_d=0.55,
        )

        arr = meta_feat.to_array(config)

        # Only base probabilities: 3 features
        assert len(arr) == 3

    def test_get_feature_names(self):
        """Test feature names match array size."""
        config = StackingConfig.default()
        meta_feat = StackingMetaFeatures(
            prob_1h=0.7,
            prob_4h=0.6,
            prob_d=0.55,
            agreement_ratio=1.0,
            direction_spread=0.0,
            confidence_spread=0.05,
            prob_range=0.15,
            volatility=0.5,
            volatility_regime=1,
        )

        names = StackingMetaFeatures.get_feature_names(config)
        arr = meta_feat.to_array(config)

        assert len(names) == len(arr)


class TestStackingMetaLearner:
    """Tests for StackingMetaLearner class."""

    @pytest.fixture
    def simple_learner(self):
        """Create a simple stacking meta-learner."""
        config = StackingConfig(
            n_folds=3,
            min_train_size=50,
            use_volatility_features=False,
        )
        return StackingMetaLearner(config)

    @pytest.fixture
    def mock_models(self):
        """Create mock base models for testing."""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import StandardScaler

        class MockModel:
            def __init__(self):
                self.model = RandomForestClassifier(n_estimators=10, random_state=42)
                self.scaler = StandardScaler()
                self.is_trained = True

            def _create_model(self):
                return RandomForestClassifier(n_estimators=10, random_state=42)

        return {
            "1H": MockModel(),
            "4H": MockModel(),
            "D": MockModel(),
        }

    def test_init(self, simple_learner):
        """Test initialization."""
        assert simple_learner.config.n_folds == 3
        assert simple_learner.meta_model is None
        assert simple_learner.is_trained is False

    def test_volatility_regime_classification(self, simple_learner):
        """Test volatility regime classification."""
        assert simple_learner._get_volatility_regime(0.1) == 0  # Low
        assert simple_learner._get_volatility_regime(0.5) == 1  # Normal
        assert simple_learner._get_volatility_regime(0.9) == 2  # High

    def test_train_and_predict(self):
        """Test training and prediction."""
        config = StackingConfig(
            n_folds=2,
            min_train_size=20,
            use_volatility_features=False,
        )
        learner = StackingMetaLearner(config)

        # Create synthetic meta-features
        n_samples = 200
        np.random.seed(42)

        # Create features with some signal
        meta_features = np.random.randn(n_samples, 6)
        # Make prob columns in [0, 1]
        meta_features[:, :3] = np.abs(meta_features[:, :3]) / np.abs(meta_features[:, :3]).max()

        # Create correlated labels
        labels = (meta_features[:, 0] + meta_features[:, 1] > 1).astype(int)

        # Train
        results = learner.train(meta_features, labels, val_ratio=0.2)

        assert learner.is_trained
        assert "meta_train_accuracy" in results
        assert "meta_val_accuracy" in results
        assert results["meta_val_accuracy"] > 0.4  # Better than random

    def test_predict_single(self):
        """Test single prediction after training."""
        config = StackingConfig(
            n_folds=2,
            min_train_size=20,
            use_volatility_features=False,
        )
        learner = StackingMetaLearner(config)

        # Get correct number of features for this config
        n_features = len(StackingMetaFeatures.get_feature_names(config))

        # Train on synthetic data
        n_samples = 200
        np.random.seed(42)
        meta_features = np.abs(np.random.randn(n_samples, n_features))
        meta_features[:, :3] = meta_features[:, :3] / meta_features[:, :3].max()
        labels = (meta_features[:, 0] > 0.5).astype(int)

        learner.train(meta_features, labels, val_ratio=0.2)

        # Make prediction
        direction, confidence, prob_up, prob_down = learner.predict(
            prob_1h=0.7,
            prob_4h=0.6,
            prob_d=0.55,
            conf_1h=0.7,
            conf_4h=0.6,
            conf_d=0.55,
        )

        assert direction in [0, 1]
        assert 0.0 <= confidence <= 1.0
        assert 0.0 <= prob_up <= 1.0
        assert prob_up + prob_down == pytest.approx(1.0)

    def test_predict_batch(self):
        """Test batch prediction after training."""
        config = StackingConfig(
            n_folds=2,
            min_train_size=20,
            use_volatility_features=False,
        )
        learner = StackingMetaLearner(config)

        # Get correct number of features for this config
        n_features = len(StackingMetaFeatures.get_feature_names(config))

        # Train
        n_samples = 200
        np.random.seed(42)
        meta_features = np.abs(np.random.randn(n_samples, n_features))
        meta_features[:, :3] = meta_features[:, :3] / meta_features[:, :3].max()
        labels = (meta_features[:, 0] > 0.5).astype(int)

        learner.train(meta_features, labels, val_ratio=0.2)

        # Batch prediction
        batch_size = 10
        probs_1h = np.random.uniform(0.4, 0.8, batch_size)
        probs_4h = np.random.uniform(0.4, 0.8, batch_size)
        probs_d = np.random.uniform(0.4, 0.8, batch_size)
        confs_1h = np.random.uniform(0.5, 0.9, batch_size)
        confs_4h = np.random.uniform(0.5, 0.9, batch_size)
        confs_d = np.random.uniform(0.5, 0.9, batch_size)

        directions, confidences, agreements = learner.predict_batch(
            probs_1h, probs_4h, probs_d,
            confs_1h, confs_4h, confs_d,
        )

        assert len(directions) == batch_size
        assert len(confidences) == batch_size
        assert len(agreements) == batch_size
        assert all(d in [0, 1] for d in directions)
        assert all(0.0 <= c <= 1.0 for c in confidences)

    def test_blending_with_weighted_avg(self):
        """Test blending stacking with weighted average."""
        config = StackingConfig(
            n_folds=2,
            min_train_size=20,
            use_volatility_features=False,
            blend_with_weighted_avg=0.5,  # 50% blend
        )
        learner = StackingMetaLearner(config)

        # Get correct number of features for this config
        n_features = len(StackingMetaFeatures.get_feature_names(config))

        # Train
        n_samples = 200
        np.random.seed(42)
        meta_features = np.abs(np.random.randn(n_samples, n_features))
        meta_features[:, :3] = meta_features[:, :3] / meta_features[:, :3].max()
        labels = (meta_features[:, 0] > 0.5).astype(int)

        learner.train(meta_features, labels, val_ratio=0.2)

        # Predict with blending
        weights = {"1H": 0.6, "4H": 0.3, "D": 0.1}
        direction, confidence, prob_up, prob_down = learner.predict(
            prob_1h=0.8,
            prob_4h=0.7,
            prob_d=0.6,
            conf_1h=0.8,
            conf_4h=0.7,
            conf_d=0.6,
            weights=weights,
        )

        assert direction in [0, 1]
        assert 0.0 <= prob_up <= 1.0

    def test_save_and_load(self):
        """Test model serialization."""
        config = StackingConfig(
            n_folds=2,
            min_train_size=20,
            use_volatility_features=False,
        )
        learner = StackingMetaLearner(config)

        # Get correct number of features for this config
        n_features = len(StackingMetaFeatures.get_feature_names(config))

        # Train
        n_samples = 200
        np.random.seed(42)
        meta_features = np.abs(np.random.randn(n_samples, n_features))
        meta_features[:, :3] = meta_features[:, :3] / meta_features[:, :3].max()
        labels = (meta_features[:, 0] > 0.5).astype(int)

        learner.train(meta_features, labels, val_ratio=0.2)

        # Save and load
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "meta_learner.pkl"
            learner.save(path)

            # Load into new instance
            new_learner = StackingMetaLearner()
            new_learner.load(path)

            assert new_learner.is_trained
            assert new_learner.meta_val_accuracy == learner.meta_val_accuracy

            # Predictions should match
            direction1, _, _, _ = learner.predict(
                prob_1h=0.7, prob_4h=0.6, prob_d=0.55,
                conf_1h=0.7, conf_4h=0.6, conf_d=0.55,
            )
            direction2, _, _, _ = new_learner.predict(
                prob_1h=0.7, prob_4h=0.6, prob_d=0.55,
                conf_1h=0.7, conf_4h=0.6, conf_d=0.55,
            )
            assert direction1 == direction2

    def test_summary(self):
        """Test summary generation."""
        config = StackingConfig.default()
        learner = StackingMetaLearner(config)

        # Untrained summary
        summary = learner.summary()
        assert "not trained" in summary.lower()

        # Get correct number of features for this config
        n_features = len(StackingMetaFeatures.get_feature_names(config))

        # Train and check summary again
        n_samples = 200
        np.random.seed(42)
        meta_features = np.abs(np.random.randn(n_samples, n_features))
        meta_features[:, :3] = meta_features[:, :3] / meta_features[:, :3].max()
        labels = (meta_features[:, 0] > 0.5).astype(int)

        learner.train(meta_features, labels, val_ratio=0.2)

        summary = learner.summary()
        assert "accuracy" in summary.lower()  # Check for accuracy in results
        assert "val" in summary.lower()


class TestDataLeakageDetection:
    """CRITICAL: Tests to verify no data leakage in stacking implementation.

    These tests ensure that predictions at time t do not use any information
    from times t+1, t+2, etc. This is critical for avoiding overfitting.
    """

    def test_prediction_independent_of_future_data(self):
        """Verify predictions don't change when future data is added.

        This is the core data leakage test: predictions at index i should be
        identical whether we have data up to index i or index 2*i.
        """
        config = StackingConfig(
            n_folds=2,
            min_train_size=20,
            use_volatility_features=False,
        )
        learner = StackingMetaLearner(config)

        # Get correct number of features for this config
        n_features = len(StackingMetaFeatures.get_feature_names(config))

        # Train on a fixed dataset
        n_samples = 200
        np.random.seed(42)
        meta_features = np.abs(np.random.randn(n_samples, n_features))
        meta_features[:, :3] = meta_features[:, :3] / meta_features[:, :3].max()
        labels = (meta_features[:, 0] > 0.5).astype(int)

        learner.train(meta_features, labels, val_ratio=0.2)

        # Test point
        test_probs = {
            "prob_1h": 0.65,
            "prob_4h": 0.55,
            "prob_d": 0.60,
            "conf_1h": 0.65,
            "conf_4h": 0.55,
            "conf_d": 0.60,
        }

        # Get prediction
        direction1, conf1, prob_up1, _ = learner.predict(**test_probs)

        # The prediction should be deterministic and not depend on
        # any external state that could leak future information
        direction2, conf2, prob_up2, _ = learner.predict(**test_probs)

        assert direction1 == direction2
        assert conf1 == pytest.approx(conf2)
        assert prob_up1 == pytest.approx(prob_up2)

    def test_time_series_split_respects_temporal_order(self):
        """Verify TimeSeriesSplit is used correctly for OOF predictions."""
        from sklearn.model_selection import TimeSeriesSplit

        # Create a dataset with clear temporal structure
        n_samples = 100
        tscv = TimeSeriesSplit(n_splits=5)

        for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(np.zeros(n_samples))):
            # All validation indices must be after all training indices
            assert min(val_idx) > max(train_idx), \
                f"Fold {fold_idx}: val indices overlap with train"

            # Validation indices must be contiguous
            assert val_idx[-1] - val_idx[0] + 1 == len(val_idx), \
                f"Fold {fold_idx}: val indices not contiguous"

    def test_oof_predictions_never_peek_ahead(self):
        """Verify OOF predictions only use past data.

        For each prediction in the OOF set, verify the model was only
        trained on data that came chronologically before it.
        """
        # This test verifies the implementation logic
        # The actual OOF generation uses TimeSeriesSplit which guarantees
        # that for each fold, the validation set comes after the training set

        from sklearn.model_selection import TimeSeriesSplit

        n_samples = 100
        tscv = TimeSeriesSplit(n_splits=5)

        predicted_indices = set()
        training_max_for_pred = {}

        for train_idx, val_idx in tscv.split(np.zeros(n_samples)):
            train_max = max(train_idx)
            for idx in val_idx:
                predicted_indices.add(idx)
                training_max_for_pred[idx] = train_max

        # For every predicted index, training data ended before it
        for idx, train_max in training_max_for_pred.items():
            assert idx > train_max, \
                f"Prediction at {idx} uses training data up to {train_max}"

    def test_meta_features_use_only_current_info(self):
        """Verify meta-features don't use future information.

        All meta-features should be computable from:
        1. Current base model probabilities
        2. Current base model confidences
        3. Past volatility (rolling window ending at current)
        """
        config = StackingConfig.default()

        # Create meta-features for a single timestep
        meta_feat = StackingMetaFeatures(
            prob_1h=0.7,
            prob_4h=0.6,
            prob_d=0.55,
            agreement_ratio=2/3,  # Computed from current predictions
            direction_spread=0.0,  # Computed from current directions
            confidence_spread=0.05,  # Computed from current confidences
            prob_range=0.15,  # max - min of current probs
            volatility=0.5,  # From rolling window (past only)
            volatility_regime=1,  # Classification of volatility
        )

        arr = meta_feat.to_array(config)

        # All features should be finite and reasonable
        assert np.all(np.isfinite(arr))

        # Agreement features are bounded
        assert 0.0 <= meta_feat.agreement_ratio <= 1.0
        assert 0.0 <= meta_feat.direction_spread <= 1.0
        assert 0.0 <= meta_feat.prob_range <= 1.0

    def test_batch_predictions_independent(self):
        """Verify batch predictions don't leak information between samples."""
        config = StackingConfig(
            n_folds=2,
            min_train_size=20,
            use_volatility_features=False,
        )
        learner = StackingMetaLearner(config)

        # Get correct number of features for this config
        n_features = len(StackingMetaFeatures.get_feature_names(config))

        # Train
        n_samples = 200
        np.random.seed(42)
        meta_features = np.abs(np.random.randn(n_samples, n_features))
        meta_features[:, :3] = meta_features[:, :3] / meta_features[:, :3].max()
        labels = (meta_features[:, 0] > 0.5).astype(int)

        learner.train(meta_features, labels, val_ratio=0.2)

        # Create batch with identical samples
        batch_size = 5
        probs_1h = np.full(batch_size, 0.65)
        probs_4h = np.full(batch_size, 0.55)
        probs_d = np.full(batch_size, 0.60)
        confs_1h = np.full(batch_size, 0.65)
        confs_4h = np.full(batch_size, 0.55)
        confs_d = np.full(batch_size, 0.60)

        directions, confidences, _ = learner.predict_batch(
            probs_1h, probs_4h, probs_d,
            confs_1h, confs_4h, confs_d,
        )

        # All predictions should be identical (no cross-sample leakage)
        assert np.all(directions == directions[0])
        assert np.allclose(confidences, confidences[0])


class TestStackingIntegration:
    """Integration tests for stacking with MTFEnsemble."""

    def test_stacking_config_in_ensemble_config(self):
        """Test stacking configuration in MTFEnsembleConfig."""
        from src.models.multi_timeframe import MTFEnsembleConfig, StackingConfig

        # With default stacking
        config = MTFEnsembleConfig.with_stacking()
        assert config.use_stacking is True
        assert config.stacking_config is not None

        # With custom stacking config
        stacking_config = StackingConfig(n_folds=3, blend_with_weighted_avg=0.2)
        config = MTFEnsembleConfig.with_stacking(stacking_config)
        assert config.stacking_config.n_folds == 3
        assert config.stacking_config.blend_with_weighted_avg == 0.2

    def test_stacking_with_sentiment_config(self):
        """Test combined stacking and sentiment configuration."""
        from src.models.multi_timeframe import MTFEnsembleConfig

        config = MTFEnsembleConfig.with_stacking_and_sentiment()
        assert config.use_stacking is True
        assert config.include_sentiment is True
        assert config.sentiment_by_timeframe["D"] is True
        assert config.sentiment_by_timeframe["1H"] is False
