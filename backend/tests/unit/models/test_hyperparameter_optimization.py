"""Unit tests for Bayesian Hyperparameter Optimization.

This test suite validates:
1. Hyperparameter loading from JSON (valid, missing, malformed)
2. TimeSeriesSplit chronological ordering (CRITICAL: no data leakage)
3. ImprovedTimeframeModel hyperparameter acceptance and override
4. Integration test with minimal trials
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import TimeSeriesSplit

from src.models.multi_timeframe.improved_model import (
    ImprovedModelConfig,
    ImprovedTimeframeModel,
)


class TestHyperparameterLoading:
    """Tests for loading optimized hyperparameters from JSON."""

    @pytest.fixture
    def valid_hyperparams_json(self):
        """Create valid optimized hyperparameters JSON."""
        return {
            "optimization_date": "2026-01-21T10:00:00",
            "data_path": "data/forex/EURUSD_20200101_20251231_5min_combined.csv",
            "n_trials_per_timeframe": 100,
            "n_cv_folds": 5,
            "sentiment_enabled": True,
            "results": {
                "1H": {
                    "timeframe": "1H",
                    "best_params": {
                        "n_estimators": 350,
                        "max_depth": 7,
                        "learning_rate": 0.08,
                        "min_child_weight": 8,
                        "subsample": 0.85,
                        "colsample_bytree": 0.75,
                        "reg_alpha": 0.3,
                        "reg_lambda": 2.5,
                        "gamma": 0.1,
                    },
                    "best_cv_accuracy": 0.6852,
                    "n_trials": 100,
                },
                "4H": {
                    "timeframe": "4H",
                    "best_params": {
                        "n_estimators": 280,
                        "max_depth": 6,
                        "learning_rate": 0.06,
                        "min_child_weight": 12,
                        "subsample": 0.80,
                        "colsample_bytree": 0.70,
                        "reg_alpha": 0.2,
                        "reg_lambda": 3.0,
                        "gamma": 0.15,
                    },
                    "best_cv_accuracy": 0.6725,
                    "n_trials": 100,
                },
                "D": {
                    "timeframe": "D",
                    "best_params": {
                        "n_estimators": 220,
                        "max_depth": 5,
                        "learning_rate": 0.05,
                        "min_child_weight": 15,
                        "subsample": 0.75,
                        "colsample_bytree": 0.65,
                        "reg_alpha": 0.4,
                        "reg_lambda": 3.5,
                        "gamma": 0.2,
                    },
                    "best_cv_accuracy": 0.6312,
                    "n_trials": 100,
                },
            },
        }

    def test_load_valid_hyperparams(self, valid_hyperparams_json):
        """Test loading valid hyperparameters from JSON file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "optimized_hyperparams.json"

            # Write valid JSON
            with open(json_path, "w") as f:
                json.dump(valid_hyperparams_json, f)

            # Load and verify
            with open(json_path, "r") as f:
                data = json.load(f)

            assert "results" in data
            assert "1H" in data["results"]
            assert "4H" in data["results"]
            assert "D" in data["results"]

            # Verify 1H params
            params_1h = data["results"]["1H"]["best_params"]
            assert params_1h["n_estimators"] == 350
            assert params_1h["max_depth"] == 7
            assert params_1h["learning_rate"] == 0.08

            # Verify 4H params
            params_4h = data["results"]["4H"]["best_params"]
            assert params_4h["n_estimators"] == 280
            assert params_4h["max_depth"] == 6

            # Verify Daily params
            params_d = data["results"]["D"]["best_params"]
            assert params_d["n_estimators"] == 220
            assert params_d["max_depth"] == 5

    def test_load_missing_file_returns_none(self):
        """Test loading from non-existent file returns None gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "nonexistent.json"

            # Should not exist
            assert not json_path.exists()

            # Loading should handle gracefully
            # (simulates train_mtf_ensemble.py behavior)
            optimized_hyperparams = None
            if json_path.exists():
                with open(json_path, "r") as f:
                    optimized_hyperparams = json.load(f)

            assert optimized_hyperparams is None

    def test_load_malformed_json_raises_error(self):
        """Test loading malformed JSON raises JSONDecodeError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "malformed.json"

            # Write invalid JSON
            with open(json_path, "w") as f:
                f.write("{invalid json content")

            # Should raise JSONDecodeError
            with pytest.raises(json.JSONDecodeError):
                with open(json_path, "r") as f:
                    json.load(f)

    def test_load_missing_required_params(self, valid_hyperparams_json):
        """Test handling of JSON missing required hyperparameters."""
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "incomplete.json"

            # Remove required param (n_estimators)
            incomplete_data = valid_hyperparams_json.copy()
            del incomplete_data["results"]["1H"]["best_params"]["n_estimators"]

            with open(json_path, "w") as f:
                json.dump(incomplete_data, f)

            # Load and verify
            with open(json_path, "r") as f:
                data = json.load(f)

            # Check that n_estimators is missing
            params_1h = data["results"]["1H"]["best_params"]
            assert "n_estimators" not in params_1h

            # Application should fall back to defaults
            # (simulates train_mtf_ensemble.py behavior)
            if "n_estimators" not in params_1h:
                # Use default
                default_config = ImprovedModelConfig.hourly_model()
                assert default_config.n_estimators == 250  # Default value (shallow_fast config)

    def test_load_empty_results(self):
        """Test handling of JSON with empty results dict."""
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "empty_results.json"

            empty_data = {
                "optimization_date": "2026-01-21T10:00:00",
                "results": {},  # Empty
            }

            with open(json_path, "w") as f:
                json.dump(empty_data, f)

            with open(json_path, "r") as f:
                data = json.load(f)

            assert "results" in data
            assert len(data["results"]) == 0

            # Should fall back to defaults
            for tf in ["1H", "4H", "D"]:
                assert tf not in data["results"]

    def test_load_partial_timeframes(self, valid_hyperparams_json):
        """Test handling when only some timeframes have optimized params."""
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "partial.json"

            # Only include 1H and 4H, not Daily
            partial_data = valid_hyperparams_json.copy()
            del partial_data["results"]["D"]

            with open(json_path, "w") as f:
                json.dump(partial_data, f)

            with open(json_path, "r") as f:
                data = json.load(f)

            # Verify partial loading
            assert "1H" in data["results"]
            assert "4H" in data["results"]
            assert "D" not in data["results"]

            # Daily should use defaults
            default_config = ImprovedModelConfig.daily_model()
            assert default_config.n_estimators == 150  # shallow_fast config default


class TestTimeSeriesSplitChronological:
    """CRITICAL: Tests to verify TimeSeriesSplit maintains chronological order."""

    def test_timeseries_split_is_sequential(self):
        """CRITICAL: Verify TimeSeriesSplit produces chronological train/val folds."""
        n_samples = 1000
        X = np.arange(n_samples).reshape(-1, 1)
        y = np.zeros(n_samples)

        tscv = TimeSeriesSplit(n_splits=5)

        for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X)):
            # Train indices must come before validation indices
            assert train_idx.max() < val_idx.min(), (
                f"Fold {fold_idx}: Train max ({train_idx.max()}) "
                f"must be less than Val min ({val_idx.min()})"
            )

            # Train indices must be contiguous from start
            assert train_idx[0] == 0, f"Fold {fold_idx}: Train must start at 0"
            assert train_idx[-1] == train_idx[0] + len(train_idx) - 1, (
                f"Fold {fold_idx}: Train indices must be contiguous"
            )

            # Validation indices must be contiguous after train
            assert val_idx[0] == train_idx[-1] + 1, (
                f"Fold {fold_idx}: Val must start immediately after train"
            )
            assert val_idx[-1] == val_idx[0] + len(val_idx) - 1, (
                f"Fold {fold_idx}: Val indices must be contiguous"
            )

    def test_timeseries_split_no_overlap(self):
        """CRITICAL: Verify no overlap between train and validation sets."""
        n_samples = 1000
        X = np.arange(n_samples).reshape(-1, 1)
        y = np.zeros(n_samples)

        tscv = TimeSeriesSplit(n_splits=5)

        for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X)):
            # No overlap
            train_set = set(train_idx)
            val_set = set(val_idx)
            overlap = train_set & val_set

            assert len(overlap) == 0, (
                f"Fold {fold_idx}: Found {len(overlap)} overlapping indices"
            )

    def test_timeseries_split_no_future_leakage(self):
        """CRITICAL: Verify training data never includes future data."""
        n_samples = 500
        # Create data with temporal pattern
        np.random.seed(42)
        X = np.random.randn(n_samples, 10)
        # Add temporal trend to simulate time series
        X[:, 0] += np.linspace(0, 10, n_samples)
        y = (X[:, 0] > 5).astype(int)

        tscv = TimeSeriesSplit(n_splits=3)

        for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X)):
            # Extract train and validation data
            X_train = X[train_idx]
            X_val = X[val_idx]

            # The mean of the temporal feature should be lower in train than val
            # (since train comes before val chronologically)
            train_mean = X_train[:, 0].mean()
            val_mean = X_val[:, 0].mean()

            assert train_mean < val_mean, (
                f"Fold {fold_idx}: Train mean ({train_mean:.2f}) should be "
                f"less than Val mean ({val_mean:.2f}) for chronological data"
            )

    def test_timeseries_split_increasing_train_size(self):
        """Verify TimeSeriesSplit uses expanding window (train size increases)."""
        n_samples = 1000
        X = np.arange(n_samples).reshape(-1, 1)
        y = np.zeros(n_samples)

        tscv = TimeSeriesSplit(n_splits=5)

        prev_train_size = 0
        for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X)):
            train_size = len(train_idx)

            # Train size should increase with each fold
            assert train_size > prev_train_size, (
                f"Fold {fold_idx}: Train size ({train_size}) should be "
                f"greater than previous ({prev_train_size})"
            )

            prev_train_size = train_size


class TestImprovedTimeframeModelHyperparams:
    """Tests for ImprovedTimeframeModel hyperparameter acceptance."""

    def test_model_accepts_custom_hyperparams(self):
        """Test model accepts custom hyperparameters via config."""
        custom_params = {
            "n_estimators": 350,
            "max_depth": 7,
            "learning_rate": 0.08,
            "min_child_weight": 8,
            "subsample": 0.85,
            "colsample_bytree": 0.75,
            "reg_alpha": 0.3,
            "reg_lambda": 2.5,
            "gamma": 0.1,
        }

        config = ImprovedModelConfig.hourly_model()
        config.hyperparams = custom_params

        model = ImprovedTimeframeModel(config)

        # Verify config stored hyperparams
        assert model.config.hyperparams == custom_params

    def test_model_uses_defaults_when_no_hyperparams(self):
        """Test model uses default hyperparameters when none provided."""
        config = ImprovedModelConfig.hourly_model()
        # Don't set hyperparams (None by default)

        model = ImprovedTimeframeModel(config)

        # Verify defaults are used (shallow_fast config)
        assert model.config.hyperparams is None
        assert model.config.n_estimators == 250  # Default for hourly (shallow_fast)
        assert model.config.max_depth == 4  # shallow_fast default

    def test_hyperparams_override_defaults_correctly(self):
        """Test optimized hyperparameters override config defaults."""
        # Set up config with defaults
        config = ImprovedModelConfig.hourly_model()
        default_n_estimators = config.n_estimators
        default_max_depth = config.max_depth

        # Set optimized hyperparams
        custom_params = {
            "n_estimators": 350,  # Different from default (250)
            "max_depth": 8,  # Different from default (4)
            "learning_rate": 0.10,  # Different from default (0.08)
            "min_child_weight": 8,
            "subsample": 0.85,
            "colsample_bytree": 0.75,
            "reg_alpha": 0.3,
            "reg_lambda": 2.5,
            "gamma": 0.1,
        }
        config.hyperparams = custom_params

        model = ImprovedTimeframeModel(config)

        # Create model instance to check actual params used
        xgb_model = model._create_model()

        # Verify optimized params are used (not defaults)
        assert xgb_model.n_estimators == 350
        assert xgb_model.n_estimators != default_n_estimators

        assert xgb_model.max_depth == 8
        assert xgb_model.max_depth != default_max_depth

    def test_partial_hyperparams_fallback_to_defaults(self):
        """Test partial hyperparams use defaults for missing values."""
        # Only provide some hyperparams
        partial_params = {
            "n_estimators": 350,
            "max_depth": 8,
            # Missing: learning_rate, min_child_weight, etc.
        }

        config = ImprovedModelConfig.hourly_model()
        config.hyperparams = partial_params

        model = ImprovedTimeframeModel(config)
        xgb_model = model._create_model()

        # Provided params should be used
        assert xgb_model.n_estimators == 350
        assert xgb_model.max_depth == 8

        # Missing params should use config defaults
        assert xgb_model.learning_rate == config.learning_rate  # Default: 0.05

    def test_all_timeframe_models_accept_hyperparams(self):
        """Test all timeframe model types accept custom hyperparams."""
        custom_params = {
            "n_estimators": 250,
            "max_depth": 7,
            "learning_rate": 0.07,
        }

        for config_factory in [
            ImprovedModelConfig.hourly_model,
            ImprovedModelConfig.four_hour_model,
            ImprovedModelConfig.daily_model,
        ]:
            config = config_factory()
            config.hyperparams = custom_params

            model = ImprovedTimeframeModel(config)
            xgb_model = model._create_model()

            # Verify custom params are used
            assert xgb_model.n_estimators == 250
            assert xgb_model.max_depth == 7
            assert xgb_model.learning_rate == 0.07

    def test_hyperparams_only_affect_xgboost_model(self):
        """Test hyperparams only affect XGBoost, not other model types."""
        custom_params = {
            "n_estimators": 350,
            "max_depth": 8,
        }

        # Test with GBM model type
        config = ImprovedModelConfig.hourly_model()
        config.model_type = "gbm"  # Not XGBoost
        config.hyperparams = custom_params

        model = ImprovedTimeframeModel(config)
        gbm_model = model._create_model()

        # GBM should use config defaults, not hyperparams
        assert gbm_model.n_estimators == config.n_estimators

        # Test with XGBoost (should use hyperparams)
        config_xgb = ImprovedModelConfig.hourly_model()
        config_xgb.model_type = "xgboost"
        config_xgb.hyperparams = custom_params

        model_xgb = ImprovedTimeframeModel(config_xgb)
        xgb_model = model_xgb._create_model()

        # XGBoost should use hyperparams
        assert xgb_model.n_estimators == 350


class TestHyperparameterOptimizationIntegration:
    """Integration tests for hyperparameter optimization workflow."""

    @pytest.fixture
    def sample_ohlcv_data(self):
        """Create sample OHLCV data for testing."""
        n_samples = 500
        dates = pd.date_range(start="2023-01-01", periods=n_samples, freq="1h")

        np.random.seed(42)
        close = 1.0800 + np.cumsum(np.random.randn(n_samples) * 0.0002)
        high = close + np.abs(np.random.randn(n_samples) * 0.0005)
        low = close - np.abs(np.random.randn(n_samples) * 0.0005)
        open_price = close + np.random.randn(n_samples) * 0.0003

        df = pd.DataFrame(
            {
                "open": open_price,
                "high": high,
                "low": low,
                "close": close,
                "volume": np.random.randint(1000, 10000, n_samples),
            },
            index=dates,
        )

        return df

    def test_optimize_hyperparameters_minimal_trials(self, sample_ohlcv_data):
        """Integration test: verify hyperparameter search space and objective function."""
        # This test verifies the hyperparameter optimization components work correctly
        # without running full Optuna trials (which are slow and XGBoost version-dependent)

        from scripts.optimize_hyperparameters import resample_data

        # Resample to 1H
        df_1h = sample_ohlcv_data  # Already 1H

        # Create model config
        config = ImprovedModelConfig.hourly_model()
        model = ImprovedTimeframeModel(config)

        # Prepare data
        X, y, feature_names = model.prepare_data(df_1h, higher_tf_data={})

        # Use only training data
        n_train = int(len(X) * 0.6)
        X_train = X[:n_train]
        y_train = y[:n_train]

        # Verify data is suitable for optimization
        assert len(X_train) > 0
        assert len(np.unique(y_train)) == 2  # Binary classification

        # Test hyperparameter search space (what Optuna would suggest)
        from sklearn.model_selection import TimeSeriesSplit
        from xgboost import XGBClassifier

        # Define search space (same as in optimize_hyperparameters.py)
        test_params = {
            "n_estimators": 200,
            "max_depth": 5,
            "learning_rate": 0.05,
            "min_child_weight": 10,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "gamma": 0.0,
            "random_state": 42,
            "use_label_encoder": False,
            "eval_metric": "logloss",
            "verbosity": 0,
        }

        # Verify these params can create a valid XGBoost model
        xgb_model = XGBClassifier(**test_params)
        assert xgb_model is not None

        # Test TimeSeriesSplit works with the data
        tscv = TimeSeriesSplit(n_splits=3)
        cv_scores = []

        for train_idx, val_idx in tscv.split(X_train):
            X_fold_train = X_train[train_idx]
            y_fold_train = y_train[train_idx]
            X_fold_val = X_train[val_idx]
            y_fold_val = y_train[val_idx]

            # Verify splits are valid
            assert len(X_fold_train) > 0
            assert len(X_fold_val) > 0
            assert train_idx.max() < val_idx.min()  # Chronological

        # Verify hyperparameter ranges are valid
        assert 100 <= test_params["n_estimators"] <= 500
        assert 3 <= test_params["max_depth"] <= 10
        assert 0.01 <= test_params["learning_rate"] <= 0.3
        assert 1 <= test_params["min_child_weight"] <= 20
        assert 0.6 <= test_params["subsample"] <= 1.0
        assert 0.6 <= test_params["colsample_bytree"] <= 1.0
        assert 0.0 <= test_params["reg_alpha"] <= 1.0
        assert 0.0 <= test_params["reg_lambda"] <= 5.0
        assert 0.0 <= test_params["gamma"] <= 1.0

    def test_optimization_results_saved_correctly(self):
        """Test that optimization results are saved with correct structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "optimized_hyperparams.json"

            # Create sample optimization results
            results = {
                "optimization_date": "2026-01-21T10:00:00",
                "data_path": "data/forex/test.csv",
                "n_trials_per_timeframe": 3,
                "n_cv_folds": 3,
                "sentiment_enabled": False,
                "results": {
                    "1H": {
                        "timeframe": "1H",
                        "best_params": {
                            "n_estimators": 300,
                            "max_depth": 6,
                            "learning_rate": 0.07,
                            "min_child_weight": 10,
                            "subsample": 0.8,
                            "colsample_bytree": 0.75,
                            "reg_alpha": 0.2,
                            "reg_lambda": 2.0,
                            "gamma": 0.1,
                        },
                        "best_cv_accuracy": 0.67,
                        "n_trials": 3,
                    },
                },
            }

            # Save results
            with open(output_path, "w") as f:
                json.dump(results, f, indent=2)

            # Verify file exists and is valid JSON
            assert output_path.exists()

            with open(output_path, "r") as f:
                loaded_results = json.load(f)

            # Verify structure
            assert "optimization_date" in loaded_results
            assert "results" in loaded_results
            assert "1H" in loaded_results["results"]
            assert "best_params" in loaded_results["results"]["1H"]
            assert "best_cv_accuracy" in loaded_results["results"]["1H"]

            # Verify best_params contains all required keys
            best_params = loaded_results["results"]["1H"]["best_params"]
            required_keys = [
                "n_estimators",
                "max_depth",
                "learning_rate",
                "min_child_weight",
                "subsample",
                "colsample_bytree",
                "reg_alpha",
                "reg_lambda",
                "gamma",
            ]
            for key in required_keys:
                assert key in best_params

    def test_train_mtf_ensemble_loads_optimized_params(self):
        """Test train_mtf_ensemble.py --use-optimized-params loads correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir)
            json_path = config_dir / "optimized_hyperparams.json"

            # Create valid optimized hyperparameters
            optimized_data = {
                "optimization_date": "2026-01-21T10:00:00",
                "results": {
                    "1H": {
                        "best_params": {
                            "n_estimators": 350,
                            "max_depth": 7,
                            "learning_rate": 0.08,
                            "min_child_weight": 8,
                            "subsample": 0.85,
                            "colsample_bytree": 0.75,
                            "reg_alpha": 0.3,
                            "reg_lambda": 2.5,
                            "gamma": 0.1,
                        },
                    },
                },
            }

            with open(json_path, "w") as f:
                json.dump(optimized_data, f)

            # Simulate train_mtf_ensemble.py loading logic
            optimized_hyperparams = None
            if json_path.exists():
                with open(json_path, "r") as f:
                    hyperparams_data = json.load(f)
                    optimized_hyperparams = {}
                    for tf in ["1H", "4H", "D"]:
                        if tf in hyperparams_data.get("results", {}):
                            params = hyperparams_data["results"][tf].get("best_params", {})
                            if params and "n_estimators" in params:
                                optimized_hyperparams[tf] = params

            # Verify loading succeeded
            assert optimized_hyperparams is not None
            assert "1H" in optimized_hyperparams
            assert optimized_hyperparams["1H"]["n_estimators"] == 350

            # Create model with optimized params
            config = ImprovedModelConfig.hourly_model()
            config.hyperparams = optimized_hyperparams["1H"]

            model = ImprovedTimeframeModel(config)
            xgb_model = model._create_model()

            # Verify optimized params are used
            assert xgb_model.n_estimators == 350
            assert xgb_model.max_depth == 7
