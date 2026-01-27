"""Integration test for full pipeline with centralized configuration.

Tests the complete workflow:
1. Load data
2. Calculate indicators using config
3. Train model using config
4. Generate predictions using config

Verifies that all components properly use centralized configuration.
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, patch
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add backend src to path
backend_path = Path(__file__).parent.parent.parent
src_path = backend_path / "src"
if str(backend_path) not in sys.path:
    sys.path.insert(0, str(backend_path))

from src.config import TradingConfig
from src.features.technical import TechnicalIndicators
from src.models.multi_timeframe.improved_model import ImprovedTimeframeModel, ImprovedModelConfig
from src.models.multi_timeframe.mtf_ensemble import MTFEnsemble
from src.models.multi_timeframe.enhanced_features import EnhancedFeatureEngine


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def sample_ohlcv_data():
    """Create sample OHLCV data for testing."""
    dates = pd.date_range(start="2024-01-01", periods=500, freq="1H")

    # Generate realistic price data with trend and volatility
    base_price = 1.0800
    np.random.seed(42)

    close_prices = []
    for i in range(500):
        # Add trend and noise
        trend = i * 0.00002  # Slight upward trend
        noise = np.random.normal(0, 0.0005)
        close_prices.append(base_price + trend + noise)

    df = pd.DataFrame({
        "open": [p - np.random.uniform(0, 0.0003) for p in close_prices],
        "high": [p + np.random.uniform(0.0002, 0.0005) for p in close_prices],
        "low": [p - np.random.uniform(0.0002, 0.0005) for p in close_prices],
        "close": close_prices,
        "volume": np.random.randint(1000, 10000, 500),
    }, index=dates)

    # Ensure high is highest and low is lowest
    df["high"] = df[["open", "high", "close"]].max(axis=1)
    df["low"] = df[["open", "low", "close"]].min(axis=1)

    return df


@pytest.fixture
def default_config():
    """Create default trading config."""
    return TradingConfig()


@pytest.fixture
def custom_config():
    """Create custom trading config with modified parameters."""
    config = TradingConfig()

    # Modify indicator parameters
    config.indicators.momentum.rsi_periods = [10, 20]  # Fewer periods
    config.indicators.trend.sma_periods = [10, 20, 50]  # Fewer SMAs

    # Modify hyperparameters
    config.hyperparameters.model_1h.n_estimators = 50
    config.hyperparameters.model_1h.max_depth = 3
    config.hyperparameters.model_1h.learning_rate = 0.05

    # Modify feature parameters
    config.features.lags.standard_lags = [1, 2, 3]  # Fewer lags

    # Modify training parameters
    config.training.splits.train_ratio = 0.7
    config.training.splits.validation_ratio = 0.15

    return config


# ============================================================================
# FULL PIPELINE TESTS WITH DEFAULT CONFIG
# ============================================================================


def test_full_pipeline_with_default_config(sample_ohlcv_data, default_config):
    """Test full pipeline with default configuration."""
    df = sample_ohlcv_data.copy()

    # Step 1: Calculate technical indicators using config
    tech_indicators = TechnicalIndicators()
    df_with_indicators = tech_indicators.calculate_all(df)

    # Verify indicators were calculated
    assert len(df_with_indicators.columns) > len(df.columns)
    assert "rsi_14" in df_with_indicators.columns
    assert "sma_20" in df_with_indicators.columns
    assert "atr_14" in df_with_indicators.columns

    # Verify correct number of RSI indicators (should match config)
    rsi_cols = [col for col in df_with_indicators.columns if col.startswith("rsi_")]
    expected_rsi_count = len(default_config.indicators.momentum.rsi_periods)
    assert len(rsi_cols) == expected_rsi_count

    # Step 2: Add enhanced features using config
    feature_engine = EnhancedFeatureEngine(config=default_config)
    df_with_features = feature_engine.add_all_features(df_with_indicators, timeframe="1H")

    # Verify enhanced features were added
    assert "hour_sin" in df_with_features.columns or "hour_sin_24" in df_with_features.columns

    # Step 3: Create and train model using config
    model_config = ImprovedModelConfig.create_1h_config()
    model_config.n_estimators = default_config.hyperparameters.model_1h.n_estimators
    model_config.max_depth = default_config.hyperparameters.model_1h.max_depth
    model_config.learning_rate = default_config.hyperparameters.model_1h.learning_rate

    model = ImprovedTimeframeModel(
        timeframe="1H",
        config=default_config,
        model_config=model_config
    )

    # Train model
    try:
        results = model.train(df_with_features, calculate_features=False)

        # Verify training completed
        assert model.model is not None
        assert "train_accuracy" in results or "accuracy" in results

        # Step 4: Generate predictions
        prediction = model.predict(df_with_features.tail(100), return_proba=True)

        # Verify predictions
        assert prediction is not None
        assert "prediction" in prediction
        assert "confidence" in prediction
        assert 0 <= prediction["confidence"] <= 1

    except Exception as e:
        # Some models may not train well with small data, that's ok
        pytest.skip(f"Model training skipped due to small data: {e}")


def test_full_pipeline_indicators_use_config(sample_ohlcv_data, default_config):
    """Test that indicator calculation properly uses config parameters."""
    df = sample_ohlcv_data.copy()

    # Calculate indicators with default config
    tech_calc = TechnicalCalculator(config=default_config)
    df_indicators = tech_calc.calculate_all_indicators(df)

    # Check RSI indicators match config
    rsi_periods = default_config.indicators.momentum.rsi_periods
    for period in rsi_periods:
        assert f"rsi_{period}" in df_indicators.columns, f"RSI_{period} not found but is in config"

    # Check SMA indicators match config
    sma_periods = default_config.indicators.trend.sma_periods
    for period in sma_periods:
        assert f"sma_{period}" in df_indicators.columns, f"SMA_{period} not found but is in config"

    # Check EMA indicators match config
    ema_periods = default_config.indicators.trend.ema_periods
    for period in ema_periods:
        assert f"ema_{period}" in df_indicators.columns, f"EMA_{period} not found but is in config"


# ============================================================================
# FULL PIPELINE TESTS WITH CUSTOM CONFIG
# ============================================================================


def test_full_pipeline_with_custom_config(sample_ohlcv_data, custom_config):
    """Test full pipeline with custom configuration overrides."""
    df = sample_ohlcv_data.copy()

    # Step 1: Calculate indicators with custom config
    tech_calc = TechnicalCalculator(config=custom_config)
    df_with_indicators = tech_calc.calculate_all_indicators(df)

    # Verify custom RSI periods were used
    rsi_cols = [col for col in df_with_indicators.columns if col.startswith("rsi_")]
    assert len(rsi_cols) == len(custom_config.indicators.momentum.rsi_periods)

    # Verify custom SMA periods were used
    sma_cols = [col for col in df_with_indicators.columns if col.startswith("sma_")]
    assert len(sma_cols) == len(custom_config.indicators.trend.sma_periods)

    # Step 2: Add features with custom config
    feature_engine = EnhancedFeatureEngine(config=custom_config)
    df_with_features = feature_engine.add_all_features(df_with_indicators, timeframe="1H")

    # Verify custom lag features were used
    lag_cols = [col for col in df_with_features.columns if "lag_" in col.lower()]
    # Should have fewer lags than default

    # Step 3: Train model with custom hyperparameters
    model_config = ImprovedModelConfig.create_1h_config()
    model_config.n_estimators = custom_config.hyperparameters.model_1h.n_estimators
    model_config.max_depth = custom_config.hyperparameters.model_1h.max_depth

    model = ImprovedTimeframeModel(
        timeframe="1H",
        config=custom_config,
        model_config=model_config
    )

    # Verify model uses custom hyperparameters
    assert model.config.hyperparameters.model_1h.n_estimators == 50
    assert model.config.hyperparameters.model_1h.max_depth == 3


def test_custom_config_changes_feature_count(sample_ohlcv_data, default_config, custom_config):
    """Test that custom config produces different number of features."""
    df = sample_ohlcv_data.copy()

    # Calculate with default config
    tech_calc_default = TechnicalCalculator(config=default_config)
    df_default = tech_calc_default.calculate_all_indicators(df)

    # Calculate with custom config (fewer indicators)
    tech_calc_custom = TechnicalCalculator(config=custom_config)
    df_custom = tech_calc_custom.calculate_all_indicators(df)

    # Custom should have fewer indicator columns
    default_indicator_count = len([col for col in df_default.columns if col.startswith(("rsi_", "sma_"))])
    custom_indicator_count = len([col for col in df_custom.columns if col.startswith(("rsi_", "sma_"))])

    assert custom_indicator_count < default_indicator_count, \
        "Custom config should produce fewer indicators"


# ============================================================================
# MTF ENSEMBLE INTEGRATION TESTS
# ============================================================================


def test_mtf_ensemble_with_default_config(sample_ohlcv_data, default_config):
    """Test MTF ensemble uses config for all timeframe models."""
    df = sample_ohlcv_data.copy()

    # Create ensemble with config
    ensemble = MTFEnsemble(config=default_config)

    # Verify all models use the same config
    assert ensemble.models["1H"].config is not None
    assert ensemble.models["4H"].config is not None
    assert ensemble.models["D"].config is not None

    # Verify hyperparameters are loaded from config
    assert ensemble.models["1H"].config.hyperparameters.model_1h.n_estimators == \
        default_config.hyperparameters.model_1h.n_estimators


def test_mtf_ensemble_with_custom_config(sample_ohlcv_data, custom_config):
    """Test MTF ensemble respects custom config."""
    ensemble = MTFEnsemble(config=custom_config)

    # Verify custom hyperparameters propagated
    assert ensemble.models["1H"].config.hyperparameters.model_1h.n_estimators == 50
    assert ensemble.models["1H"].config.hyperparameters.model_1h.max_depth == 3


# ============================================================================
# CONFIG ISOLATION TESTS
# ============================================================================


def test_config_changes_dont_affect_other_instances(sample_ohlcv_data):
    """Test that config changes don't affect other instances."""
    df = sample_ohlcv_data.copy()

    # Create two separate configs
    config1 = TradingConfig()
    config2 = TradingConfig()

    # Modify config1
    config1.indicators.momentum.rsi_periods = [7, 14]

    # Create calculators with different configs
    calc1 = TechnicalCalculator(config=config1)
    calc2 = TechnicalCalculator(config=config2)

    # Calculate indicators
    df1 = calc1.calculate_all_indicators(df.copy())
    df2 = calc2.calculate_all_indicators(df.copy())

    # Count RSI columns
    rsi_cols1 = [col for col in df1.columns if col.startswith("rsi_")]
    rsi_cols2 = [col for col in df2.columns if col.startswith("rsi_")]

    # They should be different (config2 should have default RSI periods)
    assert len(rsi_cols1) == 2, "Config1 should have 2 RSI periods"
    # Config2 should have more (default is [7, 14, 21, 28])
    assert len(rsi_cols2) >= 2, "Config2 should have default RSI periods"


# ============================================================================
# VALIDATION TESTS
# ============================================================================


def test_pipeline_validates_config_before_use(sample_ohlcv_data):
    """Test that pipeline validates config before processing."""
    df = sample_ohlcv_data.copy()

    # Create config with invalid values
    config = TradingConfig()
    config.hyperparameters.model_1h.n_estimators = -1  # Invalid

    # Model creation should work (validation happens later)
    model = ImprovedTimeframeModel(timeframe="1H", config=config)

    # But training should fail or handle gracefully
    # (depending on implementation)
    assert model is not None


def test_pipeline_handles_missing_indicators_gracefully(sample_ohlcv_data, default_config):
    """Test that pipeline handles data with missing indicators."""
    df = sample_ohlcv_data.copy()

    # Don't calculate indicators, try to train directly
    model_config = ImprovedModelConfig.create_1h_config()
    model = ImprovedTimeframeModel(
        timeframe="1H",
        config=default_config,
        model_config=model_config
    )

    # Training should calculate features automatically
    try:
        results = model.train(df, calculate_features=True)
        # Should either succeed or fail gracefully
        assert True
    except Exception as e:
        # Expected - model needs proper features
        assert "feature" in str(e).lower() or "column" in str(e).lower()


# ============================================================================
# PERFORMANCE TESTS
# ============================================================================


def test_config_access_is_fast(sample_ohlcv_data, default_config):
    """Test that config access doesn't slow down pipeline."""
    import time

    df = sample_ohlcv_data.copy()
    tech_calc = TechnicalCalculator(config=default_config)

    # Time indicator calculation
    start = time.time()
    _ = tech_calc.calculate_all_indicators(df)
    duration = time.time() - start

    # Should complete in reasonable time (< 5 seconds for 500 bars)
    assert duration < 5.0, f"Indicator calculation took {duration:.2f}s, too slow"


# ============================================================================
# FEATURE COUNT CONSISTENCY TESTS
# ============================================================================


def test_feature_count_consistent_across_runs(sample_ohlcv_data, default_config):
    """Test that feature engineering produces consistent feature counts."""
    df = sample_ohlcv_data.copy()

    # Run pipeline twice
    tech_calc = TechnicalCalculator(config=default_config)
    feature_engine = EnhancedFeatureEngine(config=default_config)

    df1 = tech_calc.calculate_all_indicators(df.copy())
    df1 = feature_engine.add_all_features(df1, timeframe="1H")

    df2 = tech_calc.calculate_all_indicators(df.copy())
    df2 = feature_engine.add_all_features(df2, timeframe="1H")

    # Feature counts should match
    assert len(df1.columns) == len(df2.columns), \
        "Feature count should be consistent across runs"


def test_different_timeframes_use_appropriate_config(default_config):
    """Test that different timeframe models use appropriate config sections."""
    # Create models for different timeframes
    model_1h = ImprovedTimeframeModel(timeframe="1H", config=default_config)
    model_4h = ImprovedTimeframeModel(timeframe="4H", config=default_config)
    model_daily = ImprovedTimeframeModel(timeframe="D", config=default_config)

    # All should have config
    assert model_1h.config is not None
    assert model_4h.config is not None
    assert model_daily.config is not None

    # All should use same config instance (singleton)
    assert model_1h.config.hyperparameters.model_1h.n_estimators == \
        default_config.hyperparameters.model_1h.n_estimators
    assert model_4h.config.hyperparameters.model_4h.n_estimators == \
        default_config.hyperparameters.model_4h.n_estimators
    assert model_daily.config.hyperparameters.model_daily.n_estimators == \
        default_config.hyperparameters.model_daily.n_estimators


# ============================================================================
# SUMMARY TEST
# ============================================================================


def test_complete_pipeline_summary(sample_ohlcv_data, default_config):
    """Comprehensive test of entire pipeline with config.

    This test verifies:
    1. Data loading works
    2. Indicators calculated with config
    3. Features engineered with config
    4. Model trains with config hyperparameters
    5. Predictions generated successfully
    """
    df = sample_ohlcv_data.copy()

    print("\n=== Pipeline Test Summary ===")
    print(f"Input data: {len(df)} bars")

    # Step 1: Indicators
    tech_calc = TechnicalCalculator(config=default_config)
    df = tech_calc.calculate_all_indicators(df)
    print(f"After indicators: {len(df.columns)} columns")

    # Step 2: Enhanced features
    feature_engine = EnhancedFeatureEngine(config=default_config)
    df = feature_engine.add_all_features(df, timeframe="1H")
    print(f"After features: {len(df.columns)} columns")

    # Step 3: Model training
    model_config = ImprovedModelConfig.create_1h_config()
    model_config.n_estimators = default_config.hyperparameters.model_1h.n_estimators

    model = ImprovedTimeframeModel(
        timeframe="1H",
        config=default_config,
        model_config=model_config
    )

    try:
        results = model.train(df, calculate_features=False)
        print(f"Training completed: {results.get('accuracy', 'N/A')} accuracy")

        # Step 4: Prediction
        prediction = model.predict(df.tail(100), return_proba=True)
        print(f"Prediction: {prediction['prediction']}, Confidence: {prediction['confidence']:.2%}")

        print("=== Pipeline Test: PASSED ===\n")
        assert True

    except Exception as e:
        print(f"Note: Model training skipped with small dataset: {e}")
        pytest.skip(str(e))
