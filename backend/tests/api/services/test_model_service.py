"""Tests for ModelService focusing on data_timestamp functionality."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock, PropertyMock
from pathlib import Path


@pytest.fixture
def sample_ohlcv_1h():
    """Sample 1H OHLCV DataFrame with datetime index."""
    dates = pd.date_range(start="2024-01-15 10:00:00", periods=100, freq="1h")
    return pd.DataFrame({
        "open": np.linspace(1.085, 1.090, 100),
        "high": np.linspace(1.086, 1.091, 100),
        "low": np.linspace(1.084, 1.089, 100),
        "close": np.linspace(1.0855, 1.0905, 100),
        "volume": np.random.randint(1000, 10000, 100),
    }, index=dates)


@pytest.fixture
def sample_ohlcv_4h():
    """Sample 4H OHLCV DataFrame with datetime index."""
    dates = pd.date_range(start="2024-01-15 00:00:00", periods=50, freq="4h")
    return pd.DataFrame({
        "open": np.linspace(1.085, 1.090, 50),
        "high": np.linspace(1.086, 1.091, 50),
        "low": np.linspace(1.084, 1.089, 50),
        "close": np.linspace(1.0855, 1.0905, 50),
        "volume": np.random.randint(1000, 10000, 50),
    }, index=dates)


@pytest.fixture
def sample_ohlcv_daily():
    """Sample Daily OHLCV DataFrame with datetime index."""
    dates = pd.date_range(start="2024-01-01", periods=30, freq="D")
    return pd.DataFrame({
        "open": np.linspace(1.085, 1.090, 30),
        "high": np.linspace(1.086, 1.091, 30),
        "low": np.linspace(1.084, 1.089, 30),
        "close": np.linspace(1.0855, 1.0905, 30),
        "volume": np.random.randint(1000, 10000, 30),
    }, index=dates)


@pytest.fixture
def mock_prediction_result():
    """Mock prediction result from MTFEnsemble."""
    mock = Mock()
    mock.direction = 1
    mock.confidence = 0.75
    mock.prob_up = 0.75
    mock.prob_down = 0.25
    mock.agreement_count = 3
    mock.agreement_score = 1.0
    mock.all_agree = True
    mock.market_regime = "trending"
    mock.component_directions = {"1H": 1, "4H": 1, "D": 1}
    mock.component_confidences = {"1H": 0.72, "4H": 0.75, "D": 0.68}
    mock.component_weights = {"1H": 0.6, "4H": 0.3, "D": 0.1}
    return mock


class TestModelServiceDataTimestamp:
    """Test ModelService predict_from_pipeline returns data_timestamp."""

    def test_predict_from_pipeline_returns_data_timestamp(
        self, sample_ohlcv_1h, sample_ohlcv_4h, sample_ohlcv_daily, mock_prediction_result
    ):
        """Test predict_from_pipeline includes data_timestamp in response."""
        from src.api.services.model_service import ModelService

        service = ModelService()
        service._initialized = True

        # Mock the ensemble
        mock_ensemble = Mock()
        mock_ensemble.is_trained = True
        mock_ensemble.predict.return_value = mock_prediction_result
        service._ensemble = mock_ensemble

        # Mock pipeline service
        mock_pipeline = Mock()
        mock_pipeline.get_processed_data.side_effect = lambda tf: {
            "1h": sample_ohlcv_1h.copy(),
            "4h": sample_ohlcv_4h.copy(),
            "D": sample_ohlcv_daily.copy(),
        }.get(tf)

        with patch('src.api.services.pipeline_service.pipeline_service', mock_pipeline):
            result = service.predict_from_pipeline(symbol="EURUSD")

            # Assert data_timestamp is present
            assert "data_timestamp" in result
            assert result["data_timestamp"] is not None

            # Assert it's a valid ISO format timestamp
            assert isinstance(result["data_timestamp"], str)
            # Should be parseable as datetime
            parsed_timestamp = datetime.fromisoformat(result["data_timestamp"])
            assert isinstance(parsed_timestamp, datetime)

    def test_predict_from_pipeline_data_timestamp_matches_last_1h_bar(
        self, sample_ohlcv_1h, sample_ohlcv_4h, sample_ohlcv_daily, mock_prediction_result
    ):
        """Test data_timestamp matches the last 1H bar timestamp."""
        from src.api.services.model_service import ModelService

        service = ModelService()
        service._initialized = True

        # Mock the ensemble
        mock_ensemble = Mock()
        mock_ensemble.is_trained = True
        mock_ensemble.predict.return_value = mock_prediction_result
        service._ensemble = mock_ensemble

        # Mock pipeline service
        mock_pipeline = Mock()
        mock_pipeline.get_processed_data.side_effect = lambda tf: {
            "1h": sample_ohlcv_1h.copy(),
            "4h": sample_ohlcv_4h.copy(),
            "D": sample_ohlcv_daily.copy(),
        }.get(tf)

        with patch('src.api.services.pipeline_service.pipeline_service', mock_pipeline):
            result = service.predict_from_pipeline(symbol="EURUSD")

            # Get expected timestamp (last bar from 1H data)
            expected_timestamp = sample_ohlcv_1h.index[-1]
            expected_iso = expected_timestamp.isoformat()

            # Assert data_timestamp matches
            assert result["data_timestamp"] == expected_iso

    def test_predict_from_pipeline_data_timestamp_is_iso_format(
        self, sample_ohlcv_1h, sample_ohlcv_4h, sample_ohlcv_daily, mock_prediction_result
    ):
        """Test data_timestamp is in ISO format."""
        from src.api.services.model_service import ModelService

        service = ModelService()
        service._initialized = True

        # Mock the ensemble
        mock_ensemble = Mock()
        mock_ensemble.is_trained = True
        mock_ensemble.predict.return_value = mock_prediction_result
        service._ensemble = mock_ensemble

        # Mock pipeline service with specific timestamp
        specific_time = pd.Timestamp("2024-01-15 14:00:00")
        df_1h = sample_ohlcv_1h.copy()
        df_1h.index = pd.date_range(end=specific_time, periods=len(df_1h), freq="1h")

        mock_pipeline = Mock()
        mock_pipeline.get_processed_data.side_effect = lambda tf: {
            "1h": df_1h,
            "4h": sample_ohlcv_4h.copy(),
            "D": sample_ohlcv_daily.copy(),
        }.get(tf)

        with patch('src.api.services.pipeline_service.pipeline_service', mock_pipeline):
            result = service.predict_from_pipeline(symbol="EURUSD")

            # Assert timestamp format
            assert result["data_timestamp"] == "2024-01-15T14:00:00"

    def test_predict_from_pipeline_without_1h_data_no_data_timestamp(self):
        """Test predict_from_pipeline without 1H data doesn't crash on data_timestamp."""
        from src.api.services.model_service import ModelService

        service = ModelService()
        service._initialized = True

        # Mock the ensemble
        mock_ensemble = Mock()
        mock_ensemble.is_trained = True
        service._ensemble = mock_ensemble

        # Mock pipeline service returning None for 1H
        mock_pipeline = Mock()
        mock_pipeline.get_processed_data.return_value = None

        # Mock data_service as fallback
        mock_data_service = Mock()
        mock_data_service.get_data_for_prediction.return_value = None

        with patch('src.api.services.pipeline_service.pipeline_service', mock_pipeline):
            with patch('src.api.services.data_service.data_service', mock_data_service):
                # Should raise RuntimeError due to no data
                with pytest.raises(RuntimeError, match="No data available"):
                    service.predict_from_pipeline(symbol="EURUSD")

    def test_predict_from_pipeline_data_source_field(
        self, sample_ohlcv_1h, sample_ohlcv_4h, sample_ohlcv_daily, mock_prediction_result
    ):
        """Test predict_from_pipeline includes data_source='pipeline' field."""
        from src.api.services.model_service import ModelService

        service = ModelService()
        service._initialized = True

        # Mock the ensemble
        mock_ensemble = Mock()
        mock_ensemble.is_trained = True
        mock_ensemble.predict.return_value = mock_prediction_result
        service._ensemble = mock_ensemble

        # Mock pipeline service
        mock_pipeline = Mock()
        mock_pipeline.get_processed_data.side_effect = lambda tf: {
            "1h": sample_ohlcv_1h.copy(),
            "4h": sample_ohlcv_4h.copy(),
            "D": sample_ohlcv_daily.copy(),
        }.get(tf)

        with patch('src.api.services.pipeline_service.pipeline_service', mock_pipeline):
            result = service.predict_from_pipeline(symbol="EURUSD")

            # Assert data_source field
            assert "data_source" in result
            assert result["data_source"] == "pipeline"

    def test_predict_from_pipeline_caches_with_data_timestamp(
        self, sample_ohlcv_1h, sample_ohlcv_4h, sample_ohlcv_daily, mock_prediction_result
    ):
        """Test predict_from_pipeline caching includes data_timestamp."""
        from src.api.services.model_service import ModelService

        service = ModelService()
        service._initialized = True

        # Mock the ensemble
        mock_ensemble = Mock()
        mock_ensemble.is_trained = True
        mock_ensemble.predict.return_value = mock_prediction_result
        service._ensemble = mock_ensemble

        # Mock pipeline service
        mock_pipeline = Mock()
        mock_pipeline.get_processed_data.side_effect = lambda tf: {
            "1h": sample_ohlcv_1h.copy(),
            "4h": sample_ohlcv_4h.copy(),
            "D": sample_ohlcv_daily.copy(),
        }.get(tf)

        with patch('src.api.services.pipeline_service.pipeline_service', mock_pipeline):
            # First call
            result1 = service.predict_from_pipeline(symbol="EURUSD", use_cache=True)
            assert "data_timestamp" in result1

            # Second call (should use cache)
            result2 = service.predict_from_pipeline(symbol="EURUSD", use_cache=True)
            assert "data_timestamp" in result2

            # Should have same data_timestamp
            assert result1["data_timestamp"] == result2["data_timestamp"]

    def test_predict_from_pipeline_fallback_includes_timestamp(
        self, sample_ohlcv_1h, mock_prediction_result
    ):
        """Test predict_from_pipeline fallback to standard predict includes timestamp."""
        from src.api.services.model_service import ModelService

        service = ModelService()
        service._initialized = True

        # Mock the ensemble
        mock_ensemble = Mock()
        mock_ensemble.is_trained = True
        mock_ensemble.predict.return_value = mock_prediction_result
        service._ensemble = mock_ensemble

        # Mock pipeline service raising exception
        mock_pipeline = Mock()
        mock_pipeline.get_processed_data.side_effect = Exception("Pipeline unavailable")

        # Mock data_service for fallback
        mock_data_service = Mock()
        mock_data_service.get_data_for_prediction.return_value = sample_ohlcv_1h.copy()

        with patch('src.api.services.pipeline_service.pipeline_service', mock_pipeline):
            with patch('src.api.services.data_service.data_service', mock_data_service):
                result = service.predict_from_pipeline(symbol="EURUSD")

                # Fallback to standard predict, which includes timestamp field
                assert "timestamp" in result
                assert result["timestamp"] is not None

    def test_predict_standard_does_not_include_data_timestamp(
        self, sample_ohlcv_1h, mock_prediction_result
    ):
        """Test standard predict() method doesn't include data_timestamp (only predict_from_pipeline does)."""
        from src.api.services.model_service import ModelService

        service = ModelService()
        service._initialized = True

        # Mock the ensemble
        mock_ensemble = Mock()
        mock_ensemble.is_trained = True
        mock_ensemble.predict.return_value = mock_prediction_result
        service._ensemble = mock_ensemble

        result = service.predict(sample_ohlcv_1h.copy(), symbol="EURUSD")

        # Standard predict should NOT have data_timestamp
        assert "data_timestamp" not in result
        # But should have timestamp field
        assert "timestamp" in result


class TestModelServicePredictionResponseFields:
    """Test ModelService prediction response includes all required fields."""

    def test_predict_from_pipeline_includes_all_standard_fields(
        self, sample_ohlcv_1h, sample_ohlcv_4h, sample_ohlcv_daily, mock_prediction_result
    ):
        """Test predict_from_pipeline includes all standard prediction fields."""
        from src.api.services.model_service import ModelService

        service = ModelService()
        service._initialized = True

        # Mock the ensemble
        mock_ensemble = Mock()
        mock_ensemble.is_trained = True
        mock_ensemble.predict.return_value = mock_prediction_result
        service._ensemble = mock_ensemble

        # Mock pipeline service
        mock_pipeline = Mock()
        mock_pipeline.get_processed_data.side_effect = lambda tf: {
            "1h": sample_ohlcv_1h.copy(),
            "4h": sample_ohlcv_4h.copy(),
            "D": sample_ohlcv_daily.copy(),
        }.get(tf)

        with patch('src.api.services.pipeline_service.pipeline_service', mock_pipeline):
            result = service.predict_from_pipeline(symbol="EURUSD")

            # Assert all standard fields
            expected_fields = [
                "direction",
                "confidence",
                "prob_up",
                "prob_down",
                "should_trade",
                "agreement_count",
                "agreement_score",
                "all_agree",
                "market_regime",
                "component_directions",
                "component_confidences",
                "component_weights",
                "timestamp",
                "symbol",
            ]

            for field in expected_fields:
                assert field in result, f"Missing field: {field}"

            # Assert new fields
            assert "data_source" in result
            assert "data_timestamp" in result

    def test_predict_from_pipeline_confidence_values_clamped(
        self, sample_ohlcv_1h, sample_ohlcv_4h, sample_ohlcv_daily
    ):
        """Test predict_from_pipeline clamps confidence values to [0, 1] range."""
        from src.api.services.model_service import ModelService

        service = ModelService()
        service._initialized = True

        # Mock prediction with out-of-range values
        mock_prediction = Mock()
        mock_prediction.direction = 1
        mock_prediction.confidence = 1.5  # Out of range
        mock_prediction.prob_up = 1.2  # Out of range
        mock_prediction.prob_down = -0.1  # Out of range
        mock_prediction.agreement_count = 3
        mock_prediction.agreement_score = 1.1  # Out of range
        mock_prediction.all_agree = True
        mock_prediction.market_regime = "trending"
        mock_prediction.component_directions = {"1H": 1, "4H": 1, "D": 1}
        mock_prediction.component_confidences = {"1H": 1.5, "4H": 0.5, "D": -0.2}
        mock_prediction.component_weights = {"1H": 0.6, "4H": 0.3, "D": 0.1}

        # Mock the ensemble
        mock_ensemble = Mock()
        mock_ensemble.is_trained = True
        mock_ensemble.predict.return_value = mock_prediction
        service._ensemble = mock_ensemble

        # Mock pipeline service
        mock_pipeline = Mock()
        mock_pipeline.get_processed_data.side_effect = lambda tf: {
            "1h": sample_ohlcv_1h.copy(),
            "4h": sample_ohlcv_4h.copy(),
            "D": sample_ohlcv_daily.copy(),
        }.get(tf)

        with patch('src.api.services.pipeline_service.pipeline_service', mock_pipeline):
            result = service.predict_from_pipeline(symbol="EURUSD")

            # Assert values are clamped to [0, 1]
            assert 0.0 <= result["confidence"] <= 1.0
            assert 0.0 <= result["prob_up"] <= 1.0
            assert 0.0 <= result["prob_down"] <= 1.0
            assert 0.0 <= result["agreement_score"] <= 1.0

            # Component confidences should also be clamped
            for conf in result["component_confidences"].values():
                assert 0.0 <= conf <= 1.0

    def test_predict_from_pipeline_should_trade_threshold(
        self, sample_ohlcv_1h, sample_ohlcv_4h, sample_ohlcv_daily, mock_prediction_result
    ):
        """Test predict_from_pipeline should_trade uses 70% threshold."""
        from src.api.services.model_service import ModelService

        service = ModelService()
        service._initialized = True

        # Test case 1: confidence >= 0.70 -> should_trade = True
        mock_pred_high = Mock()
        mock_pred_high.direction = 1
        mock_pred_high.confidence = 0.75
        mock_pred_high.prob_up = 0.75
        mock_pred_high.prob_down = 0.25
        mock_pred_high.agreement_count = 3
        mock_pred_high.agreement_score = 1.0
        mock_pred_high.all_agree = True
        mock_pred_high.market_regime = "trending"
        mock_pred_high.component_directions = {"1H": 1, "4H": 1, "D": 1}
        mock_pred_high.component_confidences = {"1H": 0.75, "4H": 0.75, "D": 0.75}
        mock_pred_high.component_weights = {"1H": 0.6, "4H": 0.3, "D": 0.1}

        mock_ensemble = Mock()
        mock_ensemble.is_trained = True
        mock_ensemble.predict.return_value = mock_pred_high
        service._ensemble = mock_ensemble

        mock_pipeline = Mock()
        mock_pipeline.get_processed_data.side_effect = lambda tf: {
            "1h": sample_ohlcv_1h.copy(),
            "4h": sample_ohlcv_4h.copy(),
            "D": sample_ohlcv_daily.copy(),
        }.get(tf)

        with patch('src.api.services.pipeline_service.pipeline_service', mock_pipeline):
            result = service.predict_from_pipeline(symbol="EURUSD")
            assert result["should_trade"] is True

        # Test case 2: confidence < 0.70 -> should_trade = False
        mock_pred_low = Mock()
        mock_pred_low.direction = 1
        mock_pred_low.confidence = 0.65
        mock_pred_low.prob_up = 0.65
        mock_pred_low.prob_down = 0.35
        mock_pred_low.agreement_count = 2
        mock_pred_low.agreement_score = 0.67
        mock_pred_low.all_agree = False
        mock_pred_low.market_regime = "ranging"
        mock_pred_low.component_directions = {"1H": 1, "4H": 1, "D": -1}
        mock_pred_low.component_confidences = {"1H": 0.65, "4H": 0.62, "D": 0.58}
        mock_pred_low.component_weights = {"1H": 0.6, "4H": 0.3, "D": 0.1}

        mock_ensemble.predict.return_value = mock_pred_low

        with patch('src.api.services.pipeline_service.pipeline_service', mock_pipeline):
            result = service.predict_from_pipeline(symbol="EURUSD")
            assert result["should_trade"] is False
