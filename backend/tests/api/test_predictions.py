"""Tests for prediction endpoints using FastAPI TestClient."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock
from fastapi.testclient import TestClient
from fastapi import FastAPI


class TestPredictionEndpoints:
    """Test prediction endpoints."""

    @pytest.fixture(autouse=True)
    def setup_mocks(self):
        """Set up service mocks before each test."""
        self.mock_model_service = Mock()
        self.mock_model_service.is_loaded = True
        self.mock_model_service.get_model_info.return_value = {
            "loaded": True,
            "model_dir": "models/mtf_ensemble",
            "weights": {"1H": 0.6, "4H": 0.3, "D": 0.1},
            "agreement_bonus": 0.05,
            "sentiment_enabled": True,
            "sentiment_by_timeframe": {"1H": False, "4H": False, "D": True},
            "models": {
                "1H": {"trained": True, "val_accuracy": 0.67},
                "4H": {"trained": True, "val_accuracy": 0.65},
                "D": {"trained": True, "val_accuracy": 0.62},
            },
            "initialized_at": "2025-01-12T10:00:00Z",
        }

        self.mock_data_service = Mock()
        self.mock_data_service._initialized = True
        self.mock_data_service.get_current_price.return_value = 1.08543
        self.mock_data_service.get_latest_vix.return_value = 15.5

        # Create mock DataFrame with enough data
        self.mock_df = pd.DataFrame({
            "open": np.random.rand(200) + 1.08,
            "high": np.random.rand(200) + 1.085,
            "low": np.random.rand(200) + 1.075,
            "close": np.random.rand(200) + 1.08,
            "volume": np.random.randint(1000, 10000, 200),
        })
        self.mock_data_service.get_data_for_prediction.return_value = self.mock_df

    def test_model_status_endpoint(self):
        """Test model status endpoint returns correct information."""
        from src.api.routes import predictions

        original_model = predictions.model_service
        predictions.model_service = self.mock_model_service

        try:
            app = FastAPI()
            app.include_router(predictions.router)
            client = TestClient(app)

            response = client.get("/models/status")

            assert response.status_code == 200
            data = response.json()
            assert data["loaded"] is True
            assert data["weights"] == {"1H": 0.6, "4H": 0.3, "D": 0.1}
        finally:
            predictions.model_service = original_model

    def test_latest_prediction_model_not_loaded(self):
        """Test latest prediction returns 503 when model not loaded."""
        from src.api.routes import predictions

        original_model = predictions.model_service
        self.mock_model_service.is_loaded = False
        predictions.model_service = self.mock_model_service

        try:
            app = FastAPI()
            app.include_router(predictions.router)
            client = TestClient(app)

            response = client.get("/predictions/latest")

            assert response.status_code == 503
            assert "Model not loaded" in response.json()["detail"]
        finally:
            predictions.model_service = original_model

    def test_latest_prediction_insufficient_data(self):
        """Test latest prediction returns 503 with insufficient data."""
        from src.api.routes import predictions

        original_model = predictions.model_service
        original_data = predictions.data_service

        # Return only 10 rows - insufficient for prediction
        self.mock_data_service.get_data_for_prediction.return_value = self.mock_df.head(10)

        predictions.model_service = self.mock_model_service
        predictions.data_service = self.mock_data_service

        try:
            app = FastAPI()
            app.include_router(predictions.router)
            client = TestClient(app)

            response = client.get("/predictions/latest")

            assert response.status_code == 503
            assert "Insufficient market data" in response.json()["detail"]
        finally:
            predictions.model_service = original_model
            predictions.data_service = original_data

    def test_generate_prediction_model_not_loaded(self):
        """Test manual prediction generation with model not loaded."""
        from src.api.routes import predictions

        original_model = predictions.model_service
        self.mock_model_service.is_loaded = False
        predictions.model_service = self.mock_model_service

        try:
            app = FastAPI()
            app.include_router(predictions.router)
            client = TestClient(app)

            response = client.post("/predictions/generate")

            assert response.status_code == 503
        finally:
            predictions.model_service = original_model
