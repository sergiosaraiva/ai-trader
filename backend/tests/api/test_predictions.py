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

    def test_prediction_history_returns_should_trade(self):
        """Test prediction history endpoint returns should_trade field."""
        from src.api.routes import predictions
        from src.api.database.models import Prediction
        from src.api.database.session import get_db
        from datetime import datetime

        # Create test predictions with should_trade values
        def mock_get_db():
            mock_db = Mock()

            # Create mock predictions
            pred1 = Mock(spec=Prediction)
            pred1.id = 1
            pred1.timestamp = datetime(2024, 1, 15, 10, 0)
            pred1.symbol = "EURUSD"
            pred1.direction = "long"
            pred1.confidence = 0.75
            pred1.market_price = 1.08543
            pred1.trade_executed = True
            pred1.should_trade = True

            pred2 = Mock(spec=Prediction)
            pred2.id = 2
            pred2.timestamp = datetime(2024, 1, 15, 11, 0)
            pred2.symbol = "EURUSD"
            pred2.direction = "short"
            pred2.confidence = 0.65
            pred2.market_price = 1.08321
            pred2.trade_executed = False
            pred2.should_trade = False

            mock_query = Mock()
            mock_query.count.return_value = 2
            mock_query.order_by.return_value = mock_query
            mock_query.offset.return_value = mock_query
            mock_query.limit.return_value = mock_query
            mock_query.all.return_value = [pred1, pred2]

            mock_db.query.return_value = mock_query

            yield mock_db

        app = FastAPI()
        app.include_router(predictions.router)
        app.dependency_overrides[get_db] = mock_get_db
        client = TestClient(app)

        try:
            response = client.get("/predictions/history")

            assert response.status_code == 200
            data = response.json()

            assert data["count"] == 2
            assert data["total"] == 2

            # Verify should_trade field is present
            predictions_list = data["predictions"]
            assert len(predictions_list) == 2

            # First prediction should have should_trade=True
            assert predictions_list[0]["should_trade"] is True
            assert predictions_list[0]["confidence"] == 0.75

            # Second prediction should have should_trade=False
            assert predictions_list[1]["should_trade"] is False
            assert predictions_list[1]["confidence"] == 0.65
        finally:
            app.dependency_overrides.clear()

    def test_prediction_history_fallback_logic_for_old_records(self):
        """Test prediction history uses confidence fallback for records without should_trade."""
        from src.api.routes import predictions
        from src.api.database.models import Prediction
        from src.api.database.session import get_db
        from datetime import datetime

        # Create test predictions without should_trade (old records)
        def mock_get_db():
            mock_db = Mock()

            # Old record: high confidence (>= 0.70) -> should_trade=True
            pred1 = Mock(spec=Prediction)
            pred1.id = 1
            pred1.timestamp = datetime(2024, 1, 15, 10, 0)
            pred1.symbol = "EURUSD"
            pred1.direction = "long"
            pred1.confidence = 0.72
            pred1.market_price = 1.08543
            pred1.trade_executed = True
            pred1.should_trade = None  # Old record without should_trade

            # Old record: low confidence (< 0.70) -> should_trade=False
            pred2 = Mock(spec=Prediction)
            pred2.id = 2
            pred2.timestamp = datetime(2024, 1, 15, 11, 0)
            pred2.symbol = "EURUSD"
            pred2.direction = "short"
            pred2.confidence = 0.65
            pred2.market_price = 1.08321
            pred2.trade_executed = False
            pred2.should_trade = None  # Old record without should_trade

            # Edge case: exactly 0.70 confidence -> should_trade=True
            pred3 = Mock(spec=Prediction)
            pred3.id = 3
            pred3.timestamp = datetime(2024, 1, 15, 12, 0)
            pred3.symbol = "EURUSD"
            pred3.direction = "long"
            pred3.confidence = 0.70
            pred3.market_price = 1.08412
            pred3.trade_executed = False
            pred3.should_trade = None  # Old record without should_trade

            mock_query = Mock()
            mock_query.count.return_value = 3
            mock_query.order_by.return_value = mock_query
            mock_query.offset.return_value = mock_query
            mock_query.limit.return_value = mock_query
            mock_query.all.return_value = [pred1, pred2, pred3]

            mock_db.query.return_value = mock_query

            yield mock_db

        app = FastAPI()
        app.include_router(predictions.router)
        app.dependency_overrides[get_db] = mock_get_db
        client = TestClient(app)

        try:
            response = client.get("/predictions/history")

            assert response.status_code == 200
            data = response.json()

            assert data["count"] == 3
            predictions_list = data["predictions"]

            # First prediction: confidence 0.72 -> should_trade=True
            assert predictions_list[0]["confidence"] == 0.72
            assert predictions_list[0]["should_trade"] is True

            # Second prediction: confidence 0.65 -> should_trade=False
            assert predictions_list[1]["confidence"] == 0.65
            assert predictions_list[1]["should_trade"] is False

            # Third prediction: confidence 0.70 -> should_trade=True (edge case)
            assert predictions_list[2]["confidence"] == 0.70
            assert predictions_list[2]["should_trade"] is True
        finally:
            app.dependency_overrides.clear()

    def test_prediction_history_handles_mixed_records(self):
        """Test prediction history handles mix of new records with should_trade and old records without."""
        from src.api.routes import predictions
        from src.api.database.models import Prediction
        from src.api.database.session import get_db
        from datetime import datetime

        def mock_get_db():
            mock_db = Mock()

            # New record with should_trade=True
            pred1 = Mock(spec=Prediction)
            pred1.id = 1
            pred1.timestamp = datetime(2024, 1, 15, 10, 0)
            pred1.symbol = "EURUSD"
            pred1.direction = "long"
            pred1.confidence = 0.75
            pred1.market_price = 1.08543
            pred1.trade_executed = True
            pred1.should_trade = True

            # Old record without should_trade, confidence 0.68 (below threshold)
            pred2 = Mock(spec=Prediction)
            pred2.id = 2
            pred2.timestamp = datetime(2024, 1, 15, 11, 0)
            pred2.symbol = "EURUSD"
            pred2.direction = "short"
            pred2.confidence = 0.68
            pred2.market_price = 1.08321
            pred2.trade_executed = False
            pred2.should_trade = None

            # New record with should_trade=False (but high confidence)
            # This tests that explicit should_trade=False is respected
            pred3 = Mock(spec=Prediction)
            pred3.id = 3
            pred3.timestamp = datetime(2024, 1, 15, 12, 0)
            pred3.symbol = "EURUSD"
            pred3.direction = "long"
            pred3.confidence = 0.73
            pred3.market_price = 1.08412
            pred3.trade_executed = False
            pred3.should_trade = False  # Explicitly False despite high confidence

            mock_query = Mock()
            mock_query.count.return_value = 3
            mock_query.order_by.return_value = mock_query
            mock_query.offset.return_value = mock_query
            mock_query.limit.return_value = mock_query
            mock_query.all.return_value = [pred1, pred2, pred3]

            mock_db.query.return_value = mock_query

            yield mock_db

        app = FastAPI()
        app.include_router(predictions.router)
        app.dependency_overrides[get_db] = mock_get_db
        client = TestClient(app)

        try:
            response = client.get("/predictions/history")

            assert response.status_code == 200
            data = response.json()

            predictions_list = data["predictions"]
            assert len(predictions_list) == 3

            # New record with explicit should_trade=True
            assert predictions_list[0]["should_trade"] is True
            assert predictions_list[0]["confidence"] == 0.75

            # Old record with fallback (confidence < 0.70 -> should_trade=False)
            assert predictions_list[1]["should_trade"] is False
            assert predictions_list[1]["confidence"] == 0.68

            # New record with explicit should_trade=False (respected despite high confidence)
            assert predictions_list[2]["should_trade"] is False
            assert predictions_list[2]["confidence"] == 0.73
        finally:
            app.dependency_overrides.clear()
