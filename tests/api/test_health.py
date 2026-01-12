"""Tests for health check endpoints using FastAPI TestClient."""

import pytest
import sys
from unittest.mock import patch, Mock, MagicMock
from fastapi.testclient import TestClient
from fastapi import FastAPI


class TestHealthEndpoints:
    """Test health check endpoints."""

    @pytest.fixture(autouse=True)
    def setup_mocks(self):
        """Set up service mocks before each test."""
        # Create mock services
        self.mock_model_service = Mock()
        self.mock_model_service.is_loaded = True
        self.mock_model_service.get_model_info.return_value = {"loaded": True}

        self.mock_data_service = Mock()
        self.mock_data_service._initialized = True
        self.mock_data_service.get_current_price.return_value = 1.08543

        self.mock_trading_service = Mock()
        self.mock_trading_service._initialized = True
        self.mock_trading_service.get_status.return_value = {"status": "running"}

        self.mock_pipeline_service = Mock()
        self.mock_pipeline_service._initialized = True
        self.mock_pipeline_service.get_status.return_value = {"status": "healthy"}

    def _create_app_with_mocks(self):
        """Create a FastAPI app with mocked services."""
        # Patch services at the route module level
        with patch.dict(sys.modules, {
            'src.api.services.model_service': Mock(model_service=self.mock_model_service),
            'src.api.services.data_service': Mock(data_service=self.mock_data_service),
            'src.api.services.trading_service': Mock(trading_service=self.mock_trading_service),
            'src.api.services.pipeline_service': Mock(pipeline_service=self.mock_pipeline_service),
        }):
            # Import route with mocked dependencies
            from src.api.routes import health
            # Manually set the service references
            health.model_service = self.mock_model_service
            health.data_service = self.mock_data_service
            health.trading_service = self.mock_trading_service

            app = FastAPI()
            app.include_router(health.router)
            return app, health

    def test_health_check_returns_healthy(self):
        """Test basic health check returns healthy status."""
        # Import route module
        from src.api.routes import health

        # Replace services with mocks
        original_model = health.model_service
        original_data = health.data_service
        original_trading = health.trading_service

        health.model_service = self.mock_model_service
        health.data_service = self.mock_data_service
        health.trading_service = self.mock_trading_service

        try:
            app = FastAPI()
            app.include_router(health.router)
            client = TestClient(app)

            response = client.get("/health")

            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "healthy"
            assert data["version"] == "1.0.0"
            assert "timestamp" in data
        finally:
            # Restore originals
            health.model_service = original_model
            health.data_service = original_data
            health.trading_service = original_trading

    def test_detailed_health_returns_components(self):
        """Test detailed health returns component status."""
        from src.api.routes import health

        # Replace services with mocks
        original_model = health.model_service
        original_data = health.data_service
        original_trading = health.trading_service

        health.model_service = self.mock_model_service
        health.data_service = self.mock_data_service
        health.trading_service = self.mock_trading_service

        try:
            # Mock database session - it's imported inside the function
            with patch("src.api.database.session.get_session") as mock_get_session:
                mock_db = Mock()
                mock_db.execute.return_value = None
                mock_get_session.return_value = mock_db

                app = FastAPI()
                app.include_router(health.router)
                client = TestClient(app)

                response = client.get("/health/detailed")

            assert response.status_code == 200
            data = response.json()
            assert "components" in data
            assert "api" in data["components"]
        finally:
            health.model_service = original_model
            health.data_service = original_data
            health.trading_service = original_trading

    def test_readiness_check_all_ready(self):
        """Test readiness check when all services ready."""
        from src.api.routes import health

        original_model = health.model_service
        original_data = health.data_service
        original_trading = health.trading_service

        health.model_service = self.mock_model_service
        health.data_service = self.mock_data_service
        health.trading_service = self.mock_trading_service

        try:
            app = FastAPI()
            app.include_router(health.router)
            client = TestClient(app)

            response = client.get("/health/ready")

            assert response.status_code == 200
            data = response.json()
            assert data["ready"] is True
        finally:
            health.model_service = original_model
            health.data_service = original_data
            health.trading_service = original_trading

    def test_readiness_check_not_ready(self):
        """Test readiness check when model not loaded."""
        from src.api.routes import health

        original_model = health.model_service
        original_data = health.data_service
        original_trading = health.trading_service

        # Set model as not loaded
        self.mock_model_service.is_loaded = False

        health.model_service = self.mock_model_service
        health.data_service = self.mock_data_service
        health.trading_service = self.mock_trading_service

        try:
            app = FastAPI()
            app.include_router(health.router)
            client = TestClient(app)

            response = client.get("/health/ready")

            assert response.status_code == 200
            data = response.json()
            assert data["ready"] is False
            assert data["services"]["model"] is False
        finally:
            health.model_service = original_model
            health.data_service = original_data
            health.trading_service = original_trading
