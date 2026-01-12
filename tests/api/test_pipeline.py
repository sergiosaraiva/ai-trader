"""Tests for pipeline endpoints using FastAPI TestClient."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
from fastapi.testclient import TestClient
from fastapi import FastAPI


class TestPipelineEndpoints:
    """Test pipeline endpoints."""

    @pytest.fixture(autouse=True)
    def setup_mocks(self):
        """Set up service mocks before each test."""
        self.mock_pipeline_service = Mock()
        self.mock_pipeline_service._initialized = True
        self.mock_pipeline_service.get_status.return_value = {
            "initialized": True,
            "last_run": "2025-01-12T10:00:00Z",
        }
        self.mock_pipeline_service.run_full_pipeline.return_value = True
        self.mock_pipeline_service.get_processed_data.return_value = None

    def test_pipeline_status_returns_status(self):
        """Test pipeline status endpoint returns correct structure."""
        from src.api.routes import pipeline

        original = pipeline.pipeline_service
        pipeline.pipeline_service = self.mock_pipeline_service

        try:
            app = FastAPI()
            app.include_router(pipeline.router)
            client = TestClient(app)

            response = client.get("/pipeline/status")

            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "ok"
            assert "timestamp" in data
            assert "pipeline" in data
        finally:
            pipeline.pipeline_service = original

    def test_trigger_pipeline_sync_success(self):
        """Test synchronous pipeline trigger with success."""
        from src.api.routes import pipeline

        original = pipeline.pipeline_service
        pipeline.pipeline_service = self.mock_pipeline_service

        try:
            app = FastAPI()
            app.include_router(pipeline.router)
            client = TestClient(app)

            response = client.post("/pipeline/run-sync")

            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "success"
        finally:
            pipeline.pipeline_service = original

    def test_trigger_pipeline_sync_partial(self):
        """Test synchronous pipeline trigger with partial success."""
        from src.api.routes import pipeline

        original = pipeline.pipeline_service
        self.mock_pipeline_service.run_full_pipeline.return_value = False
        pipeline.pipeline_service = self.mock_pipeline_service

        try:
            app = FastAPI()
            app.include_router(pipeline.router)
            client = TestClient(app)

            response = client.post("/pipeline/run-sync")

            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "partial"
        finally:
            pipeline.pipeline_service = original

    def test_get_pipeline_data_invalid_timeframe(self):
        """Test getting pipeline data with invalid timeframe."""
        from src.api.routes import pipeline

        original = pipeline.pipeline_service
        pipeline.pipeline_service = self.mock_pipeline_service

        try:
            app = FastAPI()
            app.include_router(pipeline.router)
            client = TestClient(app)

            response = client.get("/pipeline/data/invalid")

            assert response.status_code == 400
            assert "Invalid timeframe" in response.json()["detail"]
        finally:
            pipeline.pipeline_service = original

    def test_get_pipeline_data_no_data(self):
        """Test getting pipeline data when no data cached."""
        from src.api.routes import pipeline

        original = pipeline.pipeline_service
        self.mock_pipeline_service.get_processed_data.return_value = None
        pipeline.pipeline_service = self.mock_pipeline_service

        try:
            app = FastAPI()
            app.include_router(pipeline.router)
            client = TestClient(app)

            response = client.get("/pipeline/data/1H")

            assert response.status_code == 404
            assert "No cached data" in response.json()["detail"]
        finally:
            pipeline.pipeline_service = original

    def test_get_pipeline_data_success(self):
        """Test getting pipeline data when data is available."""
        from src.api.routes import pipeline

        original = pipeline.pipeline_service

        # Create mock DataFrame
        dates = pd.date_range("2025-01-01", periods=100, freq="h")
        df = pd.DataFrame({
            "open": np.random.rand(100) + 1.08,
            "high": np.random.rand(100) + 1.085,
            "low": np.random.rand(100) + 1.075,
            "close": np.random.rand(100) + 1.08,
        }, index=dates)

        self.mock_pipeline_service.get_processed_data.return_value = df
        pipeline.pipeline_service = self.mock_pipeline_service

        try:
            app = FastAPI()
            app.include_router(pipeline.router)
            client = TestClient(app)

            response = client.get("/pipeline/data/1H")

            assert response.status_code == 200
            data = response.json()
            assert data["timeframe"] == "1H"
            assert data["rows"] == 100
            assert "date_range" in data
        finally:
            pipeline.pipeline_service = original
