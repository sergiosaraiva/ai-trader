"""Integration tests for performance API endpoints.

This test suite validates:
1. GET /api/v1/model/performance returns valid response
2. POST /api/v1/model/performance/reload works correctly
3. Response structure matches expected schema
4. Error handling for service failures
"""

from unittest.mock import patch, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.routes import performance


class TestPerformanceEndpoints:
    """Tests for performance endpoints."""

    @pytest.fixture(autouse=True)
    def setup_mocks(self):
        """Set up service mocks before each test."""
        self.mock_performance_service = MagicMock()
        self.mock_performance_service.is_loaded = True
        self.mock_performance_service.initialize.return_value = True
        self.mock_performance_service.get_performance_data.return_value = {
            "metrics": {
                "total_pips": 8135.6,
                "win_rate": 0.586,
                "profit_factor": 2.26,
                "total_trades": 1093,
                "high_confidence": {
                    "threshold": 0.70,
                    "win_rate": 0.621,
                    "profit_factor": 2.69,
                    "total_pips": 8693,
                    "sample_size": 966,
                },
                "full_agreement": {
                    "accuracy": 0.82,
                    "sample_size": 50,
                },
                "wfo_validation": {
                    "windows_profitable": 7,
                    "total_windows": 7,
                    "total_pips": 18136,
                    "consistency_score": 1.0,
                },
                "regime_performance": {
                    "all_profitable": True,
                    "regimes_count": 6,
                },
            },
            "highlights": [
                {
                    "type": "confidence",
                    "title": "High-Confidence Trading",
                    "value": "62.1%",
                    "description": "Win rate when model confidence exceeds 70%",
                },
                {
                    "type": "agreement",
                    "title": "Model Consensus",
                    "value": "82%",
                    "description": "Accuracy when all 3 timeframes agree",
                },
                {
                    "type": "validation",
                    "title": "Walk-Forward Validated",
                    "value": "7/7",
                    "description": "Profitable across all test periods",
                },
                {
                    "type": "robustness",
                    "title": "All-Regime Profitable",
                    "value": "6/6",
                    "description": "Works in trending and ranging markets",
                },
            ],
            "summary": {
                "headline": "Solid Performance",
                "description": "The MTF Ensemble model demonstrates solid performance with 58.6% overall win rate and 2.26x profit factor.",
            },
        }
        self.mock_performance_service.reload.return_value = True

    def test_get_model_performance_success(self):
        """Test GET /model/performance returns valid response."""
        from src.api.routes import performance

        original = performance.performance_service
        performance.performance_service = self.mock_performance_service

        try:
            app = FastAPI()
            app.include_router(performance.router)
            client = TestClient(app)

            response = client.get("/model/performance")

            assert response.status_code == 200
            data = response.json()

            # Verify structure
            assert "metrics" in data
            assert "highlights" in data
            assert "summary" in data

            # Verify metrics
            assert data["metrics"]["total_pips"] == 8135.6
            assert data["metrics"]["win_rate"] == 0.586
            assert data["metrics"]["profit_factor"] == 2.26
            assert data["metrics"]["total_trades"] == 1093

            # Verify high confidence metrics
            assert data["metrics"]["high_confidence"]["win_rate"] == 0.621
            assert data["metrics"]["high_confidence"]["threshold"] == 0.70

            # Verify highlights
            assert len(data["highlights"]) == 4
            highlight_types = [h["type"] for h in data["highlights"]]
            assert "confidence" in highlight_types
            assert "agreement" in highlight_types
            assert "validation" in highlight_types
            assert "robustness" in highlight_types

            # Verify summary
            assert data["summary"]["headline"] == "Solid Performance"
            assert "58.6%" in data["summary"]["description"]

        finally:
            performance.performance_service = original

    def test_get_model_performance_service_not_loaded(self):
        """Test GET /model/performance initializes service if not loaded."""
        from src.api.routes import performance

        # Service not loaded
        self.mock_performance_service.is_loaded = False

        original = performance.performance_service
        performance.performance_service = self.mock_performance_service

        try:
            app = FastAPI()
            app.include_router(performance.router)
            client = TestClient(app)

            response = client.get("/model/performance")

            assert response.status_code == 200
            # Should have called initialize
            self.mock_performance_service.initialize.assert_called_once()

        finally:
            performance.performance_service = original

    def test_get_model_performance_initialization_fails(self):
        """Test GET /model/performance when initialization fails."""
        from src.api.routes import performance

        # Service not loaded and initialization fails
        self.mock_performance_service.is_loaded = False
        self.mock_performance_service.initialize.return_value = False

        original = performance.performance_service
        performance.performance_service = self.mock_performance_service

        try:
            app = FastAPI()
            app.include_router(performance.router)
            client = TestClient(app)

            response = client.get("/model/performance")

            # Should still return 200 with defaults
            assert response.status_code == 200
            data = response.json()
            assert "metrics" in data

        finally:
            performance.performance_service = original

    def test_get_model_performance_service_exception(self):
        """Test GET /model/performance handles service exceptions."""
        from src.api.routes import performance

        # Make get_performance_data raise an exception
        self.mock_performance_service.get_performance_data.side_effect = Exception("Test error")

        original = performance.performance_service
        performance.performance_service = self.mock_performance_service

        try:
            app = FastAPI()
            app.include_router(performance.router)
            client = TestClient(app)

            response = client.get("/model/performance")

            assert response.status_code == 500
            data = response.json()
            assert "detail" in data

        finally:
            performance.performance_service = original

    def test_reload_performance_data_success(self):
        """Test POST /model/performance/reload returns success."""
        from src.api.routes import performance

        original = performance.performance_service
        performance.performance_service = self.mock_performance_service

        try:
            app = FastAPI()
            app.include_router(performance.router)
            client = TestClient(app)

            response = client.post("/model/performance/reload")

            assert response.status_code == 200
            data = response.json()

            assert data["status"] == "success"
            assert data["message"] == "Performance data reloaded successfully"

            # Verify reload was called
            self.mock_performance_service.reload.assert_called_once()

        finally:
            performance.performance_service = original

    def test_reload_performance_data_partial_success(self):
        """Test POST /model/performance/reload when reload encounters issues."""
        from src.api.routes import performance

        # Make reload return False (encountered issues but used defaults)
        self.mock_performance_service.reload.return_value = False

        original = performance.performance_service
        performance.performance_service = self.mock_performance_service

        try:
            app = FastAPI()
            app.include_router(performance.router)
            client = TestClient(app)

            response = client.post("/model/performance/reload")

            assert response.status_code == 200
            data = response.json()

            assert data["status"] == "warning"
            assert "using defaults" in data["message"]

        finally:
            performance.performance_service = original

    def test_reload_performance_data_exception(self):
        """Test POST /model/performance/reload handles exceptions."""
        from src.api.routes import performance

        # Make reload raise an exception
        self.mock_performance_service.reload.side_effect = Exception("Reload error")

        original = performance.performance_service
        performance.performance_service = self.mock_performance_service

        try:
            app = FastAPI()
            app.include_router(performance.router)
            client = TestClient(app)

            response = client.post("/model/performance/reload")

            assert response.status_code == 500
            data = response.json()
            assert "detail" in data

        finally:
            performance.performance_service = original


class TestPerformanceResponseStructure:
    """Tests for response structure validation."""

    @pytest.fixture(autouse=True)
    def setup_mocks(self):
        """Set up service mocks before each test."""
        self.mock_performance_service = MagicMock()
        self.mock_performance_service.is_loaded = True

    def test_response_has_all_required_fields(self):
        """Test response contains all required fields."""
        from src.api.routes import performance

        self.mock_performance_service.get_performance_data.return_value = {
            "metrics": {
                "total_pips": 8000,
                "win_rate": 0.58,
                "profit_factor": 2.2,
                "total_trades": 1000,
                "high_confidence": {"win_rate": 0.62},
                "full_agreement": {"accuracy": 0.80},
                "wfo_validation": {"consistency_score": 1.0},
                "regime_performance": {"all_profitable": True},
            },
            "highlights": [],
            "summary": {"headline": "Test", "description": "Test description"},
        }

        original = performance.performance_service
        performance.performance_service = self.mock_performance_service

        try:
            app = FastAPI()
            app.include_router(performance.router)
            client = TestClient(app)

            response = client.get("/model/performance")
            data = response.json()

            # Verify top-level fields
            assert "metrics" in data
            assert "highlights" in data
            assert "summary" in data

            # Verify metrics fields
            assert "total_pips" in data["metrics"]
            assert "win_rate" in data["metrics"]
            assert "profit_factor" in data["metrics"]
            assert "total_trades" in data["metrics"]
            assert "high_confidence" in data["metrics"]
            assert "full_agreement" in data["metrics"]
            assert "wfo_validation" in data["metrics"]
            assert "regime_performance" in data["metrics"]

            # Verify summary fields
            assert "headline" in data["summary"]
            assert "description" in data["summary"]

        finally:
            performance.performance_service = original

    def test_highlights_have_correct_structure(self):
        """Test highlights have correct structure."""
        from src.api.routes import performance

        self.mock_performance_service.get_performance_data.return_value = {
            "metrics": {},
            "highlights": [
                {
                    "type": "confidence",
                    "title": "Test Title",
                    "value": "62.1%",
                    "description": "Test description",
                }
            ],
            "summary": {"headline": "Test", "description": "Test"},
        }

        original = performance.performance_service
        performance.performance_service = self.mock_performance_service

        try:
            app = FastAPI()
            app.include_router(performance.router)
            client = TestClient(app)

            response = client.get("/model/performance")
            data = response.json()

            highlights = data["highlights"]
            assert len(highlights) == 1

            highlight = highlights[0]
            assert "type" in highlight
            assert "title" in highlight
            assert "value" in highlight
            assert "description" in highlight

        finally:
            performance.performance_service = original

    def test_metrics_values_are_correct_types(self):
        """Test metrics values have correct types."""
        from src.api.routes import performance

        self.mock_performance_service.get_performance_data.return_value = {
            "metrics": {
                "total_pips": 8135.6,
                "win_rate": 0.586,
                "profit_factor": 2.26,
                "total_trades": 1093,
                "high_confidence": {
                    "threshold": 0.70,
                    "win_rate": 0.621,
                },
                "full_agreement": {
                    "accuracy": 0.82,
                    "sample_size": 50,
                },
                "wfo_validation": {
                    "windows_profitable": 7,
                    "total_windows": 7,
                    "consistency_score": 1.0,
                },
                "regime_performance": {
                    "all_profitable": True,
                    "regimes_count": 6,
                },
            },
            "highlights": [],
            "summary": {"headline": "Test", "description": "Test"},
        }

        original = performance.performance_service
        performance.performance_service = self.mock_performance_service

        try:
            app = FastAPI()
            app.include_router(performance.router)
            client = TestClient(app)

            response = client.get("/model/performance")
            data = response.json()

            metrics = data["metrics"]

            # Verify types
            assert isinstance(metrics["total_pips"], (int, float))
            assert isinstance(metrics["win_rate"], float)
            assert isinstance(metrics["profit_factor"], float)
            assert isinstance(metrics["total_trades"], int)

            assert isinstance(metrics["high_confidence"]["threshold"], float)
            assert isinstance(metrics["high_confidence"]["win_rate"], float)

            assert isinstance(metrics["full_agreement"]["accuracy"], float)
            assert isinstance(metrics["full_agreement"]["sample_size"], int)

            assert isinstance(metrics["wfo_validation"]["consistency_score"], float)
            assert isinstance(metrics["regime_performance"]["all_profitable"], bool)

        finally:
            performance.performance_service = original
