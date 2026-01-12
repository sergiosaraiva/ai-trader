"""Tests for trading endpoints using FastAPI TestClient."""

import pytest
from datetime import datetime
from unittest.mock import Mock, MagicMock
from fastapi.testclient import TestClient
from fastapi import FastAPI


class TestTradingEndpoints:
    """Test trading endpoints."""

    @pytest.fixture(autouse=True)
    def setup_mocks(self):
        """Set up service mocks before each test."""
        self.mock_trading_service = Mock()
        self.mock_trading_service.is_loaded = True
        self.mock_trading_service._initialized = True

        self.mock_trading_service.get_status.return_value = {
            "mode": "paper",
            "balance": 100000.0,
            "equity": 100000.0,
            "unrealized_pnl": 0.0,
            "open_position": None,
            "has_position": False,
        }

        self.mock_trading_service.get_performance.return_value = {
            "total_trades": 10,
            "winning_trades": 6,
            "losing_trades": 4,
            "win_rate": 0.6,
            "total_pips": 150.5,
            "total_pnl_usd": 1505.0,
            "avg_pips_per_trade": 15.05,
            "profit_factor": 2.5,
            "initial_balance": 100000.0,
            "current_balance": 101505.0,
            "return_pct": 1.505,
        }

        self.mock_trading_service.get_equity_curve.return_value = [
            {"timestamp": "2024-01-01T00:00:00", "balance": 100000.0, "equity": 100000.0},
            {"timestamp": "2024-01-02T00:00:00", "balance": 100150.0, "equity": 100150.0},
            {"timestamp": "2024-01-03T00:00:00", "balance": 101505.0, "equity": 101505.0},
        ]

        self.mock_data_service = Mock()
        self.mock_data_service.get_current_price.return_value = 1.08543

    def test_trading_status_no_position(self):
        """Test trading status endpoint with no open position."""
        from src.api.routes import trading

        original_trading = trading.trading_service
        original_data = trading.data_service
        trading.trading_service = self.mock_trading_service
        trading.data_service = self.mock_data_service

        try:
            app = FastAPI()
            app.include_router(trading.router)
            client = TestClient(app)

            response = client.get("/trading/status")

            assert response.status_code == 200
            data = response.json()
            assert data["mode"] == "paper"
            assert data["balance"] == 100000.0
            assert data["has_position"] is False
            assert data["open_position"] is None
        finally:
            trading.trading_service = original_trading
            trading.data_service = original_data

    def test_trading_status_with_position(self):
        """Test trading status endpoint with open position."""
        from src.api.routes import trading

        self.mock_trading_service.get_status.return_value = {
            "mode": "paper",
            "balance": 100000.0,
            "equity": 100150.0,
            "unrealized_pnl": 150.0,
            "open_position": {
                "id": 1,
                "symbol": "EURUSD",
                "direction": "long",
                "entry_price": 1.08500,
                "entry_time": datetime(2024, 1, 5, 10, 0, 0),
                "lot_size": 0.1,
                "take_profit": 1.08750,
                "stop_loss": 1.08350,
                "confidence": 0.72,
            },
            "has_position": True,
        }

        original_trading = trading.trading_service
        original_data = trading.data_service
        trading.trading_service = self.mock_trading_service
        trading.data_service = self.mock_data_service

        try:
            app = FastAPI()
            app.include_router(trading.router)
            client = TestClient(app)

            response = client.get("/trading/status")

            assert response.status_code == 200
            data = response.json()
            assert data["has_position"] is True
            assert data["open_position"] is not None
            assert data["open_position"]["direction"] == "long"
        finally:
            trading.trading_service = original_trading
            trading.data_service = original_data

    def test_trading_performance(self):
        """Test trading performance endpoint."""
        from src.api.routes import trading

        original_trading = trading.trading_service
        trading.trading_service = self.mock_trading_service

        try:
            app = FastAPI()
            app.include_router(trading.router)
            client = TestClient(app)

            response = client.get("/trading/performance")

            assert response.status_code == 200
            data = response.json()
            assert data["total_trades"] == 10
            assert data["win_rate"] == 0.6
            assert data["profit_factor"] == 2.5
            assert data["total_pips"] == 150.5
        finally:
            trading.trading_service = original_trading

    def test_equity_curve(self):
        """Test equity curve endpoint."""
        from src.api.routes import trading
        from src.api.database.session import get_db

        original_trading = trading.trading_service
        trading.trading_service = self.mock_trading_service

        try:
            app = FastAPI()
            app.include_router(trading.router)

            # Override database dependency
            mock_db = Mock()
            app.dependency_overrides[get_db] = lambda: mock_db

            client = TestClient(app)

            response = client.get("/trading/equity-curve")

            assert response.status_code == 200
            data = response.json()
            assert "data" in data
            assert data["count"] == 3
            assert len(data["data"]) == 3
        finally:
            trading.trading_service = original_trading
            app.dependency_overrides.clear()

    def test_close_position_no_position(self):
        """Test closing position when no position is open."""
        from src.api.routes import trading
        from src.api.database.session import get_db

        original_trading = trading.trading_service
        trading.trading_service = self.mock_trading_service

        try:
            app = FastAPI()
            app.include_router(trading.router)

            mock_db = Mock()
            app.dependency_overrides[get_db] = lambda: mock_db

            client = TestClient(app)

            response = client.post("/trading/close-position")

            assert response.status_code == 400
            assert "No open position" in response.json()["detail"]
        finally:
            trading.trading_service = original_trading
            app.dependency_overrides.clear()

    def test_close_position_success(self):
        """Test successfully closing a position."""
        from src.api.routes import trading
        from src.api.database.session import get_db

        self.mock_trading_service.get_status.return_value = {
            "has_position": True,
            "open_position": {"id": 1, "direction": "long"},
        }
        self.mock_trading_service.close_position.return_value = {
            "id": 1,
            "pips": 25.0,
            "pnl_usd": 250.0,
            "is_winner": True,
        }

        original_trading = trading.trading_service
        original_data = trading.data_service
        trading.trading_service = self.mock_trading_service
        trading.data_service = self.mock_data_service

        try:
            app = FastAPI()
            app.include_router(trading.router)

            mock_db = Mock()
            app.dependency_overrides[get_db] = lambda: mock_db

            client = TestClient(app)

            response = client.post("/trading/close-position")

            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "success"
        finally:
            trading.trading_service = original_trading
            trading.data_service = original_data
            app.dependency_overrides.clear()

    def test_legacy_positions_endpoint(self):
        """Test legacy positions endpoint."""
        from src.api.routes import trading

        original_trading = trading.trading_service
        trading.trading_service = self.mock_trading_service

        try:
            app = FastAPI()
            app.include_router(trading.router)
            client = TestClient(app)

            response = client.get("/positions")

            assert response.status_code == 200
            data = response.json()
            assert isinstance(data, list)
            assert len(data) == 0  # No open positions
        finally:
            trading.trading_service = original_trading

    def test_legacy_performance_endpoint(self):
        """Test legacy performance endpoint."""
        from src.api.routes import trading

        original_trading = trading.trading_service
        trading.trading_service = self.mock_trading_service

        try:
            app = FastAPI()
            app.include_router(trading.router)
            client = TestClient(app)

            response = client.get("/performance")

            assert response.status_code == 200
            data = response.json()
            assert "total_trades" in data
        finally:
            trading.trading_service = original_trading

    def test_risk_metrics_endpoint(self):
        """Test risk metrics endpoint."""
        from src.api.routes import trading

        original_trading = trading.trading_service
        trading.trading_service = self.mock_trading_service

        try:
            app = FastAPI()
            app.include_router(trading.router)
            client = TestClient(app)

            response = client.get("/risk/metrics")

            assert response.status_code == 200
            data = response.json()
            assert "account_balance" in data
            assert data["current_exposure"] == 0.0
        finally:
            trading.trading_service = original_trading

    def test_service_not_loaded_returns_error(self):
        """Test that endpoints handle uninitialized service gracefully."""
        from src.api.routes import trading

        self.mock_trading_service.is_loaded = False

        original_trading = trading.trading_service
        trading.trading_service = self.mock_trading_service

        try:
            app = FastAPI()
            app.include_router(trading.router)
            client = TestClient(app)

            response = client.get("/performance")

            assert response.status_code == 200
            data = response.json()
            assert "error" in data
        finally:
            trading.trading_service = original_trading


class TestTradeHistoryEndpoint:
    """Test trade history endpoint."""

    @pytest.fixture(autouse=True)
    def setup_mocks(self):
        """Set up mocks for trade history tests."""
        from unittest.mock import patch

        # Mock database models
        self.mock_trade = Mock()
        self.mock_trade.id = 1
        self.mock_trade.symbol = "EURUSD"
        self.mock_trade.direction = "long"
        self.mock_trade.entry_price = 1.08500
        self.mock_trade.entry_time = datetime(2024, 1, 5, 10, 0, 0)
        self.mock_trade.exit_price = 1.08750
        self.mock_trade.exit_time = datetime(2024, 1, 5, 15, 0, 0)
        self.mock_trade.exit_reason = "tp"
        self.mock_trade.lot_size = 0.1
        self.mock_trade.take_profit = 1.08750
        self.mock_trade.stop_loss = 1.08350
        self.mock_trade.pips = 25.0
        self.mock_trade.pnl_usd = 250.0
        self.mock_trade.is_winner = True
        self.mock_trade.confidence = 0.72
        self.mock_trade.status = "closed"

    def test_trade_history_with_results(self):
        """Test trade history returns trades from database."""
        from src.api.routes import trading
        from src.api.database.session import get_db

        app = FastAPI()
        app.include_router(trading.router)

        # Mock database session
        mock_db = Mock()
        mock_query = Mock()
        mock_query.filter.return_value = mock_query
        mock_query.order_by.return_value = mock_query
        mock_query.limit.return_value = mock_query
        mock_query.all.return_value = [self.mock_trade]
        mock_db.query.return_value = mock_query

        app.dependency_overrides[get_db] = lambda: mock_db

        try:
            client = TestClient(app)
            response = client.get("/trading/history")

            assert response.status_code == 200
            data = response.json()
            assert data["count"] == 1
            assert data["trades"][0]["direction"] == "long"
            assert data["trades"][0]["pips"] == 25.0
        finally:
            app.dependency_overrides.clear()

    def test_trade_history_with_status_filter(self):
        """Test trade history with status filter."""
        from src.api.routes import trading
        from src.api.database.session import get_db

        app = FastAPI()
        app.include_router(trading.router)

        mock_db = Mock()
        mock_query = Mock()
        mock_query.filter.return_value = mock_query
        mock_query.order_by.return_value = mock_query
        mock_query.limit.return_value = mock_query
        mock_query.all.return_value = []
        mock_db.query.return_value = mock_query

        app.dependency_overrides[get_db] = lambda: mock_db

        try:
            client = TestClient(app)
            response = client.get("/trading/history?status=open")

            assert response.status_code == 200
            data = response.json()
            assert data["count"] == 0
        finally:
            app.dependency_overrides.clear()

    def test_trade_history_empty(self):
        """Test trade history with no trades."""
        from src.api.routes import trading
        from src.api.database.session import get_db

        app = FastAPI()
        app.include_router(trading.router)

        mock_db = Mock()
        mock_query = Mock()
        mock_query.filter.return_value = mock_query
        mock_query.order_by.return_value = mock_query
        mock_query.limit.return_value = mock_query
        mock_query.all.return_value = []
        mock_db.query.return_value = mock_query

        app.dependency_overrides[get_db] = lambda: mock_db

        try:
            client = TestClient(app)
            response = client.get("/trading/history")

            assert response.status_code == 200
            data = response.json()
            assert data["count"] == 0
            assert data["trades"] == []
        finally:
            app.dependency_overrides.clear()
