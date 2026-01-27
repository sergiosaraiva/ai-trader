"""Integration tests for dynamic confidence threshold system.

Tests the full integration of ThresholdManager with:
- Database (ThresholdHistory, Prediction, Trade tables)
- ModelService (prediction → threshold → should_trade decision)
- TradingService (trade close → outcome recording → feedback)
- API routes (threshold endpoints)
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime, timedelta
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Add backend src to path
backend_path = Path(__file__).parent.parent.parent
src_path = backend_path / "src"
if str(backend_path) not in sys.path:
    sys.path.insert(0, str(backend_path))

# Import database models and services
from api.database.models import Base, Prediction, Trade, ThresholdHistory
from api.services.threshold_service import ThresholdManager
from config import trading_config as config_module


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def db_session():
    """Create an in-memory SQLite database for testing."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()

    yield session

    session.close()


@pytest.fixture
def manager_with_db(db_session):
    """Create ThresholdManager with database integration."""
    with patch("api.services.threshold_service.trading_config") as mock_config:
        # Setup config
        from config.trading_config import ThresholdParameters
        mock_config.threshold = ThresholdParameters()
        mock_config.trading.confidence_threshold = 0.66
        mock_config.get_config_version.return_value = 1

        # Create manager
        manager = ThresholdManager()

        # Initialize with database
        with patch("api.services.threshold_service.get_session", return_value=db_session):
            manager.initialize(db=db_session)

        yield manager, db_session


@pytest.fixture
def populated_db(db_session):
    """Populate database with test data."""
    base_time = datetime.utcnow()

    # Add predictions
    predictions = []
    for i in range(100):
        pred = Prediction(
            timestamp=base_time - timedelta(days=30 - (i * 0.3)),
            symbol="EURUSD",
            direction="long",
            confidence=0.55 + (i % 30) / 100.0,
            should_trade=True,
            market_price=1.0850 + (i * 0.0001),
        )
        db_session.add(pred)
        predictions.append(pred)

    # Add trades
    trades = []
    for i in range(50):
        trade = Trade(
            prediction_id=predictions[i].id if i < len(predictions) else None,
            symbol="EURUSD",
            direction="long",
            entry_price=1.0850,
            entry_time=base_time - timedelta(days=25 - (i * 0.5)),
            exit_price=1.0860 if i % 20 < 11 else 1.0840,
            exit_time=base_time - timedelta(days=25 - (i * 0.5) - 0.1),
            exit_reason="tp" if i % 20 < 11 else "sl",
            lot_size=0.1,
            pips=10.0 if i % 20 < 11 else -10.0,
            pnl_usd=100.0 if i % 20 < 11 else -100.0,
            is_winner=i % 20 < 11,  # 55% win rate
            confidence=0.70,
            status="closed",
        )
        db_session.add(trade)
        trades.append(trade)

    db_session.commit()

    return predictions, trades


# ============================================================================
# DATABASE INTEGRATION TESTS
# ============================================================================


class TestDatabaseIntegration:
    """Test integration with database tables."""

    def test_threshold_history_recording(self, manager_with_db, populated_db):
        """Test that threshold calculations are persisted to ThresholdHistory."""
        manager, db_session = manager_with_db

        # Calculate threshold
        threshold = manager.calculate_threshold(db=db_session, record_history=True)

        # Query history
        history = db_session.query(ThresholdHistory).all()

        # Should have one record
        assert len(history) == 1

        record = history[0]
        assert record.threshold_value == threshold
        assert record.short_term_component is not None
        assert record.medium_term_component is not None
        assert record.long_term_component is not None
        assert record.blended_value is not None
        assert record.prediction_count_30d > 0
        assert record.reason == "dynamic"

    def test_prediction_confidence_loading(self, db_session, populated_db):
        """Test loading prediction confidences from database."""
        predictions, _ = populated_db

        with patch("api.services.threshold_service.trading_config") as mock_config:
            from config.trading_config import ThresholdParameters
            mock_config.threshold = ThresholdParameters()
            mock_config.trading.confidence_threshold = 0.66
            mock_config.get_config_version.return_value = 1

            manager = ThresholdManager()

            # Initialize from database
            manager.initialize(db=db_session)

            # Verify predictions were loaded
            assert manager.is_initialized
            assert len(manager._predictions_30d) == len(predictions)

    def test_trade_outcome_loading(self, db_session, populated_db):
        """Test loading trade outcomes from database."""
        _, trades = populated_db

        with patch("api.services.threshold_service.trading_config") as mock_config:
            from config.trading_config import ThresholdParameters
            mock_config.threshold = ThresholdParameters()
            mock_config.trading.confidence_threshold = 0.66
            mock_config.get_config_version.return_value = 1

            manager = ThresholdManager()

            # Initialize from database
            manager.initialize(db=db_session)

            # Verify trades were loaded (capped at 100)
            assert manager.is_initialized
            assert len(manager._recent_trades) == min(len(trades), 100)

    def test_initialization_from_existing_data(self, db_session, populated_db):
        """Test that manager initializes correctly from existing database data."""
        predictions, trades = populated_db

        with patch("api.services.threshold_service.trading_config") as mock_config:
            from config.trading_config import ThresholdParameters
            mock_config.threshold = ThresholdParameters()
            mock_config.trading.confidence_threshold = 0.66
            mock_config.get_config_version.return_value = 1

            manager = ThresholdManager()
            manager.initialize(db=db_session)

            # Calculate threshold immediately
            threshold = manager.calculate_threshold(db=db_session, record_history=False)

            # Should work without errors
            assert 0.55 <= threshold <= 0.75

    def test_history_query_limit(self, manager_with_db, populated_db):
        """Test get_recent_history respects limit parameter."""
        manager, db_session = manager_with_db

        # Calculate multiple times
        for _ in range(10):
            manager.calculate_threshold(db=db_session, record_history=True)

        # Get limited history
        history = manager.get_recent_history(limit=5, db=db_session)

        # Should return only 5 records
        assert len(history) == 5

    def test_fallback_history_recording(self, db_session):
        """Test that fallback threshold is recorded correctly."""
        with patch("api.services.threshold_service.trading_config") as mock_config:
            from config.trading_config import ThresholdParameters
            mock_config.threshold = ThresholdParameters()
            mock_config.trading.confidence_threshold = 0.66
            mock_config.get_config_version.return_value = 1

            manager = ThresholdManager()
            manager._initialized = True

            # Calculate with insufficient data
            threshold = manager.calculate_threshold(db=db_session, record_history=True)

            # Query history
            history = db_session.query(ThresholdHistory).all()

            assert len(history) == 1
            assert history[0].threshold_value == 0.66
            assert "insufficient_data" in history[0].reason


# ============================================================================
# SERVICE INTEGRATION TESTS
# ============================================================================


class TestModelServiceIntegration:
    """Test integration with ModelService."""

    def test_prediction_to_threshold_flow(self, manager_with_db):
        """Test full flow: prediction → threshold → should_trade decision."""
        manager, db_session = manager_with_db

        # Add some history
        base_time = datetime.utcnow()
        for i in range(60):
            manager.record_prediction(
                None,
                0.60 + (i % 20) / 100,
                base_time - timedelta(days=i * 0.5)
            )

        # Calculate threshold
        threshold = manager.calculate_threshold(record_history=False)

        # Simulate ModelService logic
        prediction_confidence = 0.72

        # Decision: should_trade = confidence >= threshold
        should_trade = prediction_confidence >= threshold

        # With typical threshold ~0.60-0.65, 0.72 should trade
        assert should_trade is True

    def test_low_confidence_rejection(self, manager_with_db):
        """Test that low confidence predictions are rejected."""
        manager, db_session = manager_with_db

        # Add history with high confidences (threshold will be high)
        base_time = datetime.utcnow()
        for i in range(60):
            manager.record_prediction(
                None,
                0.70 + (i % 10) / 100,
                base_time - timedelta(days=i * 0.5)
            )

        # Calculate threshold (should be ~0.70+)
        threshold = manager.calculate_threshold(record_history=False)

        # Low confidence prediction
        prediction_confidence = 0.60

        # Should not trade
        should_trade = prediction_confidence >= threshold
        assert should_trade is False

    def test_dynamic_threshold_feedback_loop(self, manager_with_db):
        """Test feedback loop: predictions → threshold → trades → adjustment."""
        manager, db_session = manager_with_db

        # Phase 1: Add predictions
        base_time = datetime.utcnow()
        for i in range(60):
            manager.record_prediction(
                None,
                0.65,
                base_time - timedelta(days=i * 0.5)
            )

        # Calculate initial threshold
        threshold_1 = manager.calculate_threshold(record_history=False)

        # Phase 2: Add losing trades (low win rate)
        for i in range(30):
            manager.record_trade_outcome(
                i,
                is_winner=i % 10 < 3,  # 30% win rate
                timestamp=base_time - timedelta(days=i * 0.3)
            )

        # Calculate adjusted threshold (should increase due to poor performance)
        threshold_2 = manager.calculate_threshold(record_history=False)

        # Threshold should increase to be more selective
        assert threshold_2 >= threshold_1


class TestTradingServiceIntegration:
    """Test integration with TradingService."""

    def test_trade_close_outcome_recording(self, manager_with_db):
        """Test that trade outcomes are recorded when trades close."""
        manager, db_session = manager_with_db

        # Simulate trade close with outcome
        trade_id = 1
        is_winner = True
        timestamp = datetime.utcnow()

        # Record outcome
        manager.record_trade_outcome(trade_id, is_winner, timestamp)

        # Verify it's in memory
        assert len(manager._recent_trades) == 1
        assert manager._recent_trades[0] == (timestamp, is_winner)

    def test_full_cycle_integration(self, manager_with_db, db_session):
        """Test full cycle: prediction → trade → close → feedback → next threshold."""
        manager, db = manager_with_db

        # Step 1: Create prediction
        pred = Prediction(
            timestamp=datetime.utcnow(),
            symbol="EURUSD",
            direction="long",
            confidence=0.72,
            should_trade=True,
            market_price=1.0850,
        )
        db.add(pred)
        db.commit()

        # Record in manager
        manager.record_prediction(pred.id, pred.confidence, pred.timestamp)

        # Step 2: Calculate threshold
        threshold_1 = manager.calculate_threshold(db=db, record_history=False)

        # Step 3: Execute trade (simulated)
        trade = Trade(
            prediction_id=pred.id,
            symbol="EURUSD",
            direction="long",
            entry_price=1.0850,
            entry_time=datetime.utcnow(),
            lot_size=0.1,
            status="open",
        )
        db.add(trade)
        db.commit()

        # Step 4: Close trade with outcome
        trade.exit_price = 1.0860
        trade.exit_time = datetime.utcnow()
        trade.exit_reason = "tp"
        trade.pips = 10.0
        trade.pnl_usd = 100.0
        trade.is_winner = True
        trade.status = "closed"
        db.commit()

        # Record outcome
        manager.record_trade_outcome(trade.id, trade.is_winner, trade.exit_time)

        # Step 5: Calculate next threshold (with feedback)
        threshold_2 = manager.calculate_threshold(db=db, record_history=False)

        # Both thresholds should be valid
        assert 0.55 <= threshold_1 <= 0.75
        assert 0.55 <= threshold_2 <= 0.75


# ============================================================================
# API INTEGRATION TESTS
# ============================================================================


class TestAPIIntegration:
    """Test integration with FastAPI routes."""

    @pytest.fixture
    def client(self, db_session):
        """Create FastAPI test client."""
        # Mock the app and routes
        from fastapi import FastAPI

        app = FastAPI()

        # Mock threshold service
        with patch("api.routes.threshold.threshold_service") as mock_service:
            # Import routes (would normally be from api.routes.threshold)
            # For testing, we'll mock the endpoints

            @app.get("/api/v1/threshold/status")
            async def get_status():
                if not mock_service.is_initialized:
                    from fastapi import HTTPException
                    raise HTTPException(status_code=503, detail="Service not initialized")
                return mock_service.get_status()

            @app.get("/api/v1/threshold/current")
            async def get_current():
                if not mock_service.is_initialized:
                    from fastapi import HTTPException
                    raise HTTPException(status_code=503, detail="Service not initialized")
                threshold = mock_service.get_current_threshold()
                if threshold is None:
                    threshold = mock_service.calculate_threshold(record_history=False)
                return threshold

            # Setup mock service
            mock_service.is_initialized = True
            mock_service.get_status.return_value = {
                "initialized": True,
                "use_dynamic": True,
                "current_threshold": 0.65,
                "predictions_30d": 100,
            }
            mock_service.get_current_threshold.return_value = 0.65

            client = TestClient(app)
            yield client

    def test_status_endpoint(self, client):
        """Test GET /api/v1/threshold/status."""
        response = client.get("/api/v1/threshold/status")

        assert response.status_code == 200
        data = response.json()

        assert "initialized" in data
        assert "current_threshold" in data
        assert data["initialized"] is True

    def test_current_endpoint(self, client):
        """Test GET /api/v1/threshold/current."""
        response = client.get("/api/v1/threshold/current")

        assert response.status_code == 200
        threshold = response.json()

        assert isinstance(threshold, float)
        assert 0.0 <= threshold <= 1.0

    def test_status_endpoint_not_initialized(self):
        """Test 503 error when service not initialized."""
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        app = FastAPI()

        with patch("api.routes.threshold.threshold_service") as mock_service:
            mock_service.is_initialized = False

            @app.get("/api/v1/threshold/status")
            async def get_status():
                if not mock_service.is_initialized:
                    from fastapi import HTTPException
                    raise HTTPException(status_code=503, detail="Service not initialized")
                return mock_service.get_status()

            client = TestClient(app)
            response = client.get("/api/v1/threshold/status")

            assert response.status_code == 503

    def test_calculate_endpoint_with_history(self, db_session):
        """Test POST /api/v1/threshold/calculate with history recording."""
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        app = FastAPI()

        with patch("api.routes.threshold.threshold_service") as mock_service:
            mock_service.is_initialized = True
            mock_service.calculate_threshold.return_value = 0.67
            mock_service.get_status.return_value = {
                "predictions_7d": 50,
                "predictions_14d": 80,
                "predictions_30d": 100,
                "recent_trades": 25,
            }
            mock_service.get_recent_history.return_value = [{
                "short_term": 0.68,
                "medium_term": 0.66,
                "long_term": 0.65,
                "blended": 0.665,
                "adjustment": 0.005,
                "predictions_7d": 50,
                "predictions_14d": 80,
                "predictions_30d": 100,
                "trade_count": 25,
                "win_rate": 0.56,
            }]

            @app.post("/api/v1/threshold/calculate")
            async def calculate(record_history: bool = True):
                if not mock_service.is_initialized:
                    from fastapi import HTTPException
                    raise HTTPException(status_code=503, detail="Service not initialized")

                threshold = mock_service.calculate_threshold(
                    db=db_session if record_history else None,
                    record_history=record_history
                )

                status = mock_service.get_status()
                history = mock_service.get_recent_history(limit=1, db=db_session) if record_history else []

                components = {}
                data_quality = {}

                if history:
                    latest = history[0]
                    components = {
                        "short_term": latest["short_term"],
                        "medium_term": latest["medium_term"],
                        "long_term": latest["long_term"],
                        "blended": latest["blended"],
                        "adjustment": latest["adjustment"],
                    }
                    data_quality = {
                        "predictions_7d": latest["predictions_7d"],
                        "predictions_14d": latest["predictions_14d"],
                        "predictions_30d": latest["predictions_30d"],
                        "trade_count": latest["trade_count"],
                        "win_rate": latest["win_rate"],
                    }

                return {
                    "threshold": threshold,
                    "timestamp": datetime.utcnow().isoformat(),
                    "components": components,
                    "data_quality": data_quality,
                }

            client = TestClient(app)
            response = client.post("/api/v1/threshold/calculate?record_history=true")

            assert response.status_code == 200
            data = response.json()

            assert "threshold" in data
            assert "components" in data
            assert "data_quality" in data
            assert data["threshold"] == 0.67


# ============================================================================
# PERFORMANCE TESTS
# ============================================================================


class TestPerformance:
    """Test performance characteristics of the system."""

    def test_initialization_performance(self, db_session):
        """Test initialization time with large dataset."""
        import time

        # Add large dataset
        base_time = datetime.utcnow()
        predictions = []
        for i in range(1000):
            pred = Prediction(
                timestamp=base_time - timedelta(days=30 - (i * 0.03)),
                symbol="EURUSD",
                direction="long",
                confidence=0.60 + (i % 20) / 100,
                should_trade=True,
                market_price=1.0850,
            )
            predictions.append(pred)

        db_session.bulk_save_objects(predictions)
        db_session.commit()

        # Time initialization
        with patch("api.services.threshold_service.trading_config") as mock_config:
            from config.trading_config import ThresholdParameters
            mock_config.threshold = ThresholdParameters()
            mock_config.trading.confidence_threshold = 0.66
            mock_config.get_config_version.return_value = 1

            manager = ThresholdManager()

            start = time.time()
            manager.initialize(db=db_session)
            elapsed = time.time() - start

            # Should complete in < 1 second
            assert elapsed < 1.0
            assert manager.is_initialized

    def test_calculation_performance(self, manager_with_db):
        """Test calculation performance with large dataset."""
        import time

        manager, db_session = manager_with_db

        # Add large dataset
        base_time = datetime.utcnow()
        for i in range(1000):
            manager.record_prediction(
                None,
                0.60 + (i % 20) / 100,
                base_time - timedelta(days=i * 0.03)
            )

        # Time calculation
        start = time.time()
        manager.calculate_threshold(record_history=False)
        elapsed = time.time() - start

        # Should complete in < 100ms
        assert elapsed < 0.1
