"""Integration tests for Conservative Hybrid position sizing system.

Tests end-to-end integration of:
- Position sizer + trading service
- Circuit breakers + trading service
- Configuration hot reload
- Trade execution with risk tracking
- Database persistence
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime, timedelta, timezone
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Add backend src to path
backend_path = Path(__file__).parent.parent.parent
src_path = backend_path / "src"
if str(backend_path) not in sys.path:
    sys.path.insert(0, str(backend_path))

# Import directly from module files to avoid dependency issues
import importlib.util
import logging

# Load position_sizer module
position_sizer_spec = importlib.util.spec_from_file_location(
    "position_sizer",
    src_path / "trading" / "position_sizer.py"
)
position_sizer_module = importlib.util.module_from_spec(position_sizer_spec)
position_sizer_module.logger = logging.getLogger(__name__)
position_sizer_spec.loader.exec_module(position_sizer_module)

ConservativeHybridSizer = position_sizer_module.ConservativeHybridSizer

# Load circuit_breakers module
circuit_breakers_spec = importlib.util.spec_from_file_location(
    "circuit_breakers",
    src_path / "trading" / "circuit_breakers.py"
)
circuit_breakers_module = importlib.util.module_from_spec(circuit_breakers_spec)
circuit_breakers_module.logger = logging.getLogger(__name__)
circuit_breakers_spec.loader.exec_module(circuit_breakers_module)

TradingCircuitBreaker = circuit_breakers_module.TradingCircuitBreaker

# Load trading_config module
config_spec = importlib.util.spec_from_file_location(
    "trading_config",
    src_path / "config" / "trading_config.py"
)
config_module = importlib.util.module_from_spec(config_spec)
config_module.logger = logging.getLogger(__name__)
config_spec.loader.exec_module(config_module)

TradingConfig = config_module.TradingConfig
ConservativeHybridParameters = config_module.ConservativeHybridParameters

# Import models normally (they work fine)
from src.api.database.models import Base, Trade, CircuitBreakerEvent


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def db_session():
    """Create an in-memory SQLite database session for testing."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()


@pytest.fixture
def trading_config():
    """Create a trading config instance with test settings."""
    config = TradingConfig()
    # Reset to defaults for testing
    config.conservative_hybrid = ConservativeHybridParameters(
        base_risk_percent=1.5,
        confidence_scaling_factor=0.5,
        min_risk_percent=0.8,
        max_risk_percent=2.5,
        confidence_threshold=0.70,
        daily_loss_limit_percent=-3.0,
        consecutive_loss_limit=5,
    )
    return config


@pytest.fixture
def position_sizer():
    """Create position sizer instance."""
    return ConservativeHybridSizer()


@pytest.fixture
def circuit_breaker(trading_config):
    """Create circuit breaker instance."""
    return TradingCircuitBreaker(trading_config.conservative_hybrid)


@pytest.fixture
def mock_trading_service(trading_config, position_sizer, circuit_breaker):
    """Create a mock trading service with position sizing and circuit breakers."""

    class MockTradingService:
        """Simplified trading service for integration testing."""

        def __init__(self, config, sizer, breaker):
            self.config = config
            self.position_sizer = sizer
            self.circuit_breaker = breaker
            self.balance = 10000.0

        def calculate_position_size(self, confidence, sl_pips, db_session):
            """Calculate position size using position sizer."""
            # Check circuit breakers first
            can_trade, reason = self.circuit_breaker.can_trade(db_session, self.balance)
            if not can_trade:
                return 0.0, 0.0, {"reason": f"circuit_breaker: {reason}"}

            # Calculate position
            position_lots, risk_pct, metadata = self.position_sizer.calculate_position_size(
                balance=self.balance,
                confidence=confidence,
                sl_pips=sl_pips,
                config=self.config.conservative_hybrid
            )

            return position_lots, risk_pct, metadata

        def execute_trade(self, db_session, direction, confidence, sl_pips, entry_price=1.0850):
            """Execute a trade and record it in the database."""
            # Calculate position
            position_lots, risk_pct, metadata = self.calculate_position_size(
                confidence, sl_pips, db_session
            )

            if position_lots == 0.0:
                return None, metadata

            # Create trade record
            trade = Trade(
                symbol="EURUSD",
                direction=direction,
                entry_price=entry_price,
                entry_time=datetime.now(timezone.utc),
                lot_size=position_lots,
                confidence=confidence,
                risk_percentage_used=risk_pct,
                status="open",
                stop_loss=entry_price - (sl_pips * 0.0001) if direction == "long" else entry_price + (sl_pips * 0.0001)
            )

            db_session.add(trade)
            db_session.commit()

            return trade, metadata

        def close_trade(self, db_session, trade, exit_price, exit_reason="tp"):
            """Close a trade and update P&L."""
            trade.exit_price = exit_price
            trade.exit_time = datetime.now(timezone.utc)
            trade.exit_reason = exit_reason
            trade.status = "closed"

            # Calculate P&L
            if trade.direction == "long":
                pips = (exit_price - trade.entry_price) / 0.0001
            else:
                pips = (trade.entry_price - exit_price) / 0.0001

            trade.pips = pips
            trade.pnl_usd = pips * 10.0 * trade.lot_size  # $10 per pip per 0.1 lot
            trade.is_winner = trade.pnl_usd > 0

            # Update balance
            self.balance += trade.pnl_usd

            db_session.commit()

            return trade

    return MockTradingService(trading_config, position_sizer, circuit_breaker)


# ============================================================================
# POSITION SIZING INTEGRATION TESTS
# ============================================================================


class TestTradingServicePositionSizing:
    """Test that trading service uses position sizer correctly."""

    def test_trading_service_uses_position_sizer(self, mock_trading_service, db_session):
        """Test trading service calculates positions correctly."""
        confidence = 0.75
        sl_pips = 15.0

        position_lots, risk_pct, metadata = mock_trading_service.calculate_position_size(
            confidence, sl_pips, db_session
        )

        # Verify position was calculated
        assert position_lots > 0.0
        assert risk_pct > 0.0
        assert "confidence" in metadata
        assert metadata["confidence"] == confidence

    def test_trading_service_respects_circuit_breakers(self, mock_trading_service, db_session):
        """Test trading service respects circuit breaker blocks."""
        now = datetime.now(timezone.utc)

        # Create trades that trigger daily loss limit
        for i in range(5):
            trade = Trade(
                symbol="EURUSD",
                direction="long",
                entry_price=1.0850,
                entry_time=now - timedelta(hours=i),
                exit_price=1.0840,
                exit_time=now - timedelta(hours=i) + timedelta(minutes=30),
                exit_reason="sl",
                lot_size=0.1,
                status="closed",
                pnl_usd=-60.0,
                is_winner=False
            )
            db_session.add(trade)

        db_session.commit()

        # Try to calculate position
        position_lots, risk_pct, metadata = mock_trading_service.calculate_position_size(
            0.75, 15.0, db_session
        )

        # Should be blocked
        assert position_lots == 0.0
        assert risk_pct == 0.0
        assert "circuit_breaker" in metadata["reason"]

    def test_execute_trade_records_risk_percentage(self, mock_trading_service, db_session):
        """Test execute_trade saves risk_percentage_used to database."""
        confidence = 0.75
        sl_pips = 15.0

        trade, metadata = mock_trading_service.execute_trade(
            db_session, "long", confidence, sl_pips
        )

        # Verify trade was created
        assert trade is not None
        assert trade.risk_percentage_used is not None
        assert trade.risk_percentage_used > 0.0
        assert trade.confidence == confidence
        assert trade.lot_size > 0.0

        # Verify it's in the database
        db_trade = db_session.query(Trade).filter_by(id=trade.id).first()
        assert db_trade is not None
        assert db_trade.risk_percentage_used == trade.risk_percentage_used


# ============================================================================
# CONFIGURATION HOT RELOAD TESTS
# ============================================================================


class TestConfigurationHotReload:
    """Test configuration changes apply to position sizing."""

    def test_config_hot_reload_updates_position_sizing(self, mock_trading_service, db_session):
        """Test that config changes update position sizing."""
        confidence = 0.75
        sl_pips = 15.0

        # Calculate position with default config
        pos1, risk1, meta1 = mock_trading_service.calculate_position_size(
            confidence, sl_pips, db_session
        )

        # Update config (increase base risk)
        mock_trading_service.config.conservative_hybrid.base_risk_percent = 2.0

        # Recalculate position
        pos2, risk2, meta2 = mock_trading_service.calculate_position_size(
            confidence, sl_pips, db_session
        )

        # Position should increase with higher base risk
        assert pos2 > pos1
        assert risk2 > risk1

    def test_config_change_threshold(self, mock_trading_service, db_session):
        """Test changing confidence threshold."""
        confidence = 0.68
        sl_pips = 15.0

        # With default threshold (0.70), should not trade
        pos1, risk1, meta1 = mock_trading_service.calculate_position_size(
            confidence, sl_pips, db_session
        )
        assert pos1 == 0.0

        # Lower threshold to 0.65
        mock_trading_service.config.conservative_hybrid.confidence_threshold = 0.65

        # Now should trade
        pos2, risk2, meta2 = mock_trading_service.calculate_position_size(
            confidence, sl_pips, db_session
        )
        assert pos2 > 0.0


# ============================================================================
# CIRCUIT BREAKER INTEGRATION TESTS
# ============================================================================


class TestCircuitBreakerIntegration:
    """Test circuit breaker integration with trading flow."""

    def test_daily_loss_limit_integration(self, mock_trading_service, db_session):
        """Test daily loss limit stops trading after -3% loss."""
        initial_balance = mock_trading_service.balance
        confidence = 0.75
        sl_pips = 15.0

        # Execute trades until we hit -3% loss
        # -3% of $10,000 = -$300
        trades_executed = 0
        total_loss = 0.0

        while total_loss > -300.0 and trades_executed < 10:
            # Execute trade
            trade, metadata = mock_trading_service.execute_trade(
                db_session, "long", confidence, sl_pips
            )

            if trade is None:
                break

            # Close trade with loss
            mock_trading_service.close_trade(
                db_session, trade, exit_price=1.0840, exit_reason="sl"
            )

            total_loss += trade.pnl_usd
            trades_executed += 1

        # Try to execute another trade
        position_lots, risk_pct, metadata = mock_trading_service.calculate_position_size(
            confidence, sl_pips, db_session
        )

        # Should be blocked
        assert position_lots == 0.0
        assert "circuit_breaker" in metadata["reason"]

        # Verify circuit breaker event was persisted
        events = db_session.query(CircuitBreakerEvent).filter_by(
            breaker_type="daily_loss_limit"
        ).all()
        assert len(events) > 0

    def test_consecutive_loss_integration(self, mock_trading_service, db_session):
        """Test consecutive loss limit stops trading after 5 losses."""
        confidence = 0.75
        sl_pips = 15.0

        # Execute 5 losing trades
        for i in range(5):
            trade, metadata = mock_trading_service.execute_trade(
                db_session, "long", confidence, sl_pips
            )

            assert trade is not None

            # Close with loss
            mock_trading_service.close_trade(
                db_session, trade, exit_price=1.0840, exit_reason="sl"
            )

        # Try to execute another trade
        position_lots, risk_pct, metadata = mock_trading_service.calculate_position_size(
            confidence, sl_pips, db_session
        )

        # Should be blocked
        assert position_lots == 0.0
        assert "circuit_breaker" in metadata["reason"]

        # Verify it's the consecutive loss breaker
        events = db_session.query(CircuitBreakerEvent).filter_by(
            breaker_type="consecutive_losses"
        ).all()
        assert len(events) > 0

    def test_circuit_breaker_persists_across_restarts(self, trading_config, db_session):
        """Test circuit breaker state persists across service restarts."""
        now = datetime.now(timezone.utc)

        # Create a persisted circuit breaker event
        event = CircuitBreakerEvent(
            breaker_type="daily_loss_limit",
            action="triggered",
            triggered_at=now,
            value=-3.5,
            metadata={"daily_pnl": -350.0, "balance": 10000.0}
        )
        db_session.add(event)
        db_session.commit()

        # Simulate restart by creating new circuit breaker instance
        new_breaker = TradingCircuitBreaker(trading_config.conservative_hybrid)

        # Check if trading is allowed
        can_trade, reason = new_breaker.can_trade(db_session, 10000.0)

        # Should still be blocked
        assert can_trade is False
        assert "Daily loss limit breached" in reason


# ============================================================================
# END-TO-END WORKFLOW TESTS
# ============================================================================


class TestEndToEndWorkflow:
    """Test complete trading workflows."""

    def test_full_trade_lifecycle_with_position_sizing(self, mock_trading_service, db_session):
        """Test full trade lifecycle: open, size correctly, close, record."""
        confidence = 0.75
        sl_pips = 15.0

        # Execute trade
        trade, metadata = mock_trading_service.execute_trade(
            db_session, "long", confidence, sl_pips, entry_price=1.0850
        )

        # Verify trade was opened
        assert trade is not None
        assert trade.status == "open"
        assert trade.lot_size > 0.0
        assert trade.risk_percentage_used > 0.0

        # Close trade with profit
        closed_trade = mock_trading_service.close_trade(
            db_session, trade, exit_price=1.0875, exit_reason="tp"
        )

        # Verify trade was closed
        assert closed_trade.status == "closed"
        assert closed_trade.pnl_usd > 0.0
        assert closed_trade.is_winner is True
        assert closed_trade.pips > 0.0

    def test_progressive_position_reduction_with_losses(self, mock_trading_service, db_session):
        """Test that position size reduces as losses accumulate (balance decreases)."""
        confidence = 0.75
        sl_pips = 15.0

        initial_balance = mock_trading_service.balance

        # First trade
        trade1, _ = mock_trading_service.execute_trade(db_session, "long", confidence, sl_pips)
        size1 = trade1.lot_size

        # Close with loss
        mock_trading_service.close_trade(db_session, trade1, exit_price=1.0840, exit_reason="sl")

        # Balance should have decreased
        assert mock_trading_service.balance < initial_balance

        # Second trade with reduced balance
        trade2, _ = mock_trading_service.execute_trade(db_session, "long", confidence, sl_pips)
        size2 = trade2.lot_size

        # Position should be smaller due to reduced balance
        assert size2 < size1

    def test_mixed_trade_outcomes(self, mock_trading_service, db_session):
        """Test trading with mixed wins and losses."""
        confidence = 0.75
        sl_pips = 15.0

        # Execute 10 trades with alternating outcomes
        for i in range(10):
            trade, metadata = mock_trading_service.execute_trade(
                db_session, "long", confidence, sl_pips
            )

            if trade is None:
                break

            # Alternate between wins and losses
            if i % 2 == 0:
                # Win
                mock_trading_service.close_trade(
                    db_session, trade, exit_price=1.0875, exit_reason="tp"
                )
            else:
                # Loss
                mock_trading_service.close_trade(
                    db_session, trade, exit_price=1.0840, exit_reason="sl"
                )

        # Verify trades were recorded
        trades = db_session.query(Trade).all()
        assert len(trades) == 10

        # Count wins and losses
        wins = sum(1 for t in trades if t.is_winner)
        losses = sum(1 for t in trades if not t.is_winner)

        assert wins == 5
        assert losses == 5

    def test_confidence_scaling_effect(self, mock_trading_service, db_session):
        """Test that higher confidence results in larger positions."""
        sl_pips = 15.0

        # Low confidence (just above threshold)
        trade1, _ = mock_trading_service.execute_trade(db_session, "long", 0.71, sl_pips)

        # Medium confidence
        trade2, _ = mock_trading_service.execute_trade(db_session, "long", 0.75, sl_pips)

        # High confidence
        trade3, _ = mock_trading_service.execute_trade(db_session, "long", 0.80, sl_pips)

        # Verify position sizes increase with confidence
        assert trade1.lot_size < trade2.lot_size < trade3.lot_size
        assert trade1.risk_percentage_used < trade2.risk_percentage_used < trade3.risk_percentage_used


# ============================================================================
# ERROR RECOVERY TESTS
# ============================================================================


class TestErrorRecovery:
    """Test error handling and recovery scenarios."""

    def test_invalid_inputs_dont_create_trades(self, mock_trading_service, db_session):
        """Test that invalid inputs don't create database records."""
        # Try with confidence below threshold
        trade1, metadata1 = mock_trading_service.execute_trade(
            db_session, "long", confidence=0.65, sl_pips=15.0
        )

        assert trade1 is None

        # Try with invalid SL
        trade2, metadata2 = mock_trading_service.execute_trade(
            db_session, "long", confidence=0.75, sl_pips=0.0
        )

        assert trade2 is None

        # Verify no trades were created
        trades = db_session.query(Trade).all()
        assert len(trades) == 0

    def test_database_rollback_on_error(self, mock_trading_service, db_session):
        """Test that database rollback works on errors."""
        confidence = 0.75
        sl_pips = 15.0

        # Execute valid trade
        trade, _ = mock_trading_service.execute_trade(db_session, "long", confidence, sl_pips)
        assert trade is not None

        initial_trade_count = db_session.query(Trade).count()

        # Try to create invalid trade (this should fail)
        try:
            with patch.object(db_session, 'commit', side_effect=Exception("DB error")):
                mock_trading_service.execute_trade(db_session, "long", confidence, sl_pips)
        except Exception:
            db_session.rollback()

        # Verify trade count didn't change
        final_trade_count = db_session.query(Trade).count()
        assert final_trade_count == initial_trade_count
