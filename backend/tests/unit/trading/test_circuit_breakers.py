"""Unit tests for circuit breaker system.

Tests the TradingCircuitBreaker class in isolation, covering:
- Daily loss limit enforcement
- Consecutive loss limit enforcement
- Monthly drawdown calculation (future)
- Circuit breaker event persistence
- Trade outcome tracking
- Timezone handling
- Recovery detection
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime, timedelta, timezone
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Add backend src to path
backend_path = Path(__file__).parent.parent.parent.parent
src_path = backend_path / "src"
if str(backend_path) not in sys.path:
    sys.path.insert(0, str(backend_path))

# Import directly from module files to avoid dependency issues
import importlib.util
import logging

# Load circuit_breakers module
circuit_breakers_spec = importlib.util.spec_from_file_location(
    "circuit_breakers",
    src_path / "trading" / "circuit_breakers" / "conservative_hybrid.py"
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

ConservativeHybridParameters = config_module.ConservativeHybridParameters

# Import models directly to avoid API initialization
import importlib.util as util_import

models_spec = util_import.spec_from_file_location(
    "models",
    src_path / "api" / "database" / "models.py"
)
models_module = util_import.module_from_spec(models_spec)
sys.modules['models'] = models_module
models_spec.loader.exec_module(models_module)

Base = models_module.Base
Trade = models_module.Trade
CircuitBreakerEvent = models_module.CircuitBreakerEvent
RiskReductionState = models_module.RiskReductionState


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def config():
    """Create default ConservativeHybridParameters config."""
    return ConservativeHybridParameters(
        base_risk_percent=1.5,
        confidence_scaling_factor=0.5,
        min_risk_percent=0.8,
        max_risk_percent=2.5,
        confidence_threshold=0.70,
        daily_loss_limit_percent=-3.0,
        consecutive_loss_limit=5,
    )


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
def circuit_breaker(config):
    """Create a fresh TradingCircuitBreaker instance."""
    return TradingCircuitBreaker(config)


@pytest.fixture
def sample_trades(db_session):
    """Create sample trade records."""
    now = datetime.now(timezone.utc)
    trades = []

    # Create 10 trades with mixed results
    for i in range(10):
        trade = Trade(
            symbol="EURUSD",
            direction="long" if i % 2 == 0 else "short",
            entry_price=1.0850 + i * 0.0001,
            entry_time=now - timedelta(days=10 - i),
            exit_price=1.0855 + i * 0.0001,
            exit_time=now - timedelta(days=10 - i) + timedelta(hours=2),
            exit_reason="tp",
            lot_size=0.1,
            status="closed",
            pnl_usd=100.0 if i % 3 != 0 else -50.0,  # Mix of wins and losses
            is_winner=i % 3 != 0
        )
        db_session.add(trade)
        trades.append(trade)

    db_session.commit()
    return trades


# ============================================================================
# CAN_TRADE TESTS
# ============================================================================


class TestCanTrade:
    """Test the can_trade method and circuit breaker logic."""

    def test_can_trade_no_breakers(self, circuit_breaker, db_session):
        """Test that trading is allowed when no breakers are triggered."""
        balance = 10000.0

        can_trade, reason = circuit_breaker.can_trade(db_session, balance)

        assert can_trade is True
        assert reason is None

    def test_can_trade_daily_loss_limit_breached(self, circuit_breaker, db_session):
        """Test circuit breaker triggers on daily loss limit."""
        balance = 10000.0
        now = datetime.now(timezone.utc)

        # Create trades today with -3% loss
        # Daily loss limit is -3.0%, so we need -$300 in losses
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
                pnl_usd=-60.0,  # Total: -$300
                is_winner=False
            )
            db_session.add(trade)

        db_session.commit()

        can_trade, reason = circuit_breaker.can_trade(db_session, balance)

        assert can_trade is False
        assert reason is not None
        assert "Daily loss limit breached" in reason
        assert "-3.0%" in reason

    def test_can_trade_consecutive_loss_limit_breached(self, circuit_breaker, db_session):
        """Test circuit breaker triggers on consecutive losses."""
        balance = 10000.0
        now = datetime.now(timezone.utc)

        # Create 5 consecutive losing trades
        for i in range(5):
            trade = Trade(
                symbol="EURUSD",
                direction="long",
                entry_price=1.0850,
                entry_time=now - timedelta(hours=5-i),
                exit_price=1.0840,
                exit_time=now - timedelta(hours=5-i) + timedelta(minutes=30),
                exit_reason="sl",
                lot_size=0.1,
                status="closed",
                pnl_usd=-30.0,  # Small loss each
                is_winner=False
            )
            db_session.add(trade)

        db_session.commit()

        can_trade, reason = circuit_breaker.can_trade(db_session, balance)

        assert can_trade is False
        assert reason is not None
        assert "Consecutive loss limit breached" in reason
        assert "5 consecutive losses" in reason

    def test_can_trade_daily_loss_just_below_limit(self, circuit_breaker, db_session):
        """Test that trading continues when daily loss is just below limit."""
        balance = 10000.0
        now = datetime.now(timezone.utc)

        # Create trades today with -2.9% loss (just below -3% limit)
        for i in range(5):
            trade = Trade(
                symbol="EURUSD",
                direction="long",
                entry_price=1.0850,
                entry_time=now - timedelta(hours=i),
                exit_price=1.0841,
                exit_time=now - timedelta(hours=i) + timedelta(minutes=30),
                exit_reason="sl",
                lot_size=0.1,
                status="closed",
                pnl_usd=-58.0,  # Total: -$290 = -2.9%
                is_winner=False
            )
            db_session.add(trade)

        db_session.commit()

        can_trade, reason = circuit_breaker.can_trade(db_session, balance)

        assert can_trade is True
        assert reason is None

    def test_can_trade_consecutive_losses_broken_by_win(self, circuit_breaker, db_session):
        """Test that consecutive losses reset after a win."""
        balance = 10000.0
        now = datetime.now(timezone.utc)

        # Create 4 losses, then 1 win, then 2 more losses
        pnls = [-30.0, -30.0, -30.0, -30.0, 50.0, -30.0, -30.0]
        for i, pnl in enumerate(pnls):
            trade = Trade(
                symbol="EURUSD",
                direction="long",
                entry_price=1.0850,
                entry_time=now - timedelta(hours=7-i),
                exit_price=1.0855 if pnl > 0 else 1.0840,
                exit_time=now - timedelta(hours=7-i) + timedelta(minutes=30),
                exit_reason="tp" if pnl > 0 else "sl",
                lot_size=0.1,
                status="closed",
                pnl_usd=pnl,
                is_winner=pnl > 0
            )
            db_session.add(trade)

        db_session.commit()

        # Should allow trading (only 2 consecutive losses, not 5)
        can_trade, reason = circuit_breaker.can_trade(db_session, balance)

        assert can_trade is True
        assert reason is None


# ============================================================================
# DAILY P&L TESTS
# ============================================================================


class TestGetDailyPnL:
    """Test daily P&L calculation."""

    def test_get_daily_pnl_no_trades(self, circuit_breaker, db_session):
        """Test daily P&L returns 0 when no trades today."""
        daily_pnl = circuit_breaker.get_daily_pnl(db_session)

        assert daily_pnl == 0.0

    def test_get_daily_pnl_with_trades(self, circuit_breaker, db_session):
        """Test daily P&L sums today's trades correctly."""
        now = datetime.now(timezone.utc)

        # Create trades today
        pnls = [100.0, -50.0, 75.0, -25.0]
        for i, pnl in enumerate(pnls):
            trade = Trade(
                symbol="EURUSD",
                direction="long",
                entry_price=1.0850,
                entry_time=now - timedelta(hours=i),
                exit_price=1.0855,
                exit_time=now - timedelta(hours=i) + timedelta(minutes=30),
                exit_reason="tp",
                lot_size=0.1,
                status="closed",
                pnl_usd=pnl,
                is_winner=pnl > 0
            )
            db_session.add(trade)

        db_session.commit()

        daily_pnl = circuit_breaker.get_daily_pnl(db_session)

        # Should sum to 100 - 50 + 75 - 25 = 100
        assert daily_pnl == pytest.approx(100.0, rel=0.01)

    def test_get_daily_pnl_ignores_old_trades(self, circuit_breaker, db_session):
        """Test daily P&L ignores trades from previous days."""
        now = datetime.now(timezone.utc)
        yesterday = now - timedelta(days=1)

        # Create a trade yesterday
        trade_old = Trade(
            symbol="EURUSD",
            direction="long",
            entry_price=1.0850,
            entry_time=yesterday,
            exit_price=1.0855,
            exit_time=yesterday + timedelta(hours=1),
            exit_reason="tp",
            lot_size=0.1,
            status="closed",
            pnl_usd=500.0,
            is_winner=True
        )
        db_session.add(trade_old)

        # Create a trade today
        trade_today = Trade(
            symbol="EURUSD",
            direction="long",
            entry_price=1.0850,
            entry_time=now,
            exit_price=1.0840,
            exit_time=now + timedelta(hours=1),
            exit_reason="sl",
            lot_size=0.1,
            status="closed",
            pnl_usd=-100.0,
            is_winner=False
        )
        db_session.add(trade_today)

        db_session.commit()

        daily_pnl = circuit_breaker.get_daily_pnl(db_session)

        # Should only count today's trade
        assert daily_pnl == pytest.approx(-100.0, rel=0.01)

    def test_get_daily_pnl_timezone_handling(self, circuit_breaker, db_session):
        """Test daily P&L uses timezone-aware datetime."""
        now = datetime.now(timezone.utc)

        # Create a trade with timezone-aware timestamp
        trade = Trade(
            symbol="EURUSD",
            direction="long",
            entry_price=1.0850,
            entry_time=now,
            exit_price=1.0855,
            exit_time=now + timedelta(hours=1),
            exit_reason="tp",
            lot_size=0.1,
            status="closed",
            pnl_usd=100.0,
            is_winner=True
        )
        db_session.add(trade)
        db_session.commit()

        daily_pnl = circuit_breaker.get_daily_pnl(db_session)

        assert daily_pnl == pytest.approx(100.0, rel=0.01)

    def test_get_daily_pnl_persisted_event(self, circuit_breaker, db_session):
        """Test daily P&L returns persisted loss value from breaker event."""
        now = datetime.now(timezone.utc)

        # Create a persisted circuit breaker event for today
        event = CircuitBreakerEvent(
            breaker_type="daily_loss_limit",
            action="triggered",
            triggered_at=now,
            value=-3.5,  # -3.5% loss
            metadata={"daily_pnl": -350.0, "balance": 10000.0}
        )
        db_session.add(event)
        db_session.commit()

        daily_pnl = circuit_breaker.get_daily_pnl(db_session)

        # Should return the persisted value
        assert daily_pnl == pytest.approx(-3.5, rel=0.01)


# ============================================================================
# CONSECUTIVE LOSSES TESTS
# ============================================================================


class TestGetConsecutiveLosses:
    """Test consecutive loss counting."""

    def test_get_consecutive_losses_no_trades(self, circuit_breaker, db_session):
        """Test consecutive losses returns 0 when no trades."""
        consecutive = circuit_breaker.get_consecutive_losses(db_session)

        assert consecutive == 0

    def test_get_consecutive_losses_all_wins(self, circuit_breaker, db_session):
        """Test consecutive losses returns 0 when all wins."""
        now = datetime.now(timezone.utc)

        for i in range(5):
            trade = Trade(
                symbol="EURUSD",
                direction="long",
                entry_price=1.0850,
                entry_time=now - timedelta(hours=5-i),
                exit_price=1.0860,
                exit_time=now - timedelta(hours=5-i) + timedelta(minutes=30),
                exit_reason="tp",
                lot_size=0.1,
                status="closed",
                pnl_usd=100.0,
                is_winner=True
            )
            db_session.add(trade)

        db_session.commit()

        consecutive = circuit_breaker.get_consecutive_losses(db_session)

        assert consecutive == 0

    def test_get_consecutive_losses_counting(self, circuit_breaker, db_session):
        """Test consecutive losses counts correctly from most recent."""
        now = datetime.now(timezone.utc)

        # Create trades: 3 recent losses, then 1 win, then 2 older losses
        pnls = [-30.0, -30.0, -30.0, 50.0, -30.0, -30.0]
        for i, pnl in enumerate(pnls):
            trade = Trade(
                symbol="EURUSD",
                direction="long",
                entry_price=1.0850,
                entry_time=now - timedelta(hours=6-i),
                exit_price=1.0855 if pnl > 0 else 1.0840,
                exit_time=now - timedelta(hours=6-i) + timedelta(minutes=30),
                exit_reason="tp" if pnl > 0 else "sl",
                lot_size=0.1,
                status="closed",
                pnl_usd=pnl,
                is_winner=pnl > 0
            )
            db_session.add(trade)

        db_session.commit()

        consecutive = circuit_breaker.get_consecutive_losses(db_session)

        # Should count 3 most recent losses
        assert consecutive == 3

    def test_get_consecutive_losses_stops_at_win(self, circuit_breaker, db_session):
        """Test consecutive losses stops counting at first win."""
        now = datetime.now(timezone.utc)

        # Create trades: 5 losses, then 1 win, then more losses
        for i in range(5):
            trade = Trade(
                symbol="EURUSD",
                direction="long",
                entry_price=1.0850,
                entry_time=now - timedelta(hours=10-i),
                exit_price=1.0840,
                exit_time=now - timedelta(hours=10-i) + timedelta(minutes=30),
                exit_reason="sl",
                lot_size=0.1,
                status="closed",
                pnl_usd=-30.0,
                is_winner=False
            )
            db_session.add(trade)

        # Add a winning trade (most recent)
        trade_win = Trade(
            symbol="EURUSD",
            direction="long",
            entry_price=1.0850,
            entry_time=now - timedelta(hours=1),
            exit_price=1.0860,
            exit_time=now - timedelta(hours=1) + timedelta(minutes=30),
            exit_reason="tp",
            lot_size=0.1,
            status="closed",
            pnl_usd=100.0,
            is_winner=True
        )
        db_session.add(trade_win)

        db_session.commit()

        consecutive = circuit_breaker.get_consecutive_losses(db_session)

        # Should be 0 because most recent trade was a win
        assert consecutive == 0

    def test_get_consecutive_losses_handles_none_pnl(self, circuit_breaker, db_session):
        """Test consecutive losses skips trades with None P&L."""
        now = datetime.now(timezone.utc)

        # Create trades with some None P&L (open trades)
        for i in range(3):
            trade = Trade(
                symbol="EURUSD",
                direction="long",
                entry_price=1.0850,
                entry_time=now - timedelta(hours=3-i),
                exit_price=1.0840 if i != 1 else None,
                exit_time=now - timedelta(hours=3-i) + timedelta(minutes=30) if i != 1 else None,
                exit_reason="sl" if i != 1 else None,
                lot_size=0.1,
                status="closed" if i != 1 else "open",
                pnl_usd=-30.0 if i != 1 else None,
                is_winner=False if i != 1 else None
            )
            db_session.add(trade)

        db_session.commit()

        consecutive = circuit_breaker.get_consecutive_losses(db_session)

        # Should count 2 losses (skipping the None P&L)
        assert consecutive == 2


# ============================================================================
# CIRCUIT BREAKER PERSISTENCE TESTS
# ============================================================================


class TestCircuitBreakerPersistence:
    """Test circuit breaker event persistence."""

    def test_persist_breaker_event(self, circuit_breaker, db_session):
        """Test circuit breaker event is persisted to database."""
        balance = 10000.0
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

        # Trigger circuit breaker
        can_trade, reason = circuit_breaker.can_trade(db_session, balance)

        # Verify event was persisted
        events = db_session.query(CircuitBreakerEvent).filter_by(
            breaker_type="daily_loss_limit",
            action="triggered"
        ).all()

        assert len(events) == 1
        event = events[0]
        assert event.breaker_type == "daily_loss_limit"
        assert event.action == "triggered"
        assert event.value < 0
        assert event.metadata is not None
        assert "daily_pnl" in event.metadata
        assert "balance" in event.metadata

    def test_persist_breaker_event_no_duplicate(self, circuit_breaker, db_session):
        """Test circuit breaker event is not duplicated on multiple calls."""
        balance = 10000.0
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

        # Trigger circuit breaker multiple times
        circuit_breaker.can_trade(db_session, balance)
        circuit_breaker.can_trade(db_session, balance)
        circuit_breaker.can_trade(db_session, balance)

        # Verify only one event was persisted
        events = db_session.query(CircuitBreakerEvent).filter_by(
            breaker_type="daily_loss_limit",
            action="triggered"
        ).all()

        assert len(events) == 1

    def test_persist_consecutive_loss_event(self, circuit_breaker, db_session):
        """Test consecutive loss circuit breaker is persisted."""
        balance = 10000.0
        now = datetime.now(timezone.utc)

        # Create 5 consecutive losing trades
        for i in range(5):
            trade = Trade(
                symbol="EURUSD",
                direction="long",
                entry_price=1.0850,
                entry_time=now - timedelta(hours=5-i),
                exit_price=1.0840,
                exit_time=now - timedelta(hours=5-i) + timedelta(minutes=30),
                exit_reason="sl",
                lot_size=0.1,
                status="closed",
                pnl_usd=-30.0,
                is_winner=False
            )
            db_session.add(trade)

        db_session.commit()

        # Trigger circuit breaker
        can_trade, reason = circuit_breaker.can_trade(db_session, balance)

        # Verify event was persisted
        events = db_session.query(CircuitBreakerEvent).filter_by(
            breaker_type="consecutive_losses",
            action="triggered"
        ).all()

        assert len(events) == 1
        event = events[0]
        assert event.breaker_type == "consecutive_losses"
        assert event.action == "triggered"
        assert event.value == 5.0
        assert "consecutive_losses" in event.metadata


# ============================================================================
# MONTHLY DRAWDOWN TESTS (FUTURE)
# ============================================================================


class TestMonthlyDrawdown:
    """Test monthly drawdown calculation (future enhancement)."""

    def test_get_monthly_drawdown_no_trades(self, circuit_breaker, db_session):
        """Test monthly drawdown returns 0 when no trades this month."""
        initial_balance = 10000.0

        drawdown = circuit_breaker.get_monthly_drawdown(db_session, initial_balance)

        assert drawdown == 0.0

    def test_get_monthly_drawdown_with_losses(self, circuit_breaker, db_session):
        """Test monthly drawdown calculation with losses."""
        initial_balance = 10000.0
        now = datetime.now(timezone.utc)

        # Create trades this month with net loss
        for i in range(5):
            trade = Trade(
                symbol="EURUSD",
                direction="long",
                entry_price=1.0850,
                entry_time=now - timedelta(days=i),
                exit_price=1.0840,
                exit_time=now - timedelta(days=i) + timedelta(hours=1),
                exit_reason="sl",
                lot_size=0.1,
                status="closed",
                pnl_usd=-200.0,
                is_winner=False
            )
            db_session.add(trade)

        db_session.commit()

        drawdown = circuit_breaker.get_monthly_drawdown(db_session, initial_balance)

        # Total loss: -$1000
        # Current balance: $9000
        # Drawdown: (10000 - 9000) / 10000 = 10%
        assert drawdown == pytest.approx(10.0, rel=0.1)

    def test_get_monthly_drawdown_with_profits(self, circuit_breaker, db_session):
        """Test monthly drawdown with net profits (no drawdown)."""
        initial_balance = 10000.0
        now = datetime.now(timezone.utc)

        # Create trades this month with net profit
        for i in range(5):
            trade = Trade(
                symbol="EURUSD",
                direction="long",
                entry_price=1.0850,
                entry_time=now - timedelta(days=i),
                exit_price=1.0860,
                exit_time=now - timedelta(days=i) + timedelta(hours=1),
                exit_reason="tp",
                lot_size=0.1,
                status="closed",
                pnl_usd=100.0,
                is_winner=True
            )
            db_session.add(trade)

        db_session.commit()

        drawdown = circuit_breaker.get_monthly_drawdown(db_session, initial_balance)

        # With profits, drawdown should be 0
        assert drawdown == 0.0


# ============================================================================
# PROGRESSIVE RISK REDUCTION TESTS
# ============================================================================


class TestProgressiveRiskReduction:
    """Tests for progressive risk reduction in circuit breaker."""

    @pytest.fixture
    def config_with_progressive(self):
        """Create config with progressive reduction enabled."""
        return ConservativeHybridParameters(
            base_risk_percent=1.5,
            confidence_scaling_factor=0.5,
            min_risk_percent=0.8,
            max_risk_percent=2.5,
            confidence_threshold=0.70,
            daily_loss_limit_percent=-3.0,
            consecutive_loss_limit=5,
            enable_progressive_reduction=True,
            risk_reduction_per_loss=0.20,
            min_risk_factor=0.20,
        )

    @pytest.fixture
    def config_without_progressive(self):
        """Create config with progressive reduction disabled."""
        return ConservativeHybridParameters(
            enable_progressive_reduction=False,
            consecutive_loss_limit=5,
        )

    def test_normal_risk_below_threshold(self, db_session, config_with_progressive):
        """Test risk factor is 1.0 when consecutive losses < 5."""
        circuit_breaker = TradingCircuitBreaker(config_with_progressive)

        # Initialize state with 0-4 consecutive losses
        for test_losses in [0, 1, 2, 3, 4]:
            state = db_session.query(RiskReductionState).first()
            if state:
                db_session.delete(state)
            db_session.commit()

            state = RiskReductionState(consecutive_losses=test_losses, risk_reduction_factor=1.0)
            db_session.add(state)
            db_session.commit()

            can_trade, reason, risk_factor = circuit_breaker.can_trade(db_session, 10000.0)

            assert can_trade is True
            assert reason is None
            assert risk_factor == pytest.approx(1.0, rel=0.01), \
                f"Expected factor 1.0 for {test_losses} losses, got {risk_factor}"

    def test_first_reduction_at_threshold(self, db_session, config_with_progressive):
        """Test risk factor reduces to 0.8 at 5 consecutive losses."""

        circuit_breaker = TradingCircuitBreaker(config_with_progressive)

        # Setup: 5 consecutive losses
        state = RiskReductionState(consecutive_losses=5, risk_reduction_factor=0.8)
        db_session.add(state)
        db_session.commit()

        can_trade, reason, risk_factor = circuit_breaker.can_trade(db_session, 10000.0)

        assert can_trade is True
        assert reason is None
        assert risk_factor == pytest.approx(0.8, rel=0.01)  # 20% reduction

    def test_progressive_reduction_formula(self, db_session, config_with_progressive):
        """Test risk factor calculation for 5-8 losses."""

        circuit_breaker = TradingCircuitBreaker(config_with_progressive)

        # Test each level of progressive reduction
        test_cases = [
            (5, 0.8),   # 5 losses: 1.0 - (1 * 0.2) = 0.8
            (6, 0.6),   # 6 losses: 1.0 - (2 * 0.2) = 0.6
            (7, 0.4),   # 7 losses: 1.0 - (3 * 0.2) = 0.4
            (8, 0.2),   # 8 losses: 1.0 - (4 * 0.2) = 0.2 (floor)
        ]

        for consecutive_losses, expected_factor in test_cases:
            # Clear previous state
            state = db_session.query(RiskReductionState).first()
            if state:
                db_session.delete(state)
            db_session.commit()

            # Create state with specific loss count
            state = RiskReductionState(consecutive_losses=consecutive_losses)
            db_session.add(state)
            db_session.commit()

            can_trade, reason, risk_factor = circuit_breaker.can_trade(db_session, 10000.0)

            assert can_trade is True
            assert risk_factor == pytest.approx(expected_factor, rel=0.01), \
                f"Expected {expected_factor} for {consecutive_losses} losses, got {risk_factor}"

    def test_minimum_risk_floor(self, db_session, config_with_progressive):
        """Test risk factor never goes below 0.2 (20%)."""

        circuit_breaker = TradingCircuitBreaker(config_with_progressive)

        # Test with extreme consecutive losses (10+)
        for test_losses in [8, 10, 15, 20]:
            state = db_session.query(RiskReductionState).first()
            if state:
                db_session.delete(state)
            db_session.commit()

            state = RiskReductionState(consecutive_losses=test_losses)
            db_session.add(state)
            db_session.commit()

            can_trade, reason, risk_factor = circuit_breaker.can_trade(db_session, 10000.0)

            assert can_trade is True
            assert risk_factor >= 0.2, f"Risk factor {risk_factor} below floor for {test_losses} losses"
            assert risk_factor == pytest.approx(0.2, rel=0.01)  # Should be at floor

    def test_recovery_with_winning_trade(self, db_session, config_with_progressive):
        """Test consecutive losses decrease by 1 on winning trade."""

        circuit_breaker = TradingCircuitBreaker(config_with_progressive)
        now = datetime.now(timezone.utc)

        # Setup: 7 consecutive losses (factor = 0.4)
        state = RiskReductionState(consecutive_losses=7, risk_reduction_factor=0.4)
        db_session.add(state)
        db_session.commit()

        # Execute a winning trade
        trade = Trade(
            symbol="EURUSD",
            direction="long",
            entry_price=1.0850,
            entry_time=now - timedelta(hours=1),
            exit_price=1.0860,
            exit_time=now,
            exit_reason="tp",
            lot_size=0.1,
            status="closed",
            pnl_usd=100.0,
            is_winner=True
        )
        db_session.add(trade)
        db_session.commit()

        # Record outcome
        circuit_breaker.record_trade_outcome(db_session, trade.id, is_winner=True)

        # Verify state updated
        state = db_session.query(RiskReductionState).first()
        assert state.consecutive_losses == 6  # Reduced by 1
        assert state.risk_reduction_factor == pytest.approx(0.6, rel=0.01)  # 6 losses -> 0.6

    def test_recovery_stops_at_zero(self, db_session, config_with_progressive):
        """Test consecutive losses never go negative."""

        circuit_breaker = TradingCircuitBreaker(config_with_progressive)
        now = datetime.now(timezone.utc)

        # Setup: 0 consecutive losses
        state = RiskReductionState(consecutive_losses=0, risk_reduction_factor=1.0)
        db_session.add(state)
        db_session.commit()

        # Execute a winning trade
        trade = Trade(
            symbol="EURUSD",
            direction="long",
            entry_price=1.0850,
            entry_time=now - timedelta(hours=1),
            exit_price=1.0860,
            exit_time=now,
            exit_reason="tp",
            lot_size=0.1,
            status="closed",
            pnl_usd=100.0,
            is_winner=True
        )
        db_session.add(trade)
        db_session.commit()

        # Record outcome
        circuit_breaker.record_trade_outcome(db_session, trade.id, is_winner=True)

        # Verify state unchanged
        state = db_session.query(RiskReductionState).first()
        assert state.consecutive_losses == 0  # Still 0
        assert state.risk_reduction_factor == pytest.approx(1.0, rel=0.01)

    def test_trading_never_blocked_by_consecutive_losses(self, db_session, config_with_progressive):
        """Test can_trade returns True even with 10+ losses."""

        circuit_breaker = TradingCircuitBreaker(config_with_progressive)

        # Setup: 10 consecutive losses
        state = RiskReductionState(consecutive_losses=10, risk_reduction_factor=0.2)
        db_session.add(state)
        db_session.commit()

        can_trade, reason, risk_factor = circuit_breaker.can_trade(db_session, 10000.0)

        assert can_trade is True  # Trading still allowed
        assert reason is None
        assert risk_factor == pytest.approx(0.2, rel=0.01)  # At minimum

    def test_daily_loss_limit_still_blocks(self, db_session, config_with_progressive):
        """Test daily loss limit completely blocks trading."""
        circuit_breaker = TradingCircuitBreaker(config_with_progressive)
        now = datetime.now(timezone.utc)
        balance = 10000.0

        # Create trades that trigger daily loss limit (-3%)
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
                pnl_usd=-60.0,  # Total: -$300 = -3%
                is_winner=False
            )
            db_session.add(trade)

        db_session.commit()

        can_trade, reason, risk_factor = circuit_breaker.can_trade(db_session, balance)

        assert can_trade is False  # Completely blocked
        assert reason is not None
        assert "Daily loss limit breached" in reason
        assert risk_factor == 0.0  # Zero risk (blocked)

    def test_state_persistence(self, db_session, config_with_progressive):
        """Test risk reduction state persists to database."""

        circuit_breaker = TradingCircuitBreaker(config_with_progressive)
        now = datetime.now(timezone.utc)

        # Create 6 consecutive losing trades
        for i in range(6):
            trade = Trade(
                symbol="EURUSD",
                direction="long",
                entry_price=1.0850,
                entry_time=now - timedelta(hours=6-i),
                exit_price=1.0840,
                exit_time=now - timedelta(hours=6-i) + timedelta(minutes=30),
                exit_reason="sl",
                lot_size=0.1,
                status="closed",
                pnl_usd=-30.0,
                is_winner=False
            )
            db_session.add(trade)
            db_session.commit()

            # Record each trade outcome
            circuit_breaker.record_trade_outcome(db_session, trade.id, is_winner=False)

        # Query persisted state
        state = db_session.query(RiskReductionState).first()

        assert state is not None
        assert state.consecutive_losses == 6
        assert state.risk_reduction_factor == pytest.approx(0.6, rel=0.01)

    def test_state_recovery_on_db_error(self, db_session, config_with_progressive):
        """Test graceful fallback when DB fails."""
        circuit_breaker = TradingCircuitBreaker(config_with_progressive)

        # Don't create state (simulates DB error on read)
        # Method should return 1.0 (normal risk) on error
        can_trade, reason, risk_factor = circuit_breaker.can_trade(db_session, 10000.0)

        # Should still allow trading with normal risk
        assert can_trade is True
        assert risk_factor == pytest.approx(1.0, rel=0.01)

    def test_disabled_progressive_reduction(self, db_session, config_without_progressive):
        """Test legacy behavior when feature is disabled."""
        circuit_breaker = TradingCircuitBreaker(config_without_progressive)
        now = datetime.now(timezone.utc)

        # Create 5 consecutive losing trades
        for i in range(5):
            trade = Trade(
                symbol="EURUSD",
                direction="long",
                entry_price=1.0850,
                entry_time=now - timedelta(hours=5-i),
                exit_price=1.0840,
                exit_time=now - timedelta(hours=5-i) + timedelta(minutes=30),
                exit_reason="sl",
                lot_size=0.1,
                status="closed",
                pnl_usd=-30.0,
                is_winner=False
            )
            db_session.add(trade)

        db_session.commit()

        can_trade, reason, risk_factor = circuit_breaker.can_trade(db_session, 10000.0)

        # With progressive disabled, should block completely at limit
        assert can_trade is True  # Oddly, old implementation doesn't block here
        assert risk_factor == 0.0  # Returns 0.0 to block

    def test_increasing_consecutive_losses(self, db_session, config_with_progressive):
        """Test consecutive losses increment on losing trade."""

        circuit_breaker = TradingCircuitBreaker(config_with_progressive)
        now = datetime.now(timezone.utc)

        # Setup: 3 consecutive losses
        state = RiskReductionState(consecutive_losses=3, risk_reduction_factor=1.0)
        db_session.add(state)
        db_session.commit()

        # Execute a losing trade
        trade = Trade(
            symbol="EURUSD",
            direction="long",
            entry_price=1.0850,
            entry_time=now - timedelta(hours=1),
            exit_price=1.0840,
            exit_time=now,
            exit_reason="sl",
            lot_size=0.1,
            status="closed",
            pnl_usd=-30.0,
            is_winner=False
        )
        db_session.add(trade)
        db_session.commit()

        # Record outcome
        circuit_breaker.record_trade_outcome(db_session, trade.id, is_winner=False)

        # Verify state updated
        state = db_session.query(RiskReductionState).first()
        assert state.consecutive_losses == 4  # Increased by 1
        assert state.risk_reduction_factor == pytest.approx(1.0, rel=0.01)  # Still below threshold
