"""Integration tests for progressive risk reduction system.

Tests the complete flow of progressive risk reduction across:
- Circuit breaker state management
- Position sizer integration
- Trading service integration
- State persistence and recovery
- Cross-service coordination
"""

import pytest
import sys
from pathlib import Path
from datetime import datetime, timedelta, timezone
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Add backend src to path
backend_path = Path(__file__).parent.parent.parent
src_path = backend_path / "src"
if str(backend_path) not in sys.path:
    sys.path.insert(0, str(backend_path))

# Import directly from module files
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

# Load position_sizer module
position_sizer_spec = importlib.util.spec_from_file_location(
    "position_sizer",
    src_path / "trading" / "position_sizer.py"
)
position_sizer_module = importlib.util.module_from_spec(position_sizer_spec)
position_sizer_module.logger = logging.getLogger(__name__)
position_sizer_spec.loader.exec_module(position_sizer_module)

ConservativeHybridSizer = position_sizer_module.ConservativeHybridSizer

# Load trading_config module
config_spec = importlib.util.spec_from_file_location(
    "trading_config",
    src_path / "config" / "trading_config.py"
)
config_module = importlib.util.module_from_spec(config_spec)
config_module.logger = logging.getLogger(__name__)
config_spec.loader.exec_module(config_module)

ConservativeHybridParameters = config_module.ConservativeHybridParameters

# Import models normally
from src.api.database.models import Base, Trade, RiskReductionState


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def config():
    """Create default ConservativeHybridParameters config with progressive reduction."""
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
def config_disabled():
    """Create config with progressive reduction disabled."""
    return ConservativeHybridParameters(
        enable_progressive_reduction=False,
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
def position_sizer():
    """Create a fresh ConservativeHybridSizer instance."""
    return ConservativeHybridSizer()


# ============================================================================
# FULL CYCLE TESTS
# ============================================================================


class TestProgressiveReductionIntegration:
    """Integration tests for progressive risk reduction system."""

    def test_full_reduction_and_recovery_cycle(self, db_session, config, circuit_breaker, position_sizer):
        """Test complete cycle: normal → reduced → recovery → normal."""
        balance = 10000.0
        confidence = 0.75
        sl_pips = 15.0
        now = datetime.now(timezone.utc)

        # Step 1: Execute 8 losing trades (reach minimum risk)
        for i in range(8):
            trade = Trade(
                symbol="EURUSD",
                direction="long",
                entry_price=1.0850,
                entry_time=now - timedelta(hours=8-i),
                exit_price=1.0840,
                exit_time=now - timedelta(hours=8-i) + timedelta(minutes=30),
                exit_reason="sl",
                lot_size=0.1,
                status="closed",
                pnl_usd=-30.0,
                is_winner=False
            )
            db_session.add(trade)
            db_session.commit()
            circuit_breaker.record_trade_outcome(db_session, trade.id, is_winner=False)

        # Step 2: Verify risk factor = 0.2 (minimum)
        can_trade, reason, risk_factor = circuit_breaker.can_trade(db_session, balance)
        assert can_trade is True
        assert risk_factor == pytest.approx(0.2, rel=0.01)

        # Verify position size is 20% of normal
        pos_reduced, _, _ = position_sizer.calculate_position_size(
            balance, confidence, sl_pips, config, risk_reduction_factor=risk_factor
        )
        pos_normal, _, _ = position_sizer.calculate_position_size(
            balance, confidence, sl_pips, config, risk_reduction_factor=1.0
        )
        assert pos_reduced == pytest.approx(pos_normal * 0.2, rel=0.01)

        # Step 3: Execute 3 winning trades (recovery)
        for i in range(3):
            trade = Trade(
                symbol="EURUSD",
                direction="long",
                entry_price=1.0850,
                entry_time=now + timedelta(hours=i),
                exit_price=1.0860,
                exit_time=now + timedelta(hours=i) + timedelta(minutes=30),
                exit_reason="tp",
                lot_size=0.1,
                status="closed",
                pnl_usd=100.0,
                is_winner=True
            )
            db_session.add(trade)
            db_session.commit()
            circuit_breaker.record_trade_outcome(db_session, trade.id, is_winner=True)

        # Step 4: Verify risk factor = 0.8 (5 consecutive losses)
        state = db_session.query(RiskReductionState).first()
        assert state.consecutive_losses == 5
        assert state.risk_reduction_factor == pytest.approx(0.8, rel=0.01)

        # Step 5: Execute 1 more winning trade
        trade = Trade(
            symbol="EURUSD",
            direction="long",
            entry_price=1.0850,
            entry_time=now + timedelta(hours=5),
            exit_price=1.0860,
            exit_time=now + timedelta(hours=5) + timedelta(minutes=30),
            exit_reason="tp",
            lot_size=0.1,
            status="closed",
            pnl_usd=100.0,
            is_winner=True
        )
        db_session.add(trade)
        db_session.commit()
        circuit_breaker.record_trade_outcome(db_session, trade.id, is_winner=True)

        # Step 6: Verify risk factor = 1.0 (normal - 4 losses, below threshold)
        state = db_session.query(RiskReductionState).first()
        assert state.consecutive_losses == 4
        assert state.risk_reduction_factor == pytest.approx(1.0, rel=0.01)

    def test_trading_service_integration(self, db_session, config, circuit_breaker, position_sizer):
        """Test TradingService workflow uses risk reduction correctly."""
        balance = 10000.0
        confidence = 0.75
        sl_pips = 15.0
        now = datetime.now(timezone.utc)

        # Step 1: Create 6 consecutive losses
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
            circuit_breaker.record_trade_outcome(db_session, trade.id, is_winner=False)

        # Step 2: Check if trading allowed and get risk factor
        can_trade, reason, risk_factor = circuit_breaker.can_trade(db_session, balance)
        assert can_trade is True
        assert risk_factor == pytest.approx(0.6, rel=0.01)  # 6 losses -> 60%

        # Step 3: Calculate position size with risk reduction
        position_lots, risk_pct, metadata = position_sizer.calculate_position_size(
            balance=balance,
            confidence=confidence,
            sl_pips=sl_pips,
            config=config,
            risk_reduction_factor=risk_factor
        )

        # Step 4: Verify position size is 60% of normal
        pos_normal, _, _ = position_sizer.calculate_position_size(
            balance, confidence, sl_pips, config, risk_reduction_factor=1.0
        )
        assert position_lots == pytest.approx(pos_normal * 0.6, rel=0.01)

        # Step 5: Verify trade metadata includes risk_reduction_factor
        assert metadata["risk_reduction_factor"] == risk_factor

    def test_state_persists_across_restart(self, config, position_sizer):
        """Test risk reduction state survives service restart."""
        # Create first database session (simulates first run)
        engine1 = create_engine("sqlite:///./test_restart.db")
        Base.metadata.create_all(engine1)
        Session1 = sessionmaker(bind=engine1)
        db_session1 = Session1()

        try:
            circuit_breaker1 = TradingCircuitBreaker(config)
            now = datetime.now(timezone.utc)

            # Create 7 consecutive losses (factor = 0.4)
            for i in range(7):
                trade = Trade(
                    symbol="EURUSD",
                    direction="long",
                    entry_price=1.0850,
                    entry_time=now - timedelta(hours=7-i),
                    exit_price=1.0840,
                    exit_time=now - timedelta(hours=7-i) + timedelta(minutes=30),
                    exit_reason="sl",
                    lot_size=0.1,
                    status="closed",
                    pnl_usd=-30.0,
                    is_winner=False
                )
                db_session1.add(trade)
                db_session1.commit()
                circuit_breaker1.record_trade_outcome(db_session1, trade.id, is_winner=False)

            # Verify state
            state = db_session1.query(RiskReductionState).first()
            assert state.consecutive_losses == 7
            assert state.risk_reduction_factor == pytest.approx(0.4, rel=0.01)

            db_session1.close()

            # Create new session (simulates restart)
            Session2 = sessionmaker(bind=engine1)
            db_session2 = Session2()
            circuit_breaker2 = TradingCircuitBreaker(config)

            # Verify factor still 0.4 after restart
            can_trade, reason, risk_factor = circuit_breaker2.can_trade(db_session2, 10000.0)
            assert can_trade is True
            assert risk_factor == pytest.approx(0.4, rel=0.01)

            db_session2.close()

        finally:
            # Cleanup
            import os
            if os.path.exists("./test_restart.db"):
                os.remove("./test_restart.db")

    def test_daily_loss_overrides_progressive_reduction(self, db_session, config, circuit_breaker):
        """Test daily loss blocks even with low risk factor."""
        balance = 10000.0
        now = datetime.now(timezone.utc)

        # Setup: Create state with minimum risk (8 losses)
        state = RiskReductionState(consecutive_losses=8, risk_reduction_factor=0.2)
        db_session.add(state)
        db_session.commit()

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

        # Check trading status
        can_trade, reason, risk_factor = circuit_breaker.can_trade(db_session, balance)

        # Daily loss should override progressive reduction
        assert can_trade is False
        assert "Daily loss limit breached" in reason
        assert risk_factor == 0.0  # Completely blocked

    def test_disabled_progressive_reduction(self, db_session, config_disabled):
        """Test legacy behavior when feature is disabled."""
        circuit_breaker = TradingCircuitBreaker(config_disabled)
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

        # With progressive disabled, should return 0.0 at limit
        can_trade, reason, risk_factor = circuit_breaker.can_trade(db_session, 10000.0)

        # Legacy behavior: returns 0.0 to block trading
        assert risk_factor == 0.0

    def test_concurrent_trades_handling(self, db_session, config, circuit_breaker):
        """Test multiple trades in quick succession update state correctly."""
        now = datetime.now(timezone.utc)

        # Execute 3 losing trades rapidly (within same minute)
        for i in range(3):
            trade = Trade(
                symbol="EURUSD",
                direction="long",
                entry_price=1.0850,
                entry_time=now + timedelta(seconds=i*10),
                exit_price=1.0840,
                exit_time=now + timedelta(seconds=i*10+30),
                exit_reason="sl",
                lot_size=0.1,
                status="closed",
                pnl_usd=-30.0,
                is_winner=False
            )
            db_session.add(trade)
            db_session.commit()
            circuit_breaker.record_trade_outcome(db_session, trade.id, is_winner=False)

        # Verify state updates correctly (no race conditions)
        state = db_session.query(RiskReductionState).first()
        assert state.consecutive_losses == 3
        assert state.risk_reduction_factor == pytest.approx(1.0, rel=0.01)  # Below threshold

    def test_mixed_trade_outcomes(self, db_session, config, circuit_breaker):
        """Test complex sequence of wins and losses."""
        now = datetime.now(timezone.utc)

        # Complex sequence: L L L W L L W L L L L L (net: 8 consecutive at end)
        outcomes = [False, False, False, True, False, False, True, False, False, False, False, False]

        for i, is_winner in enumerate(outcomes):
            trade = Trade(
                symbol="EURUSD",
                direction="long",
                entry_price=1.0850,
                entry_time=now + timedelta(hours=i),
                exit_price=1.0860 if is_winner else 1.0840,
                exit_time=now + timedelta(hours=i) + timedelta(minutes=30),
                exit_reason="tp" if is_winner else "sl",
                lot_size=0.1,
                status="closed",
                pnl_usd=100.0 if is_winner else -30.0,
                is_winner=is_winner
            )
            db_session.add(trade)
            db_session.commit()
            circuit_breaker.record_trade_outcome(db_session, trade.id, is_winner=is_winner)

        # Final state should show 5 consecutive losses (last 5 after second win)
        state = db_session.query(RiskReductionState).first()
        assert state.consecutive_losses == 5
        assert state.risk_reduction_factor == pytest.approx(0.8, rel=0.01)

    def test_position_size_progression_with_losses(self, db_session, config, circuit_breaker, position_sizer):
        """Test position size progressively reduces as losses accumulate."""
        balance = 10000.0
        confidence = 0.75
        sl_pips = 15.0
        now = datetime.now(timezone.utc)

        position_sizes = []

        # Create 8 consecutive losses and track position sizes
        for i in range(8):
            # Check current risk factor
            can_trade, _, risk_factor = circuit_breaker.can_trade(db_session, balance)

            # Calculate position size
            pos, _, _ = position_sizer.calculate_position_size(
                balance, confidence, sl_pips, config, risk_reduction_factor=risk_factor
            )
            position_sizes.append(pos)

            # Record loss
            trade = Trade(
                symbol="EURUSD",
                direction="long",
                entry_price=1.0850,
                entry_time=now + timedelta(hours=i),
                exit_price=1.0840,
                exit_time=now + timedelta(hours=i) + timedelta(minutes=30),
                exit_reason="sl",
                lot_size=0.1,
                status="closed",
                pnl_usd=-30.0,
                is_winner=False
            )
            db_session.add(trade)
            db_session.commit()
            circuit_breaker.record_trade_outcome(db_session, trade.id, is_winner=False)

        # Verify progressive reduction
        # First 5 should be same (below threshold)
        for i in range(4):
            assert position_sizes[i] == pytest.approx(position_sizes[i+1], rel=0.01)

        # After threshold, should progressively reduce
        assert position_sizes[5] < position_sizes[4]  # 5th loss: 0.8x
        assert position_sizes[6] < position_sizes[5]  # 6th loss: 0.6x
        assert position_sizes[7] < position_sizes[6]  # 7th loss: 0.4x

    def test_recovery_progression_with_wins(self, db_session, config, circuit_breaker, position_sizer):
        """Test position size progressively increases as recovery occurs."""
        balance = 10000.0
        confidence = 0.75
        sl_pips = 15.0
        now = datetime.now(timezone.utc)

        # Setup: Start with 8 consecutive losses
        state = RiskReductionState(consecutive_losses=8, risk_reduction_factor=0.2)
        db_session.add(state)
        db_session.commit()

        position_sizes = []

        # Execute 4 winning trades and track position sizes
        for i in range(4):
            # Check current risk factor
            can_trade, _, risk_factor = circuit_breaker.can_trade(db_session, balance)

            # Calculate position size
            pos, _, _ = position_sizer.calculate_position_size(
                balance, confidence, sl_pips, config, risk_reduction_factor=risk_factor
            )
            position_sizes.append(pos)

            # Record win
            trade = Trade(
                symbol="EURUSD",
                direction="long",
                entry_price=1.0850,
                entry_time=now + timedelta(hours=i),
                exit_price=1.0860,
                exit_time=now + timedelta(hours=i) + timedelta(minutes=30),
                exit_reason="tp",
                lot_size=0.1,
                status="closed",
                pnl_usd=100.0,
                is_winner=True
            )
            db_session.add(trade)
            db_session.commit()
            circuit_breaker.record_trade_outcome(db_session, trade.id, is_winner=True)

        # Verify progressive recovery (positions should increase)
        # 8 losses (0.2) -> 7 losses (0.4) -> 6 losses (0.6) -> 5 losses (0.8) -> 4 losses (1.0)
        for i in range(len(position_sizes) - 1):
            assert position_sizes[i+1] > position_sizes[i], \
                f"Position at win {i+1} should be > position at win {i}"


# ============================================================================
# ERROR HANDLING TESTS
# ============================================================================


class TestProgressiveReductionErrorHandling:
    """Test error handling in progressive reduction system."""

    def test_missing_state_initialization(self, db_session, config, circuit_breaker):
        """Test system initializes state if missing."""
        # Don't create state - should auto-initialize
        can_trade, reason, risk_factor = circuit_breaker.can_trade(db_session, 10000.0)

        # Should initialize with normal risk
        assert can_trade is True
        assert risk_factor == pytest.approx(1.0, rel=0.01)

        # State should be created
        state = db_session.query(RiskReductionState).first()
        assert state is not None
        assert state.consecutive_losses == 0
        assert state.risk_reduction_factor == pytest.approx(1.0, rel=0.01)

    def test_corrupted_state_recovery(self, db_session, config, circuit_breaker):
        """Test recovery from corrupted state values."""
        # Create state with invalid values
        state = RiskReductionState(consecutive_losses=-5, risk_reduction_factor=2.0)
        db_session.add(state)
        db_session.commit()

        # System should handle gracefully (return 1.0 on error)
        can_trade, reason, risk_factor = circuit_breaker.can_trade(db_session, 10000.0)

        # Should allow trading (fail-safe)
        assert can_trade is True
        # Risk factor calculation should handle negative losses
        assert risk_factor >= 0.0
