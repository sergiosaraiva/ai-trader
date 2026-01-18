"""Tests for Account Manager."""

import pytest
from datetime import date
import tempfile
import os

from src.trading.account.manager import (
    AccountManager,
    AccountState,
    DailyStats,
)


class TestDailyStats:
    """Tests for DailyStats dataclass."""

    def test_daily_stats_creation(self):
        """Test daily stats creation."""
        stats = DailyStats(
            date=date.today(),
            starting_equity=100000,
            ending_equity=101000,
            high_water_mark=101500,
            low_water_mark=99500,
            realized_pnl=1000,
            unrealized_pnl=0,
            trade_count=10,
            winning_trades=6,
            losing_trades=4,
            gross_profit=2000,
            gross_loss=-1000,
            commission=50,
        )

        assert stats.net_pnl == 950  # 1000 + 0 - 50
        assert stats.daily_return == pytest.approx(1.0)
        assert stats.win_rate == pytest.approx(0.6)
        assert stats.profit_factor == 2.0
        assert stats.max_drawdown == pytest.approx(1.97, rel=0.01)

    def test_daily_stats_to_dict(self):
        """Test daily stats serialization."""
        stats = DailyStats(
            date=date(2024, 1, 15),
            starting_equity=100000,
            ending_equity=101000,
            high_water_mark=101500,
            low_water_mark=99500,
            realized_pnl=1000,
            unrealized_pnl=0,
            trade_count=10,
            winning_trades=6,
            losing_trades=4,
            gross_profit=2000,
            gross_loss=-1000,
            commission=50,
        )

        d = stats.to_dict()
        assert d["date"] == "2024-01-15"
        assert d["starting_equity"] == 100000
        assert d["ending_equity"] == 101000


class TestAccountState:
    """Tests for AccountState dataclass."""

    def test_account_state_creation(self):
        """Test account state creation."""
        from datetime import datetime

        state = AccountState(
            timestamp=datetime.now(),
            balance=100000,
            equity=101000,
            margin_used=5000,
            margin_available=95000,
            unrealized_pnl=1000,
            realized_pnl=500,
        )

        assert state.balance == 100000
        assert state.equity == 101000
        assert state.margin_level == pytest.approx(2020.0)  # 101000/5000*100

    def test_account_state_margin_level_no_margin(self):
        """Test margin level with no margin used."""
        from datetime import datetime

        state = AccountState(
            timestamp=datetime.now(),
            balance=100000,
            equity=100000,
            margin_used=0,
            margin_available=100000,
            unrealized_pnl=0,
            realized_pnl=0,
        )

        assert state.margin_level == float('inf')


class TestAccountManager:
    """Tests for AccountManager."""

    def test_initialization(self):
        """Test account manager initialization."""
        manager = AccountManager(
            initial_balance=100000,
            leverage=10.0,
            margin_requirement=0.01,
        )

        assert manager.balance == 100000
        assert manager.equity == 100000
        assert manager.margin_available == pytest.approx(1000000)  # 100000 * 10

    def test_equity_with_unrealized_pnl(self):
        """Test equity calculation with unrealized PnL."""
        manager = AccountManager(initial_balance=100000)

        manager.update_unrealized_pnl(1500)

        assert manager.balance == 100000  # Balance unchanged
        assert manager.equity == 101500  # Includes unrealized

    def test_margin_available(self):
        """Test available margin calculation."""
        manager = AccountManager(
            initial_balance=100000,
            leverage=10.0,
        )

        manager.update_margin(50000)

        assert manager.margin_used == 50000
        assert manager.margin_available == pytest.approx(950000)

    def test_margin_level(self):
        """Test margin level calculation."""
        manager = AccountManager(initial_balance=100000)

        manager.update_margin(10000)

        assert manager.margin_level == pytest.approx(1000.0)

    def test_record_trade_result_profit(self):
        """Test recording profitable trade."""
        manager = AccountManager(initial_balance=100000)

        manager.record_trade_result(
            realized_pnl=500,
            commission=10,
        )

        assert manager.balance == 100490  # 100000 + 500 - 10
        assert manager._winning_trades_today == 1
        assert manager._gross_profit_today == 500

    def test_record_trade_result_loss(self):
        """Test recording losing trade."""
        manager = AccountManager(initial_balance=100000)

        manager.record_trade_result(
            realized_pnl=-300,
            commission=10,
        )

        assert manager.balance == 99690  # 100000 - 300 - 10
        assert manager._losing_trades_today == 1
        assert manager._gross_loss_today == -300

    def test_daily_pnl(self):
        """Test daily PnL calculation."""
        manager = AccountManager(initial_balance=100000)

        manager.record_trade_result(realized_pnl=500, commission=10)
        manager.record_trade_result(realized_pnl=-200, commission=10)
        manager.update_unrealized_pnl(100)

        assert manager.daily_pnl == pytest.approx(380)  # 500-200+100-20

    def test_total_realized_pnl(self):
        """Test total realized PnL calculation."""
        manager = AccountManager(initial_balance=100000)

        manager.record_trade_result(realized_pnl=500, commission=10)
        manager.record_trade_result(realized_pnl=300, commission=10)

        # Total realized = current balance - initial balance
        assert manager.total_realized_pnl == pytest.approx(780)

    def test_drawdown(self):
        """Test drawdown calculation."""
        manager = AccountManager(initial_balance=100000)

        # Create a high water mark
        manager.update_unrealized_pnl(5000)  # Equity = 105000
        assert manager._high_water_mark == 105000

        # Draw down
        manager.update_unrealized_pnl(-5000)  # Equity = 95000

        # Drawdown = (105000 - 95000) / 105000 * 100
        assert manager.drawdown == pytest.approx(9.52, rel=0.01)

    def test_high_water_mark_tracking(self):
        """Test high water mark updates correctly."""
        manager = AccountManager(initial_balance=100000)

        manager.update_unrealized_pnl(1000)
        assert manager._high_water_mark == 101000

        manager.update_unrealized_pnl(2000)
        assert manager._high_water_mark == 102000

        manager.update_unrealized_pnl(500)
        assert manager._high_water_mark == 102000  # Doesn't decrease

    def test_can_open_position(self):
        """Test position opening check."""
        manager = AccountManager(
            initial_balance=100000,
            leverage=10.0,
            margin_requirement=0.01,
        )

        # Can open position worth 500,000 (needs 5000 margin)
        # Available margin = 100000 * 10 = 1,000,000
        assert manager.can_open_position(500000) is True

        # Use most of margin - leave only 2000 available
        # Available margin = 1,000,000 - 998,000 = 2,000
        manager.update_margin(998000)

        # Can open position worth 200,000 (needs 2000 margin)
        assert manager.can_open_position(200000) is True

        # Cannot open position worth 500,000 (needs 5000 margin)
        assert manager.can_open_position(500000) is False

    def test_get_max_position_value(self):
        """Test maximum position value calculation."""
        manager = AccountManager(
            initial_balance=100000,
            leverage=10.0,
            margin_requirement=0.01,
        )

        # Available margin = 1,000,000
        # Max position = 1,000,000 / 0.01 = 100,000,000
        max_pos = manager.get_max_position_value()
        assert max_pos == pytest.approx(100000000)

    def test_take_snapshot(self):
        """Test account snapshot."""
        manager = AccountManager(initial_balance=100000)

        manager.update_unrealized_pnl(500)
        manager.update_margin(1000)

        snapshot = manager.take_snapshot()

        assert snapshot.balance == 100000
        assert snapshot.equity == 100500
        assert snapshot.margin_used == 1000
        assert len(manager._snapshots) == 1

    def test_get_state(self):
        """Test getting current state without storing."""
        manager = AccountManager(initial_balance=100000)

        state = manager.get_state()

        assert state.balance == 100000
        assert len(manager._snapshots) == 0  # Not stored

    def test_get_stats(self):
        """Test getting comprehensive statistics."""
        manager = AccountManager(initial_balance=100000)

        manager.record_trade_result(realized_pnl=500, commission=10)
        manager.record_trade_result(realized_pnl=-200, commission=10)
        manager.update_unrealized_pnl(300)

        stats = manager.get_stats()

        assert stats["balance"] == pytest.approx(100280)
        assert stats["equity"] == pytest.approx(100580)
        assert stats["daily_trades"] == 2
        assert stats["daily_win_rate"] == 0.5

    def test_save_and_load_state(self):
        """Test state persistence."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = os.path.join(tmpdir, "account_state.json")

            # Create and modify manager
            manager1 = AccountManager(
                initial_balance=100000,
                state_file=state_file,
            )
            manager1.record_trade_result(realized_pnl=1000, commission=10)
            manager1.update_unrealized_pnl(500)
            manager1.save_state()

            # Create new manager and load state
            manager2 = AccountManager(
                initial_balance=100000,
                state_file=state_file,
            )
            success = manager2.load_state()

            assert success is True
            assert manager2.balance == pytest.approx(100990)

    def test_reset(self):
        """Test account reset."""
        manager = AccountManager(initial_balance=100000)

        manager.record_trade_result(realized_pnl=500, commission=10)
        manager.update_unrealized_pnl(300)

        manager.reset()

        assert manager.balance == 100000
        assert manager.equity == 100000
        assert manager._trades_today == 0
        assert manager._high_water_mark == 100000

    def test_reset_daily_counters(self):
        """Test daily counter reset."""
        manager = AccountManager(initial_balance=100000)

        manager.record_trade_result(realized_pnl=500, commission=10)
        manager.record_trade_result(realized_pnl=-200, commission=10)

        manager.reset_daily_counters()

        # Daily counters reset
        assert manager._trades_today == 0
        assert manager._realized_pnl_today == 0
        assert manager._winning_trades_today == 0

        # Balance preserved
        assert manager.balance == pytest.approx(100280)
