"""Tests for kill switch safety mechanism."""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from src.trading.safety.kill_switch import (
    KillSwitch,
    KillSwitchConfig,
    KillSwitchState,
    KillSwitchTrigger,
    TriggerType,
)


class TestTriggerType:
    """Tests for TriggerType enum."""

    def test_all_trigger_types(self):
        """Test all trigger types exist."""
        assert TriggerType.MANUAL.value == "manual"
        assert TriggerType.DAILY_LOSS.value == "daily_loss"
        assert TriggerType.DAILY_TRADES.value == "daily_trades"
        assert TriggerType.POSITION_VALUE.value == "position_value"
        assert TriggerType.CONNECTIVITY.value == "connectivity"


class TestKillSwitchTrigger:
    """Tests for KillSwitchTrigger dataclass."""

    def test_trigger_creation(self):
        """Test trigger creation."""
        trigger = KillSwitchTrigger(
            trigger_type=TriggerType.DAILY_LOSS,
            reason="Loss limit exceeded",
            value=10.0,
            threshold=5.0,
        )

        assert trigger.trigger_type == TriggerType.DAILY_LOSS
        assert trigger.reason == "Loss limit exceeded"
        assert trigger.value == 10.0
        assert trigger.threshold == 5.0

    def test_to_dict(self):
        """Test trigger to dictionary conversion."""
        trigger = KillSwitchTrigger(
            trigger_type=TriggerType.DAILY_LOSS,
            reason="Loss limit exceeded",
        )

        d = trigger.to_dict()

        assert d["trigger_type"] == "daily_loss"
        assert d["reason"] == "Loss limit exceeded"
        assert "timestamp" in d


class TestKillSwitchConfig:
    """Tests for KillSwitchConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = KillSwitchConfig()

        assert config.max_daily_loss_pct == 5.0
        assert config.max_daily_loss_amount == 10000.0
        assert config.max_daily_trades == 100
        assert config.require_authorization_code is True

    def test_custom_config(self):
        """Test custom configuration."""
        config = KillSwitchConfig(
            max_daily_loss_pct=3.0,
            max_daily_trades=50,
            require_authorization_code=False,
        )

        assert config.max_daily_loss_pct == 3.0
        assert config.max_daily_trades == 50
        assert config.require_authorization_code is False


class TestKillSwitch:
    """Tests for KillSwitch."""

    @pytest.fixture
    def kill_switch(self):
        """Create kill switch for testing."""
        config = KillSwitchConfig(
            max_daily_loss_pct=5.0,
            max_daily_loss_amount=1000.0,
            max_daily_trades=10,
            max_total_position_value=100000.0,
            max_disconnection_seconds=30.0,
            require_authorization_code=False,
        )
        return KillSwitch(config)

    def test_initial_state(self, kill_switch):
        """Test initial state is inactive."""
        assert kill_switch.is_active is False
        assert kill_switch.state.trigger is None

    def test_manual_trigger(self, kill_switch):
        """Test manual trigger."""
        kill_switch.trigger(
            reason="Manual test trigger",
            trigger_type=TriggerType.MANUAL,
        )

        assert kill_switch.is_active is True
        assert kill_switch.state.trigger.trigger_type == TriggerType.MANUAL
        assert kill_switch.state.manual_triggers == 1

    def test_daily_loss_pct_trigger(self, kill_switch):
        """Test daily loss percentage trigger."""
        triggered = kill_switch.check_daily_loss(loss_pct=6.0, loss_amount=0.0)

        assert triggered is True
        assert kill_switch.is_active is True
        assert kill_switch.state.trigger.trigger_type == TriggerType.DAILY_LOSS

    def test_daily_loss_amount_trigger(self, kill_switch):
        """Test daily loss amount trigger."""
        triggered = kill_switch.check_daily_loss(loss_pct=0.0, loss_amount=1500.0)

        assert triggered is True
        assert kill_switch.is_active is True

    def test_daily_loss_no_trigger(self, kill_switch):
        """Test daily loss check without trigger."""
        triggered = kill_switch.check_daily_loss(loss_pct=2.0, loss_amount=500.0)

        assert triggered is False
        assert kill_switch.is_active is False

    def test_daily_trades_trigger(self, kill_switch):
        """Test daily trades limit trigger."""
        triggered = kill_switch.check_daily_trades(15)

        assert triggered is True
        assert kill_switch.is_active is True
        assert kill_switch.state.trigger.trigger_type == TriggerType.DAILY_TRADES

    def test_daily_trades_no_trigger(self, kill_switch):
        """Test daily trades check without trigger."""
        triggered = kill_switch.check_daily_trades(5)

        assert triggered is False
        assert kill_switch.is_active is False

    def test_position_value_trigger(self, kill_switch):
        """Test position value trigger."""
        triggered = kill_switch.check_position_value(150000.0)

        assert triggered is True
        assert kill_switch.is_active is True
        assert kill_switch.state.trigger.trigger_type == TriggerType.POSITION_VALUE

    def test_position_value_no_trigger(self, kill_switch):
        """Test position value check without trigger."""
        triggered = kill_switch.check_position_value(50000.0)

        assert triggered is False
        assert kill_switch.is_active is False

    def test_connectivity_trigger(self, kill_switch):
        """Test connectivity loss trigger."""
        # First disconnection - starts timer
        triggered = kill_switch.check_connectivity(False)
        assert triggered is False

        # Simulate time passing
        with patch.object(kill_switch, '_disconnection_start', datetime.now() - timedelta(seconds=60)):
            triggered = kill_switch.check_connectivity(False)

        assert triggered is True
        assert kill_switch.is_active is True
        assert kill_switch.state.trigger.trigger_type == TriggerType.CONNECTIVITY

    def test_connectivity_recovery(self, kill_switch):
        """Test connectivity recovery resets timer."""
        kill_switch.check_connectivity(False)  # Start disconnect timer
        triggered = kill_switch.check_connectivity(True)  # Reconnect

        assert triggered is False
        assert kill_switch._disconnection_start is None

    def test_check_all(self, kill_switch):
        """Test check_all method."""
        # No triggers
        triggered = kill_switch.check_all(
            daily_loss_pct=2.0,
            daily_loss_amount=500.0,
            trade_count=5,
            total_position_value=50000.0,
            is_connected=True,
        )

        assert triggered is False
        assert kill_switch.is_active is False

    def test_check_all_triggers(self, kill_switch):
        """Test check_all with triggering values."""
        triggered = kill_switch.check_all(
            daily_loss_pct=10.0,  # Exceeds limit
            daily_loss_amount=500.0,
            trade_count=5,
            total_position_value=50000.0,
            is_connected=True,
        )

        assert triggered is True
        assert kill_switch.is_active is True

    def test_reset_without_authorization(self, kill_switch):
        """Test reset without requiring authorization."""
        kill_switch.trigger("Test")

        success = kill_switch.reset()

        assert success is True
        assert kill_switch.is_active is False
        assert kill_switch.state.resets == 1

    def test_reset_with_authorization(self):
        """Test reset with authorization code."""
        config = KillSwitchConfig(require_authorization_code=True)
        ks = KillSwitch(config)

        ks.trigger("Test")
        code = ks.get_reset_code()

        # Wrong code
        success = ks.reset(authorization="WRONGCODE")
        assert success is False
        assert ks.is_active is True

        # Correct code
        success = ks.reset(authorization=code)
        assert success is True
        assert ks.is_active is False

    def test_reset_code_expiry(self):
        """Test reset code expiration."""
        config = KillSwitchConfig(require_authorization_code=True)
        ks = KillSwitch(config)

        ks.trigger("Test")
        code = ks.get_reset_code()

        # Simulate code expiry
        ks._reset_code_expires = datetime.now() - timedelta(minutes=10)

        success = ks.reset(authorization=code)
        assert success is False
        assert ks.is_active is True

    def test_force_reset(self):
        """Test force reset without authorization."""
        config = KillSwitchConfig(require_authorization_code=True)
        ks = KillSwitch(config)

        ks.trigger("Test")
        success = ks.reset(force=True)

        assert success is True
        assert ks.is_active is False

    def test_already_active_trigger_ignored(self, kill_switch):
        """Test that new triggers are ignored when already active."""
        kill_switch.trigger("First trigger")

        # This should be ignored
        kill_switch.trigger("Second trigger")

        assert kill_switch.state.total_triggers == 1

    def test_trigger_history(self, kill_switch):
        """Test trigger history."""
        kill_switch.trigger("Trigger 1")
        kill_switch.reset()
        kill_switch.trigger("Trigger 2")
        kill_switch.reset()
        kill_switch.trigger("Trigger 3")

        history = kill_switch.get_trigger_history(limit=10)

        assert len(history) == 3
        assert history[0].reason == "Trigger 1"
        assert history[2].reason == "Trigger 3"

    def test_notification_callback(self):
        """Test notification callback on trigger."""
        callback = Mock()
        config = KillSwitchConfig(notification_callbacks=[callback])
        ks = KillSwitch(config)

        ks.trigger("Test trigger")

        callback.assert_called_once()
        trigger_arg = callback.call_args[0][0]
        assert trigger_arg.reason == "Test trigger"

    def test_reset_daily_counters(self, kill_switch):
        """Test daily counter reset."""
        kill_switch._daily_trades = 50
        kill_switch.reset_daily_counters()

        assert kill_switch._daily_trades == 0

    def test_auto_reset_at_scheduled_time(self, kill_switch):
        """Test auto-reset at scheduled time."""
        kill_switch.trigger(
            reason="Test",
            auto_reset_hours=0.0,  # Immediate reset
        )

        # Set auto-reset time in the past
        kill_switch._state.trigger.auto_reset_at = datetime.now() - timedelta(hours=1)

        # is_active property should trigger auto-reset
        assert kill_switch.is_active is False

    def test_get_stats(self, kill_switch):
        """Test get_stats method."""
        kill_switch.trigger("Test")
        stats = kill_switch.get_stats()

        assert stats["is_active"] is True
        assert stats["total_triggers"] == 1
        assert "daily_trades" in stats
        assert "last_connectivity_check" in stats
