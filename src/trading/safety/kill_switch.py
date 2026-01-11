"""
Kill Switch Safety System.

Emergency halt mechanism for production trading with multiple trigger conditions.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Callable
import threading
import logging
import hashlib
import secrets

logger = logging.getLogger(__name__)


class TriggerType(Enum):
    """Types of kill switch triggers."""
    MANUAL = "manual"
    DAILY_LOSS = "daily_loss"
    DAILY_TRADES = "daily_trades"
    POSITION_VALUE = "position_value"
    CONNECTIVITY = "connectivity"
    MODEL_DEGRADATION = "model_degradation"
    EXTERNAL_SIGNAL = "external_signal"
    MAINTENANCE = "maintenance"
    MARKET_CLOSED = "market_closed"


@dataclass
class KillSwitchTrigger:
    """Record of a kill switch trigger event."""
    trigger_type: TriggerType
    reason: str
    timestamp: datetime = field(default_factory=datetime.now)
    value: Optional[float] = None
    threshold: Optional[float] = None
    auto_reset_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "trigger_type": self.trigger_type.value,
            "reason": self.reason,
            "timestamp": self.timestamp.isoformat(),
            "value": self.value,
            "threshold": self.threshold,
            "auto_reset_at": self.auto_reset_at.isoformat() if self.auto_reset_at else None,
        }


@dataclass
class KillSwitchState:
    """Current state of the kill switch."""
    is_active: bool = False
    trigger: Optional[KillSwitchTrigger] = None
    activated_at: Optional[datetime] = None
    trigger_history: List[KillSwitchTrigger] = field(default_factory=list)

    # Counters
    total_triggers: int = 0
    manual_triggers: int = 0
    automatic_triggers: int = 0
    resets: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "is_active": self.is_active,
            "trigger": self.trigger.to_dict() if self.trigger else None,
            "activated_at": self.activated_at.isoformat() if self.activated_at else None,
            "total_triggers": self.total_triggers,
            "manual_triggers": self.manual_triggers,
            "automatic_triggers": self.automatic_triggers,
            "resets": self.resets,
            "trigger_history_count": len(self.trigger_history),
        }


@dataclass
class KillSwitchConfig:
    """Configuration for kill switch."""
    # Daily loss limit
    max_daily_loss_pct: float = 5.0
    max_daily_loss_amount: float = 10000.0

    # Daily trade limits
    max_daily_trades: int = 100
    max_trades_per_hour: int = 20

    # Position limits
    max_total_position_value: float = 1000000.0
    max_single_position_value: float = 100000.0

    # Connectivity
    max_disconnection_seconds: float = 60.0
    connectivity_check_interval: float = 5.0

    # Auto-reset settings
    auto_reset_next_day: bool = True
    auto_reset_after_hours: float = 0.0  # 0 = no auto reset

    # Authorization
    require_authorization_code: bool = True
    authorization_code_length: int = 8

    # Notifications
    notify_on_trigger: bool = True
    notification_callbacks: List[Callable[[KillSwitchTrigger], None]] = field(default_factory=list)


class KillSwitch:
    """
    Kill switch for emergency trading halt.

    Features:
    - Multiple trigger conditions (loss, trades, position value, connectivity)
    - Manual and automatic triggers
    - Authorized reset with confirmation code
    - Auto-reset after specified time
    - Trigger history and audit trail
    - Thread-safe operation

    Usage:
        kill_switch = KillSwitch(config)

        # Check before trading
        if kill_switch.is_active:
            return  # Don't trade

        # Update metrics
        kill_switch.check_daily_loss(current_loss_pct, current_loss_amount)
        kill_switch.check_daily_trades(trade_count)

        # Manual trigger
        kill_switch.trigger(reason="Manual halt for review")

        # Reset with authorization
        code = kill_switch.get_reset_code()
        kill_switch.reset(authorization=code)
    """

    def __init__(self, config: Optional[KillSwitchConfig] = None):
        """
        Initialize kill switch.

        Args:
            config: Kill switch configuration
        """
        self.config = config or KillSwitchConfig()
        self._state = KillSwitchState()
        self._lock = threading.RLock()

        # Authorization
        self._reset_code: Optional[str] = None
        self._reset_code_expires: Optional[datetime] = None

        # Monitoring
        self._last_connectivity_check: datetime = datetime.now()
        self._disconnection_start: Optional[datetime] = None

        # Daily counters
        self._daily_trades = 0
        self._hourly_trades = 0
        self._last_hour_reset: datetime = datetime.now()
        self._last_day_reset: datetime = datetime.now().replace(hour=0, minute=0, second=0)

        logger.info("Kill switch initialized")

    @property
    def is_active(self) -> bool:
        """Check if kill switch is active."""
        with self._lock:
            # Check for auto-reset
            if self._state.is_active and self._state.trigger:
                if self._state.trigger.auto_reset_at:
                    if datetime.now() >= self._state.trigger.auto_reset_at:
                        self._do_reset("Auto-reset after scheduled time")
                        return False

            return self._state.is_active

    @property
    def state(self) -> KillSwitchState:
        """Get current state."""
        with self._lock:
            return self._state

    def trigger(
        self,
        reason: str,
        trigger_type: TriggerType = TriggerType.MANUAL,
        value: Optional[float] = None,
        threshold: Optional[float] = None,
        auto_reset_hours: Optional[float] = None,
    ) -> None:
        """
        Trigger the kill switch.

        Args:
            reason: Reason for trigger
            trigger_type: Type of trigger
            value: Current value that triggered (optional)
            threshold: Threshold that was exceeded (optional)
            auto_reset_hours: Hours until auto-reset (optional)
        """
        with self._lock:
            if self._state.is_active:
                logger.warning("Kill switch already active, ignoring new trigger")
                return

            # Calculate auto-reset time
            auto_reset_at = None
            if auto_reset_hours and auto_reset_hours > 0:
                auto_reset_at = datetime.now() + timedelta(hours=auto_reset_hours)
            elif self.config.auto_reset_after_hours > 0:
                auto_reset_at = datetime.now() + timedelta(hours=self.config.auto_reset_after_hours)

            # Create trigger record
            trigger = KillSwitchTrigger(
                trigger_type=trigger_type,
                reason=reason,
                value=value,
                threshold=threshold,
                auto_reset_at=auto_reset_at,
            )

            # Update state
            self._state.is_active = True
            self._state.trigger = trigger
            self._state.activated_at = datetime.now()
            self._state.trigger_history.append(trigger)
            self._state.total_triggers += 1

            if trigger_type == TriggerType.MANUAL:
                self._state.manual_triggers += 1
            else:
                self._state.automatic_triggers += 1

            logger.critical(
                f"KILL SWITCH TRIGGERED: {trigger_type.value} - {reason}"
            )

            # Send notifications
            if self.config.notify_on_trigger:
                for callback in self.config.notification_callbacks:
                    try:
                        callback(trigger)
                    except Exception as e:
                        logger.error(f"Error in notification callback: {e}")

    def check_daily_loss(
        self,
        loss_pct: float,
        loss_amount: float,
    ) -> bool:
        """
        Check daily loss limits.

        Args:
            loss_pct: Current daily loss percentage (positive = loss)
            loss_amount: Current daily loss amount (positive = loss)

        Returns:
            True if kill switch triggered
        """
        with self._lock:
            if self._state.is_active:
                return True

            triggered = False

            if loss_pct > 0 and loss_pct >= self.config.max_daily_loss_pct:
                self.trigger(
                    reason=f"Daily loss limit exceeded: {loss_pct:.2f}% >= {self.config.max_daily_loss_pct:.2f}%",
                    trigger_type=TriggerType.DAILY_LOSS,
                    value=loss_pct,
                    threshold=self.config.max_daily_loss_pct,
                )
                triggered = True

            if loss_amount > 0 and loss_amount >= self.config.max_daily_loss_amount:
                self.trigger(
                    reason=f"Daily loss amount exceeded: ${loss_amount:.2f} >= ${self.config.max_daily_loss_amount:.2f}",
                    trigger_type=TriggerType.DAILY_LOSS,
                    value=loss_amount,
                    threshold=self.config.max_daily_loss_amount,
                )
                triggered = True

            return triggered

    def check_daily_trades(self, trade_count: int) -> bool:
        """
        Check daily trade limits.

        Args:
            trade_count: Current daily trade count

        Returns:
            True if kill switch triggered
        """
        with self._lock:
            if self._state.is_active:
                return True

            self._daily_trades = trade_count

            if trade_count >= self.config.max_daily_trades:
                self.trigger(
                    reason=f"Daily trade limit exceeded: {trade_count} >= {self.config.max_daily_trades}",
                    trigger_type=TriggerType.DAILY_TRADES,
                    value=float(trade_count),
                    threshold=float(self.config.max_daily_trades),
                )
                return True

            return False

    def check_position_value(
        self,
        total_value: float,
        max_single_value: float = 0.0,
    ) -> bool:
        """
        Check position value limits.

        Args:
            total_value: Total position value
            max_single_value: Maximum value of a single position

        Returns:
            True if kill switch triggered
        """
        with self._lock:
            if self._state.is_active:
                return True

            triggered = False

            if total_value >= self.config.max_total_position_value:
                self.trigger(
                    reason=f"Total position value exceeded: ${total_value:.2f} >= ${self.config.max_total_position_value:.2f}",
                    trigger_type=TriggerType.POSITION_VALUE,
                    value=total_value,
                    threshold=self.config.max_total_position_value,
                )
                triggered = True

            if max_single_value >= self.config.max_single_position_value:
                self.trigger(
                    reason=f"Single position value exceeded: ${max_single_value:.2f} >= ${self.config.max_single_position_value:.2f}",
                    trigger_type=TriggerType.POSITION_VALUE,
                    value=max_single_value,
                    threshold=self.config.max_single_position_value,
                )
                triggered = True

            return triggered

    def check_connectivity(self, is_connected: bool) -> bool:
        """
        Check connectivity status.

        Args:
            is_connected: Whether connected to broker

        Returns:
            True if kill switch triggered due to disconnection
        """
        with self._lock:
            if self._state.is_active:
                return True

            now = datetime.now()
            self._last_connectivity_check = now

            if is_connected:
                self._disconnection_start = None
                return False

            # Track disconnection
            if self._disconnection_start is None:
                self._disconnection_start = now
                logger.warning("Broker disconnection detected")
                return False

            # Check if disconnected too long
            disconnection_duration = (now - self._disconnection_start).total_seconds()

            if disconnection_duration >= self.config.max_disconnection_seconds:
                self.trigger(
                    reason=f"Broker disconnected for {disconnection_duration:.0f}s >= {self.config.max_disconnection_seconds:.0f}s",
                    trigger_type=TriggerType.CONNECTIVITY,
                    value=disconnection_duration,
                    threshold=self.config.max_disconnection_seconds,
                )
                return True

            return False

    def check_all(
        self,
        daily_loss_pct: float = 0.0,
        daily_loss_amount: float = 0.0,
        trade_count: int = 0,
        total_position_value: float = 0.0,
        max_single_position: float = 0.0,
        is_connected: bool = True,
    ) -> bool:
        """
        Check all conditions at once.

        Args:
            daily_loss_pct: Current daily loss percentage
            daily_loss_amount: Current daily loss amount
            trade_count: Daily trade count
            total_position_value: Total position value
            max_single_position: Maximum single position value
            is_connected: Broker connection status

        Returns:
            True if any condition triggered kill switch
        """
        if self._state.is_active:
            return True

        # Check in order of severity
        if self.check_connectivity(is_connected):
            return True

        if self.check_daily_loss(daily_loss_pct, daily_loss_amount):
            return True

        if self.check_position_value(total_position_value, max_single_position):
            return True

        if self.check_daily_trades(trade_count):
            return True

        return False

    def get_reset_code(self) -> str:
        """
        Generate a reset authorization code.

        Returns:
            Authorization code for reset
        """
        with self._lock:
            # Generate random code
            code = secrets.token_hex(self.config.authorization_code_length // 2).upper()

            # Store code with expiry
            self._reset_code = code
            self._reset_code_expires = datetime.now() + timedelta(minutes=5)

            logger.info(f"Reset authorization code generated (expires in 5 minutes)")

            return code

    def reset(self, authorization: str = "", force: bool = False) -> bool:
        """
        Reset the kill switch.

        Args:
            authorization: Authorization code from get_reset_code()
            force: Force reset without authorization (dangerous!)

        Returns:
            True if reset successful
        """
        with self._lock:
            if not self._state.is_active:
                logger.info("Kill switch not active, nothing to reset")
                return True

            # Check authorization
            if self.config.require_authorization_code and not force:
                if not authorization:
                    logger.warning("Reset attempted without authorization code")
                    return False

                if self._reset_code is None:
                    logger.warning("No reset code generated")
                    return False

                if self._reset_code_expires and datetime.now() > self._reset_code_expires:
                    logger.warning("Reset code expired")
                    self._reset_code = None
                    return False

                if authorization.upper() != self._reset_code:
                    logger.warning("Invalid reset authorization code")
                    return False

            self._do_reset("Manual reset with authorization")
            return True

    def _do_reset(self, reason: str) -> None:
        """Internal reset implementation."""
        self._state.is_active = False
        self._state.trigger = None
        self._state.activated_at = None
        self._state.resets += 1

        self._reset_code = None
        self._reset_code_expires = None
        self._disconnection_start = None

        logger.info(f"Kill switch reset: {reason}")

    def reset_daily_counters(self) -> None:
        """Reset daily counters (call at start of trading day)."""
        with self._lock:
            self._daily_trades = 0
            self._last_day_reset = datetime.now()

            # Auto-reset if configured
            if self.config.auto_reset_next_day and self._state.is_active:
                if self._state.trigger and self._state.trigger.trigger_type in (
                    TriggerType.DAILY_LOSS,
                    TriggerType.DAILY_TRADES,
                ):
                    self._do_reset("Auto-reset at start of new trading day")

            logger.info("Daily counters reset")

    def add_notification_callback(
        self,
        callback: Callable[[KillSwitchTrigger], None],
    ) -> None:
        """Add a callback for kill switch triggers."""
        self.config.notification_callbacks.append(callback)

    def get_trigger_history(self, limit: int = 10) -> List[KillSwitchTrigger]:
        """Get recent trigger history."""
        with self._lock:
            return self._state.trigger_history[-limit:]

    def get_stats(self) -> Dict[str, Any]:
        """Get kill switch statistics."""
        with self._lock:
            return {
                **self._state.to_dict(),
                "daily_trades": self._daily_trades,
                "last_connectivity_check": self._last_connectivity_check.isoformat(),
                "disconnection_duration": (
                    (datetime.now() - self._disconnection_start).total_seconds()
                    if self._disconnection_start else 0
                ),
            }
