"""Position tracking module for the trading agent.

Tracks open positions and checks for exit conditions based on
triple barrier method (TP, SL, timeout).
"""

import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta

from .broker_manager import BrokerManager
from .models import ExitSignal

logger = logging.getLogger(__name__)


class PositionTracker:
    """Tracks open positions and checks for exit conditions.

    Uses triple barrier method:
    1. Take Profit: Price reaches TP level
    2. Stop Loss: Price reaches SL level
    3. Timeout: Maximum holding period exceeded
    """

    def __init__(
        self,
        broker_manager: BrokerManager,
        db_session_factory,
    ):
        """Initialize position tracker.

        Args:
            broker_manager: Broker connection manager
            db_session_factory: Factory to create database sessions
        """
        self.broker_manager = broker_manager
        self.db_session_factory = db_session_factory

        # Track positions: trade_id -> position_info
        self._tracked_positions: Dict[int, Dict] = {}

        logger.info("PositionTracker initialized")

    def track_position(
        self,
        trade_id: int,
        entry_price: float,
        direction: str,
        tp: float,
        sl: float,
        max_bars: int,
    ) -> None:
        """Start tracking a position.

        Args:
            trade_id: Trade ID from database
            entry_price: Entry price
            direction: "long" or "short"
            tp: Take profit price
            sl: Stop loss price
            max_bars: Maximum bars to hold position
        """
        self._tracked_positions[trade_id] = {
            "trade_id": trade_id,
            "entry_price": entry_price,
            "direction": direction,
            "take_profit": tp,
            "stop_loss": sl,
            "max_bars": max_bars,
            "entry_time": datetime.now(),
            "bars_elapsed": 0,
        }

        logger.info(
            f"Tracking position {trade_id}: "
            f"{direction} @ {entry_price:.5f}, "
            f"TP={tp:.5f}, SL={sl:.5f}, max_bars={max_bars}"
        )

    async def check_exits(self) -> List[ExitSignal]:
        """Check all tracked positions for exit conditions.

        Returns:
            List of exit signals for positions that should be closed
        """
        exit_signals = []

        if not self.broker_manager.is_connected():
            logger.warning("Cannot check exits - broker not connected")
            return exit_signals

        try:
            # Get current positions from broker
            broker_positions = await self.broker_manager.get_open_positions()

            # Check each tracked position
            for trade_id, position_info in list(self._tracked_positions.items()):
                # Find matching broker position
                symbol = position_info.get("symbol", "EURUSD")
                broker_position = next(
                    (p for p in broker_positions if p.get("symbol") == symbol),
                    None
                )

                if not broker_position:
                    # Position already closed
                    logger.info(f"Position {trade_id} already closed in broker")
                    del self._tracked_positions[trade_id]
                    continue

                # Get current price
                current_price = broker_position["current_price"]

                # Check exit conditions
                exit_signal = self._check_exit_conditions(
                    trade_id=trade_id,
                    position_info=position_info,
                    current_price=current_price,
                )

                if exit_signal:
                    exit_signals.append(exit_signal)
                    # Remove from tracking
                    del self._tracked_positions[trade_id]

        except Exception as e:
            logger.error(f"Error checking exits: {e}")

        return exit_signals

    def _check_exit_conditions(
        self,
        trade_id: int,
        position_info: Dict,
        current_price: float,
    ) -> Optional[ExitSignal]:
        """Check if position should be exited.

        Args:
            trade_id: Trade ID
            position_info: Position information dictionary
            current_price: Current market price

        Returns:
            ExitSignal if position should be closed, None otherwise
        """
        direction = position_info["direction"]
        entry_price = position_info["entry_price"]
        tp_price = position_info.get("take_profit")
        sl_price = position_info.get("stop_loss")

        # Check Take Profit
        if tp_price:
            if direction == "long" and current_price >= tp_price:
                logger.info(
                    f"Position {trade_id} hit take profit: "
                    f"{current_price:.5f} >= {tp_price:.5f}"
                )
                return ExitSignal(
                    trade_id=trade_id,
                    reason="take_profit",
                    exit_price=current_price,
                )

            if direction == "short" and current_price <= tp_price:
                logger.info(
                    f"Position {trade_id} hit take profit: "
                    f"{current_price:.5f} <= {tp_price:.5f}"
                )
                return ExitSignal(
                    trade_id=trade_id,
                    reason="take_profit",
                    exit_price=current_price,
                )

        # Check Stop Loss
        if sl_price:
            if direction == "long" and current_price <= sl_price:
                logger.info(
                    f"Position {trade_id} hit stop loss: "
                    f"{current_price:.5f} <= {sl_price:.5f}"
                )
                return ExitSignal(
                    trade_id=trade_id,
                    reason="stop_loss",
                    exit_price=current_price,
                )

            if direction == "short" and current_price >= sl_price:
                logger.info(
                    f"Position {trade_id} hit stop loss: "
                    f"{current_price:.5f} >= {sl_price:.5f}"
                )
                return ExitSignal(
                    trade_id=trade_id,
                    reason="stop_loss",
                    exit_price=current_price,
                )

        # Check Timeout
        max_bars = position_info.get("max_bars", 24)
        entry_time = position_info["entry_time"]
        time_elapsed = datetime.now() - entry_time

        # Assuming 1H bars
        bars_elapsed = time_elapsed.total_seconds() / 3600

        if bars_elapsed >= max_bars:
            logger.info(
                f"Position {trade_id} exceeded max holding period: "
                f"{bars_elapsed:.1f} bars >= {max_bars} bars"
            )
            return ExitSignal(
                trade_id=trade_id,
                reason="timeout",
                exit_price=current_price,
            )

        return None

    def stop_tracking(self, trade_id: int) -> None:
        """Stop tracking a position.

        Args:
            trade_id: Trade ID to stop tracking
        """
        if trade_id in self._tracked_positions:
            del self._tracked_positions[trade_id]
            logger.debug(f"Stopped tracking position {trade_id}")

    def get_tracked_positions(self) -> List[Dict]:
        """Get all tracked positions.

        Returns:
            List of position information dictionaries
        """
        return list(self._tracked_positions.values())

    def get_tracked_count(self) -> int:
        """Get number of tracked positions.

        Returns:
            Number of positions being tracked
        """
        return len(self._tracked_positions)

    def clear(self) -> None:
        """Clear all tracked positions."""
        self._tracked_positions.clear()
        logger.info("Position tracker cleared")
