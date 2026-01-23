"""Unit tests for PositionTracker.

Tests triple barrier exit conditions: take profit, stop loss, and timeout.
Tests that symbol is correctly tracked (not hardcoded).
"""

import pytest
from unittest.mock import Mock, AsyncMock
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent.parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# Import from conftest
from .conftest import AgentConfig

from agent.position_tracker import PositionTracker
from agent.broker_manager import BrokerManager
from agent.models import ExitSignal


@pytest.fixture
def mock_broker_manager():
    """Create mock broker manager."""
    manager = Mock(spec=BrokerManager)
    manager.is_connected = Mock(return_value=True)
    manager.get_open_positions = AsyncMock(return_value=[])
    return manager


@pytest.fixture
def mock_db_session_factory():
    """Create mock database session factory."""
    return Mock()


class TestPositionTrackerTracking:
    """Test position tracking functionality."""

    def test_track_position(self, mock_broker_manager, mock_db_session_factory):
        """Test starting to track a position."""
        tracker = PositionTracker(mock_broker_manager, mock_db_session_factory)

        tracker.track_position(
            trade_id=1,
            entry_price=1.08500,
            direction="long",
            tp=1.09000,
            sl=1.08000,
            max_bars=24,
        )

        assert tracker.get_tracked_count() == 1

        positions = tracker.get_tracked_positions()
        assert len(positions) == 1
        assert positions[0]["trade_id"] == 1
        assert positions[0]["entry_price"] == 1.08500
        assert positions[0]["direction"] == "long"
        assert positions[0]["take_profit"] == 1.09000
        assert positions[0]["stop_loss"] == 1.08000
        assert positions[0]["max_bars"] == 24

    def test_track_multiple_positions(self, mock_broker_manager, mock_db_session_factory):
        """Test tracking multiple positions."""
        tracker = PositionTracker(mock_broker_manager, mock_db_session_factory)

        tracker.track_position(1, 1.08500, "long", 1.09000, 1.08000, 24)
        tracker.track_position(2, 1.10000, "short", 1.09500, 1.10500, 12)

        assert tracker.get_tracked_count() == 2

    def test_stop_tracking(self, mock_broker_manager, mock_db_session_factory):
        """Test stopping tracking a position."""
        tracker = PositionTracker(mock_broker_manager, mock_db_session_factory)

        tracker.track_position(1, 1.08500, "long", 1.09000, 1.08000, 24)
        assert tracker.get_tracked_count() == 1

        tracker.stop_tracking(1)
        assert tracker.get_tracked_count() == 0

    def test_stop_tracking_nonexistent(self, mock_broker_manager, mock_db_session_factory):
        """Test stopping tracking of nonexistent position (no error)."""
        tracker = PositionTracker(mock_broker_manager, mock_db_session_factory)

        # Should not raise
        tracker.stop_tracking(999)

    def test_clear_all_positions(self, mock_broker_manager, mock_db_session_factory):
        """Test clearing all tracked positions."""
        tracker = PositionTracker(mock_broker_manager, mock_db_session_factory)

        tracker.track_position(1, 1.08500, "long", 1.09000, 1.08000, 24)
        tracker.track_position(2, 1.10000, "short", 1.09500, 1.10500, 12)

        assert tracker.get_tracked_count() == 2

        tracker.clear()

        assert tracker.get_tracked_count() == 0


class TestPositionTrackerExitConditions:
    """Test triple barrier exit condition detection."""

    @pytest.mark.asyncio
    async def test_check_exits_long_take_profit(
        self, mock_broker_manager, mock_db_session_factory
    ):
        """Test exit detection when long position hits take profit."""
        tracker = PositionTracker(mock_broker_manager, mock_db_session_factory)

        tracker.track_position(
            trade_id=1,
            entry_price=1.08500,
            direction="long",
            tp=1.09000,
            sl=1.08000,
            max_bars=24,
        )

        # Add symbol to position info
        tracker._tracked_positions[1]["symbol"] = "EURUSD"

        # Mock broker position at take profit
        mock_broker_manager.get_open_positions = AsyncMock(return_value=[{
            "symbol": "EURUSD",
            "current_price": 1.09100,  # Above TP
        }])

        exit_signals = await tracker.check_exits()

        assert len(exit_signals) == 1
        assert exit_signals[0].trade_id == 1
        assert exit_signals[0].reason == "take_profit"
        assert exit_signals[0].exit_price == 1.09100

        # Position should be removed from tracking
        assert tracker.get_tracked_count() == 0

    @pytest.mark.asyncio
    async def test_check_exits_long_stop_loss(
        self, mock_broker_manager, mock_db_session_factory
    ):
        """Test exit detection when long position hits stop loss."""
        tracker = PositionTracker(mock_broker_manager, mock_db_session_factory)

        tracker.track_position(
            trade_id=1,
            entry_price=1.08500,
            direction="long",
            tp=1.09000,
            sl=1.08000,
            max_bars=24,
        )

        tracker._tracked_positions[1]["symbol"] = "EURUSD"

        # Mock broker position at stop loss
        mock_broker_manager.get_open_positions = AsyncMock(return_value=[{
            "symbol": "EURUSD",
            "current_price": 1.07900,  # Below SL
        }])

        exit_signals = await tracker.check_exits()

        assert len(exit_signals) == 1
        assert exit_signals[0].trade_id == 1
        assert exit_signals[0].reason == "stop_loss"
        assert exit_signals[0].exit_price == 1.07900

    @pytest.mark.asyncio
    async def test_check_exits_short_take_profit(
        self, mock_broker_manager, mock_db_session_factory
    ):
        """Test exit detection when short position hits take profit."""
        tracker = PositionTracker(mock_broker_manager, mock_db_session_factory)

        tracker.track_position(
            trade_id=1,
            entry_price=1.10000,
            direction="short",
            tp=1.09500,  # Take profit below entry for short
            sl=1.10500,  # Stop loss above entry for short
            max_bars=24,
        )

        tracker._tracked_positions[1]["symbol"] = "EURUSD"

        # Mock broker position at take profit
        mock_broker_manager.get_open_positions = AsyncMock(return_value=[{
            "symbol": "EURUSD",
            "current_price": 1.09400,  # Below TP for short = profit
        }])

        exit_signals = await tracker.check_exits()

        assert len(exit_signals) == 1
        assert exit_signals[0].trade_id == 1
        assert exit_signals[0].reason == "take_profit"
        assert exit_signals[0].exit_price == 1.09400

    @pytest.mark.asyncio
    async def test_check_exits_short_stop_loss(
        self, mock_broker_manager, mock_db_session_factory
    ):
        """Test exit detection when short position hits stop loss."""
        tracker = PositionTracker(mock_broker_manager, mock_db_session_factory)

        tracker.track_position(
            trade_id=1,
            entry_price=1.10000,
            direction="short",
            tp=1.09500,
            sl=1.10500,
            max_bars=24,
        )

        tracker._tracked_positions[1]["symbol"] = "EURUSD"

        # Mock broker position at stop loss
        mock_broker_manager.get_open_positions = AsyncMock(return_value=[{
            "symbol": "EURUSD",
            "current_price": 1.10600,  # Above SL for short = loss
        }])

        exit_signals = await tracker.check_exits()

        assert len(exit_signals) == 1
        assert exit_signals[0].trade_id == 1
        assert exit_signals[0].reason == "stop_loss"
        assert exit_signals[0].exit_price == 1.10600

    @pytest.mark.asyncio
    async def test_check_exits_timeout(
        self, mock_broker_manager, mock_db_session_factory
    ):
        """Test exit detection when max holding period exceeded."""
        tracker = PositionTracker(mock_broker_manager, mock_db_session_factory)

        tracker.track_position(
            trade_id=1,
            entry_price=1.08500,
            direction="long",
            tp=1.09000,
            sl=1.08000,
            max_bars=2,  # 2 hours
        )

        # Set entry time to 3 hours ago
        tracker._tracked_positions[1]["entry_time"] = datetime.now() - timedelta(hours=3)
        tracker._tracked_positions[1]["symbol"] = "EURUSD"

        # Mock broker position (price within bounds)
        mock_broker_manager.get_open_positions = AsyncMock(return_value=[{
            "symbol": "EURUSD",
            "current_price": 1.08700,  # Between SL and TP
        }])

        exit_signals = await tracker.check_exits()

        assert len(exit_signals) == 1
        assert exit_signals[0].trade_id == 1
        assert exit_signals[0].reason == "timeout"
        assert exit_signals[0].exit_price == 1.08700

    @pytest.mark.asyncio
    async def test_check_exits_no_exit_needed(
        self, mock_broker_manager, mock_db_session_factory
    ):
        """Test no exit when position is within bounds."""
        tracker = PositionTracker(mock_broker_manager, mock_db_session_factory)

        tracker.track_position(
            trade_id=1,
            entry_price=1.08500,
            direction="long",
            tp=1.09000,
            sl=1.08000,
            max_bars=24,
        )

        tracker._tracked_positions[1]["symbol"] = "EURUSD"

        # Mock broker position (price within bounds)
        mock_broker_manager.get_open_positions = AsyncMock(return_value=[{
            "symbol": "EURUSD",
            "current_price": 1.08700,  # Between SL and TP
        }])

        exit_signals = await tracker.check_exits()

        assert len(exit_signals) == 0
        # Position should still be tracked
        assert tracker.get_tracked_count() == 1

    @pytest.mark.asyncio
    async def test_check_exits_position_already_closed(
        self, mock_broker_manager, mock_db_session_factory
    ):
        """Test handling when position is already closed in broker."""
        tracker = PositionTracker(mock_broker_manager, mock_db_session_factory)

        tracker.track_position(
            trade_id=1,
            entry_price=1.08500,
            direction="long",
            tp=1.09000,
            sl=1.08000,
            max_bars=24,
        )

        tracker._tracked_positions[1]["symbol"] = "EURUSD"

        # No positions in broker
        mock_broker_manager.get_open_positions = AsyncMock(return_value=[])

        exit_signals = await tracker.check_exits()

        # No exit signal (already closed)
        assert len(exit_signals) == 0

        # Position should be removed from tracking
        assert tracker.get_tracked_count() == 0

    @pytest.mark.asyncio
    async def test_check_exits_broker_not_connected(
        self, mock_broker_manager, mock_db_session_factory
    ):
        """Test check exits when broker not connected."""
        mock_broker_manager.is_connected = Mock(return_value=False)

        tracker = PositionTracker(mock_broker_manager, mock_db_session_factory)

        tracker.track_position(1, 1.08500, "long", 1.09000, 1.08000, 24)

        exit_signals = await tracker.check_exits()

        assert exit_signals == []

    @pytest.mark.asyncio
    async def test_check_exits_handles_errors(
        self, mock_broker_manager, mock_db_session_factory
    ):
        """Test check exits handles errors gracefully."""
        tracker = PositionTracker(mock_broker_manager, mock_db_session_factory)

        tracker.track_position(1, 1.08500, "long", 1.09000, 1.08000, 24)

        # Mock error
        mock_broker_manager.get_open_positions = AsyncMock(
            side_effect=Exception("Network error")
        )

        # Should not raise
        exit_signals = await tracker.check_exits()

        assert exit_signals == []


class TestPositionTrackerSymbolHandling:
    """Test that symbol is correctly tracked (not hardcoded)."""

    @pytest.mark.asyncio
    async def test_symbol_stored_correctly(
        self, mock_broker_manager, mock_db_session_factory
    ):
        """Test that symbol is stored from position info (not hardcoded)."""
        tracker = PositionTracker(mock_broker_manager, mock_db_session_factory)

        tracker.track_position(
            trade_id=1,
            entry_price=1.30000,
            direction="long",
            tp=1.31000,
            sl=1.29000,
            max_bars=24,
        )

        # Set symbol to non-default
        tracker._tracked_positions[1]["symbol"] = "GBPUSD"

        # Mock broker with GBPUSD position
        mock_broker_manager.get_open_positions = AsyncMock(return_value=[{
            "symbol": "GBPUSD",
            "current_price": 1.31100,
        }])

        exit_signals = await tracker.check_exits()

        assert len(exit_signals) == 1
        # Verify it matched the correct symbol
        assert exit_signals[0].reason == "take_profit"

    @pytest.mark.asyncio
    async def test_symbol_defaults_to_eurusd(
        self, mock_broker_manager, mock_db_session_factory
    ):
        """Test that symbol defaults to EURUSD if not set."""
        tracker = PositionTracker(mock_broker_manager, mock_db_session_factory)

        tracker.track_position(
            trade_id=1,
            entry_price=1.08500,
            direction="long",
            tp=1.09000,
            sl=1.08000,
            max_bars=24,
        )

        # Don't set symbol - should default to EURUSD

        # Mock broker with EURUSD position
        mock_broker_manager.get_open_positions = AsyncMock(return_value=[{
            "symbol": "EURUSD",
            "current_price": 1.09100,
        }])

        exit_signals = await tracker.check_exits()

        assert len(exit_signals) == 1


class TestPositionTrackerMultiplePositions:
    """Test tracking multiple positions simultaneously."""

    @pytest.mark.asyncio
    async def test_check_exits_multiple_positions(
        self, mock_broker_manager, mock_db_session_factory
    ):
        """Test checking multiple positions with different exit conditions."""
        tracker = PositionTracker(mock_broker_manager, mock_db_session_factory)

        # Position 1: Will hit TP
        tracker.track_position(
            trade_id=1,
            entry_price=1.08500,
            direction="long",
            tp=1.09000,
            sl=1.08000,
            max_bars=24,
        )
        tracker._tracked_positions[1]["symbol"] = "EURUSD"

        # Position 2: Will hit SL
        tracker.track_position(
            trade_id=2,
            entry_price=1.30000,
            direction="short",
            tp=1.29500,
            sl=1.30500,
            max_bars=24,
        )
        tracker._tracked_positions[2]["symbol"] = "GBPUSD"

        # Position 3: Still open
        tracker.track_position(
            trade_id=3,
            entry_price=1.05000,
            direction="long",
            tp=1.06000,
            sl=1.04000,
            max_bars=24,
        )
        tracker._tracked_positions[3]["symbol"] = "USDCHF"

        # Mock broker positions
        mock_broker_manager.get_open_positions = AsyncMock(return_value=[
            {"symbol": "EURUSD", "current_price": 1.09100},  # Hit TP
            {"symbol": "GBPUSD", "current_price": 1.30600},  # Hit SL
            {"symbol": "USDCHF", "current_price": 1.05200},  # Still open
        ])

        exit_signals = await tracker.check_exits()

        # Should have 2 exit signals
        assert len(exit_signals) == 2

        # Check reasons
        reasons = [sig.reason for sig in exit_signals]
        assert "take_profit" in reasons
        assert "stop_loss" in reasons

        # Position 3 should still be tracked
        assert tracker.get_tracked_count() == 1
        remaining = tracker.get_tracked_positions()
        assert remaining[0]["trade_id"] == 3

    @pytest.mark.asyncio
    async def test_check_exits_removes_only_exited_positions(
        self, mock_broker_manager, mock_db_session_factory
    ):
        """Test that only exited positions are removed from tracking."""
        tracker = PositionTracker(mock_broker_manager, mock_db_session_factory)

        # Track 3 positions
        tracker.track_position(1, 1.08500, "long", 1.09000, 1.08000, 24)
        tracker._tracked_positions[1]["symbol"] = "EURUSD"

        tracker.track_position(2, 1.30000, "short", 1.29500, 1.30500, 24)
        tracker._tracked_positions[2]["symbol"] = "GBPUSD"

        tracker.track_position(3, 1.05000, "long", 1.06000, 1.04000, 24)
        tracker._tracked_positions[3]["symbol"] = "USDCHF"

        # Mock: Position 1 hit TP, position 2 still open, position 3 already closed
        mock_broker_manager.get_open_positions = AsyncMock(return_value=[
            {"symbol": "EURUSD", "current_price": 1.09100},  # Hit TP
            {"symbol": "GBPUSD", "current_price": 1.29700},  # Still open
            # USDCHF not in broker (already closed)
        ])

        exit_signals = await tracker.check_exits()

        # Should have 1 exit signal (position 1)
        assert len(exit_signals) == 1
        assert exit_signals[0].trade_id == 1

        # Position 2 should still be tracked, 1 and 3 removed
        assert tracker.get_tracked_count() == 1
        remaining = tracker.get_tracked_positions()
        assert remaining[0]["trade_id"] == 2
