"""Tests for Position Manager."""

import pytest
from datetime import datetime

from src.trading.positions.manager import (
    PositionManager,
    Position,
    PositionSide,
    PositionStatus,
)
from src.trading.orders.manager import Order, OrderSide, OrderType, OrderStatus


class TestPosition:
    """Tests for Position dataclass."""

    def test_position_creation(self):
        """Test basic position creation."""
        position = Position(
            position_id="pos-001",
            symbol="EURUSD",
            side=PositionSide.LONG,
            quantity=1.0,
            average_entry_price=1.1000,
            current_price=1.1000,
            total_cost=1.1000,
        )

        assert position.position_id == "pos-001"
        assert position.symbol == "EURUSD"
        assert position.side == PositionSide.LONG
        assert position.quantity == 1.0
        assert position.status == PositionStatus.OPEN

    def test_position_is_long(self):
        """Test is_long property."""
        position = Position(
            position_id="pos-001",
            symbol="EURUSD",
            side=PositionSide.LONG,
            quantity=1.0,
            average_entry_price=1.1000,
            current_price=1.1000,
        )

        assert position.is_long is True
        assert position.is_short is False

    def test_position_is_short(self):
        """Test is_short property."""
        position = Position(
            position_id="pos-001",
            symbol="EURUSD",
            side=PositionSide.SHORT,
            quantity=1.0,
            average_entry_price=1.1000,
            current_price=1.1000,
        )

        assert position.is_long is False
        assert position.is_short is True

    def test_position_market_value(self):
        """Test market value calculation."""
        position = Position(
            position_id="pos-001",
            symbol="EURUSD",
            side=PositionSide.LONG,
            quantity=2.0,
            average_entry_price=1.1000,
            current_price=1.1050,
        )

        assert position.market_value == pytest.approx(2.21)

    def test_position_update_price_long(self):
        """Test price update for long position."""
        position = Position(
            position_id="pos-001",
            symbol="EURUSD",
            side=PositionSide.LONG,
            quantity=1.0,
            average_entry_price=1.1000,
            current_price=1.1000,
        )

        # Price goes up
        position.update_price(1.1100)

        assert position.current_price == 1.1100
        assert position.unrealized_pnl == pytest.approx(0.01)

    def test_position_update_price_short(self):
        """Test price update for short position."""
        position = Position(
            position_id="pos-001",
            symbol="EURUSD",
            side=PositionSide.SHORT,
            quantity=1.0,
            average_entry_price=1.1000,
            current_price=1.1000,
        )

        # Price goes down (profit for short)
        position.update_price(1.0900)

        assert position.current_price == 1.0900
        assert position.unrealized_pnl == pytest.approx(0.01)

    def test_position_pnl_percentage(self):
        """Test PnL percentage calculation."""
        position = Position(
            position_id="pos-001",
            symbol="EURUSD",
            side=PositionSide.LONG,
            quantity=1.0,
            average_entry_price=1.1000,
            current_price=1.1000,
            total_cost=1.1000,
            unrealized_pnl=0.0110,
        )

        assert position.pnl_percentage == pytest.approx(1.0)

    def test_position_to_dict(self):
        """Test position serialization."""
        position = Position(
            position_id="pos-001",
            symbol="EURUSD",
            side=PositionSide.LONG,
            quantity=1.0,
            average_entry_price=1.1000,
            current_price=1.1000,
        )

        d = position.to_dict()
        assert d["position_id"] == "pos-001"
        assert d["symbol"] == "EURUSD"
        assert d["side"] == "long"


class TestPositionManager:
    """Tests for PositionManager."""

    def test_initialization(self):
        """Test position manager initialization."""
        manager = PositionManager()

        assert len(manager.positions) == 0
        assert len(manager.closed_positions) == 0

    def test_process_fill_open_long(self):
        """Test opening a long position from fill."""
        manager = PositionManager()

        order = Order(
            order_id="ord-001",
            symbol="EURUSD",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=1.0,
            status=OrderStatus.FILLED,
            filled_quantity=1.0,
            average_fill_price=1.1000,
            commission=2.0,
        )

        position = manager.process_fill(order)

        assert position is not None
        assert position.side == PositionSide.LONG
        assert position.quantity == 1.0
        assert position.average_entry_price == 1.1000
        assert position.total_commission == 2.0
        assert "EURUSD" in manager.positions

    def test_process_fill_open_short(self):
        """Test opening a short position from fill."""
        manager = PositionManager()

        order = Order(
            order_id="ord-001",
            symbol="EURUSD",
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=1.0,
            status=OrderStatus.FILLED,
            filled_quantity=1.0,
            average_fill_price=1.1000,
        )

        position = manager.process_fill(order)

        assert position is not None
        assert position.side == PositionSide.SHORT

    def test_process_fill_add_to_position(self):
        """Test adding to existing position."""
        manager = PositionManager()

        # Open position
        order1 = Order(
            order_id="ord-001",
            symbol="EURUSD",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=1.0,
            status=OrderStatus.FILLED,
            filled_quantity=1.0,
            average_fill_price=1.1000,
        )
        manager.process_fill(order1)

        # Add to position at different price
        order2 = Order(
            order_id="ord-002",
            symbol="EURUSD",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=1.0,
            status=OrderStatus.FILLED,
            filled_quantity=1.0,
            average_fill_price=1.1100,
        )
        position = manager.process_fill(order2)

        assert position.quantity == 2.0
        assert position.average_entry_price == pytest.approx(1.1050)
        assert position.trade_count == 2

    def test_process_fill_close_position(self):
        """Test closing a position."""
        manager = PositionManager()

        # Open long position
        order1 = Order(
            order_id="ord-001",
            symbol="EURUSD",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=1.0,
            status=OrderStatus.FILLED,
            filled_quantity=1.0,
            average_fill_price=1.1000,
        )
        manager.process_fill(order1)

        # Close position (sell)
        order2 = Order(
            order_id="ord-002",
            symbol="EURUSD",
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=1.0,
            status=OrderStatus.FILLED,
            filled_quantity=1.0,
            average_fill_price=1.1050,
        )
        manager.process_fill(order2)

        # Position should be closed
        assert "EURUSD" not in manager.positions
        assert len(manager.closed_positions) == 1
        assert manager.closed_positions[0].status == PositionStatus.CLOSED
        assert manager.closed_positions[0].realized_pnl == pytest.approx(0.005)

    def test_process_fill_partial_close(self):
        """Test partial position close."""
        manager = PositionManager()

        # Open long position
        order1 = Order(
            order_id="ord-001",
            symbol="EURUSD",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=2.0,
            status=OrderStatus.FILLED,
            filled_quantity=2.0,
            average_fill_price=1.1000,
        )
        manager.process_fill(order1)

        # Partial close
        order2 = Order(
            order_id="ord-002",
            symbol="EURUSD",
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=1.0,
            status=OrderStatus.FILLED,
            filled_quantity=1.0,
            average_fill_price=1.1050,
        )
        position = manager.process_fill(order2)

        assert position.quantity == 1.0
        assert position.realized_pnl == pytest.approx(0.005)
        assert position.status == PositionStatus.PARTIAL

    def test_process_fill_reversal(self):
        """Test position reversal (close and open opposite)."""
        manager = PositionManager()

        # Open long position
        order1 = Order(
            order_id="ord-001",
            symbol="EURUSD",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=1.0,
            status=OrderStatus.FILLED,
            filled_quantity=1.0,
            average_fill_price=1.1000,
        )
        manager.process_fill(order1)

        # Sell more than we have (reversal)
        order2 = Order(
            order_id="ord-002",
            symbol="EURUSD",
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=2.0,
            status=OrderStatus.FILLED,
            filled_quantity=2.0,
            average_fill_price=1.1050,
        )
        position = manager.process_fill(order2)

        # Should have new short position
        assert position.side == PositionSide.SHORT
        assert position.quantity == 1.0
        assert len(manager.closed_positions) == 1

    def test_get_position(self):
        """Test getting position by symbol."""
        manager = PositionManager()

        order = Order(
            order_id="ord-001",
            symbol="EURUSD",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=1.0,
            status=OrderStatus.FILLED,
            filled_quantity=1.0,
            average_fill_price=1.1000,
        )
        manager.process_fill(order)

        position = manager.get_position("EURUSD")
        assert position is not None

        no_position = manager.get_position("GBPUSD")
        assert no_position is None

    def test_get_all_positions(self):
        """Test getting all positions."""
        manager = PositionManager()

        # Open two positions
        for symbol in ["EURUSD", "GBPUSD"]:
            order = Order(
                order_id=f"ord-{symbol}",
                symbol=symbol,
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=1.0,
                status=OrderStatus.FILLED,
                filled_quantity=1.0,
                average_fill_price=1.1000,
            )
            manager.process_fill(order)

        positions = manager.get_all_positions()
        assert len(positions) == 2

    def test_update_positions(self):
        """Test updating all positions with prices."""
        prices = {"EURUSD": 1.1050, "GBPUSD": 1.2550}
        manager = PositionManager(get_price_callback=lambda s: prices.get(s, 1.0))

        # Open positions
        for symbol in ["EURUSD", "GBPUSD"]:
            order = Order(
                order_id=f"ord-{symbol}",
                symbol=symbol,
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=1.0,
                status=OrderStatus.FILLED,
                filled_quantity=1.0,
                average_fill_price=1.1000 if symbol == "EURUSD" else 1.2500,
            )
            manager.process_fill(order)

        manager.update_positions()

        eurusd = manager.get_position("EURUSD")
        assert eurusd.current_price == 1.1050
        assert eurusd.unrealized_pnl == pytest.approx(0.005)

    def test_calculate_total_pnl(self):
        """Test total PnL calculation."""
        manager = PositionManager(get_price_callback=lambda s: 1.1050)

        # Open position
        order1 = Order(
            order_id="ord-001",
            symbol="EURUSD",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=1.0,
            status=OrderStatus.FILLED,
            filled_quantity=1.0,
            average_fill_price=1.1000,
            commission=1.0,
        )
        manager.process_fill(order1)
        manager.update_positions()

        pnl = manager.calculate_total_pnl()

        # unrealized_pnl = (1.1050 - 1.1000) * 1.0 = 0.005
        assert pnl["unrealized_pnl"] == pytest.approx(0.005)
        assert pnl["total_commission"] == 1.0
        # net_pnl = unrealized + realized - commission = 0.005 + 0 - 1.0 = -0.995
        assert pnl["net_pnl"] == pytest.approx(-0.995)

    def test_calculate_exposure(self):
        """Test exposure calculation."""
        manager = PositionManager(get_price_callback=lambda s: 1.1000)

        # Open long position
        order1 = Order(
            order_id="ord-001",
            symbol="EURUSD",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=1.0,
            status=OrderStatus.FILLED,
            filled_quantity=1.0,
            average_fill_price=1.1000,
        )
        manager.process_fill(order1)
        manager.update_positions()

        exposure = manager.calculate_exposure()

        assert exposure["long_exposure"] == pytest.approx(1.1)
        assert exposure["short_exposure"] == 0.0
        assert exposure["net_exposure"] == pytest.approx(1.1)

    def test_close_position(self):
        """Test manually closing a position."""
        manager = PositionManager()

        order = Order(
            order_id="ord-001",
            symbol="EURUSD",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=1.0,
            status=OrderStatus.FILLED,
            filled_quantity=1.0,
            average_fill_price=1.1000,
        )
        manager.process_fill(order)

        position = manager.close_position("EURUSD", exit_price=1.1100)

        assert position is not None
        assert position.realized_pnl == pytest.approx(0.01)
        assert "EURUSD" not in manager.positions
        assert len(manager.closed_positions) == 1

    def test_close_all_positions(self):
        """Test closing all positions."""
        prices = {"EURUSD": 1.1050, "GBPUSD": 1.2550}
        manager = PositionManager(get_price_callback=lambda s: prices.get(s, 1.0))

        # Open positions
        for symbol in ["EURUSD", "GBPUSD"]:
            order = Order(
                order_id=f"ord-{symbol}",
                symbol=symbol,
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=1.0,
                status=OrderStatus.FILLED,
                filled_quantity=1.0,
                average_fill_price=1.1000 if symbol == "EURUSD" else 1.2500,
            )
            manager.process_fill(order)

        closed = manager.close_all_positions()

        assert len(closed) == 2
        assert len(manager.positions) == 0
        assert len(manager.closed_positions) == 2

    def test_on_position_opened_callback(self):
        """Test position opened callback."""
        manager = PositionManager()

        opened_positions = []
        manager.on_position_opened(lambda p: opened_positions.append(p))

        order = Order(
            order_id="ord-001",
            symbol="EURUSD",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=1.0,
            status=OrderStatus.FILLED,
            filled_quantity=1.0,
            average_fill_price=1.1000,
        )
        manager.process_fill(order)

        assert len(opened_positions) == 1

    def test_on_position_closed_callback(self):
        """Test position closed callback."""
        manager = PositionManager()

        closed_positions = []
        manager.on_position_closed(lambda p: closed_positions.append(p))

        # Open position
        order1 = Order(
            order_id="ord-001",
            symbol="EURUSD",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=1.0,
            status=OrderStatus.FILLED,
            filled_quantity=1.0,
            average_fill_price=1.1000,
        )
        manager.process_fill(order1)

        # Close position
        order2 = Order(
            order_id="ord-002",
            symbol="EURUSD",
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=1.0,
            status=OrderStatus.FILLED,
            filled_quantity=1.0,
            average_fill_price=1.1050,
        )
        manager.process_fill(order2)

        assert len(closed_positions) == 1

    def test_get_stats(self):
        """Test getting statistics."""
        manager = PositionManager()

        # Open and close a position
        order1 = Order(
            order_id="ord-001",
            symbol="EURUSD",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=1.0,
            status=OrderStatus.FILLED,
            filled_quantity=1.0,
            average_fill_price=1.1000,
        )
        manager.process_fill(order1)

        order2 = Order(
            order_id="ord-002",
            symbol="EURUSD",
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=1.0,
            status=OrderStatus.FILLED,
            filled_quantity=1.0,
            average_fill_price=1.1050,
        )
        manager.process_fill(order2)

        stats = manager.get_stats()

        assert stats["open_positions"] == 0
        assert stats["closed_positions"] == 1
        assert stats["winning_trades"] == 1
        assert stats["win_rate"] == 1.0

    def test_reset(self):
        """Test manager reset."""
        manager = PositionManager()

        order = Order(
            order_id="ord-001",
            symbol="EURUSD",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=1.0,
            status=OrderStatus.FILLED,
            filled_quantity=1.0,
            average_fill_price=1.1000,
        )
        manager.process_fill(order)

        manager.reset()

        assert len(manager.positions) == 0
        assert len(manager.closed_positions) == 0
