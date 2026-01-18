"""Tests for Simulation Execution Module."""

import pytest
from datetime import datetime, timedelta

from src.trading.execution.simulation import (
    # Slippage models
    SlippageModel,
    FixedSlippageModel,
    VolumeBasedSlippageModel,
    VolatilitySlippageModel,
    # Latency models
    LatencyModel,
    FixedLatencyModel,
    RandomLatencyModel,
    # Commission models
    CommissionModel,
    FixedCommissionModel,
    PercentageCommissionModel,
    TieredCommissionModel,
    # Fill simulator
    FillSimulator,
    FillEvent,
    # Execution engine
    SimulationExecutionEngine,
    SimulationConfig,
)
from src.trading.orders.manager import Order, OrderType, OrderSide, OrderStatus


class TestFixedSlippageModel:
    """Tests for FixedSlippageModel."""

    def test_buy_slippage_positive(self):
        """Test BUY order gets positive slippage (higher price)."""
        model = FixedSlippageModel(slippage_pct=0.001)  # 0.1%
        order = Order(
            order_id="test-001",
            symbol="EURUSD",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=1.0,
        )

        slippage = model.calculate_slippage(order, market_price=1.1000)

        assert slippage > 0  # Positive = worse price for buyer
        assert slippage == pytest.approx(0.0011, rel=0.01)

    def test_sell_slippage_negative(self):
        """Test SELL order gets negative slippage (lower price)."""
        model = FixedSlippageModel(slippage_pct=0.001)
        order = Order(
            order_id="test-001",
            symbol="EURUSD",
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=1.0,
        )

        slippage = model.calculate_slippage(order, market_price=1.1000)

        assert slippage < 0  # Negative = worse price for seller
        assert slippage == pytest.approx(-0.0011, rel=0.01)


class TestVolumeBasedSlippageModel:
    """Tests for VolumeBasedSlippageModel."""

    def test_small_order_low_slippage(self):
        """Test small orders have low slippage."""
        model = VolumeBasedSlippageModel(
            base_slippage_pct=0.0001,
            volume_impact_factor=0.1,
        )
        order = Order(
            order_id="test-001",
            symbol="EURUSD",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=0.01,  # Small
        )

        slippage = model.calculate_slippage(
            order,
            market_price=1.1000,
            market_data={"avg_volume": 1000000},
        )

        # Small order should have close to base slippage
        assert abs(slippage) < 0.001

    def test_large_order_higher_slippage(self):
        """Test large orders have higher slippage."""
        model = VolumeBasedSlippageModel(
            base_slippage_pct=0.0001,
            volume_impact_factor=0.1,
        )
        small_order = Order(
            order_id="small",
            symbol="EURUSD",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=0.01,
        )
        large_order = Order(
            order_id="large",
            symbol="EURUSD",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=100.0,  # Much larger
        )

        small_slippage = model.calculate_slippage(
            small_order, market_price=1.1000, market_data={"avg_volume": 1000}
        )
        large_slippage = model.calculate_slippage(
            large_order, market_price=1.1000, market_data={"avg_volume": 1000}
        )

        assert abs(large_slippage) > abs(small_slippage)


class TestVolatilitySlippageModel:
    """Tests for VolatilitySlippageModel."""

    def test_high_volatility_higher_slippage(self):
        """Test high volatility increases slippage."""
        model = VolatilitySlippageModel(
            base_slippage_pct=0.0001,
            volatility_multiplier=2.0,
            normal_atr_pct=0.005,
        )
        order = Order(
            order_id="test-001",
            symbol="EURUSD",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=1.0,
        )

        # Normal volatility
        normal_slippage = model.calculate_slippage(
            order, market_price=1.1000, market_data={"atr": 0.0055}
        )

        # High volatility
        high_slippage = model.calculate_slippage(
            order, market_price=1.1000, market_data={"atr": 0.0110}  # Double
        )

        assert abs(high_slippage) > abs(normal_slippage)


class TestFixedLatencyModel:
    """Tests for FixedLatencyModel."""

    def test_fixed_latency(self):
        """Test fixed latency returns constant value."""
        model = FixedLatencyModel(latency_ms=50.0)

        latency1 = model.get_latency()
        latency2 = model.get_latency()

        assert latency1 == latency2
        assert latency1 == timedelta(milliseconds=50)


class TestRandomLatencyModel:
    """Tests for RandomLatencyModel."""

    def test_latency_within_bounds(self):
        """Test latency is within specified bounds."""
        model = RandomLatencyModel(
            min_latency_ms=10.0,
            max_latency_ms=100.0,
            spike_probability=0.0,  # No spikes
        )

        for _ in range(100):
            latency = model.get_latency()
            ms = latency.total_seconds() * 1000
            assert 10.0 <= ms <= 100.0

    def test_latency_spikes(self):
        """Test latency spikes occur occasionally."""
        model = RandomLatencyModel(
            min_latency_ms=10.0,
            max_latency_ms=100.0,
            spike_probability=1.0,  # Always spike
            spike_multiplier=5.0,
        )

        latency = model.get_latency()
        ms = latency.total_seconds() * 1000

        # With spike, should be >= min * multiplier (10 * 5 = 50)
        assert ms >= 50.0


class TestFixedCommissionModel:
    """Tests for FixedCommissionModel."""

    def test_fixed_commission(self):
        """Test fixed commission per trade."""
        model = FixedCommissionModel(commission_per_trade=5.0)
        order = Order(
            order_id="test-001",
            symbol="EURUSD",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=100.0,
        )

        commission = model.calculate_commission(order, fill_price=1.1000, fill_quantity=100.0)

        assert commission == 5.0


class TestPercentageCommissionModel:
    """Tests for PercentageCommissionModel."""

    def test_percentage_commission(self):
        """Test percentage-based commission."""
        model = PercentageCommissionModel(commission_pct=0.001)  # 0.1%
        order = Order(
            order_id="test-001",
            symbol="EURUSD",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=100.0,
        )

        commission = model.calculate_commission(order, fill_price=1.1000, fill_quantity=100.0)

        # 100 * 1.1 = 110, 110 * 0.001 = 0.11
        assert commission == pytest.approx(0.11, rel=0.01)

    def test_minimum_commission(self):
        """Test minimum commission floor."""
        model = PercentageCommissionModel(
            commission_pct=0.0001,
            min_commission=5.0,
        )
        order = Order(
            order_id="test-001",
            symbol="EURUSD",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=1.0,
        )

        commission = model.calculate_commission(order, fill_price=1.1000, fill_quantity=1.0)

        assert commission == 5.0  # Minimum applied

    def test_maximum_commission(self):
        """Test maximum commission cap."""
        model = PercentageCommissionModel(
            commission_pct=0.01,  # 1%
            max_commission=10.0,
        )
        order = Order(
            order_id="test-001",
            symbol="EURUSD",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=10000.0,
        )

        commission = model.calculate_commission(order, fill_price=1.1000, fill_quantity=10000.0)

        assert commission == 10.0  # Maximum applied


class TestTieredCommissionModel:
    """Tests for TieredCommissionModel."""

    def test_tiered_commission(self):
        """Test tiered commission rates."""
        model = TieredCommissionModel(
            tiers=[
                (0, 0.001),       # 0.1% for values < 10k
                (10000, 0.0005),  # 0.05% for values 10k-100k
                (100000, 0.0001), # 0.01% for values > 100k
            ]
        )
        order = Order(
            order_id="test-001",
            symbol="EURUSD",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=100.0,
        )

        # Small trade: 100 * 100 = 10000 -> 0.05% tier
        commission = model.calculate_commission(order, fill_price=100.0, fill_quantity=100.0)
        assert commission == pytest.approx(5.0, rel=0.01)  # 10000 * 0.0005

        # Large trade: 1000 * 100 = 100000 -> 0.01% tier
        commission_large = model.calculate_commission(order, fill_price=1000.0, fill_quantity=100.0)
        assert commission_large == pytest.approx(10.0, rel=0.01)  # 100000 * 0.0001


class TestFillSimulator:
    """Tests for FillSimulator."""

    @pytest.fixture
    def simulator(self):
        """Create fill simulator."""
        return FillSimulator(
            slippage_model=FixedSlippageModel(slippage_pct=0.0001),
            latency_model=FixedLatencyModel(latency_ms=50.0),
            commission_model=PercentageCommissionModel(commission_pct=0.0001),
        )

    def test_market_order_fills(self, simulator):
        """Test market order fills immediately."""
        order = Order(
            order_id="test-001",
            symbol="EURUSD",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=1.0,
        )

        fill = simulator.simulate_fill(
            order=order,
            current_price=1.1000,
            current_time=datetime.now(),
        )

        assert fill is not None
        assert fill.fill_quantity == 1.0
        assert fill.fill_price > 1.1000  # Slippage for buy
        assert fill.commission > 0

    def test_limit_order_fills_at_limit(self, simulator):
        """Test limit order fills when price reaches limit."""
        order = Order(
            order_id="test-001",
            symbol="EURUSD",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=1.0,
            limit_price=1.0950,
        )

        # Price doesn't reach limit
        fill1 = simulator.simulate_fill(
            order=order,
            current_price=1.1000,
            current_time=datetime.now(),
            market_data={"high": 1.1050, "low": 1.0980},
        )
        assert fill1 is None

        # Price reaches limit
        fill2 = simulator.simulate_fill(
            order=order,
            current_price=1.0940,
            current_time=datetime.now(),
            market_data={"high": 1.0980, "low": 1.0920},
        )
        assert fill2 is not None
        assert fill2.fill_price <= 1.0950 + 0.001  # Near limit

    def test_stop_order_triggers(self, simulator):
        """Test stop order triggers at stop price."""
        order = Order(
            order_id="test-001",
            symbol="EURUSD",
            side=OrderSide.SELL,
            order_type=OrderType.STOP,
            quantity=1.0,
            stop_price=1.0950,
        )

        # Price above stop
        fill1 = simulator.simulate_fill(
            order=order,
            current_price=1.1000,
            current_time=datetime.now(),
            market_data={"high": 1.1050, "low": 1.0980},
        )
        assert fill1 is None

        # Price reaches stop
        fill2 = simulator.simulate_fill(
            order=order,
            current_price=1.0940,
            current_time=datetime.now(),
            market_data={"high": 1.0960, "low": 1.0920},
        )
        assert fill2 is not None


class TestSimulationExecutionEngine:
    """Tests for SimulationExecutionEngine."""

    @pytest.fixture
    def engine(self):
        """Create execution engine."""
        config = SimulationConfig(
            initial_capital=100000.0,
            slippage_model=FixedSlippageModel(slippage_pct=0.0001),
            latency_model=FixedLatencyModel(latency_ms=50.0),
            commission_model=PercentageCommissionModel(commission_pct=0.0001),
        )
        return SimulationExecutionEngine(config)

    def test_submit_order(self, engine):
        """Test order submission."""
        order = Order(
            order_id="test-001",
            symbol="EURUSD",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=1.0,
        )

        success = engine.submit_order(order)

        assert success is True
        assert order.status in [OrderStatus.SUBMITTED, OrderStatus.ACCEPTED]

    def test_market_order_fills_on_update(self, engine):
        """Test market order fills on market update."""
        order = Order(
            order_id="test-001",
            symbol="EURUSD",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=1.0,
        )
        engine.submit_order(order)

        fills = engine.update_market_data(
            symbol="EURUSD",
            price=1.1000,
            timestamp=datetime.now(),
        )

        assert len(fills) == 1
        assert fills[0].symbol == "EURUSD"
        assert order.status == OrderStatus.FILLED

    def test_cancel_order(self, engine):
        """Test order cancellation."""
        order = Order(
            order_id="test-001",
            symbol="EURUSD",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=1.0,
            limit_price=1.0900,
        )
        engine.submit_order(order)

        success = engine.cancel_order("test-001")

        assert success is True
        assert order.status == OrderStatus.CANCELLED

    def test_get_pending_orders(self, engine):
        """Test getting pending orders."""
        order1 = Order(
            order_id="test-001",
            symbol="EURUSD",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=1.0,
            limit_price=1.0900,
        )
        order2 = Order(
            order_id="test-002",
            symbol="GBPUSD",
            side=OrderSide.SELL,
            order_type=OrderType.LIMIT,
            quantity=1.0,
            limit_price=1.2500,
        )
        engine.submit_order(order1)
        engine.submit_order(order2)

        all_pending = engine.get_pending_orders()
        eurusd_pending = engine.get_pending_orders("EURUSD")

        assert len(all_pending) == 2
        assert len(eurusd_pending) == 1

    def test_get_stats(self, engine):
        """Test getting execution statistics."""
        order = Order(
            order_id="test-001",
            symbol="EURUSD",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=1.0,
        )
        engine.submit_order(order)
        engine.update_market_data("EURUSD", 1.1000, datetime.now())

        stats = engine.get_stats()

        assert stats["total_fills"] == 1
        assert "total_slippage" in stats
        assert "total_commission" in stats

    def test_reset(self, engine):
        """Test engine reset."""
        order = Order(
            order_id="test-001",
            symbol="EURUSD",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=1.0,
        )
        engine.submit_order(order)
        engine.update_market_data("EURUSD", 1.1000, datetime.now())

        engine.reset()

        assert len(engine.get_pending_orders()) == 0
        assert len(engine.get_fill_history()) == 0
