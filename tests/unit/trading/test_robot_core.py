"""Tests for Trading Robot Core."""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import MagicMock, patch, AsyncMock
import tempfile
import os

from src.trading.robot.core import (
    TradingRobot,
    RobotStatus,
    RobotState,
    TradingCycleResult,
)
from src.trading.robot.config import RobotConfig, SimulationConfig
from src.trading.orders.manager import ExecutionMode
from src.trading.risk.profiles import RiskLevel, load_risk_profile
from src.trading.signals.actions import Action


class TestRobotStatus:
    """Tests for RobotStatus enum."""

    def test_status_values(self):
        """Test all status values exist."""
        assert RobotStatus.STOPPED.value == "stopped"
        assert RobotStatus.STARTING.value == "starting"
        assert RobotStatus.RUNNING.value == "running"
        assert RobotStatus.PAUSED.value == "paused"
        assert RobotStatus.STOPPING.value == "stopping"
        assert RobotStatus.ERROR.value == "error"


class TestRobotState:
    """Tests for RobotState dataclass."""

    def test_robot_state_creation(self):
        """Test robot state creation."""
        state = RobotState(
            status=RobotStatus.RUNNING,
            timestamp=datetime.now(),
            cycle_count=100,
            last_signal=None,
            last_prediction=None,
            account_equity=100000,
            open_positions=2,
            active_brackets=1,
            circuit_breaker_state="active",
        )

        assert state.status == RobotStatus.RUNNING
        assert state.cycle_count == 100
        assert state.account_equity == 100000

    def test_robot_state_to_dict(self):
        """Test robot state serialization."""
        state = RobotState(
            status=RobotStatus.RUNNING,
            timestamp=datetime.now(),
            cycle_count=100,
            last_signal=None,
            last_prediction=None,
            account_equity=100000,
            open_positions=2,
            active_brackets=1,
            circuit_breaker_state="active",
        )

        d = state.to_dict()
        assert d["status"] == "running"
        assert d["cycle_count"] == 100


class TestTradingCycleResult:
    """Tests for TradingCycleResult dataclass."""

    def test_cycle_result_creation(self):
        """Test cycle result creation."""
        result = TradingCycleResult(
            timestamp=datetime.now(),
            cycle_number=1,
            prediction_made=True,
            signal_generated=True,
            order_submitted=True,
            action_taken=Action.BUY,
            reason="LONG: prob=75%, conf=80%",
            duration_ms=50.5,
        )

        assert result.prediction_made is True
        assert result.action_taken == Action.BUY
        assert result.duration_ms == 50.5

    def test_cycle_result_to_dict(self):
        """Test cycle result serialization."""
        result = TradingCycleResult(
            timestamp=datetime.now(),
            cycle_number=1,
            prediction_made=True,
            signal_generated=True,
            order_submitted=True,
            action_taken=Action.BUY,
            reason="Test",
            duration_ms=50.5,
        )

        d = result.to_dict()
        assert d["cycle_number"] == 1
        assert d["action_taken"] == "buy"


class TestTradingRobot:
    """Tests for TradingRobot."""

    @pytest.fixture
    def basic_config(self):
        """Create basic robot configuration."""
        return RobotConfig(
            name="Test Robot",
            version="1.0.0",
            symbol="EURUSD",
            mode="simulation",
            cycle_interval_seconds=1,
            risk_profile_name="moderate",
            simulation_config=SimulationConfig(
                initial_capital=100000,
            ),
        )

    @pytest.fixture
    def robot(self, basic_config):
        """Create a basic trading robot."""
        return TradingRobot(
            config=basic_config,
            execution_mode=ExecutionMode.SIMULATION,
            get_price_callback=lambda s: 1.1000,
        )

    def test_initialization(self, basic_config):
        """Test robot initialization."""
        robot = TradingRobot(
            config=basic_config,
            execution_mode=ExecutionMode.SIMULATION,
        )

        assert robot.config == basic_config
        assert robot.execution_mode == ExecutionMode.SIMULATION
        assert robot._status == RobotStatus.STOPPED
        assert robot._cycle_count == 0

    def test_initialization_with_risk_profile(self, basic_config):
        """Test robot initialization with custom risk profile."""
        risk_profile = load_risk_profile("conservative")

        robot = TradingRobot(
            config=basic_config,
            risk_profile=risk_profile,
            execution_mode=ExecutionMode.SIMULATION,
        )

        assert robot.risk_profile.level == RiskLevel.CONSERVATIVE

    def test_get_status(self, robot):
        """Test getting robot status."""
        status = robot.get_status()

        assert isinstance(status, RobotState)
        assert status.status == RobotStatus.STOPPED
        assert status.cycle_count == 0

    def test_get_stats(self, robot):
        """Test getting comprehensive stats."""
        stats = robot.get_stats()

        assert "robot" in stats
        assert "account" in stats
        assert "positions" in stats
        assert "orders" in stats
        assert "circuit_breakers" in stats
        assert stats["robot"]["status"] == "stopped"

    def test_pause_resume(self, robot):
        """Test pause and resume."""
        robot.pause()
        assert robot._status == RobotStatus.PAUSED
        assert not robot._pause_event.is_set()

        robot.resume()
        assert robot._status == RobotStatus.RUNNING
        assert robot._pause_event.is_set()

    def test_force_halt(self, robot):
        """Test force halt."""
        robot.force_halt("Test halt")

        state = robot.circuit_breaker_manager.get_status()
        assert state["overall_state"] == "halted"

    def test_force_resume(self, robot):
        """Test force resume after halt."""
        robot.force_halt("Test halt")
        robot.force_resume()

        state = robot.circuit_breaker_manager.get_status()
        assert state["overall_state"] == "active"

    def test_reset(self, robot):
        """Test robot reset."""
        # Modify state
        robot._cycle_count = 100
        robot._error_message = "Test error"

        robot.reset()

        assert robot._cycle_count == 0
        assert robot._error_message == ""

    def test_reset_while_running_raises(self, robot):
        """Test reset while running raises error."""
        robot._running = True

        with pytest.raises(RuntimeError):
            robot.reset()

    def test_get_recent_cycles(self, robot):
        """Test getting recent cycle results."""
        # Add some cycle results
        for i in range(5):
            result = TradingCycleResult(
                timestamp=datetime.now(),
                cycle_number=i,
                prediction_made=True,
                signal_generated=True,
                order_submitted=False,
                action_taken=Action.HOLD,
                reason="Test",
                duration_ms=10.0,
            )
            robot._cycle_history.append(result)

        recent = robot.get_recent_cycles(3)
        assert len(recent) == 3
        assert recent[-1].cycle_number == 4

    @pytest.mark.asyncio
    async def test_start_stop(self, basic_config):
        """Test starting and stopping robot."""
        robot = TradingRobot(
            config=basic_config,
            execution_mode=ExecutionMode.SIMULATION,
        )

        # Start robot in background
        start_task = asyncio.create_task(robot.start())

        # Let it run briefly
        await asyncio.sleep(0.1)

        assert robot._status == RobotStatus.RUNNING

        # Stop robot
        await robot.stop()

        # Wait for start task to complete
        await asyncio.wait_for(start_task, timeout=2.0)

        assert robot._status == RobotStatus.STOPPED

    @pytest.mark.asyncio
    async def test_trading_cycle_no_predictor(self, basic_config):
        """Test trading cycle without predictor."""
        robot = TradingRobot(
            config=basic_config,
            execution_mode=ExecutionMode.SIMULATION,
            get_price_callback=lambda s: 1.1000,
        )

        result = await robot._trading_cycle()

        assert result.prediction_made is False
        assert result.action_taken == Action.HOLD
        assert "No prediction available" in result.reason

    @pytest.mark.asyncio
    async def test_trading_cycle_with_mock_predictor(self, basic_config):
        """Test trading cycle with mocked predictor."""
        # Create mock predictor
        mock_predictor = MagicMock()
        mock_prediction = MagicMock()
        mock_prediction.direction = 1
        mock_prediction.direction_probability = 0.75
        mock_prediction.confidence = 0.80
        mock_prediction.agreement_score = 0.9
        mock_prediction.component_predictions = {}
        mock_prediction.market_regime = "trending"
        mock_prediction.volatility_level = "normal"
        mock_prediction.to_dict = MagicMock(return_value={})
        mock_predictor.predict = MagicMock(return_value=mock_prediction)

        def get_features(symbol):
            return {"short_term": [], "medium_term": [], "long_term": []}

        robot = TradingRobot(
            config=basic_config,
            ensemble_predictor=mock_predictor,
            execution_mode=ExecutionMode.SIMULATION,
            get_price_callback=lambda s: 1.1000,
            get_features_callback=get_features,
            get_atr_callback=lambda s: 0.0050,
        )

        result = await robot._trading_cycle()

        assert result.prediction_made is True
        assert result.signal_generated is True
        assert mock_predictor.predict.called

    @pytest.mark.asyncio
    async def test_trading_cycle_circuit_breaker_halted(self, basic_config):
        """Test trading cycle when circuit breaker halted."""
        # Create mock predictor so we test circuit breaker check
        mock_predictor = MagicMock()
        mock_prediction = MagicMock()
        mock_prediction.direction = 1
        mock_prediction.direction_probability = 0.75
        mock_prediction.confidence = 0.80
        mock_prediction.agreement_score = 0.9
        mock_prediction.component_predictions = {}
        mock_prediction.market_regime = "trending"
        mock_prediction.volatility_level = "normal"
        mock_prediction.to_dict = MagicMock(return_value={})
        mock_predictor.predict = MagicMock(return_value=mock_prediction)

        def get_features(symbol):
            return {"short_term": [], "medium_term": [], "long_term": []}

        robot = TradingRobot(
            config=basic_config,
            ensemble_predictor=mock_predictor,
            execution_mode=ExecutionMode.SIMULATION,
            get_price_callback=lambda s: 1.1000,
            get_features_callback=get_features,
        )

        # Force halt
        robot.circuit_breaker_manager.force_halt("Test halt")

        result = await robot._trading_cycle()

        # Circuit breaker should prevent signal generation (but allow price/position updates)
        assert result.signal_generated is True  # Signal is still generated
        assert result.action_taken == Action.HOLD  # But action is HOLD due to circuit breaker
        assert "Circuit breaker" in result.reason

    def test_state_persistence(self, basic_config):
        """Test state save and load."""
        with tempfile.TemporaryDirectory() as tmpdir:
            robot = TradingRobot(
                config=basic_config,
                execution_mode=ExecutionMode.SIMULATION,
                state_dir=tmpdir,
            )

            # Modify state
            robot._cycle_count = 100
            robot.account_manager.record_trade_result(realized_pnl=500, commission=10)

            # Save state
            robot._save_state()

            # Create new robot and load state
            robot2 = TradingRobot(
                config=basic_config,
                execution_mode=ExecutionMode.SIMULATION,
                state_dir=tmpdir,
            )
            robot2._load_state()

            assert robot2._cycle_count == 100

    def test_callback_integration(self, robot):
        """Test callbacks integrate correctly between components."""
        # Get the order manager
        order_manager = robot.order_manager
        position_manager = robot.position_manager

        # Simulate a filled order
        from src.trading.orders.manager import Order, OrderSide, OrderType, OrderStatus

        order = Order(
            order_id="test-001",
            symbol="EURUSD",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=0.01,
            status=OrderStatus.FILLED,
            filled_quantity=0.01,
            average_fill_price=1.1000,
            commission=1.0,
        )

        # Process the fill
        position_manager.process_fill(order)

        # Should have position
        position = position_manager.get_position("EURUSD")
        assert position is not None
        assert position.quantity == 0.01

    def test_component_initialization(self, robot):
        """Test all components are properly initialized."""
        assert robot.signal_generator is not None
        assert robot.order_manager is not None
        assert robot.position_manager is not None
        assert robot.account_manager is not None
        assert robot.circuit_breaker_manager is not None

    def test_risk_profile_loading(self, basic_config):
        """Test risk profile is loaded correctly."""
        basic_config.risk_profile_name = "conservative"

        robot = TradingRobot(
            config=basic_config,
            execution_mode=ExecutionMode.SIMULATION,
        )

        assert robot.risk_profile.level == RiskLevel.CONSERVATIVE

    @pytest.mark.asyncio
    async def test_cleanup_cancels_orders(self, basic_config):
        """Test cleanup cancels open orders."""
        robot = TradingRobot(
            config=basic_config,
            execution_mode=ExecutionMode.SIMULATION,
        )

        # Create an open order
        from src.trading.orders.manager import Order, OrderSide, OrderType

        order = Order(
            order_id="test-001",
            symbol="EURUSD",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=1.0,
            limit_price=1.0950,
        )
        robot.order_manager.submit_order(order)

        # Run cleanup
        await robot._cleanup()

        # Order should be cancelled
        assert order.status.value == "cancelled"
