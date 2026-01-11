"""
Trading Robot Core.

Main trading robot with async trading cycle, circuit breaker integration,
and graceful shutdown support.
"""

import asyncio
import logging
import signal
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
import json

from .config import RobotConfig
from ..signals.generator import SignalGenerator, EnsemblePrediction, Position as SignalPosition
from ..signals.actions import Action, TradingSignal
from ..orders.manager import OrderManager, ExecutionMode, Order, BracketOrder
from ..positions.manager import PositionManager, Position
from ..account.manager import AccountManager
from ..risk.profiles import RiskProfile, load_risk_profile
from ..circuit_breakers.manager import CircuitBreakerManager
from ..circuit_breakers.base import TradeResult, TradingState

logger = logging.getLogger(__name__)


class RobotStatus(Enum):
    """Robot status enumeration."""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    ERROR = "error"


@dataclass
class RobotState:
    """Current robot state snapshot."""
    status: RobotStatus
    timestamp: datetime
    cycle_count: int
    last_signal: Optional[TradingSignal]
    last_prediction: Optional[Dict[str, Any]]
    account_equity: float
    open_positions: int
    active_brackets: int
    circuit_breaker_state: str
    error_message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "status": self.status.value,
            "timestamp": self.timestamp.isoformat(),
            "cycle_count": self.cycle_count,
            "last_signal": self.last_signal.to_dict() if self.last_signal else None,
            "account_equity": self.account_equity,
            "open_positions": self.open_positions,
            "active_brackets": self.active_brackets,
            "circuit_breaker_state": self.circuit_breaker_state,
            "error_message": self.error_message,
        }


@dataclass
class TradingCycleResult:
    """Result of a single trading cycle."""
    timestamp: datetime
    cycle_number: int
    prediction_made: bool
    signal_generated: bool
    order_submitted: bool
    action_taken: Action
    reason: str
    duration_ms: float
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "cycle_number": self.cycle_number,
            "prediction_made": self.prediction_made,
            "signal_generated": self.signal_generated,
            "order_submitted": self.order_submitted,
            "action_taken": self.action_taken.value,
            "reason": self.reason,
            "duration_ms": self.duration_ms,
            "error": self.error,
        }


class TradingRobot:
    """
    Main trading robot.

    Coordinates all trading components:
    - EnsemblePredictor for predictions
    - SignalGenerator for trading signals
    - OrderManager for order execution
    - PositionManager for position tracking
    - AccountManager for account management
    - CircuitBreakerManager for risk protection
    """

    def __init__(
        self,
        config: RobotConfig,
        ensemble_predictor: Optional[Any] = None,
        risk_profile: Optional[RiskProfile] = None,
        execution_mode: ExecutionMode = ExecutionMode.SIMULATION,
        get_price_callback: Optional[Callable[[str], float]] = None,
        get_features_callback: Optional[Callable[[str], Dict]] = None,
        get_atr_callback: Optional[Callable[[str], float]] = None,
        state_dir: Optional[str] = None,
    ):
        """
        Initialize trading robot.

        Args:
            config: Robot configuration
            ensemble_predictor: EnsemblePredictor instance for predictions
            risk_profile: Risk profile (loads from config if not provided)
            execution_mode: Execution mode (simulation/paper/production)
            get_price_callback: Callback to get current price for a symbol
            get_features_callback: Callback to get features for prediction
            get_atr_callback: Callback to get ATR for stop-loss calculation
            state_dir: Directory for state persistence
        """
        self.config = config
        self.execution_mode = execution_mode
        self.state_dir = Path(state_dir) if state_dir else None

        # Callbacks for market data
        self._get_price = get_price_callback or (lambda s: 1.0)
        self._get_features = get_features_callback
        self._get_atr = get_atr_callback or (lambda s: 0.0)

        # Load risk profile
        self.risk_profile = risk_profile or load_risk_profile(config.risk_profile_name)

        # Initialize components
        self.ensemble_predictor = ensemble_predictor

        self.signal_generator = SignalGenerator(
            risk_profile=self.risk_profile,
        )

        self.order_manager = OrderManager(
            execution_mode=execution_mode,
            get_price_callback=self._get_price,
        )

        self.position_manager = PositionManager(
            get_price_callback=self._get_price,
        )

        self.account_manager = AccountManager(
            initial_balance=config.simulation_config.initial_capital,
            state_file=str(self.state_dir / "account_state.json") if self.state_dir else None,
        )

        self.circuit_breaker_manager = CircuitBreakerManager(
            risk_profile=self.risk_profile,
            initial_equity=config.simulation_config.initial_capital,
        )

        # Robot state
        self._status = RobotStatus.STOPPED
        self._cycle_count = 0
        self._last_signal: Optional[TradingSignal] = None
        self._last_prediction: Optional[Dict[str, Any]] = None
        self._error_message = ""

        # Async control
        self._running = False
        self._shutdown_event = asyncio.Event()
        self._pause_event = asyncio.Event()
        self._pause_event.set()  # Not paused initially

        # Cycle history
        self._cycle_history: List[TradingCycleResult] = []
        self._max_history = 1000

        # Register callbacks
        self._setup_callbacks()

        logger.info(
            f"TradingRobot initialized: {config.name} v{config.version}, "
            f"mode={execution_mode.value}, profile={self.risk_profile.name}"
        )

    def _setup_callbacks(self) -> None:
        """Set up internal callbacks between components."""
        # When orders fill, update positions
        def on_order_fill(order: Order):
            self.position_manager.process_fill(order)
            # Update account margin
            exposure = self.position_manager.calculate_exposure()
            self.account_manager.update_margin(
                exposure['gross_exposure'] * self.account_manager.margin_requirement
            )

        self.order_manager.on_fill(on_order_fill)

        # When positions close, record trade result
        def on_position_closed(position: Position):
            self.account_manager.record_trade_result(
                realized_pnl=position.realized_pnl,
                commission=position.total_commission,
            )
            # Update circuit breakers
            trade_result = TradeResult(
                symbol=position.symbol,
                side=position.side.value,
                entry_price=position.average_entry_price,
                exit_price=position.current_price,
                quantity=position.quantity,
                pnl=position.realized_pnl,
                pnl_pct=position.pnl_percentage,
                entry_time=position.opened_at,
                exit_time=position.closed_at or datetime.now(),
            )
            self.circuit_breaker_manager.record_trade(trade_result)

        self.position_manager.on_position_closed(on_position_closed)

    async def start(self) -> None:
        """Start the trading robot."""
        if self._running:
            logger.warning("Robot is already running")
            return

        logger.info("Starting trading robot...")
        self._status = RobotStatus.STARTING
        self._running = True
        self._shutdown_event.clear()

        # Load state if available
        if self.state_dir:
            self._load_state()

        self._status = RobotStatus.RUNNING
        logger.info("Trading robot started")

        try:
            await self._main_loop()
        except asyncio.CancelledError:
            logger.info("Robot main loop cancelled")
        except Exception as e:
            logger.error(f"Robot error: {e}", exc_info=True)
            self._status = RobotStatus.ERROR
            self._error_message = str(e)
        finally:
            await self._cleanup()

    async def stop(self) -> None:
        """Gracefully stop the trading robot."""
        if not self._running:
            logger.warning("Robot is not running")
            return

        logger.info("Stopping trading robot...")
        self._status = RobotStatus.STOPPING
        self._shutdown_event.set()

    async def _main_loop(self) -> None:
        """Main trading loop."""
        cycle_interval = self.config.cycle_interval_seconds

        while self._running and not self._shutdown_event.is_set():
            # Check for pause
            await self._pause_event.wait()

            # Run trading cycle
            try:
                cycle_start = datetime.now()
                result = await self._trading_cycle()
                self._cycle_history.append(result)

                # Trim history
                if len(self._cycle_history) > self._max_history:
                    self._cycle_history = self._cycle_history[-self._max_history:]

            except Exception as e:
                logger.error(f"Error in trading cycle: {e}", exc_info=True)

            # Wait for next cycle
            try:
                await asyncio.wait_for(
                    self._shutdown_event.wait(),
                    timeout=cycle_interval,
                )
                break  # Shutdown requested
            except asyncio.TimeoutError:
                pass  # Continue to next cycle

    async def _trading_cycle(self) -> TradingCycleResult:
        """
        Execute a single trading cycle.

        Steps:
        1. Check if trading allowed (circuit breakers)
        2. Get latest market data
        3. Generate prediction
        4. Generate signal
        5. Execute if actionable
        6. Update state
        """
        cycle_start = datetime.now()
        self._cycle_count += 1

        result = TradingCycleResult(
            timestamp=cycle_start,
            cycle_number=self._cycle_count,
            prediction_made=False,
            signal_generated=False,
            order_submitted=False,
            action_taken=Action.HOLD,
            reason="",
            duration_ms=0,
        )

        try:
            symbol = self.config.symbol

            # Step 1: Check circuit breakers
            breaker_state = self.circuit_breaker_manager.check_all(
                current_equity=self.account_manager.equity,
            )

            # Step 2: Get current price
            current_price = self._get_price(symbol)

            # Step 3: Update positions with current prices
            self.position_manager.update_positions()
            pnl = self.position_manager.calculate_total_pnl()
            self.account_manager.update_unrealized_pnl(pnl['unrealized_pnl'])

            # Step 4: Update bracket orders
            self.order_manager.update_bracket_status(symbol, current_price)

            # Step 5: Generate prediction (if predictor available)
            prediction = None
            if self.ensemble_predictor and self._get_features:
                features = self._get_features(symbol)
                if features:
                    prediction = self.ensemble_predictor.predict(
                        features=features,
                        symbol=symbol,
                    )
                    result.prediction_made = True
                    self._last_prediction = prediction.to_dict() if hasattr(prediction, 'to_dict') else None

            # If no predictor, skip signal generation
            if not prediction:
                result.reason = "No prediction available"
                return result

            # Step 6: Get current position
            position = self.position_manager.get_position(symbol)
            current_position = None
            if position:
                current_position = SignalPosition(
                    symbol=position.symbol,
                    side=position.side.value,
                    quantity=position.quantity,
                    entry_price=position.average_entry_price,
                    current_price=position.current_price,
                    unrealized_pnl=position.unrealized_pnl,
                )

            # Step 7: Get ATR for stop-loss
            atr = self._get_atr(symbol)

            # Step 8: Generate signal
            signal = self.signal_generator.generate_signal_from_predictor(
                predictor_output=prediction,
                symbol=symbol,
                current_price=current_price,
                breaker_state=breaker_state,
                current_position=current_position,
                atr=atr,
            )
            result.signal_generated = True
            result.action_taken = signal.action
            result.reason = signal.reason
            self._last_signal = signal

            # Step 9: Execute if actionable
            if signal.is_actionable and signal.position_size_pct > 0:
                # Create and submit bracket order
                bracket = self.order_manager.create_bracket_order(
                    signal=signal,
                    account_equity=self.account_manager.equity,
                )
                order_result = self.order_manager.submit_bracket_order(bracket)
                result.order_submitted = order_result.success

                logger.info(
                    f"Cycle {self._cycle_count}: {signal.action.value} {symbol}, "
                    f"confidence={signal.confidence:.1%}, "
                    f"size={signal.position_size_pct:.1%}"
                )
            else:
                logger.debug(
                    f"Cycle {self._cycle_count}: HOLD - {signal.reason}"
                )

        except Exception as e:
            result.error = str(e)
            logger.error(f"Error in cycle {self._cycle_count}: {e}")

        finally:
            cycle_end = datetime.now()
            result.duration_ms = (cycle_end - cycle_start).total_seconds() * 1000

        return result

    async def _cleanup(self) -> None:
        """Cleanup on shutdown."""
        logger.info("Cleaning up trading robot...")

        # Cancel open orders
        open_orders = self.order_manager.get_open_orders()
        for order in open_orders:
            self.order_manager.cancel_order(order.order_id)

        # Close all positions (optional based on config)
        # self.position_manager.close_all_positions()

        # Save state
        if self.state_dir:
            self._save_state()

        self._running = False
        self._status = RobotStatus.STOPPED
        logger.info("Trading robot stopped")

    def pause(self) -> None:
        """Pause trading (stops making new trades)."""
        self._pause_event.clear()
        self._status = RobotStatus.PAUSED
        logger.info("Trading robot paused")

    def resume(self) -> None:
        """Resume trading after pause."""
        self._pause_event.set()
        self._status = RobotStatus.RUNNING
        logger.info("Trading robot resumed")

    def force_halt(self, reason: str) -> None:
        """Force trading halt via circuit breaker."""
        self.circuit_breaker_manager.force_halt(reason)
        logger.warning(f"Trading halted: {reason}")

    def force_resume(self) -> None:
        """Force resume after halt (use with caution)."""
        self.circuit_breaker_manager.force_resume()
        logger.warning("Trading forcibly resumed")

    def get_status(self) -> RobotState:
        """Get current robot status."""
        return RobotState(
            status=self._status,
            timestamp=datetime.now(),
            cycle_count=self._cycle_count,
            last_signal=self._last_signal,
            last_prediction=self._last_prediction,
            account_equity=self.account_manager.equity,
            open_positions=len(self.position_manager.get_all_positions()),
            active_brackets=len(self.order_manager.get_active_brackets()),
            circuit_breaker_state=self.circuit_breaker_manager.current_state.overall_state.value,
            error_message=self._error_message,
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics."""
        return {
            "robot": {
                "status": self._status.value,
                "cycle_count": self._cycle_count,
                "execution_mode": self.execution_mode.value,
            },
            "account": self.account_manager.get_stats(),
            "positions": self.position_manager.get_stats(),
            "orders": self.order_manager.get_stats(),
            "circuit_breakers": self.circuit_breaker_manager.get_status(),
        }

    def get_recent_cycles(self, n: int = 10) -> List[TradingCycleResult]:
        """Get recent trading cycle results."""
        return self._cycle_history[-n:]

    def _save_state(self) -> None:
        """Save robot state to disk."""
        if not self.state_dir:
            return

        self.state_dir.mkdir(parents=True, exist_ok=True)

        # Save account state
        self.account_manager.save_state()

        # Save robot state
        state = {
            "timestamp": datetime.now().isoformat(),
            "cycle_count": self._cycle_count,
            "status": self._status.value,
        }

        state_path = self.state_dir / "robot_state.json"
        with open(state_path, 'w') as f:
            json.dump(state, f, indent=2)

        logger.info(f"Robot state saved to {self.state_dir}")

    def _load_state(self) -> None:
        """Load robot state from disk."""
        if not self.state_dir:
            return

        # Load account state
        self.account_manager.load_state()

        # Load robot state
        state_path = self.state_dir / "robot_state.json"
        if state_path.exists():
            try:
                with open(state_path) as f:
                    state = json.load(f)
                self._cycle_count = state.get('cycle_count', 0)
                logger.info(f"Robot state loaded from {self.state_dir}")
            except Exception as e:
                logger.error(f"Error loading robot state: {e}")

    def reset(self) -> None:
        """Reset robot to initial state."""
        if self._running:
            raise RuntimeError("Cannot reset while running")

        self.order_manager.reset()
        self.position_manager.reset()
        self.account_manager.reset()
        self.circuit_breaker_manager.force_resume()

        self._cycle_count = 0
        self._last_signal = None
        self._last_prediction = None
        self._error_message = ""
        self._cycle_history.clear()

        logger.info("Trading robot reset")


def setup_signal_handlers(robot: TradingRobot) -> None:
    """
    Set up signal handlers for graceful shutdown.

    Args:
        robot: TradingRobot instance
    """
    loop = asyncio.get_event_loop()

    def handle_shutdown(sig):
        logger.info(f"Received signal {sig.name}, initiating shutdown...")
        asyncio.create_task(robot.stop())

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda s=sig: handle_shutdown(s))


async def run_robot(
    config: RobotConfig,
    ensemble_predictor: Optional[Any] = None,
    **kwargs,
) -> TradingRobot:
    """
    Convenience function to create and run a trading robot.

    Args:
        config: Robot configuration
        ensemble_predictor: EnsemblePredictor instance
        **kwargs: Additional arguments for TradingRobot

    Returns:
        TradingRobot instance
    """
    robot = TradingRobot(
        config=config,
        ensemble_predictor=ensemble_predictor,
        **kwargs,
    )

    try:
        setup_signal_handlers(robot)
    except RuntimeError:
        # Signal handlers can only be set in main thread
        pass

    await robot.start()
    return robot
