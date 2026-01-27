"""Agent runner - orchestrates existing services for autonomous trading.

The AgentRunner manages the main trading loop and coordinates between
command handling, state management, and existing backend services.
"""

import asyncio
import logging
from datetime import datetime
from enum import Enum
from typing import Optional, Dict, Any

# Import existing backend services
from ..api.services.model_service import model_service
from ..api.database.session import get_session

from .config import AgentConfig
from .command_handler import CommandHandler
from .state_manager import StateManager
from .trading_cycle import TradingCycle
from .broker_manager import BrokerManager
from .trade_executor import TradeExecutor
from .safety_manager import SafetyManager
from .metrics import agent_metrics, CycleMetrics

logger = logging.getLogger(__name__)

# Graceful shutdown signal
_shutdown_event = asyncio.Event()


class AgentStatus(str, Enum):
    """Agent operational status."""

    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    ERROR = "error"


class AgentRunner:
    """Main agent orchestrator.

    Coordinates:
    - Command polling and execution
    - State persistence
    - Trading cycle execution
    - Service coordination (model_service, trading_service, etc.)
    """

    def __init__(self, config: AgentConfig):
        """Initialize agent runner.

        Args:
            config: Agent configuration
        """
        self.config = config
        self._status = AgentStatus.STOPPED

        # Initialize managers
        self._command_handler = CommandHandler(get_session)
        self._state_manager = StateManager(get_session)

        # Initialize safety manager (now uses centralized TradingConfig)
        self._safety_manager = SafetyManager(
            initial_equity=config.initial_capital,
            db_session_factory=get_session,
        )

        # Initialize broker and trade executor for paper/live mode
        self._broker_manager: Optional[BrokerManager] = None
        self._trade_executor: Optional[TradeExecutor] = None

        if config.mode in ("paper", "live"):
            self._broker_manager = BrokerManager(config)
            self._trade_executor = TradeExecutor(
                broker_manager=self._broker_manager,
                config=config,
                db_session_factory=get_session,
            )

        # Initialize trading cycle
        self._trading_cycle = TradingCycle(
            config=config,
            model_service=model_service,
            db_session_factory=get_session,
            broker_manager=self._broker_manager,
            trade_executor=self._trade_executor,
            safety_manager=self._safety_manager,
        )

        # Main loop control
        self._running = False
        self._main_task: Optional[asyncio.Task] = None
        self._cycle_count = 0

        # Error tracking
        self._last_error: Optional[str] = None

    @property
    def status(self) -> AgentStatus:
        """Get current agent status."""
        return self._status

    async def start(self) -> bool:
        """Start the agent.

        Returns:
            True if started successfully, False otherwise
        """
        if self._running:
            logger.warning("Agent already running")
            return False

        logger.info("Starting agent...")
        self._status = AgentStatus.STARTING
        self._state_manager.update_status(self._status.value)

        try:
            # Initialize state manager
            if not self._state_manager.initialize(self.config):
                raise RuntimeError("Failed to initialize state manager")

            # Load previous state for crash recovery
            previous_state = self._state_manager.get_state()
            if previous_state:
                self._cycle_count = previous_state.get("cycle_count", 0)
                logger.info(
                    f"Recovered from previous state: "
                    f"cycle_count={self._cycle_count}, "
                    f"status={previous_state.get('status')}"
                )

            # Initialize model service
            logger.info("Initializing model service...")
            if not model_service.is_initialized:
                if not model_service.initialize(warm_up=True):
                    raise RuntimeError("Failed to initialize model service")
            logger.info("Model service ready")

            # Connect to MT5 broker if in paper/live mode
            if self.config.mode in ("paper", "live") and self._broker_manager:
                logger.info(f"Connecting to MT5 broker ({self.config.mode} mode)...")
                if not await self._broker_manager.connect():
                    raise RuntimeError("Failed to connect to MT5 broker")
                logger.info("MT5 broker connected")

            # Start command polling
            await self._command_handler.start_polling()

            # Mark as running
            self._running = True
            self._status = AgentStatus.RUNNING
            self._state_manager.set_started()

            # Start main loop
            self._main_task = asyncio.create_task(self._main_loop())

            # Start metrics collection
            agent_metrics.start()

            logger.info(
                f"Agent started successfully in {self.config.mode} mode "
                f"(confidence threshold: {self.config.confidence_threshold})"
            )
            return True

        except Exception as e:
            self._last_error = str(e)
            self._status = AgentStatus.ERROR
            self._state_manager.update_status(self._status.value, error_message=str(e))
            logger.error(f"Failed to start agent: {e}", exc_info=True)
            return False

    async def stop(self) -> bool:
        """Stop the agent gracefully with timeout protection.

        Uses configurable shutdown_timeout_seconds for graceful shutdown.
        If shutdown takes too long, force stops the agent.

        Returns:
            True if stopped successfully, False otherwise
        """
        if not self._running:
            logger.warning("Agent not running")
            return False

        logger.info("Stopping agent...")
        self._status = AgentStatus.STOPPING
        self._state_manager.update_status(self._status.value)

        # Signal shutdown to main loop
        _shutdown_event.set()

        try:
            # Use timeout for graceful shutdown
            shutdown_timeout = self.config.shutdown_timeout_seconds

            async def _graceful_shutdown():
                # Stop main loop
                self._running = False

                # Cancel main task
                if self._main_task:
                    self._main_task.cancel()
                    try:
                        await self._main_task
                    except asyncio.CancelledError:
                        pass

                # Close open positions if configured
                if (
                    self.config.close_positions_on_shutdown
                    and self._trade_executor
                    and self.config.mode in ("paper", "live")
                ):
                    logger.info("Closing open positions...")
                    closed_count = await self._trade_executor.close_all_positions("agent_stopped")
                    if closed_count > 0:
                        logger.info(f"Closed {closed_count} open positions")

                # Disconnect from MT5 broker
                if self._broker_manager and self.config.mode in ("paper", "live"):
                    logger.info("Disconnecting from MT5 broker...")
                    await self._broker_manager.disconnect()

                # Stop command polling
                await self._command_handler.stop_polling()

            # Execute graceful shutdown with timeout
            try:
                await asyncio.wait_for(_graceful_shutdown(), timeout=shutdown_timeout)
                logger.info("Graceful shutdown completed")
            except asyncio.TimeoutError:
                logger.warning(
                    f"Graceful shutdown timed out after {shutdown_timeout}s, "
                    f"forcing shutdown..."
                )
                # Force stop
                self._running = False
                if self._main_task and not self._main_task.done():
                    self._main_task.cancel()

            # Mark as stopped
            self._status = AgentStatus.STOPPED
            self._state_manager.set_stopped()

            # Stop metrics
            agent_metrics.stop()

            # Clear shutdown event for potential restart
            _shutdown_event.clear()

            logger.info("Agent stopped successfully")
            return True

        except Exception as e:
            self._last_error = str(e)
            logger.error(f"Error stopping agent: {e}", exc_info=True)
            return False

    def pause(self) -> bool:
        """Pause trading (stop executing trades but keep monitoring).

        Returns:
            True if paused successfully, False otherwise
        """
        if self._status != AgentStatus.RUNNING:
            logger.warning(f"Cannot pause agent in status {self._status}")
            return False

        logger.info("Pausing agent...")
        self._status = AgentStatus.PAUSED
        self._state_manager.update_status(self._status.value)
        logger.info("Agent paused")
        return True

    def resume(self) -> bool:
        """Resume trading after pause.

        Returns:
            True if resumed successfully, False otherwise
        """
        if self._status != AgentStatus.PAUSED:
            logger.warning(f"Cannot resume agent in status {self._status}")
            return False

        logger.info("Resuming agent...")
        self._status = AgentStatus.RUNNING
        self._state_manager.update_status(self._status.value)
        logger.info("Agent resumed")
        return True

    async def _get_current_equity(self) -> float:
        """Get current account equity from broker or fallback to initial capital.

        Returns:
            Current account equity
        """
        if self._broker_manager and self._broker_manager.is_connected():
            try:
                account_info = await self._broker_manager.get_account_info()
                if account_info and account_info.get("equity"):
                    return account_info["equity"]
            except Exception as e:
                logger.warning(f"Failed to get equity from broker: {e}")
        return self.config.initial_capital

    def get_status(self) -> Dict[str, Any]:
        """Get current agent status and statistics.

        Returns:
            Dictionary with status information
        """
        status_dict = {
            "status": self._status.value,
            "mode": self.config.mode,
            "cycle_count": self._cycle_count,
            "running": self._running,
            "confidence_threshold": self.config.confidence_threshold,
            "cycle_interval_seconds": self.config.cycle_interval_seconds,
            "last_error": self._last_error,
            "model_loaded": model_service.is_loaded,
        }

        # Add broker connection status
        if self._broker_manager:
            status_dict["broker_connected"] = self._broker_manager.is_connected()
            status_dict["broker_stats"] = self._broker_manager.get_connection_stats()

        # Add open trades count
        if self._trade_executor:
            status_dict["open_trades"] = self._trade_executor.get_open_trade_count()

        # Add safety status
        status_dict["safety"] = self._safety_manager.get_status()

        # Add metrics
        status_dict["metrics"] = agent_metrics.get_summary()
        status_dict["health_indicators"] = agent_metrics.get_health_indicators()

        return status_dict

    async def _main_loop(self) -> None:
        """Main trading loop."""
        logger.info("Main loop started")

        try:
            while self._running:
                # Check for commands
                await self._process_commands()

                # Check broker connection health (if in paper/live mode)
                if self._broker_manager and self.config.mode in ("paper", "live"):
                    if not await self._broker_manager.check_connection_health():
                        logger.warning("Broker connection unhealthy, attempting reconnection...")
                        if not await self._broker_manager.reconnect():
                            logger.error("Failed to reconnect to broker")
                            # Continue in degraded state (will retry next cycle)

                # Execute trading cycle if running (not paused)
                if self._status == AgentStatus.RUNNING:
                    # Check safety before cycle
                    broker_connected = (
                        await self._broker_manager.check_connection_health()
                        if self._broker_manager
                        else True
                    )

                    # Get current equity from broker or fallback to initial capital
                    current_equity = await self._get_current_equity()

                    safety_status = self._safety_manager.check_safety(
                        current_equity=current_equity,
                        is_broker_connected=broker_connected,
                    )

                    # Update state with circuit breaker status
                    self._state_manager.update_circuit_breaker(
                        circuit_breaker_state=safety_status.circuit_breaker_state,
                        kill_switch_active=safety_status.kill_switch_active,
                    )

                    # Only execute cycle if safe
                    if safety_status.is_safe_to_trade:
                        await self._execute_cycle()
                    else:
                        logger.warning(
                            f"Trading not safe, skipping cycle: "
                            f"reasons={safety_status.breaker_reasons}"
                        )

                        # If circuit breaker halted, pause the agent
                        if safety_status.circuit_breaker_triggered:
                            self.pause()
                            logger.warning("Agent paused due to circuit breaker")

                # Wait for next cycle
                await asyncio.sleep(self.config.cycle_interval_seconds)

        except asyncio.CancelledError:
            logger.info("Main loop cancelled")
            raise
        except Exception as e:
            self._last_error = str(e)
            self._status = AgentStatus.ERROR
            self._state_manager.update_status(self._status.value, error_message=str(e))
            logger.error(f"Error in main loop: {e}", exc_info=True)
            raise

    async def _process_commands(self) -> None:
        """Check for and process pending commands."""
        commands = await self._command_handler.poll_commands()

        for cmd in commands:
            command_id = cmd["id"]
            command_name = cmd["command"]
            payload = cmd["payload"]

            logger.info(f"Processing command: {command_name} (id={command_id})")
            self._command_handler.mark_processing(command_id)

            try:
                result = await self._execute_command(command_name, payload)
                self._command_handler.mark_completed(command_id, result)

            except Exception as e:
                error_msg = str(e)
                logger.error(f"Command {command_name} failed: {error_msg}")
                self._command_handler.mark_failed(command_id, error_msg)

    async def _execute_command(self, command: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a command.

        Args:
            command: Command name
            payload: Command parameters

        Returns:
            Command result dictionary

        Raises:
            ValueError: If command is unknown
        """
        if command == "start":
            success = await self.start()
            return {"success": success, "status": self._status.value}

        elif command == "stop":
            success = await self.stop()
            return {"success": success, "status": self._status.value}

        elif command == "pause":
            success = self.pause()
            return {"success": success, "status": self._status.value}

        elif command == "resume":
            success = self.resume()
            return {"success": success, "status": self._status.value}

        elif command == "kill":
            # Emergency stop - trigger kill switch
            reason = payload.get("reason", "Manual kill command")
            logger.warning(f"KILL command received: {reason}")
            self._safety_manager.trigger_kill_switch(reason)
            self._state_manager.update_circuit_breaker(
                circuit_breaker_state="kill_switch",
                kill_switch_active=True,
            )
            success = await self.stop()
            return {"success": success, "status": "killed", "reason": reason}

        elif command == "reset_kill_switch":
            # Reset kill switch
            authorization = payload.get("authorization", "")
            force = payload.get("force", False)
            success = self._safety_manager.reset_kill_switch(
                authorization=authorization, force=force
            )
            if success:
                self._state_manager.update_circuit_breaker(
                    circuit_breaker_state="active",
                    kill_switch_active=False,
                )
            return {"success": success}

        elif command == "update_config":
            # Update configuration
            self.config.update_from_dict(payload)
            self._state_manager.update_config(self.config)
            return {"success": True, "config": self.config.to_dict()}

        elif command == "reset_circuit_breaker":
            # Reset specific circuit breaker
            breaker_name = payload.get("breaker_name")
            if not breaker_name:
                raise ValueError("breaker_name required in payload")
            success = self._safety_manager.reset_circuit_breaker(breaker_name)
            return {"success": success, "breaker_name": breaker_name}

        else:
            raise ValueError(f"Unknown command: {command}")

    async def _execute_cycle(self) -> None:
        """Execute one trading cycle.

        Steps:
        1. Increment cycle counter
        2. Execute trading cycle (predict → signal → trade)
        3. Record metrics
        4. Update state with cycle results
        5. Handle errors gracefully

        The actual trading logic is in TradingCycle.execute().
        """
        self._cycle_count += 1
        start_time = datetime.now()

        logger.info(f"Executing cycle {self._cycle_count}")

        try:
            # Execute trading cycle
            result = await self._trading_cycle.execute(self._cycle_count)

            # Record cycle metrics
            cycle_metrics = CycleMetrics(
                cycle_number=self._cycle_count,
                start_time=start_time,
                duration_ms=result.duration_ms,
                prediction_made=result.prediction_made,
                signal_generated=result.signal_generated,
                trade_executed=result.action_taken == "trade_executed",
                error=result.error,
            )
            agent_metrics.record_cycle(cycle_metrics)

            # Log result
            if result.error:
                logger.warning(
                    f"Cycle {self._cycle_count} completed with error: {result.error}"
                )
            else:
                logger.info(
                    f"Cycle {self._cycle_count} completed - "
                    f"action={result.action_taken}, "
                    f"duration={result.duration_ms:.1f}ms"
                )

            # Get current equity for state tracking
            current_equity = await self._get_current_equity()

            # Update state with cycle results
            self._state_manager.update_cycle(
                cycle_count=self._cycle_count,
                last_prediction=result.prediction,
                last_signal=result.signal,
                account_equity=current_equity,
                open_positions=0,  # Phase 5: get from position manager
            )

            # Log cycle summary
            logger.debug(
                f"Cycle {self._cycle_count} summary: "
                f"prediction={result.prediction_made}, "
                f"signal={result.signal_generated}, "
                f"action={result.action_taken}, "
                f"reason={result.reason}"
            )

        except Exception as e:
            # Catch any unexpected errors to prevent crash
            logger.error(
                f"Cycle {self._cycle_count} failed with unexpected error: {e}",
                exc_info=True,
            )
            # Update state to show error occurred
            self._state_manager.update_cycle(
                cycle_count=self._cycle_count,
                last_prediction=None,
                last_signal=None,
                account_equity=self.config.initial_capital,
                open_positions=0,
            )
