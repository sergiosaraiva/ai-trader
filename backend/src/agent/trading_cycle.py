"""Trading cycle execution logic.

Implements the core trading cycle: predict → signal → trade.
This module connects the AgentRunner with the existing trading infrastructure.
"""

import logging
from datetime import datetime
from typing import Optional, Callable

from sqlalchemy.orm import Session

from .config import AgentConfig
from .models import CycleResult, PredictionData, SignalData
from ..api.database.models import Prediction

logger = logging.getLogger(__name__)


class TradingCycle:
    """Executes a single trading cycle: predict → signal → trade.

    This class coordinates between:
    - model_service: For predictions
    - Database: For storing predictions
    - SignalGenerator: For trading signals (Phase 5)
    - broker_manager: For MT5 connection (Phase 5)
    - trade_executor: For trade execution (Phase 5)
    """

    def __init__(
        self,
        config: AgentConfig,
        model_service,
        db_session_factory: Callable[[], Session],
        broker_manager=None,
        trade_executor=None,
        safety_manager=None,
    ):
        """Initialize trading cycle.

        Args:
            config: Agent configuration
            model_service: Model service singleton
            db_session_factory: Factory function to create database sessions
            broker_manager: Broker manager (for paper/live mode)
            trade_executor: Trade executor (for paper/live mode)
            safety_manager: Safety manager for circuit breakers and kill switch
        """
        self.config = config
        self.model_service = model_service
        self.db_session_factory = db_session_factory
        self.broker_manager = broker_manager
        self.trade_executor = trade_executor
        self.safety_manager = safety_manager

        logger.info(
            f"TradingCycle initialized: "
            f"mode={config.mode}, "
            f"threshold={config.confidence_threshold}"
        )

    async def execute(self, cycle_number: int) -> CycleResult:
        """Execute one trading cycle.

        Steps:
        1. Check model service is ready
        2. Generate prediction using model_service
        3. Store prediction in database
        4. Check if confidence meets threshold
        5. Log signal (Phase 5: actually generate signal)
        6. Return cycle result

        Args:
            cycle_number: Current cycle number (for tracking)

        Returns:
            CycleResult with details of what happened
        """
        start_time = datetime.now()
        result = CycleResult(
            cycle_number=cycle_number,
            timestamp=start_time,
        )

        try:
            # Step 1: Check model service is ready
            if not self.model_service.is_loaded:
                result.error = "Model service not loaded"
                result.reason = "Waiting for model initialization"
                logger.warning(f"Cycle {cycle_number}: Model not loaded")
                return result

            # Step 2: Generate prediction
            try:
                prediction_dict = self._get_prediction()
                prediction_data = PredictionData.from_service_output(prediction_dict)
                result.prediction = prediction_data.to_dict()
                result.prediction_made = True

                logger.info(
                    f"Cycle {cycle_number}: Prediction made - "
                    f"direction={prediction_data.direction}, "
                    f"confidence={prediction_data.confidence:.1%}, "
                    f"should_trade={prediction_data.should_trade}"
                )

            except Exception as e:
                result.error = f"Prediction failed: {str(e)}"
                result.reason = "Error generating prediction"
                logger.error(f"Cycle {cycle_number}: Prediction error - {e}")
                return result

            # Step 3: Store prediction in database
            try:
                prediction_id = self._store_prediction(
                    prediction_data,
                    cycle_number,
                )
                logger.debug(f"Cycle {cycle_number}: Prediction stored (id={prediction_id})")

            except Exception as e:
                # Non-critical error - continue with cycle
                logger.warning(f"Cycle {cycle_number}: Failed to store prediction - {e}")

            # Step 4: Check safety status before trading
            if self.safety_manager:
                # Get actual broker equity if available (Critical Issue 5 fix)
                current_equity = None
                if self.broker_manager and self.broker_manager.is_connected():
                    try:
                        account_info = await self.broker_manager.get_account_info()
                        if account_info:
                            current_equity = account_info.get("equity")
                    except Exception as e:
                        logger.warning(f"Failed to get broker equity: {e}")

                safety_status = self.safety_manager.check_safety(
                    current_equity=current_equity,
                    confidence=prediction_data.confidence,
                    ensemble_agreement=prediction_data.agreement_score,
                    is_broker_connected=(
                        self.broker_manager.is_connected()
                        if self.broker_manager
                        else True
                    ),
                )

                if not safety_status.is_safe_to_trade:
                    result.action_taken = "hold"
                    result.reason = f"Safety check failed: {'; '.join(safety_status.breaker_reasons)}"
                    logger.warning(f"Cycle {cycle_number}: HOLD - {result.reason}")
                    return result

                # Apply safety multipliers to position size
                if safety_status.size_multiplier < 1.0:
                    logger.info(
                        f"Cycle {cycle_number}: Safety size multiplier applied: {safety_status.size_multiplier:.2f}"
                    )

                # Apply confidence override if breaker requires higher confidence
                effective_threshold = self.config.confidence_threshold
                if safety_status.min_confidence_override:
                    effective_threshold = max(
                        effective_threshold, safety_status.min_confidence_override
                    )
                    logger.info(
                        f"Cycle {cycle_number}: Safety confidence override: {effective_threshold:.1%}"
                    )

                # Check confidence threshold (with safety override)
                if prediction_data.confidence < effective_threshold:
                    result.action_taken = "hold"
                    result.reason = (
                        f"Confidence {prediction_data.confidence:.1%} below "
                        f"threshold {effective_threshold:.1%}"
                    )
                    logger.debug(f"Cycle {cycle_number}: HOLD - {result.reason}")
                    return result
            else:
                # No safety manager - check basic confidence threshold
                if prediction_data.confidence < self.config.confidence_threshold:
                    result.action_taken = "hold"
                    result.reason = (
                        f"Confidence {prediction_data.confidence:.1%} below "
                        f"threshold {self.config.confidence_threshold:.1%}"
                    )
                    logger.debug(f"Cycle {cycle_number}: HOLD - {result.reason}")
                    return result

            # Step 5: Check open positions first
            if self.trade_executor and self.config.mode in ("paper", "live"):
                try:
                    positions_to_close = await self.trade_executor.check_open_positions()
                    for pos in positions_to_close:
                        logger.info(
                            f"Closing position {pos.trade_id} - "
                            f"reason={pos.close_reason}, "
                            f"price={pos.current_price:.5f}"
                        )
                        closed = await self.trade_executor.close_position(
                            pos.trade_id,
                            pos.close_reason,
                        )

                        # Record trade result to safety manager (Critical Issue 4 fix)
                        if closed and self.safety_manager:
                            from ..trading.circuit_breakers.base import TradeResult as CBTradeResult
                            try:
                                # Create TradeResult for circuit breaker tracking
                                trade_result = CBTradeResult(
                                    pnl=pos.unrealized_pnl or 0.0,
                                    is_winner=(pos.unrealized_pnl or 0) > 0,
                                    timestamp=datetime.now(),
                                )
                                self.safety_manager.record_trade_result(trade_result)
                                logger.debug(
                                    f"Trade result recorded: pnl={pos.unrealized_pnl:.2f}"
                                )
                            except Exception as e:
                                logger.warning(f"Failed to record trade result: {e}")

                except Exception as e:
                    logger.warning(f"Error checking positions: {e}")

            # Step 6: Generate and execute signal
            if self.config.mode == "simulation":
                # Simulation mode: just log what would happen
                signal_data = self._create_placeholder_signal(prediction_data)
                result.signal = signal_data.to_dict()
                result.signal_generated = True
                result.action_taken = "signal_generated"
                result.reason = (
                    f"{prediction_data.direction.upper()} signal generated - "
                    f"confidence {prediction_data.confidence:.1%}"
                )

                logger.info(
                    f"Cycle {cycle_number}: [SIMULATION] SIGNAL GENERATED - "
                    f"{prediction_data.direction.upper()} "
                    f"(confidence={prediction_data.confidence:.1%})"
                )

            elif self.config.mode in ("paper", "live") and self.trade_executor:
                # Paper/Live mode: execute actual trades
                try:
                    from ..trading.signals.actions import TradingSignal, Action

                    # Create trading signal
                    action = Action.BUY if prediction_data.direction == "long" else Action.SELL
                    trading_signal = TradingSignal(
                        action=action,
                        symbol=self.config.symbol,
                        timestamp=datetime.now(),
                        confidence=prediction_data.confidence,
                        direction_probability=prediction_data.prob_up,
                        position_size_pct=min(
                            self.config.max_position_size,
                            self.config.max_position_size * (prediction_data.confidence - 0.5) * 2,
                        ),
                        stop_loss_pct=0.02,  # 2% default
                        take_profit_pct=0.04,  # 4% default (2:1 R:R)
                    )

                    # Execute trade
                    trade_result = await self.trade_executor.execute_signal(trading_signal)

                    if trade_result.success:
                        result.action_taken = "trade_executed"
                        result.signal_generated = True
                        result.reason = (
                            f"Trade executed: ID={trade_result.trade_id}, "
                            f"ticket={trade_result.mt5_ticket}, "
                            f"price={trade_result.entry_price:.5f}"
                        )

                        logger.info(
                            f"Cycle {cycle_number}: [{self.config.mode.upper()}] TRADE EXECUTED - "
                            f"ID={trade_result.trade_id}, "
                            f"ticket={trade_result.mt5_ticket}"
                        )
                    else:
                        result.action_taken = "trade_failed"
                        result.reason = f"Trade execution failed: {trade_result.error}"
                        logger.error(
                            f"Cycle {cycle_number}: TRADE FAILED - {trade_result.error}"
                        )

                except Exception as e:
                    result.action_taken = "trade_error"
                    result.reason = f"Trade execution error: {str(e)}"
                    logger.error(
                        f"Cycle {cycle_number}: Trade execution error - {e}",
                        exc_info=True,
                    )

        except Exception as e:
            result.error = str(e)
            result.reason = "Unexpected error in trading cycle"
            logger.error(f"Cycle {cycle_number}: Unexpected error - {e}", exc_info=True)

        finally:
            # Calculate duration
            end_time = datetime.now()
            result.duration_ms = (end_time - start_time).total_seconds() * 1000

        return result

    def _get_prediction(self) -> dict:
        """Get prediction from model service.

        Uses pipeline_service processed data if available, otherwise
        falls back to standard prediction.

        Returns:
            Prediction dictionary from model_service

        Raises:
            RuntimeError: If prediction fails
        """
        try:
            # Try pipeline prediction first (uses pre-processed features)
            prediction = self.model_service.predict_from_pipeline(
                use_cache=True,
                symbol=self.config.symbol,
            )
            return prediction

        except Exception as e:
            logger.warning(f"Pipeline prediction failed, trying fallback: {e}")

            # Fallback to standard prediction
            try:
                from ..api.services.data_service import data_service

                df = data_service.get_data_for_prediction()
                if df is None or df.empty:
                    raise RuntimeError("No data available for prediction")

                prediction = self.model_service.predict(
                    df_5min=df,
                    use_cache=True,
                    symbol=self.config.symbol,
                )
                return prediction

            except Exception as fallback_error:
                logger.error(f"Fallback prediction also failed: {fallback_error}")
                raise RuntimeError(f"All prediction methods failed: {fallback_error}")

    def _store_prediction(
        self,
        prediction_data: PredictionData,
        cycle_number: int,
    ) -> int:
        """Store prediction in database.

        Args:
            prediction_data: Prediction to store
            cycle_number: Current cycle number

        Returns:
            Prediction ID

        Raises:
            Exception: If database operation fails
        """
        session = None
        try:
            session = self.db_session_factory()

            # Create Prediction record
            prediction = Prediction(
                timestamp=prediction_data.timestamp,
                symbol=prediction_data.symbol,
                direction=prediction_data.direction,
                confidence=prediction_data.confidence,
                prob_up=prediction_data.prob_up,
                prob_down=prediction_data.prob_down,
                pred_1h=prediction_data.component_directions.get("1H"),
                conf_1h=prediction_data.component_confidences.get("1H"),
                pred_4h=prediction_data.component_directions.get("4H"),
                conf_4h=prediction_data.component_confidences.get("4H"),
                pred_d=prediction_data.component_directions.get("D"),
                conf_d=prediction_data.component_confidences.get("D"),
                agreement_count=prediction_data.agreement_count,
                agreement_score=prediction_data.agreement_score,
                market_regime=prediction_data.market_regime,
                should_trade=prediction_data.should_trade,
                used_by_agent=True,
                agent_cycle_number=cycle_number,
            )

            session.add(prediction)
            session.commit()

            prediction_id = prediction.id
            return prediction_id

        except Exception as e:
            if session:
                session.rollback()
            raise e
        finally:
            if session:
                session.close()

    def _create_placeholder_signal(self, prediction_data: PredictionData) -> SignalData:
        """Create a placeholder signal for Phase 4.

        In Phase 5, this will be replaced with actual SignalGenerator.

        Args:
            prediction_data: Prediction data

        Returns:
            Placeholder signal data
        """
        # Map prediction direction to signal action
        action = "buy" if prediction_data.direction == "long" else "sell"

        # Simple position sizing based on confidence (will be improved in Phase 5)
        position_size_pct = min(
            self.config.max_position_size,
            self.config.max_position_size * (prediction_data.confidence - 0.5) * 2,
        )

        return SignalData(
            action=action,
            confidence=prediction_data.confidence,
            reason=f"{prediction_data.direction.upper()} signal from prediction",
            position_size_pct=position_size_pct,
            stop_loss_pct=0.02,  # 2% default stop loss
            take_profit_pct=0.04,  # 4% default take profit (2:1 R:R)
            timestamp=datetime.now(),
        )
