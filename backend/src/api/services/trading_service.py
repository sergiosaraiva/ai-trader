"""Paper trading service for simulated trade execution.

This service provides:
- Virtual account with $100K starting balance
- Trade execution based on predictions
- Position tracking and P&L calculation
- Trade history recording
"""

import logging
from datetime import datetime, timedelta
from threading import Lock
from typing import Dict, List, Optional, Any, Tuple

from sqlalchemy.orm import Session

from ..database.models import Trade, PerformanceSnapshot, Prediction
from ..database.session import get_session
from ..utils.validation import safe_division
from ...config import trading_config
from ...trading.position_sizer import ConservativeHybridSizer
from ...trading.circuit_breakers import TradingCircuitBreaker

logger = logging.getLogger(__name__)


class TradingService:
    """Service for paper trading simulation.

    Manages virtual account, executes trades based on predictions,
    and tracks performance metrics.

    Configuration is loaded from the centralized trading_config system.
    """

    def __init__(self):
        self._lock = Lock()

        # Load trading parameters from centralized config
        self._load_config()

        # Account state (in-memory, synced with DB)
        self._balance = self.INITIAL_BALANCE
        self._equity = self.INITIAL_BALANCE
        self._open_position: Optional[Dict] = None

        # Performance tracking
        self._total_trades = 0
        self._winning_trades = 0
        self._losing_trades = 0
        self._total_pips = 0.0
        self._total_pnl = 0.0

        # Initialized flag
        self._initialized = False

        # Position sizing and circuit breakers
        self.position_sizer = ConservativeHybridSizer()
        self.circuit_breaker = TradingCircuitBreaker(trading_config.conservative_hybrid)

        # Register config change callbacks for hot reload
        trading_config.register_callback("trading", self._on_config_change)
        trading_config.register_callback("conservative_hybrid", self._on_conservative_hybrid_change)

    def _load_config(self) -> None:
        """Load configuration from centralized config."""
        params = trading_config.trading
        self.INITIAL_BALANCE = params.initial_balance
        self.DEFAULT_LOT_SIZE = params.default_lot_size
        self.PIP_VALUE = params.pip_value
        self.CONFIDENCE_THRESHOLD = params.confidence_threshold
        self.DEFAULT_TP_PIPS = params.default_tp_pips
        self.DEFAULT_SL_PIPS = params.default_sl_pips
        self.MAX_HOLDING_HOURS = params.max_holding_hours

    def _on_config_change(self, params) -> None:
        """Callback for configuration changes (hot reload).

        Args:
            params: TradingParameters object with new values
        """
        logger.info("Trading configuration changed, reloading parameters...")
        with self._lock:
            self.INITIAL_BALANCE = params.initial_balance
            self.DEFAULT_LOT_SIZE = params.default_lot_size
            self.PIP_VALUE = params.pip_value
            self.CONFIDENCE_THRESHOLD = params.confidence_threshold
            self.DEFAULT_TP_PIPS = params.default_tp_pips
            self.DEFAULT_SL_PIPS = params.default_sl_pips
            self.MAX_HOLDING_HOURS = params.max_holding_hours
        logger.info("Trading parameters reloaded successfully")

    def _on_conservative_hybrid_change(self, params) -> None:
        """Callback for conservative hybrid configuration changes.

        Args:
            params: ConservativeHybridParameters object with new values
        """
        logger.info("Conservative hybrid configuration changed, reloading circuit breaker...")
        with self._lock:
            self.circuit_breaker = TradingCircuitBreaker(params)
        logger.info("Circuit breaker reloaded successfully")

    @property
    def is_loaded(self) -> bool:
        """Check if service is initialized and ready."""
        return self._initialized

    def initialize(self, db: Optional[Session] = None) -> bool:
        """Initialize trading service from database state."""
        if self._initialized:
            return True

        logger.info("Initializing TradingService...")

        # Use provided session or create new one
        should_close = db is None
        if db is None:
            db = get_session()

        try:
            # Load open position if any
            open_trade = db.query(Trade).filter(
                Trade.status == "open"
            ).first()

            if open_trade:
                self._open_position = self._trade_to_dict(open_trade)
                logger.info(f"Loaded open position: {self._open_position['direction']} at {self._open_position['entry_price']}")

            # Calculate performance from closed trades
            closed_trades = db.query(Trade).filter(
                Trade.status == "closed"
            ).all()

            self._total_trades = len(closed_trades)
            self._winning_trades = sum(1 for t in closed_trades if t.is_winner)
            self._losing_trades = self._total_trades - self._winning_trades
            self._total_pips = sum(t.pips or 0 for t in closed_trades)
            self._total_pnl = sum(t.pnl_usd or 0 for t in closed_trades)

            # Calculate current balance
            self._balance = self.INITIAL_BALANCE + self._total_pnl

            # Load most recent performance snapshot for validation
            latest = db.query(PerformanceSnapshot).order_by(
                PerformanceSnapshot.timestamp.desc()
            ).first()

            if latest:
                # Use snapshot balance as source of truth
                self._balance = latest.balance

            self._equity = self._balance
            self._initialized = True

            logger.info(f"TradingService initialized: Balance=${self._balance:,.2f}, Trades={self._total_trades}")
            return True

        finally:
            if should_close:
                db.close()

    def _trade_to_dict(self, trade: Trade) -> Dict:
        """Convert Trade ORM object to dict."""
        return {
            "id": trade.id,
            "symbol": trade.symbol,
            "direction": trade.direction,
            "entry_price": trade.entry_price,
            "entry_time": trade.entry_time,
            "exit_price": trade.exit_price,
            "exit_time": trade.exit_time,
            "exit_reason": trade.exit_reason,
            "lot_size": trade.lot_size,
            "take_profit": trade.take_profit,
            "stop_loss": trade.stop_loss,
            "pips": trade.pips,
            "pnl_usd": trade.pnl_usd,
            "is_winner": trade.is_winner,
            "confidence": trade.confidence,
            "status": trade.status,
        }

    def execute_trade(
        self,
        prediction: Dict,
        current_price: float,
        db: Optional[Session] = None,
    ) -> Optional[Dict]:
        """Execute a paper trade based on prediction.

        Args:
            prediction: Prediction dict from ModelService
            current_price: Current market price
            db: Database session

        Returns:
            Trade dict if executed, None otherwise
        """
        # Check if we should trade
        if not prediction.get("should_trade", False):
            logger.info(f"Skipping trade: confidence {prediction['confidence']:.1%} below threshold")
            return None

        # Check for existing position
        if self._open_position is not None:
            logger.info("Skipping trade: position already open")
            return None

        # Create session if not provided
        should_close = db is None
        if db is None:
            db = get_session()

        try:
            # Check circuit breakers (returns 3-tuple with risk_reduction_factor)
            can_trade, breaker_reason, risk_reduction_factor = self.circuit_breaker.can_trade(db, self._balance)
            if not can_trade:
                logger.warning(f"Circuit breaker triggered: {breaker_reason}")
                return None

            # Determine trade direction and levels
            direction = prediction["direction"]
            confidence = prediction["confidence"]

            # Calculate position size using Conservative Hybrid sizer
            config = trading_config.conservative_hybrid
            position_lots, risk_pct_used, metadata = self.position_sizer.calculate_position_size(
                balance=self._balance,
                confidence=confidence,
                sl_pips=self.DEFAULT_SL_PIPS,
                config=config,
                pip_value=self.PIP_VALUE,
                lot_size=config.lot_size,
                risk_reduction_factor=risk_reduction_factor
            )

            if position_lots <= 0:
                logger.info(f"Position size is zero, skipping trade. Reason: {metadata.get('reason', 'unknown')}")
                return None

            # Calculate TP and SL levels
            pip_size = 0.0001  # For EURUSD
            if direction == "long":
                take_profit = current_price + (self.DEFAULT_TP_PIPS * pip_size)
                stop_loss = current_price - (self.DEFAULT_SL_PIPS * pip_size)
            else:
                take_profit = current_price - (self.DEFAULT_TP_PIPS * pip_size)
                stop_loss = current_price + (self.DEFAULT_SL_PIPS * pip_size)

            # Create trade record
            trade = Trade(
                symbol="EURUSD",
                direction=direction,
                entry_price=current_price,
                entry_time=datetime.utcnow(),
                lot_size=position_lots,
                take_profit=take_profit,
                stop_loss=stop_loss,
                max_holding_bars=self.MAX_HOLDING_HOURS,
                confidence=confidence,
                risk_percentage_used=risk_pct_used,
                status="open",
            )

            db.add(trade)
            db.commit()
            db.refresh(trade)

            # Update in-memory state
            with self._lock:
                self._open_position = self._trade_to_dict(trade)

            logger.info(
                f"Executed {direction.upper()} at {current_price:.5f} "
                f"(TP: {take_profit:.5f}, SL: {stop_loss:.5f}, Conf: {confidence:.1%}, "
                f"Position: {position_lots:.4f} lots, Risk: {risk_pct_used:.2f}%)"
            )

            return self._open_position

        except Exception as e:
            logger.error(f"Failed to execute trade: {e}")
            db.rollback()
            return None

        finally:
            if should_close:
                db.close()

    def check_and_close_position(
        self,
        current_price: float,
        db: Optional[Session] = None,
    ) -> Optional[Dict]:
        """Check if open position should be closed and close it.

        Checks:
        1. Take profit hit
        2. Stop loss hit
        3. Max holding time exceeded

        Args:
            current_price: Current market price
            db: Database session

        Returns:
            Closed trade dict if position closed, None otherwise
        """
        if self._open_position is None:
            return None

        position = self._open_position
        direction = position["direction"]
        entry_price = position["entry_price"]
        entry_time = position["entry_time"]
        take_profit = position["take_profit"]
        stop_loss = position["stop_loss"]

        # Check exit conditions
        exit_reason = None
        pip_size = 0.0001

        if direction == "long":
            if current_price >= take_profit:
                exit_reason = "tp"
            elif current_price <= stop_loss:
                exit_reason = "sl"
        else:  # short
            if current_price <= take_profit:
                exit_reason = "tp"
            elif current_price >= stop_loss:
                exit_reason = "sl"

        # Check max holding time
        if exit_reason is None:
            try:
                if isinstance(entry_time, str):
                    entry_time = datetime.fromisoformat(entry_time)
                hours_held = (datetime.utcnow() - entry_time).total_seconds() / 3600
                if hours_held >= self.MAX_HOLDING_HOURS:
                    exit_reason = "timeout"
            except (ValueError, TypeError) as e:
                logger.error(f"Failed to parse entry_time: {entry_time}, error: {e}")
                # Continue without timeout check if time parsing fails

        if exit_reason is None:
            return None

        # Close the position
        return self.close_position(current_price, exit_reason, db)

    def close_position(
        self,
        exit_price: float,
        exit_reason: str,
        db: Optional[Session] = None,
    ) -> Optional[Dict]:
        """Close the open position.

        Args:
            exit_price: Exit price
            exit_reason: Reason for exit (tp, sl, timeout, manual)
            db: Database session

        Returns:
            Closed trade dict
        """
        if self._open_position is None:
            logger.warning("No position to close")
            return None

        position = self._open_position
        direction = position["direction"]
        entry_price = position["entry_price"]

        # Calculate P&L
        pip_size = 0.0001
        if direction == "long":
            pips = (exit_price - entry_price) / pip_size
        else:
            pips = (entry_price - exit_price) / pip_size

        pnl_usd = pips * self.PIP_VALUE
        is_winner = pips > 0

        # Update database
        should_close = db is None
        if db is None:
            db = get_session()

        try:
            trade = db.query(Trade).filter(Trade.id == position["id"]).first()
            if trade:
                trade.exit_price = exit_price
                trade.exit_time = datetime.utcnow()
                trade.exit_reason = exit_reason
                trade.pips = pips
                trade.pnl_usd = pnl_usd
                trade.is_winner = is_winner
                trade.status = "closed"
                db.commit()

            # Update in-memory state
            with self._lock:
                self._open_position = None
                self._total_trades += 1
                if is_winner:
                    self._winning_trades += 1
                else:
                    self._losing_trades += 1
                self._total_pips += pips
                self._total_pnl += pnl_usd
                self._balance += pnl_usd
                self._equity = self._balance

            # Record trade outcome for threshold service
            try:
                from .threshold_service import threshold_service
                if threshold_service.is_initialized:
                    threshold_service.record_trade_outcome(
                        trade_id=position["id"],
                        is_winner=is_winner,
                        timestamp=datetime.utcnow()
                    )
            except Exception as e:
                logger.error(f"Failed to record trade outcome to threshold service: {e}")

            # Record trade outcome for circuit breaker (progressive risk reduction)
            try:
                self.circuit_breaker.record_trade_outcome(
                    db=db,
                    trade_id=position["id"],
                    is_winner=is_winner
                )
            except Exception as e:
                logger.error(f"Failed to record trade outcome to circuit breaker: {e}")

            result = {
                **position,
                "exit_price": exit_price,
                "exit_time": datetime.utcnow().isoformat(),
                "exit_reason": exit_reason,
                "pips": pips,
                "pnl_usd": pnl_usd,
                "is_winner": is_winner,
                "status": "closed",
            }

            logger.info(
                f"Closed {direction.upper()}: {pips:+.1f} pips (${pnl_usd:+.2f}) - {exit_reason}"
            )

            return result

        except Exception as e:
            logger.error(f"Failed to close position: {e}")
            db.rollback()
            return None

        finally:
            if should_close:
                db.close()

    def get_status(self) -> Dict[str, Any]:
        """Get current trading status."""
        unrealized_pnl = 0.0
        if self._open_position:
            # Note: Would need current price to calculate unrealized P&L
            pass

        return {
            "mode": "paper",
            "balance": self._balance,
            "equity": self._equity,
            "unrealized_pnl": unrealized_pnl,
            "open_position": self._open_position,
            "has_position": self._open_position is not None,
        }

    def get_performance(self) -> Dict[str, Any]:
        """Get trading performance metrics."""
        win_rate = safe_division(
            self._winning_trades,
            self._total_trades,
            default=0.0
        )

        avg_pips = safe_division(
            self._total_pips,
            self._total_trades,
            default=0.0
        )

        # Profit factor: (winning_trades * avg_win) / (losing_trades * avg_loss)
        # Requires both numerator and denominator to be non-zero
        avg_win = self.DEFAULT_TP_PIPS if self._winning_trades > 0 else 0
        avg_loss = self.DEFAULT_SL_PIPS if self._losing_trades > 0 else 0

        total_wins = self._winning_trades * avg_win
        total_losses = self._losing_trades * avg_loss

        profit_factor = safe_division(
            total_wins,
            total_losses,
            default=0.0
        )

        return {
            "total_trades": self._total_trades,
            "winning_trades": self._winning_trades,
            "losing_trades": self._losing_trades,
            "win_rate": win_rate,
            "total_pips": self._total_pips,
            "total_pnl_usd": self._total_pnl,
            "avg_pips_per_trade": avg_pips,
            "profit_factor": profit_factor,
            "initial_balance": self.INITIAL_BALANCE,
            "current_balance": self._balance,
            "return_pct": (self._balance - self.INITIAL_BALANCE) / self.INITIAL_BALANCE * 100,
        }

    def get_trade_history(
        self,
        limit: int = 50,
        db: Optional[Session] = None,
    ) -> List[Dict]:
        """Get trade history.

        Args:
            limit: Maximum number of trades to return
            db: Database session

        Returns:
            List of trade dicts
        """
        should_close = db is None
        if db is None:
            db = get_session()

        try:
            trades = db.query(Trade).order_by(
                Trade.entry_time.desc()
            ).limit(limit).all()

            return [self._trade_to_dict(t) for t in trades]

        finally:
            if should_close:
                db.close()

    def get_equity_curve(
        self,
        db: Optional[Session] = None,
    ) -> List[Dict]:
        """Get equity curve data.

        Returns list of {timestamp, balance, equity} points.
        """
        should_close = db is None
        if db is None:
            db = get_session()

        try:
            # Get closed trades ordered by exit time
            trades = db.query(Trade).filter(
                Trade.status == "closed"
            ).order_by(Trade.exit_time.asc()).all()

            curve = [
                {
                    "timestamp": datetime(2024, 1, 1).isoformat(),  # Start
                    "balance": self.INITIAL_BALANCE,
                    "equity": self.INITIAL_BALANCE,
                }
            ]

            running_balance = self.INITIAL_BALANCE
            for trade in trades:
                if trade.pnl_usd is not None:
                    running_balance += trade.pnl_usd
                    curve.append({
                        "timestamp": (
                            trade.exit_time.isoformat()
                            if trade.exit_time else datetime.utcnow().isoformat()
                        ),
                        "balance": running_balance,
                        "equity": running_balance,
                    })

            return curve

        finally:
            if should_close:
                db.close()

    def save_performance_snapshot(
        self,
        db: Optional[Session] = None,
    ) -> None:
        """Save current performance to database."""
        should_close = db is None
        if db is None:
            db = get_session()

        try:
            perf = self.get_performance()

            snapshot = PerformanceSnapshot(
                timestamp=datetime.utcnow(),
                balance=self._balance,
                equity=self._equity,
                unrealized_pnl=0.0,
                total_trades=self._total_trades,
                winning_trades=self._winning_trades,
                losing_trades=self._losing_trades,
                win_rate=perf["win_rate"],
                total_pips=self._total_pips,
                total_pnl_usd=self._total_pnl,
                profit_factor=perf["profit_factor"],
                avg_pips_per_trade=perf["avg_pips_per_trade"],
            )

            db.add(snapshot)
            db.commit()

            logger.debug("Performance snapshot saved")

        except Exception as e:
            logger.error(f"Failed to save performance snapshot: {e}")
            db.rollback()

        finally:
            if should_close:
                db.close()

    def get_daily_pnl(self, db: Optional[Session] = None) -> float:
        """Get total P&L for today.

        Args:
            db: Database session

        Returns:
            Total P&L in USD for today
        """
        should_close = db is None
        if db is None:
            db = get_session()

        try:
            return self.circuit_breaker.get_daily_pnl(db)
        finally:
            if should_close:
                db.close()

    def can_trade(self, db: Optional[Session] = None) -> Tuple[bool, Optional[str]]:
        """Check if trading is allowed (circuit breakers).

        Args:
            db: Database session

        Returns:
            Tuple of (can_trade: bool, reason: Optional[str])
        """
        should_close = db is None
        if db is None:
            db = get_session()

        try:
            return self.circuit_breaker.can_trade(db, self._balance)
        finally:
            if should_close:
                db.close()


# Singleton instance
trading_service = TradingService()
