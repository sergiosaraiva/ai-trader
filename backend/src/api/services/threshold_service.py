"""Dynamic confidence threshold service.

This service implements adaptive threshold calculation based on:
- Multi-window quantile analysis (7d, 14d, 30d)
- Recent trade performance feedback
- Configurable parameters from TradingConfig

Algorithm:
1. Calculate quantile thresholds for each window
2. Blend with configured weights (25% short, 60% medium, 15% long)
3. Adjust based on recent win rate vs target
4. Apply hard bounds and divergence checks
5. Record threshold history for monitoring
"""

import logging
from collections import deque
from datetime import datetime, timedelta
from threading import Lock
from typing import Dict, Optional, Any, Tuple

import numpy as np
from sqlalchemy.orm import Session

from ..database.models import Prediction, Trade, ThresholdHistory
from ..database.session import get_session
from ...config import trading_config

logger = logging.getLogger(__name__)


class ThresholdManager:
    """Manages dynamic confidence threshold calculation.

    Thread-safe service that:
    - Maintains in-memory deques for fast threshold calculation
    - Records prediction confidences and trade outcomes
    - Calculates adaptive thresholds based on recent history
    - Persists threshold history to database
    - Provides monitoring and metrics
    """

    def __init__(self):
        self._lock = Lock()

        # Load configuration
        self._load_config()

        # In-memory storage for fast calculation (O(1) operations)
        # Store tuples of (timestamp, confidence)
        self._predictions_7d = deque(maxlen=10080)  # 7 days * 24 hours * 60 minutes
        self._predictions_14d = deque(maxlen=20160)  # 14 days
        self._predictions_30d = deque(maxlen=43200)  # 30 days

        # Trade outcomes for performance adjustment (timestamp, is_winner)
        self._recent_trades = deque(maxlen=100)  # Keep last 100 trades

        # Current threshold cache
        self._current_threshold: Optional[float] = None
        self._last_calculation: Optional[datetime] = None
        self._calculation_count = 0

        # Register config change callback
        trading_config.register_callback("threshold", self._on_config_change)

        self._initialized = False

    def _load_config(self) -> None:
        """Load configuration from centralized config."""
        params = trading_config.threshold
        self.use_dynamic = params.use_dynamic_threshold
        self.short_term_days = params.short_term_window_days
        self.medium_term_days = params.medium_term_window_days
        self.long_term_days = params.long_term_window_days
        self.short_weight = params.short_term_weight
        self.medium_weight = params.medium_term_weight
        self.long_weight = params.long_term_weight
        self.quantile = params.quantile
        self.perf_lookback = params.performance_lookback_trades
        self.target_win_rate = params.target_win_rate
        self.adjustment_factor = params.adjustment_factor
        self.min_threshold = params.min_threshold
        self.max_threshold = params.max_threshold
        self.max_divergence = params.max_divergence_from_long_term
        self.min_predictions = params.min_predictions_required
        self.min_trades = params.min_trades_for_adjustment

        # Static fallback from trading config
        self.static_threshold = trading_config.trading.confidence_threshold

    def _on_config_change(self, params) -> None:
        """Callback for configuration changes.

        Args:
            params: ThresholdParameters object with new values
        """
        logger.info("Threshold configuration changed, reloading parameters...")
        with self._lock:
            self.use_dynamic = params.use_dynamic_threshold
            self.short_term_days = params.short_term_window_days
            self.medium_term_days = params.medium_term_window_days
            self.long_term_days = params.long_term_window_days
            self.short_weight = params.short_term_weight
            self.medium_weight = params.medium_term_weight
            self.long_weight = params.long_term_weight
            self.quantile = params.quantile
            self.perf_lookback = params.performance_lookback_trades
            self.target_win_rate = params.target_win_rate
            self.adjustment_factor = params.adjustment_factor
            self.min_threshold = params.min_threshold
            self.max_threshold = params.max_threshold
            self.max_divergence = params.max_divergence_from_long_term
            self.min_predictions = params.min_predictions_required
            self.min_trades = params.min_trades_for_adjustment

            # Invalidate cached threshold on config change
            self._current_threshold = None
            self._last_calculation = None

        logger.info("Threshold parameters reloaded successfully")

    @property
    def is_initialized(self) -> bool:
        """Check if service is initialized."""
        return self._initialized

    def initialize(self, db: Optional[Session] = None) -> bool:
        """Initialize service by loading recent predictions and trades from database.

        Args:
            db: Database session (optional)

        Returns:
            True if successful, False otherwise
        """
        if self._initialized:
            return True

        logger.info("Initializing ThresholdManager...")

        should_close = db is None
        if db is None:
            db = get_session()

        try:
            # Load recent predictions (last 30 days)
            cutoff = datetime.utcnow() - timedelta(days=self.long_term_days)
            predictions = db.query(Prediction).filter(
                Prediction.timestamp >= cutoff
            ).order_by(Prediction.timestamp.asc()).all()

            with self._lock:
                for pred in predictions:
                    self._predictions_30d.append((pred.timestamp, pred.confidence))

                    # Also add to shorter windows if within range
                    days_ago = (datetime.utcnow() - pred.timestamp).days
                    if days_ago <= self.short_term_days:
                        self._predictions_7d.append((pred.timestamp, pred.confidence))
                    if days_ago <= self.medium_term_days:
                        self._predictions_14d.append((pred.timestamp, pred.confidence))

            # Load recent trades (last 100)
            trades = db.query(Trade).filter(
                Trade.status == "closed",
                Trade.is_winner.isnot(None)
            ).order_by(Trade.exit_time.desc()).limit(100).all()

            with self._lock:
                # Reverse to get chronological order
                for trade in reversed(trades):
                    self._recent_trades.append((trade.exit_time, trade.is_winner))

            self._initialized = True
            logger.info(
                f"ThresholdManager initialized: {len(self._predictions_30d)} predictions, "
                f"{len(self._recent_trades)} trades loaded"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to initialize ThresholdManager: {e}")
            return False

        finally:
            if should_close:
                db.close()

    def record_prediction(
        self,
        pred_id: Optional[int],
        confidence: float,
        timestamp: Optional[datetime] = None
    ) -> None:
        """Record a prediction for future threshold calculations.

        Args:
            pred_id: Prediction database ID (can be None initially)
            confidence: Prediction confidence (0-1)
            timestamp: Prediction timestamp (defaults to now)
        """
        if timestamp is None:
            timestamp = datetime.utcnow()

        with self._lock:
            entry = (timestamp, confidence)
            self._predictions_7d.append(entry)
            self._predictions_14d.append(entry)
            self._predictions_30d.append(entry)

    def record_trade_outcome(
        self,
        trade_id: int,
        is_winner: bool,
        timestamp: Optional[datetime] = None
    ) -> None:
        """Record a trade outcome for performance feedback.

        Args:
            trade_id: Trade database ID
            is_winner: Whether trade was profitable
            timestamp: Trade exit timestamp (defaults to now)
        """
        if timestamp is None:
            timestamp = datetime.utcnow()

        with self._lock:
            self._recent_trades.append((timestamp, is_winner))

    def calculate_threshold(
        self,
        db: Optional[Session] = None,
        record_history: bool = True
    ) -> float:
        """Calculate dynamic confidence threshold.

        Algorithm:
        1. Extract confidence values for each window (7d, 14d, 30d)
        2. Calculate quantile (60th percentile) for each window
        3. Blend: 25% short + 60% medium + 15% long
        4. Calculate recent win rate (last 25 trades)
        5. Adjust: blended + (win_rate - target) * adjustment_factor
        6. Apply hard bounds (0.55-0.75)
        7. Apply divergence check (±0.08 from long-term)
        8. Record to history if requested

        Args:
            db: Database session for recording history (optional)
            record_history: Whether to persist calculation to database

        Returns:
            Calculated threshold (float between 0-1)
        """
        # Check if dynamic threshold is disabled
        if not self.use_dynamic:
            logger.debug(f"Dynamic threshold disabled, using static: {self.static_threshold}")
            return self.static_threshold

        with self._lock:
            # Extract confidences from each window
            confidences_7d = self._get_confidences_from_window(
                self._predictions_7d, self.short_term_days
            )
            confidences_14d = self._get_confidences_from_window(
                self._predictions_14d, self.medium_term_days
            )
            confidences_30d = self._get_confidences_from_window(
                self._predictions_30d, self.long_term_days
            )

            # Check if we have enough data
            if len(confidences_30d) < self.min_predictions:
                reason = f"insufficient_data_{len(confidences_30d)}_predictions"
                logger.info(
                    f"Insufficient predictions ({len(confidences_30d)} < {self.min_predictions}), "
                    f"using static threshold: {self.static_threshold}"
                )

                if record_history:
                    self._record_threshold_history(
                        threshold=self.static_threshold,
                        components=(None, None, None),
                        blended=None,
                        adjustment=0.0,
                        counts=(len(confidences_7d), len(confidences_14d), len(confidences_30d)),
                        win_rate=None,
                        trade_count=len(self._recent_trades),
                        reason=reason,
                        db=db
                    )

                self._current_threshold = self.static_threshold
                self._last_calculation = datetime.utcnow()
                self._calculation_count += 1
                return self.static_threshold

            # Calculate quantile for each window
            short_term = np.percentile(confidences_7d, self.quantile * 100) if confidences_7d else self.static_threshold
            medium_term = np.percentile(confidences_14d, self.quantile * 100) if confidences_14d else self.static_threshold
            long_term = np.percentile(confidences_30d, self.quantile * 100)

            # Blend components
            blended = (
                self.short_weight * short_term +
                self.medium_weight * medium_term +
                self.long_weight * long_term
            )

            # Calculate performance adjustment
            adjustment = 0.0
            win_rate = None
            trade_count = len(self._recent_trades)

            if trade_count >= self.min_trades:
                # Get recent trades within lookback window
                recent_outcomes = list(self._recent_trades)[-self.perf_lookback:]
                wins = sum(1 for _, is_winner in recent_outcomes if is_winner)
                win_rate = wins / len(recent_outcomes)

                # Calculate adjustment
                win_rate_delta = win_rate - self.target_win_rate
                adjustment = win_rate_delta * self.adjustment_factor

                logger.debug(
                    f"Performance adjustment: win_rate={win_rate:.3f}, "
                    f"target={self.target_win_rate:.3f}, delta={win_rate_delta:.3f}, "
                    f"adjustment={adjustment:+.3f}"
                )

            # Apply adjustment
            dynamic_threshold = blended + adjustment

            # Apply hard bounds
            dynamic_threshold = np.clip(dynamic_threshold, self.min_threshold, self.max_threshold)

            # Apply divergence check (prevent too much deviation from long-term)
            min_allowed = long_term - self.max_divergence
            max_allowed = long_term + self.max_divergence
            dynamic_threshold = np.clip(dynamic_threshold, min_allowed, max_allowed)

            # Record to history
            if record_history:
                self._record_threshold_history(
                    threshold=dynamic_threshold,
                    components=(short_term, medium_term, long_term),
                    blended=blended,
                    adjustment=adjustment,
                    counts=(len(confidences_7d), len(confidences_14d), len(confidences_30d)),
                    win_rate=win_rate,
                    trade_count=trade_count,
                    reason="dynamic",
                    db=db
                )

            # Update cache
            self._current_threshold = dynamic_threshold
            self._last_calculation = datetime.utcnow()
            self._calculation_count += 1

            logger.debug(
                f"Calculated dynamic threshold: {dynamic_threshold:.4f} "
                f"(short={short_term:.4f}, med={medium_term:.4f}, long={long_term:.4f}, "
                f"blended={blended:.4f}, adj={adjustment:+.4f})"
            )

            return dynamic_threshold

    def _get_confidences_from_window(
        self,
        window_deque: deque,
        max_days: int
    ) -> list:
        """Extract valid confidences from a time window.

        Args:
            window_deque: Deque containing (timestamp, confidence) tuples
            max_days: Maximum age in days

        Returns:
            List of confidence values within time window
        """
        if not window_deque:
            return []

        cutoff = datetime.utcnow() - timedelta(days=max_days)
        confidences = [
            conf for ts, conf in window_deque
            if ts >= cutoff
        ]
        return confidences

    def _record_threshold_history(
        self,
        threshold: float,
        components: Tuple[Optional[float], Optional[float], Optional[float]],
        blended: Optional[float],
        adjustment: float,
        counts: Tuple[int, int, int],
        win_rate: Optional[float],
        trade_count: int,
        reason: str,
        db: Optional[Session]
    ) -> None:
        """Persist threshold calculation to database history.

        Args:
            threshold: Final calculated threshold
            components: (short_term, medium_term, long_term) components
            blended: Blended value before adjustment
            adjustment: Performance adjustment applied
            counts: (count_7d, count_14d, count_30d) prediction counts
            win_rate: Recent win rate (or None)
            trade_count: Number of recent trades
            reason: Reason string for this calculation
            db: Database session (optional)
        """
        should_close = db is None
        if db is None:
            db = get_session()

        try:
            short, medium, long = components
            count_7d, count_14d, count_30d = counts

            history_record = ThresholdHistory(
                timestamp=datetime.utcnow(),
                threshold_value=threshold,
                short_term_component=short,
                medium_term_component=medium,
                long_term_component=long,
                blended_value=blended,
                performance_adjustment=adjustment,
                prediction_count_7d=count_7d,
                prediction_count_14d=count_14d,
                prediction_count_30d=count_30d,
                trade_win_rate_25=win_rate,
                trade_count_25=trade_count,
                reason=reason,
                config_version=trading_config.get_config_version()
            )

            db.add(history_record)
            db.commit()

        except Exception as e:
            logger.error(f"Failed to record threshold history: {e}")
            db.rollback()

        finally:
            if should_close:
                db.close()

    def get_current_threshold(self) -> Optional[float]:
        """Get the most recently calculated threshold (cached).

        Returns:
            Current threshold or None if not yet calculated
        """
        with self._lock:
            return self._current_threshold

    def get_status(self) -> Dict[str, Any]:
        """Get service status and metrics.

        Returns:
            Dictionary with status information
        """
        with self._lock:
            return {
                "initialized": self._initialized,
                "use_dynamic": self.use_dynamic,
                "current_threshold": self._current_threshold,
                "last_calculation": (
                    self._last_calculation.isoformat()
                    if self._last_calculation else None
                ),
                "calculation_count": self._calculation_count,
                "predictions_7d": len(self._predictions_7d),
                "predictions_14d": len(self._predictions_14d),
                "predictions_30d": len(self._predictions_30d),
                "recent_trades": len(self._recent_trades),
                "static_fallback": self.static_threshold,
                "config": {
                    "windows": f"{self.short_term_days}d/{self.medium_term_days}d/{self.long_term_days}d",
                    "weights": f"{self.short_weight:.0%}/{self.medium_weight:.0%}/{self.long_weight:.0%}",
                    "quantile": self.quantile,
                    "bounds": f"{self.min_threshold:.2f}-{self.max_threshold:.2f}",
                    "target_win_rate": self.target_win_rate,
                }
            }

    def get_recent_history(
        self,
        limit: int = 100,
        db: Optional[Session] = None
    ) -> list:
        """Get recent threshold calculation history.

        Args:
            limit: Maximum number of records to return
            db: Database session (optional)

        Returns:
            List of threshold history dicts
        """
        should_close = db is None
        if db is None:
            db = get_session()

        try:
            records = db.query(ThresholdHistory).order_by(
                ThresholdHistory.timestamp.desc()
            ).limit(limit).all()

            return [
                {
                    "timestamp": rec.timestamp.isoformat(),
                    "threshold": rec.threshold_value,
                    "short_term": rec.short_term_component,
                    "medium_term": rec.medium_term_component,
                    "long_term": rec.long_term_component,
                    "blended": rec.blended_value,
                    "adjustment": rec.performance_adjustment,
                    "predictions_7d": rec.prediction_count_7d,
                    "predictions_14d": rec.prediction_count_14d,
                    "predictions_30d": rec.prediction_count_30d,
                    "win_rate": rec.trade_win_rate_25,
                    "trade_count": rec.trade_count_25,
                    "reason": rec.reason,
                }
                for rec in records
            ]

        finally:
            if should_close:
                db.close()


# Singleton instance
threshold_service = ThresholdManager()
