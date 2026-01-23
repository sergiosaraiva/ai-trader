"""Metrics and telemetry for the trading agent.

Provides observability into agent operations, performance, and health.
"""

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, Optional, Deque

logger = logging.getLogger(__name__)


@dataclass
class CycleMetrics:
    """Metrics for a single trading cycle."""

    cycle_number: int
    start_time: datetime
    duration_ms: float
    prediction_made: bool
    signal_generated: bool
    trade_executed: bool
    error: Optional[str] = None


@dataclass
class TradeMetrics:
    """Metrics for a completed trade."""

    trade_id: int
    pnl_pips: float
    pnl_usd: float
    duration_seconds: float
    is_winner: bool
    exit_reason: str


class AgentMetrics:
    """Collects and exposes metrics for the trading agent.

    Thread-safe metrics collection with rolling windows for recent data.

    Features:
    - Cycle timing statistics
    - Trade performance tracking
    - Error rate monitoring
    - Broker connection health
    - Resource usage tracking
    """

    def __init__(self, max_history: int = 1000):
        """Initialize metrics collector.

        Args:
            max_history: Maximum number of entries to keep in rolling windows
        """
        self._lock = threading.Lock()
        self._max_history = max_history

        # Counters
        self._total_cycles = 0
        self._total_predictions = 0
        self._total_signals = 0
        self._total_trades = 0
        self._total_errors = 0

        # Rolling windows
        self._cycle_history: Deque[CycleMetrics] = deque(maxlen=max_history)
        self._trade_history: Deque[TradeMetrics] = deque(maxlen=max_history)
        self._error_history: Deque[Dict[str, Any]] = deque(maxlen=100)

        # Timing
        self._start_time: Optional[datetime] = None
        self._last_cycle_time: Optional[datetime] = None

        # Broker metrics
        self._broker_reconnects = 0
        self._broker_errors = 0
        self._last_broker_error: Optional[str] = None

        # Database metrics
        self._db_timeouts = 0
        self._db_errors = 0
        self._orphaned_trades = 0

        logger.info("AgentMetrics initialized")

    def start(self) -> None:
        """Mark agent start for uptime tracking."""
        with self._lock:
            self._start_time = datetime.now()

    def stop(self) -> None:
        """Mark agent stop."""
        with self._lock:
            self._start_time = None

    def record_cycle(self, metrics: CycleMetrics) -> None:
        """Record metrics for a completed cycle.

        Args:
            metrics: Cycle metrics to record
        """
        with self._lock:
            self._total_cycles += 1
            self._last_cycle_time = datetime.now()

            if metrics.prediction_made:
                self._total_predictions += 1
            if metrics.signal_generated:
                self._total_signals += 1
            if metrics.trade_executed:
                self._total_trades += 1
            if metrics.error:
                self._total_errors += 1
                self._error_history.append({
                    "cycle": metrics.cycle_number,
                    "error": metrics.error,
                    "time": datetime.now().isoformat(),
                })

            self._cycle_history.append(metrics)

    def record_trade(self, metrics: TradeMetrics) -> None:
        """Record metrics for a completed trade.

        Args:
            metrics: Trade metrics to record
        """
        with self._lock:
            self._trade_history.append(metrics)

    def record_broker_reconnect(self) -> None:
        """Record a broker reconnection attempt."""
        with self._lock:
            self._broker_reconnects += 1

    def record_broker_error(self, error: str) -> None:
        """Record a broker error.

        Args:
            error: Error message
        """
        with self._lock:
            self._broker_errors += 1
            self._last_broker_error = error

    def record_db_timeout(self) -> None:
        """Record a database timeout."""
        with self._lock:
            self._db_timeouts += 1

    def record_db_error(self) -> None:
        """Record a database error."""
        with self._lock:
            self._db_errors += 1

    def record_orphaned_trade(self) -> None:
        """Record an orphaned trade."""
        with self._lock:
            self._orphaned_trades += 1

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of all metrics.

        Returns:
            Dictionary with metric summaries
        """
        with self._lock:
            # Calculate cycle timing stats
            cycle_durations = [c.duration_ms for c in self._cycle_history]
            avg_cycle_ms = sum(cycle_durations) / len(cycle_durations) if cycle_durations else 0
            max_cycle_ms = max(cycle_durations) if cycle_durations else 0
            min_cycle_ms = min(cycle_durations) if cycle_durations else 0

            # Calculate trade stats
            trade_pnls = [t.pnl_pips for t in self._trade_history]
            winners = [t for t in self._trade_history if t.is_winner]
            win_rate = len(winners) / len(self._trade_history) if self._trade_history else 0

            # Calculate uptime
            uptime_seconds = None
            if self._start_time:
                uptime_seconds = (datetime.now() - self._start_time).total_seconds()

            # Calculate error rate (last 100 cycles)
            recent_cycles = list(self._cycle_history)[-100:]
            recent_errors = sum(1 for c in recent_cycles if c.error)
            error_rate = recent_errors / len(recent_cycles) if recent_cycles else 0

            return {
                "uptime_seconds": uptime_seconds,
                "counters": {
                    "total_cycles": self._total_cycles,
                    "total_predictions": self._total_predictions,
                    "total_signals": self._total_signals,
                    "total_trades": self._total_trades,
                    "total_errors": self._total_errors,
                },
                "cycle_timing": {
                    "avg_duration_ms": round(avg_cycle_ms, 2),
                    "max_duration_ms": round(max_cycle_ms, 2),
                    "min_duration_ms": round(min_cycle_ms, 2),
                    "last_cycle_at": self._last_cycle_time.isoformat() if self._last_cycle_time else None,
                },
                "trades": {
                    "total": len(self._trade_history),
                    "win_rate": round(win_rate, 4),
                    "total_pips": round(sum(trade_pnls), 2),
                    "avg_pips": round(sum(trade_pnls) / len(trade_pnls), 2) if trade_pnls else 0,
                },
                "errors": {
                    "total": self._total_errors,
                    "rate_last_100": round(error_rate, 4),
                    "recent": list(self._error_history)[-5:],
                },
                "broker": {
                    "reconnects": self._broker_reconnects,
                    "errors": self._broker_errors,
                    "last_error": self._last_broker_error,
                },
                "database": {
                    "timeouts": self._db_timeouts,
                    "errors": self._db_errors,
                    "orphaned_trades": self._orphaned_trades,
                },
            }

    def get_health_indicators(self) -> Dict[str, bool]:
        """Get health indicators for monitoring.

        Returns:
            Dictionary with health indicators (True = healthy)
        """
        with self._lock:
            # Calculate error rate (last 100 cycles)
            recent_cycles = list(self._cycle_history)[-100:]
            recent_errors = sum(1 for c in recent_cycles if c.error)
            error_rate = recent_errors / len(recent_cycles) if recent_cycles else 0

            # Check cycle freshness
            cycle_fresh = True
            if self._start_time and self._last_cycle_time:
                seconds_since_cycle = (datetime.now() - self._last_cycle_time).total_seconds()
                cycle_fresh = seconds_since_cycle < 300  # 5 minutes

            return {
                "running": self._start_time is not None,
                "low_error_rate": error_rate < 0.1,  # Less than 10% errors
                "recent_cycle": cycle_fresh,
                "broker_stable": self._broker_errors < 10,  # Less than 10 broker errors
                "db_stable": self._db_timeouts < 10,  # Less than 10 DB timeouts
            }

    def reset(self) -> None:
        """Reset all metrics to initial state."""
        with self._lock:
            self._total_cycles = 0
            self._total_predictions = 0
            self._total_signals = 0
            self._total_trades = 0
            self._total_errors = 0
            self._cycle_history.clear()
            self._trade_history.clear()
            self._error_history.clear()
            self._start_time = None
            self._last_cycle_time = None
            self._broker_reconnects = 0
            self._broker_errors = 0
            self._last_broker_error = None
            self._db_timeouts = 0
            self._db_errors = 0
            self._orphaned_trades = 0

        logger.info("AgentMetrics reset")


# Singleton instance
agent_metrics = AgentMetrics()
