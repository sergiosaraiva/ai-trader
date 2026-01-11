"""
Trading Monitor System.

Real-time monitoring, metrics collection, and alerting for production trading.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Deque
from collections import deque
import threading
import asyncio
import json
import logging

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of metrics."""
    # Performance
    EQUITY = "equity"
    BALANCE = "balance"
    PNL = "pnl"
    PNL_PCT = "pnl_pct"
    DRAWDOWN = "drawdown"
    SHARPE = "sharpe"

    # Trading
    TRADES_COUNT = "trades_count"
    WIN_RATE = "win_rate"
    PROFIT_FACTOR = "profit_factor"
    AVERAGE_WIN = "average_win"
    AVERAGE_LOSS = "average_loss"

    # Risk
    POSITION_VALUE = "position_value"
    EXPOSURE = "exposure"
    MARGIN_USED = "margin_used"
    MARGIN_LEVEL = "margin_level"

    # System
    LATENCY = "latency"
    CONNECTION_STATUS = "connection_status"
    ORDER_FILL_RATE = "order_fill_rate"
    MODEL_CONFIDENCE = "model_confidence"

    # Custom
    CUSTOM = "custom"


class AlertLevel(Enum):
    """Alert severity levels."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertChannel(Enum):
    """Alert delivery channels."""
    LOG = "log"
    CALLBACK = "callback"
    FILE = "file"
    WEBHOOK = "webhook"
    EMAIL = "email"


@dataclass
class Alert:
    """Alert record."""
    alert_id: str
    level: AlertLevel
    title: str
    message: str
    timestamp: datetime = field(default_factory=datetime.now)
    metric_type: Optional[MetricType] = None
    metric_value: Optional[float] = None
    threshold: Optional[float] = None
    acknowledged: bool = False
    acknowledged_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "alert_id": self.alert_id,
            "level": self.level.value,
            "title": self.title,
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
            "metric_type": self.metric_type.value if self.metric_type else None,
            "metric_value": self.metric_value,
            "threshold": self.threshold,
            "acknowledged": self.acknowledged,
            "acknowledged_at": self.acknowledged_at.isoformat() if self.acknowledged_at else None,
        }


@dataclass
class MetricValue:
    """Single metric value with timestamp."""
    value: float
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class AlertThreshold:
    """Alert threshold configuration."""
    metric_type: MetricType
    warning_threshold: Optional[float] = None
    error_threshold: Optional[float] = None
    critical_threshold: Optional[float] = None
    comparison: str = "gt"  # gt, lt, eq, ne, gte, lte
    cooldown_seconds: float = 60.0  # Minimum time between alerts

    def check(self, value: float) -> Optional[AlertLevel]:
        """Check if value triggers an alert."""
        def compare(v: float, threshold: float) -> bool:
            if self.comparison == "gt":
                return v > threshold
            elif self.comparison == "lt":
                return v < threshold
            elif self.comparison == "gte":
                return v >= threshold
            elif self.comparison == "lte":
                return v <= threshold
            elif self.comparison == "eq":
                return v == threshold
            elif self.comparison == "ne":
                return v != threshold
            return False

        if self.critical_threshold is not None and compare(value, self.critical_threshold):
            return AlertLevel.CRITICAL
        if self.error_threshold is not None and compare(value, self.error_threshold):
            return AlertLevel.ERROR
        if self.warning_threshold is not None and compare(value, self.warning_threshold):
            return AlertLevel.WARNING
        return None


@dataclass
class MonitoringConfig:
    """Configuration for trading monitor."""
    # Metric retention
    metric_history_size: int = 1000
    metric_retention_hours: float = 24.0

    # Alert settings
    alert_history_size: int = 100
    enable_alerts: bool = True
    alert_channels: List[AlertChannel] = field(default_factory=lambda: [AlertChannel.LOG])

    # Webhook settings
    webhook_url: Optional[str] = None
    webhook_timeout_seconds: float = 5.0

    # File logging
    log_file_path: Optional[str] = None

    # Alert thresholds
    thresholds: List[AlertThreshold] = field(default_factory=list)

    # Dashboard
    dashboard_update_interval: float = 1.0

    # Alert callbacks
    alert_callbacks: List[Callable[[Alert], None]] = field(default_factory=list)


class TradingMonitor:
    """
    Real-time trading monitor.

    Features:
    - Metric collection and history
    - Configurable alert thresholds
    - Multiple alert channels (log, callback, webhook)
    - Dashboard data for UI
    - Performance summaries

    Usage:
        monitor = TradingMonitor(config)
        monitor.start()

        # Record metrics
        monitor.record_metric(MetricType.EQUITY, 105000.0)
        monitor.record_metric(MetricType.PNL_PCT, 5.0)

        # Send alerts
        monitor.send_alert(AlertLevel.WARNING, "High drawdown", "Drawdown at 10%")

        # Get dashboard data
        data = monitor.get_dashboard_data()

        monitor.stop()
    """

    def __init__(self, config: Optional[MonitoringConfig] = None):
        """
        Initialize trading monitor.

        Args:
            config: Monitoring configuration
        """
        self.config = config or MonitoringConfig()

        # Metric storage
        self._metrics: Dict[MetricType, Deque[MetricValue]] = {}
        self._custom_metrics: Dict[str, Deque[MetricValue]] = {}

        # Alert storage
        self._alerts: Deque[Alert] = deque(maxlen=self.config.alert_history_size)
        self._alert_counter = 0
        self._last_alert_times: Dict[MetricType, datetime] = {}

        # Thread safety
        self._lock = threading.RLock()

        # Background tasks
        self._running = False
        self._cleanup_task: Optional[asyncio.Task] = None

        # Initialize metric storage
        for metric_type in MetricType:
            self._metrics[metric_type] = deque(maxlen=self.config.metric_history_size)

        logger.info("Trading monitor initialized")

    def start(self) -> None:
        """Start the monitor."""
        self._running = True
        logger.info("Trading monitor started")

    def stop(self) -> None:
        """Stop the monitor."""
        self._running = False

        if self._cleanup_task:
            self._cleanup_task.cancel()
            self._cleanup_task = None

        logger.info("Trading monitor stopped")

    def record_metric(
        self,
        metric_type: MetricType,
        value: float,
        timestamp: Optional[datetime] = None,
        custom_name: Optional[str] = None,
    ) -> None:
        """
        Record a metric value.

        Args:
            metric_type: Type of metric
            value: Metric value
            timestamp: Optional timestamp (default: now)
            custom_name: Name for custom metrics
        """
        with self._lock:
            ts = timestamp or datetime.now()
            metric_value = MetricValue(value=value, timestamp=ts)

            if metric_type == MetricType.CUSTOM and custom_name:
                if custom_name not in self._custom_metrics:
                    self._custom_metrics[custom_name] = deque(maxlen=self.config.metric_history_size)
                self._custom_metrics[custom_name].append(metric_value)
            else:
                self._metrics[metric_type].append(metric_value)

            # Check thresholds
            if self.config.enable_alerts:
                self._check_thresholds(metric_type, value)

    def _check_thresholds(self, metric_type: MetricType, value: float) -> None:
        """Check if metric value triggers any alerts."""
        for threshold in self.config.thresholds:
            if threshold.metric_type != metric_type:
                continue

            # Check cooldown
            last_alert = self._last_alert_times.get(metric_type)
            if last_alert:
                elapsed = (datetime.now() - last_alert).total_seconds()
                if elapsed < threshold.cooldown_seconds:
                    continue

            # Check threshold
            level = threshold.check(value)
            if level:
                self._last_alert_times[metric_type] = datetime.now()
                self.send_alert(
                    level=level,
                    title=f"{metric_type.value} threshold exceeded",
                    message=f"{metric_type.value} = {value:.4f}",
                    metric_type=metric_type,
                    metric_value=value,
                    threshold=getattr(threshold, f"{level.value}_threshold", None),
                )

    def send_alert(
        self,
        level: AlertLevel,
        title: str,
        message: str,
        metric_type: Optional[MetricType] = None,
        metric_value: Optional[float] = None,
        threshold: Optional[float] = None,
    ) -> Alert:
        """
        Send an alert.

        Args:
            level: Alert severity
            title: Alert title
            message: Alert message
            metric_type: Related metric type
            metric_value: Current metric value
            threshold: Threshold that was exceeded

        Returns:
            Created alert
        """
        with self._lock:
            self._alert_counter += 1
            alert_id = f"ALERT_{self._alert_counter:06d}"

            alert = Alert(
                alert_id=alert_id,
                level=level,
                title=title,
                message=message,
                metric_type=metric_type,
                metric_value=metric_value,
                threshold=threshold,
            )

            self._alerts.append(alert)

            # Deliver to channels
            self._deliver_alert(alert)

            return alert

    def _deliver_alert(self, alert: Alert) -> None:
        """Deliver alert to configured channels."""
        for channel in self.config.alert_channels:
            try:
                if channel == AlertChannel.LOG:
                    self._log_alert(alert)
                elif channel == AlertChannel.CALLBACK:
                    self._callback_alert(alert)
                elif channel == AlertChannel.FILE:
                    self._file_alert(alert)
                elif channel == AlertChannel.WEBHOOK:
                    self._webhook_alert(alert)
            except Exception as e:
                logger.error(f"Failed to deliver alert via {channel.value}: {e}")

    def _log_alert(self, alert: Alert) -> None:
        """Log alert."""
        log_levels = {
            AlertLevel.DEBUG: logging.DEBUG,
            AlertLevel.INFO: logging.INFO,
            AlertLevel.WARNING: logging.WARNING,
            AlertLevel.ERROR: logging.ERROR,
            AlertLevel.CRITICAL: logging.CRITICAL,
        }
        log_level = log_levels.get(alert.level, logging.INFO)
        logger.log(log_level, f"[{alert.alert_id}] {alert.title}: {alert.message}")

    def _callback_alert(self, alert: Alert) -> None:
        """Send alert to callbacks."""
        for callback in self.config.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Alert callback error: {e}")

    def _file_alert(self, alert: Alert) -> None:
        """Write alert to file."""
        if not self.config.log_file_path:
            return

        try:
            with open(self.config.log_file_path, "a") as f:
                f.write(json.dumps(alert.to_dict()) + "\n")
        except Exception as e:
            logger.error(f"Failed to write alert to file: {e}")

    def _webhook_alert(self, alert: Alert) -> None:
        """Send alert via webhook."""
        if not self.config.webhook_url:
            return

        try:
            import urllib.request
            import urllib.error

            data = json.dumps(alert.to_dict()).encode("utf-8")
            req = urllib.request.Request(
                self.config.webhook_url,
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )

            with urllib.request.urlopen(req, timeout=self.config.webhook_timeout_seconds) as response:
                if response.status != 200:
                    logger.warning(f"Webhook returned status {response.status}")

        except Exception as e:
            logger.error(f"Failed to send webhook alert: {e}")

    def acknowledge_alert(self, alert_id: str) -> bool:
        """
        Acknowledge an alert.

        Args:
            alert_id: Alert ID to acknowledge

        Returns:
            True if acknowledged
        """
        with self._lock:
            for alert in self._alerts:
                if alert.alert_id == alert_id:
                    alert.acknowledged = True
                    alert.acknowledged_at = datetime.now()
                    return True
            return False

    def get_metric(
        self,
        metric_type: MetricType,
        custom_name: Optional[str] = None,
    ) -> Optional[float]:
        """
        Get latest metric value.

        Args:
            metric_type: Type of metric
            custom_name: Name for custom metrics

        Returns:
            Latest value or None
        """
        with self._lock:
            if metric_type == MetricType.CUSTOM and custom_name:
                history = self._custom_metrics.get(custom_name)
            else:
                history = self._metrics.get(metric_type)

            if history and len(history) > 0:
                return history[-1].value
            return None

    def get_metric_history(
        self,
        metric_type: MetricType,
        custom_name: Optional[str] = None,
        limit: int = 100,
    ) -> List[MetricValue]:
        """
        Get metric history.

        Args:
            metric_type: Type of metric
            custom_name: Name for custom metrics
            limit: Maximum number of values

        Returns:
            List of metric values
        """
        with self._lock:
            if metric_type == MetricType.CUSTOM and custom_name:
                history = self._custom_metrics.get(custom_name, deque())
            else:
                history = self._metrics.get(metric_type, deque())

            return list(history)[-limit:]

    def get_alerts(
        self,
        level: Optional[AlertLevel] = None,
        unacknowledged_only: bool = False,
        limit: int = 50,
    ) -> List[Alert]:
        """
        Get alerts.

        Args:
            level: Filter by level
            unacknowledged_only: Only return unacknowledged alerts
            limit: Maximum number of alerts

        Returns:
            List of alerts
        """
        with self._lock:
            alerts = list(self._alerts)

            if level:
                alerts = [a for a in alerts if a.level == level]

            if unacknowledged_only:
                alerts = [a for a in alerts if not a.acknowledged]

            return alerts[-limit:]

    def get_dashboard_data(self) -> Dict[str, Any]:
        """
        Get data for monitoring dashboard.

        Returns:
            Dashboard data dictionary
        """
        with self._lock:
            # Get latest values for key metrics
            latest_metrics = {}
            for metric_type in [
                MetricType.EQUITY,
                MetricType.BALANCE,
                MetricType.PNL,
                MetricType.PNL_PCT,
                MetricType.DRAWDOWN,
                MetricType.TRADES_COUNT,
                MetricType.WIN_RATE,
                MetricType.POSITION_VALUE,
                MetricType.MARGIN_LEVEL,
                MetricType.LATENCY,
            ]:
                value = self.get_metric(metric_type)
                if value is not None:
                    latest_metrics[metric_type.value] = value

            # Get recent alerts
            recent_alerts = self.get_alerts(limit=10)

            # Get unacknowledged alert count by level
            alert_counts = {level.value: 0 for level in AlertLevel}
            for alert in self._alerts:
                if not alert.acknowledged:
                    alert_counts[alert.level.value] += 1

            # Build dashboard data
            return {
                "timestamp": datetime.now().isoformat(),
                "metrics": latest_metrics,
                "alerts": {
                    "recent": [a.to_dict() for a in recent_alerts],
                    "unacknowledged_counts": alert_counts,
                    "total_count": len(self._alerts),
                },
                "status": {
                    "running": self._running,
                    "metric_types_tracked": len([m for m in self._metrics.values() if len(m) > 0]),
                    "custom_metrics_tracked": len(self._custom_metrics),
                },
            }

    def get_performance_summary(
        self,
        period_hours: float = 24.0,
    ) -> Dict[str, Any]:
        """
        Get performance summary for a period.

        Args:
            period_hours: Period to summarize

        Returns:
            Performance summary
        """
        with self._lock:
            cutoff = datetime.now() - timedelta(hours=period_hours)

            def get_period_values(metric_type: MetricType) -> List[float]:
                history = self._metrics.get(metric_type, deque())
                return [m.value for m in history if m.timestamp >= cutoff]

            # Calculate statistics
            pnl_values = get_period_values(MetricType.PNL)
            equity_values = get_period_values(MetricType.EQUITY)

            summary = {
                "period_hours": period_hours,
                "start_time": cutoff.isoformat(),
                "end_time": datetime.now().isoformat(),
            }

            if pnl_values:
                summary["pnl"] = {
                    "total": sum(pnl_values),
                    "min": min(pnl_values),
                    "max": max(pnl_values),
                    "avg": sum(pnl_values) / len(pnl_values),
                }

            if equity_values:
                summary["equity"] = {
                    "start": equity_values[0] if equity_values else 0,
                    "end": equity_values[-1] if equity_values else 0,
                    "min": min(equity_values),
                    "max": max(equity_values),
                    "change_pct": (
                        ((equity_values[-1] - equity_values[0]) / equity_values[0] * 100)
                        if equity_values and equity_values[0] != 0 else 0
                    ),
                }

            # Trade stats
            trades = get_period_values(MetricType.TRADES_COUNT)
            if trades:
                summary["trades"] = {
                    "count": int(trades[-1] - trades[0]) if len(trades) > 1 else 0,
                }

            return summary

    def add_threshold(self, threshold: AlertThreshold) -> None:
        """Add an alert threshold."""
        self.config.thresholds.append(threshold)

    def add_alert_callback(self, callback: Callable[[Alert], None]) -> None:
        """Add an alert callback."""
        self.config.alert_callbacks.append(callback)

    def get_stats(self) -> Dict[str, Any]:
        """Get monitor statistics."""
        with self._lock:
            return {
                "running": self._running,
                "total_alerts": len(self._alerts),
                "unacknowledged_alerts": len([a for a in self._alerts if not a.acknowledged]),
                "metrics_tracked": {
                    mt.value: len(self._metrics[mt])
                    for mt in MetricType
                    if len(self._metrics.get(mt, [])) > 0
                },
                "custom_metrics": list(self._custom_metrics.keys()),
                "thresholds_configured": len(self.config.thresholds),
            }
