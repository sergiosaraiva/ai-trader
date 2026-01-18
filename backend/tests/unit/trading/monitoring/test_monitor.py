"""Tests for trading monitoring system."""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
import tempfile
import os

from src.trading.monitoring.monitor import (
    TradingMonitor,
    MonitoringConfig,
    MetricType,
    Alert,
    AlertLevel,
    AlertChannel,
    AlertThreshold,
    MetricValue,
)


class TestMetricType:
    """Tests for MetricType enum."""

    def test_performance_metrics(self):
        """Test performance metric types."""
        assert MetricType.EQUITY.value == "equity"
        assert MetricType.BALANCE.value == "balance"
        assert MetricType.PNL.value == "pnl"
        assert MetricType.DRAWDOWN.value == "drawdown"

    def test_trading_metrics(self):
        """Test trading metric types."""
        assert MetricType.TRADES_COUNT.value == "trades_count"
        assert MetricType.WIN_RATE.value == "win_rate"
        assert MetricType.PROFIT_FACTOR.value == "profit_factor"

    def test_risk_metrics(self):
        """Test risk metric types."""
        assert MetricType.POSITION_VALUE.value == "position_value"
        assert MetricType.EXPOSURE.value == "exposure"
        assert MetricType.MARGIN_LEVEL.value == "margin_level"


class TestAlertLevel:
    """Tests for AlertLevel enum."""

    def test_all_levels(self):
        """Test all alert levels."""
        assert AlertLevel.DEBUG.value == "debug"
        assert AlertLevel.INFO.value == "info"
        assert AlertLevel.WARNING.value == "warning"
        assert AlertLevel.ERROR.value == "error"
        assert AlertLevel.CRITICAL.value == "critical"


class TestAlert:
    """Tests for Alert dataclass."""

    def test_alert_creation(self):
        """Test alert creation."""
        alert = Alert(
            alert_id="ALERT_001",
            level=AlertLevel.WARNING,
            title="Test Alert",
            message="This is a test alert",
        )

        assert alert.alert_id == "ALERT_001"
        assert alert.level == AlertLevel.WARNING
        assert alert.title == "Test Alert"
        assert alert.acknowledged is False

    def test_to_dict(self):
        """Test alert to dictionary conversion."""
        alert = Alert(
            alert_id="ALERT_001",
            level=AlertLevel.ERROR,
            title="Test",
            message="Message",
            metric_type=MetricType.DRAWDOWN,
            metric_value=15.0,
            threshold=10.0,
        )

        d = alert.to_dict()

        assert d["alert_id"] == "ALERT_001"
        assert d["level"] == "error"
        assert d["metric_type"] == "drawdown"
        assert d["metric_value"] == 15.0


class TestAlertThreshold:
    """Tests for AlertThreshold."""

    def test_threshold_creation(self):
        """Test threshold creation."""
        threshold = AlertThreshold(
            metric_type=MetricType.DRAWDOWN,
            warning_threshold=5.0,
            error_threshold=10.0,
            critical_threshold=15.0,
        )

        assert threshold.metric_type == MetricType.DRAWDOWN
        assert threshold.warning_threshold == 5.0
        assert threshold.comparison == "gt"

    def test_check_warning(self):
        """Test threshold check returns warning."""
        threshold = AlertThreshold(
            metric_type=MetricType.DRAWDOWN,
            warning_threshold=5.0,
            error_threshold=10.0,
            critical_threshold=15.0,
        )

        level = threshold.check(7.0)
        assert level == AlertLevel.WARNING

    def test_check_error(self):
        """Test threshold check returns error."""
        threshold = AlertThreshold(
            metric_type=MetricType.DRAWDOWN,
            warning_threshold=5.0,
            error_threshold=10.0,
            critical_threshold=15.0,
        )

        level = threshold.check(12.0)
        assert level == AlertLevel.ERROR

    def test_check_critical(self):
        """Test threshold check returns critical."""
        threshold = AlertThreshold(
            metric_type=MetricType.DRAWDOWN,
            warning_threshold=5.0,
            error_threshold=10.0,
            critical_threshold=15.0,
        )

        level = threshold.check(20.0)
        assert level == AlertLevel.CRITICAL

    def test_check_no_alert(self):
        """Test threshold check returns None."""
        threshold = AlertThreshold(
            metric_type=MetricType.DRAWDOWN,
            warning_threshold=5.0,
            error_threshold=10.0,
            critical_threshold=15.0,
        )

        level = threshold.check(3.0)
        assert level is None

    def test_check_less_than(self):
        """Test threshold with less than comparison."""
        threshold = AlertThreshold(
            metric_type=MetricType.MARGIN_LEVEL,
            warning_threshold=200.0,
            error_threshold=150.0,
            critical_threshold=100.0,
            comparison="lt",
        )

        # 180 < 200 (warning threshold)
        assert threshold.check(180.0) == AlertLevel.WARNING
        # 120 < 150 (error threshold)
        assert threshold.check(120.0) == AlertLevel.ERROR
        # 80 < 100 (critical threshold)
        assert threshold.check(80.0) == AlertLevel.CRITICAL
        # 250 not < any threshold
        assert threshold.check(250.0) is None


class TestMonitoringConfig:
    """Tests for MonitoringConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = MonitoringConfig()

        assert config.metric_history_size == 1000
        assert config.alert_history_size == 100
        assert config.enable_alerts is True
        assert AlertChannel.LOG in config.alert_channels

    def test_custom_config(self):
        """Test custom configuration."""
        config = MonitoringConfig(
            metric_history_size=500,
            enable_alerts=False,
            webhook_url="https://example.com/webhook",
        )

        assert config.metric_history_size == 500
        assert config.enable_alerts is False
        assert config.webhook_url == "https://example.com/webhook"


class TestTradingMonitor:
    """Tests for TradingMonitor."""

    @pytest.fixture
    def monitor(self):
        """Create monitor for testing."""
        config = MonitoringConfig(
            enable_alerts=True,
            alert_channels=[AlertChannel.LOG],
        )
        return TradingMonitor(config)

    def test_initialization(self, monitor):
        """Test monitor initialization."""
        assert monitor._running is False
        assert len(monitor._alerts) == 0

    def test_start_stop(self, monitor):
        """Test start and stop."""
        monitor.start()
        assert monitor._running is True

        monitor.stop()
        assert monitor._running is False

    def test_record_metric(self, monitor):
        """Test recording a metric."""
        monitor.record_metric(MetricType.EQUITY, 105000.0)

        value = monitor.get_metric(MetricType.EQUITY)
        assert value == 105000.0

    def test_record_multiple_metrics(self, monitor):
        """Test recording multiple metrics."""
        monitor.record_metric(MetricType.EQUITY, 100000.0)
        monitor.record_metric(MetricType.EQUITY, 101000.0)
        monitor.record_metric(MetricType.EQUITY, 102000.0)

        value = monitor.get_metric(MetricType.EQUITY)
        assert value == 102000.0  # Latest value

    def test_record_custom_metric(self, monitor):
        """Test recording a custom metric."""
        monitor.record_metric(
            MetricType.CUSTOM,
            42.0,
            custom_name="my_custom_metric",
        )

        value = monitor.get_metric(MetricType.CUSTOM, custom_name="my_custom_metric")
        assert value == 42.0

    def test_get_metric_history(self, monitor):
        """Test getting metric history."""
        for i in range(5):
            monitor.record_metric(MetricType.PNL, float(i * 100))

        history = monitor.get_metric_history(MetricType.PNL)

        assert len(history) == 5
        assert history[0].value == 0.0
        assert history[4].value == 400.0

    def test_get_metric_not_found(self, monitor):
        """Test getting non-existent metric."""
        value = monitor.get_metric(MetricType.SHARPE)
        assert value is None

    def test_send_alert(self, monitor):
        """Test sending an alert."""
        alert = monitor.send_alert(
            level=AlertLevel.WARNING,
            title="Test Alert",
            message="This is a test",
        )

        assert alert.alert_id.startswith("ALERT_")
        assert alert.level == AlertLevel.WARNING
        assert len(monitor._alerts) == 1

    def test_send_alert_with_metric(self, monitor):
        """Test sending alert with metric info."""
        alert = monitor.send_alert(
            level=AlertLevel.ERROR,
            title="Drawdown Alert",
            message="Drawdown exceeded threshold",
            metric_type=MetricType.DRAWDOWN,
            metric_value=12.0,
            threshold=10.0,
        )

        assert alert.metric_type == MetricType.DRAWDOWN
        assert alert.metric_value == 12.0
        assert alert.threshold == 10.0

    def test_acknowledge_alert(self, monitor):
        """Test acknowledging an alert."""
        alert = monitor.send_alert(
            level=AlertLevel.WARNING,
            title="Test",
            message="Test message",
        )

        success = monitor.acknowledge_alert(alert.alert_id)

        assert success is True
        assert alert.acknowledged is True
        assert alert.acknowledged_at is not None

    def test_acknowledge_nonexistent_alert(self, monitor):
        """Test acknowledging non-existent alert."""
        success = monitor.acknowledge_alert("NONEXISTENT")
        assert success is False

    def test_get_alerts(self, monitor):
        """Test getting alerts."""
        monitor.send_alert(AlertLevel.INFO, "Info", "Info message")
        monitor.send_alert(AlertLevel.WARNING, "Warning", "Warning message")
        monitor.send_alert(AlertLevel.ERROR, "Error", "Error message")

        all_alerts = monitor.get_alerts()
        assert len(all_alerts) == 3

        warning_alerts = monitor.get_alerts(level=AlertLevel.WARNING)
        assert len(warning_alerts) == 1

    def test_get_unacknowledged_alerts(self, monitor):
        """Test getting only unacknowledged alerts."""
        alert1 = monitor.send_alert(AlertLevel.WARNING, "Alert 1", "Message 1")
        monitor.send_alert(AlertLevel.WARNING, "Alert 2", "Message 2")

        monitor.acknowledge_alert(alert1.alert_id)

        unack = monitor.get_alerts(unacknowledged_only=True)
        assert len(unack) == 1
        assert unack[0].title == "Alert 2"

    def test_threshold_triggers_alert(self, monitor):
        """Test that threshold triggers alert."""
        monitor.add_threshold(AlertThreshold(
            metric_type=MetricType.DRAWDOWN,
            warning_threshold=5.0,
            cooldown_seconds=0.0,  # No cooldown for test
        ))

        monitor.record_metric(MetricType.DRAWDOWN, 7.0)

        alerts = monitor.get_alerts()
        assert len(alerts) == 1
        assert alerts[0].level == AlertLevel.WARNING

    def test_threshold_cooldown(self, monitor):
        """Test threshold cooldown prevents repeated alerts."""
        monitor.add_threshold(AlertThreshold(
            metric_type=MetricType.DRAWDOWN,
            warning_threshold=5.0,
            cooldown_seconds=60.0,
        ))

        monitor.record_metric(MetricType.DRAWDOWN, 7.0)  # First alert
        monitor.record_metric(MetricType.DRAWDOWN, 8.0)  # Should be suppressed

        alerts = monitor.get_alerts()
        assert len(alerts) == 1

    def test_alert_callback(self):
        """Test alert callback is called."""
        callback = Mock()
        config = MonitoringConfig(
            alert_channels=[AlertChannel.CALLBACK],
            alert_callbacks=[callback],
        )
        monitor = TradingMonitor(config)

        monitor.send_alert(AlertLevel.WARNING, "Test", "Message")

        callback.assert_called_once()
        alert_arg = callback.call_args[0][0]
        assert alert_arg.title == "Test"

    def test_get_dashboard_data(self, monitor):
        """Test getting dashboard data."""
        monitor.record_metric(MetricType.EQUITY, 105000.0)
        monitor.record_metric(MetricType.PNL, 5000.0)
        monitor.send_alert(AlertLevel.WARNING, "Test", "Message")

        data = monitor.get_dashboard_data()

        assert "timestamp" in data
        assert "metrics" in data
        assert "alerts" in data
        assert data["metrics"]["equity"] == 105000.0
        assert data["metrics"]["pnl"] == 5000.0
        assert len(data["alerts"]["recent"]) == 1

    def test_get_performance_summary(self, monitor):
        """Test getting performance summary."""
        monitor.record_metric(MetricType.PNL, 100.0)
        monitor.record_metric(MetricType.PNL, 200.0)
        monitor.record_metric(MetricType.PNL, 150.0)
        monitor.record_metric(MetricType.EQUITY, 100000.0)
        monitor.record_metric(MetricType.EQUITY, 105000.0)

        summary = monitor.get_performance_summary(period_hours=24.0)

        assert "pnl" in summary
        assert summary["pnl"]["total"] == 450.0
        assert summary["pnl"]["min"] == 100.0
        assert summary["pnl"]["max"] == 200.0

    def test_get_stats(self, monitor):
        """Test getting monitor statistics."""
        monitor.record_metric(MetricType.EQUITY, 100000.0)
        monitor.send_alert(AlertLevel.WARNING, "Test", "Message")

        stats = monitor.get_stats()

        assert stats["running"] is False
        assert stats["total_alerts"] == 1
        assert stats["unacknowledged_alerts"] == 1
        assert "metrics_tracked" in stats

    def test_file_alert_channel(self):
        """Test file alert channel."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = os.path.join(tmpdir, "alerts.log")
            config = MonitoringConfig(
                alert_channels=[AlertChannel.FILE],
                log_file_path=log_file,
            )
            monitor = TradingMonitor(config)

            monitor.send_alert(AlertLevel.WARNING, "Test Alert", "Test message")

            assert os.path.exists(log_file)
            with open(log_file) as f:
                content = f.read()
                assert "Test Alert" in content

    def test_metric_history_limit(self):
        """Test metric history is limited."""
        config = MonitoringConfig(metric_history_size=5)
        monitor = TradingMonitor(config)

        for i in range(10):
            monitor.record_metric(MetricType.PNL, float(i))

        history = monitor.get_metric_history(MetricType.PNL)

        assert len(history) == 5
        assert history[0].value == 5.0  # Oldest retained
        assert history[4].value == 9.0  # Latest

    def test_alert_history_limit(self):
        """Test alert history is limited."""
        config = MonitoringConfig(alert_history_size=3)
        monitor = TradingMonitor(config)

        for i in range(5):
            monitor.send_alert(AlertLevel.INFO, f"Alert {i}", "Message")

        alerts = monitor.get_alerts()

        assert len(alerts) == 3
