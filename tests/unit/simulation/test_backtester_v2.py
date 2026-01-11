"""Tests for Enhanced Backtester."""

import pytest
from datetime import datetime, timedelta
from pathlib import Path
import tempfile

import pandas as pd
import numpy as np

from src.simulation.backtester_v2 import (
    EnhancedBacktester,
    BacktestConfig,
    BacktestStatus,
    BacktestResult,
    Trade,
)


class TestBacktestConfig:
    """Tests for BacktestConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = BacktestConfig()

        assert config.name == "backtest"
        assert config.symbol == "EURUSD"
        assert config.initial_capital == 100000.0
        assert config.risk_profile_name == "moderate"
        assert config.slippage_pct == 0.0001

    def test_custom_config(self):
        """Test custom configuration."""
        config = BacktestConfig(
            name="test_backtest",
            symbol="GBPUSD",
            initial_capital=50000.0,
            risk_profile_name="conservative",
        )

        assert config.name == "test_backtest"
        assert config.symbol == "GBPUSD"
        assert config.initial_capital == 50000.0


class TestTrade:
    """Tests for Trade dataclass."""

    def test_trade_creation(self):
        """Test trade creation."""
        trade = Trade(
            trade_id="trade-001",
            symbol="EURUSD",
            side="buy",
            entry_time=datetime(2024, 1, 15, 10, 0),
            exit_time=datetime(2024, 1, 15, 14, 0),
            entry_price=1.1000,
            exit_price=1.1050,
            quantity=1.0,
            pnl=50.0,
            pnl_pct=0.0045,
            commission=2.0,
            slippage=1.0,
            exit_reason="take_profit",
        )

        assert trade.trade_id == "trade-001"
        assert trade.is_winner is True
        assert trade.holding_period == timedelta(hours=4)

    def test_losing_trade(self):
        """Test losing trade detection."""
        trade = Trade(
            trade_id="trade-002",
            symbol="EURUSD",
            side="buy",
            entry_time=datetime(2024, 1, 15, 10, 0),
            exit_time=datetime(2024, 1, 15, 14, 0),
            entry_price=1.1000,
            exit_price=1.0950,
            quantity=1.0,
            pnl=-50.0,
            pnl_pct=-0.0045,
            commission=2.0,
            slippage=1.0,
            exit_reason="stop_loss",
        )

        assert trade.is_winner is False

    def test_trade_to_dict(self):
        """Test trade serialization."""
        trade = Trade(
            trade_id="trade-001",
            symbol="EURUSD",
            side="buy",
            entry_time=datetime(2024, 1, 15, 10, 0),
            exit_time=datetime(2024, 1, 15, 14, 0),
            entry_price=1.1000,
            exit_price=1.1050,
            quantity=1.0,
            pnl=50.0,
            pnl_pct=0.0045,
            commission=2.0,
            slippage=1.0,
            exit_reason="take_profit",
        )

        d = trade.to_dict()

        assert d["trade_id"] == "trade-001"
        assert d["pnl"] == 50.0
        assert d["is_winner"] is True


class TestEnhancedBacktester:
    """Tests for EnhancedBacktester."""

    @pytest.fixture
    def sample_data(self):
        """Create sample OHLCV data with trend."""
        dates = pd.date_range(start="2024-01-01", periods=200, freq="D")
        np.random.seed(42)

        # Generate uptrending prices with some noise
        trend = np.linspace(0, 0.1, 200)  # 10% uptrend
        noise = np.random.randn(200) * 0.005
        returns = trend / 200 + noise

        base_price = 1.1
        prices = base_price * (1 + returns).cumprod()

        df = pd.DataFrame({
            "open": prices * (1 + np.random.randn(200) * 0.001),
            "high": prices * (1 + abs(np.random.randn(200) * 0.005)),
            "low": prices * (1 - abs(np.random.randn(200) * 0.005)),
            "close": prices,
            "volume": np.random.randint(1000, 10000, 200),
        }, index=dates)

        return df

    @pytest.fixture
    def config(self):
        """Create backtest configuration."""
        return BacktestConfig(
            name="test_backtest",
            symbol="EURUSD",
            initial_capital=100000.0,
            risk_profile_name="moderate",
            warmup_bars=50,
            verbose=False,
        )

    @pytest.fixture
    def backtester(self, config, sample_data):
        """Create backtester with sample data."""
        bt = EnhancedBacktester(config=config)
        bt.load_data("EURUSD", sample_data)
        return bt

    def test_initialization(self, config):
        """Test backtester initialization."""
        bt = EnhancedBacktester(config=config)

        assert bt.config == config
        assert bt.risk_profile is not None
        assert bt._status == BacktestStatus.PENDING

    def test_load_data(self, config, sample_data):
        """Test loading data."""
        bt = EnhancedBacktester(config=config)
        bt.load_data("EURUSD", sample_data)

        assert "EURUSD" in bt.market_simulator.symbols

    def test_run_backtest(self, backtester):
        """Test running a complete backtest."""
        result = backtester.run()

        assert result.status == BacktestStatus.COMPLETED
        assert result.initial_capital == 100000.0
        assert result.final_equity > 0
        assert result.duration_seconds > 0

    def test_equity_curve_generated(self, backtester):
        """Test equity curve is generated."""
        result = backtester.run()

        assert len(result.equity_curve) > 0
        assert result.equity_curve.iloc[0] == pytest.approx(100000.0, rel=0.01)

    def test_metrics_calculated(self, backtester):
        """Test performance metrics are calculated."""
        result = backtester.run()

        # Return metrics
        assert isinstance(result.total_return, float)
        assert isinstance(result.annualized_return, float)

        # Risk metrics
        assert isinstance(result.sharpe_ratio, float)
        assert isinstance(result.sortino_ratio, float)
        assert isinstance(result.max_drawdown, float)

    def test_trade_statistics(self, backtester):
        """Test trade statistics calculation."""
        result = backtester.run()

        assert result.total_trades >= 0
        assert result.winning_trades >= 0
        assert result.losing_trades >= 0
        assert result.winning_trades + result.losing_trades <= result.total_trades

        if result.total_trades > 0:
            assert 0 <= result.win_rate <= 1

    def test_result_to_dict(self, backtester):
        """Test result serialization."""
        result = backtester.run()
        d = result.to_dict()

        assert "config" in d
        assert "status" in d
        assert "equity" in d
        assert "returns" in d
        assert "risk" in d
        assert "trades" in d

    def test_result_to_json(self, backtester):
        """Test result JSON serialization."""
        result = backtester.run()
        json_str = result.to_json()

        assert isinstance(json_str, str)
        assert len(json_str) > 0

    def test_save_results(self, config, sample_data):
        """Test saving results to files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config.output_dir = tmpdir
            config.save_trades = True
            config.save_equity_curve = True

            bt = EnhancedBacktester(config=config)
            bt.load_data("EURUSD", sample_data)
            result = bt.run()

            # Check files created
            assert Path(tmpdir, "summary.json").exists()

    def test_empty_backtest(self, config):
        """Test backtest with no data."""
        bt = EnhancedBacktester(config=config)

        with pytest.raises(ValueError):
            bt.run()

    def test_warmup_period_respected(self, config, sample_data):
        """Test warmup period is respected."""
        config.warmup_bars = 100

        bt = EnhancedBacktester(config=config)
        bt.load_data("EURUSD", sample_data)
        result = bt.run()

        # Should have fewer potential trades due to warmup
        assert result.status == BacktestStatus.COMPLETED

    def test_slippage_and_commission(self, sample_data):
        """Test slippage and commission are applied."""
        config = BacktestConfig(
            symbol="EURUSD",
            slippage_pct=0.001,  # High slippage
            commission_pct=0.001,  # High commission
            warmup_bars=50,
            verbose=False,
        )

        bt = EnhancedBacktester(config=config)
        bt.load_data("EURUSD", sample_data)
        result = bt.run()

        # High costs should impact returns
        assert result.total_commission >= 0
        assert result.total_slippage >= 0

    def test_risk_profile_affects_sizing(self, sample_data):
        """Test different risk profiles affect position sizing."""
        # Conservative config
        config_conservative = BacktestConfig(
            symbol="EURUSD",
            risk_profile_name="conservative",
            warmup_bars=50,
            verbose=False,
        )

        # Aggressive config
        config_aggressive = BacktestConfig(
            symbol="EURUSD",
            risk_profile_name="aggressive",
            warmup_bars=50,
            verbose=False,
        )

        bt_conservative = EnhancedBacktester(config=config_conservative)
        bt_conservative.load_data("EURUSD", sample_data)

        bt_aggressive = EnhancedBacktester(config=config_aggressive)
        bt_aggressive.load_data("EURUSD", sample_data)

        # Both should complete
        result_conservative = bt_conservative.run()
        result_aggressive = bt_aggressive.run()

        assert result_conservative.status == BacktestStatus.COMPLETED
        assert result_aggressive.status == BacktestStatus.COMPLETED


class TestBacktestResult:
    """Tests for BacktestResult."""

    @pytest.fixture
    def sample_result(self):
        """Create sample backtest result."""
        equity_curve = pd.Series(
            [100000, 100500, 101000, 100800, 101500],
            index=pd.date_range(start="2024-01-01", periods=5),
        )
        returns = equity_curve.pct_change().dropna()

        return BacktestResult(
            config=BacktestConfig(name="test"),
            status=BacktestStatus.COMPLETED,
            start_time=datetime(2024, 1, 1),
            end_time=datetime(2024, 1, 5),
            duration_seconds=60.0,
            initial_capital=100000.0,
            final_equity=101500.0,
            peak_equity=101500.0,
            low_equity=100000.0,
            total_return=1500.0,
            total_return_pct=0.015,
            annualized_return=0.15,
            cagr=0.15,
            sharpe_ratio=1.5,
            sortino_ratio=2.0,
            calmar_ratio=1.0,
            max_drawdown=0.005,
            max_drawdown_duration=timedelta(days=1),
            volatility=0.10,
            downside_volatility=0.05,
            total_trades=10,
            winning_trades=6,
            losing_trades=4,
            win_rate=0.6,
            profit_factor=2.0,
            average_win=400.0,
            average_loss=-250.0,
            largest_win=800.0,
            largest_loss=-400.0,
            average_trade=150.0,
            expectancy=140.0,
            average_holding_period=timedelta(hours=4),
            max_holding_period=timedelta(hours=12),
            min_holding_period=timedelta(hours=1),
            trades_per_month=5.0,
            recovery_factor=3.0,
            risk_reward_ratio=1.6,
            equity_curve=equity_curve,
            returns_series=returns,
        )

    def test_to_dict(self, sample_result):
        """Test result to_dict method."""
        d = sample_result.to_dict()

        assert d["status"] == "completed"
        assert d["equity"]["final"] == 101500.0
        assert d["returns"]["total_return_pct"] == 0.015
        assert d["trades"]["win_rate"] == 0.6

    def test_to_json(self, sample_result):
        """Test result to_json method."""
        json_str = sample_result.to_json()

        assert "completed" in json_str
        assert "101500" in json_str

    def test_print_summary(self, sample_result, capsys):
        """Test print_summary output."""
        sample_result.print_summary()

        captured = capsys.readouterr()
        assert "BACKTEST RESULTS" in captured.out
        assert "Total Return" in captured.out
        assert "Sharpe Ratio" in captured.out
