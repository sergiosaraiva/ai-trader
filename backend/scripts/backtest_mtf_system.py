#!/usr/bin/env python3
"""Backtest the multi-timeframe scalper system.

This script:
1. Loads trained MTF models
2. Simulates trading on historical data
3. Reports comprehensive performance metrics
"""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import List, Optional, Dict

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd

from src.models.multi_timeframe import (
    MultiTimeframeModel,
    TimeframeConfig,
    MTFPredictor,
    MTFSignalGenerator,
    ScalperConfig,
    ScalperSignal,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class Trade:
    """Completed trade record."""

    entry_time: datetime
    exit_time: datetime
    direction: str  # 'long' or 'short'
    entry_price: float
    exit_price: float
    stop_loss: float
    take_profit: float
    position_size: float
    pnl_pips: float
    pnl_pct: float
    exit_reason: str  # 'tp', 'sl', 'signal', 'timeout'
    confidence: float
    strength: str
    alignment: str


@dataclass
class BacktestResult:
    """Backtest results container."""

    trades: List[Trade] = field(default_factory=list)
    initial_equity: float = 10000.0
    final_equity: float = 10000.0
    equity_curve: List[float] = field(default_factory=list)

    @property
    def total_trades(self) -> int:
        return len(self.trades)

    @property
    def winning_trades(self) -> int:
        return sum(1 for t in self.trades if t.pnl_pct > 0)

    @property
    def losing_trades(self) -> int:
        return sum(1 for t in self.trades if t.pnl_pct <= 0)

    @property
    def win_rate(self) -> float:
        return self.winning_trades / self.total_trades if self.total_trades > 0 else 0

    @property
    def total_return(self) -> float:
        return (self.final_equity - self.initial_equity) / self.initial_equity

    @property
    def avg_win(self) -> float:
        wins = [t.pnl_pct for t in self.trades if t.pnl_pct > 0]
        return np.mean(wins) if wins else 0

    @property
    def avg_loss(self) -> float:
        losses = [t.pnl_pct for t in self.trades if t.pnl_pct <= 0]
        return np.mean(losses) if losses else 0

    @property
    def profit_factor(self) -> float:
        gross_profit = sum(t.pnl_pct for t in self.trades if t.pnl_pct > 0)
        gross_loss = abs(sum(t.pnl_pct for t in self.trades if t.pnl_pct <= 0))
        return gross_profit / gross_loss if gross_loss > 0 else float("inf")

    @property
    def max_drawdown(self) -> float:
        if not self.equity_curve:
            return 0
        peak = self.equity_curve[0]
        max_dd = 0
        for eq in self.equity_curve:
            if eq > peak:
                peak = eq
            dd = (peak - eq) / peak
            max_dd = max(max_dd, dd)
        return max_dd

    @property
    def sharpe_ratio(self) -> float:
        if len(self.trades) < 2:
            return 0
        returns = [t.pnl_pct for t in self.trades]
        if np.std(returns) == 0:
            return 0
        # Annualized Sharpe (assuming ~250 trading days, ~50 trades/day for scalper)
        return np.mean(returns) / np.std(returns) * np.sqrt(250 * 50)

    @property
    def avg_trade_duration(self) -> timedelta:
        if not self.trades:
            return timedelta(0)
        durations = [(t.exit_time - t.entry_time) for t in self.trades]
        avg_seconds = np.mean([d.total_seconds() for d in durations])
        return timedelta(seconds=avg_seconds)


class MTFBacktester:
    """Backtester for multi-timeframe scalper system."""

    def __init__(
        self,
        mtf_model: MultiTimeframeModel,
        predictor: MTFPredictor,
        signal_generator: MTFSignalGenerator,
        initial_equity: float = 10000.0,
        pip_value: float = 0.0001,
        spread_pips: float = 1.0,  # 1 pip spread
        max_bars_in_trade: int = 60,  # Max 5 hours for 5min bars
    ):
        """Initialize backtester.

        Args:
            mtf_model: Trained MTF model
            predictor: MTF predictor
            signal_generator: Signal generator
            initial_equity: Starting equity
            pip_value: Value of 1 pip
            spread_pips: Spread cost in pips
            max_bars_in_trade: Maximum bars before forced exit
        """
        self.mtf_model = mtf_model
        self.predictor = predictor
        self.signal_generator = signal_generator
        self.initial_equity = initial_equity
        self.pip_value = pip_value
        self.spread_pips = spread_pips
        self.max_bars_in_trade = max_bars_in_trade

    def run(
        self,
        df_5min: pd.DataFrame,
        lookback_bars: int = 100,
        progress_interval: int = 10000,
    ) -> BacktestResult:
        """Run backtest on 5-minute data.

        Args:
            df_5min: 5-minute OHLCV data
            lookback_bars: Bars needed for prediction
            progress_interval: Log progress every N bars

        Returns:
            BacktestResult with all metrics
        """
        result = BacktestResult(initial_equity=self.initial_equity)
        equity = self.initial_equity
        result.equity_curve = [equity]

        # Position tracking
        position: Optional[Dict] = None  # None or {direction, entry_price, sl, tp, size, entry_bar, signal}

        total_bars = len(df_5min)
        logger.info(f"Starting backtest on {total_bars} bars")

        for i in range(lookback_bars, total_bars - 1):
            current_time = df_5min.index[i]
            current_bar = df_5min.iloc[i]
            next_bar = df_5min.iloc[i + 1]

            # Progress logging
            if i % progress_interval == 0:
                logger.info(
                    f"Progress: {i}/{total_bars} ({i/total_bars*100:.1f}%), "
                    f"Trades: {len(result.trades)}, Equity: ${equity:,.2f}"
                )

            # Check existing position
            if position is not None:
                # Check exit conditions
                exit_reason, exit_price = self._check_exit(
                    position, current_bar, next_bar, i
                )

                if exit_reason:
                    # Close position
                    trade = self._close_position(
                        position, exit_price, current_time, exit_reason
                    )
                    result.trades.append(trade)

                    # Update equity
                    equity *= 1 + trade.pnl_pct * trade.position_size
                    result.equity_curve.append(equity)

                    position = None

            # Generate signal if no position
            if position is None:
                # Get lookback data
                lookback_data = df_5min.iloc[i - lookback_bars : i + 1].copy()

                # Generate prediction and signal
                prediction = self.predictor.predict(lookback_data)

                if prediction.should_trade:
                    signal = self.signal_generator.generate_signal(
                        prediction, current_bar["close"]
                    )

                    if signal.is_trade_signal:
                        # Open position
                        position = {
                            "direction": "long" if signal.is_long else "short",
                            "entry_price": current_bar["close"],
                            "stop_loss": signal.stop_loss_price,
                            "take_profit": signal.take_profit_price,
                            "position_size": signal.position_size_pct,
                            "entry_bar": i,
                            "entry_time": current_time,
                            "signal": signal,
                        }

        # Close any remaining position at end
        if position is not None:
            exit_price = df_5min.iloc[-1]["close"]
            trade = self._close_position(
                position, exit_price, df_5min.index[-1], "end_of_data"
            )
            result.trades.append(trade)
            equity *= 1 + trade.pnl_pct * trade.position_size
            result.equity_curve.append(equity)

        result.final_equity = equity
        logger.info(f"Backtest complete: {len(result.trades)} trades")

        return result

    def _check_exit(
        self,
        position: Dict,
        current_bar: pd.Series,
        next_bar: pd.Series,
        current_bar_idx: int,
    ) -> tuple[Optional[str], float]:
        """Check if position should be exited.

        Returns:
            Tuple of (exit_reason, exit_price) or (None, 0)
        """
        direction = position["direction"]
        sl = position["stop_loss"]
        tp = position["take_profit"]
        entry_bar = position["entry_bar"]

        # Check timeout
        bars_in_trade = current_bar_idx - entry_bar
        if bars_in_trade >= self.max_bars_in_trade:
            return "timeout", current_bar["close"]

        # Check SL/TP hit on next bar (using high/low)
        if direction == "long":
            # Check SL (low touches SL)
            if next_bar["low"] <= sl:
                return "sl", sl
            # Check TP (high touches TP)
            if next_bar["high"] >= tp:
                return "tp", tp
        else:  # short
            # Check SL (high touches SL)
            if next_bar["high"] >= sl:
                return "sl", sl
            # Check TP (low touches TP)
            if next_bar["low"] <= tp:
                return "tp", tp

        return None, 0

    def _close_position(
        self,
        position: Dict,
        exit_price: float,
        exit_time: datetime,
        exit_reason: str,
    ) -> Trade:
        """Close position and create trade record."""
        direction = position["direction"]
        entry_price = position["entry_price"]
        signal = position["signal"]

        # Calculate PnL
        if direction == "long":
            pnl_pips = (exit_price - entry_price) / self.pip_value - self.spread_pips
        else:
            pnl_pips = (entry_price - exit_price) / self.pip_value - self.spread_pips

        pnl_pct = pnl_pips * self.pip_value / entry_price

        return Trade(
            entry_time=position["entry_time"],
            exit_time=exit_time,
            direction=direction,
            entry_price=entry_price,
            exit_price=exit_price,
            stop_loss=position["stop_loss"],
            take_profit=position["take_profit"],
            position_size=position["position_size"],
            pnl_pips=pnl_pips,
            pnl_pct=pnl_pct,
            exit_reason=exit_reason,
            confidence=signal.confidence,
            strength=signal.strength.value,
            alignment=signal.alignment.value,
        )


def load_5min_data(data_path: Path) -> pd.DataFrame:
    """Load 5-minute data."""
    if data_path.suffix == ".parquet":
        df = pd.read_parquet(data_path)
    else:
        df = pd.read_csv(data_path)

    df.columns = [c.lower() for c in df.columns]

    time_col = None
    for col in ["timestamp", "time", "date", "datetime"]:
        if col in df.columns:
            time_col = col
            break

    if time_col:
        df[time_col] = pd.to_datetime(df[time_col])
        df = df.set_index(time_col)

    return df.sort_index()


def print_results(result: BacktestResult, title: str):
    """Print backtest results."""
    print(f"\n{'=' * 70}")
    print(f"{title}")
    print("=" * 70)

    print(f"\nTRADE STATISTICS")
    print("-" * 40)
    print(f"Total Trades:       {result.total_trades}")
    print(f"Winning Trades:     {result.winning_trades}")
    print(f"Losing Trades:      {result.losing_trades}")
    print(f"Win Rate:           {result.win_rate:.2%}")
    print(f"Avg Win:            {result.avg_win:.4%}")
    print(f"Avg Loss:           {result.avg_loss:.4%}")
    print(f"Profit Factor:      {result.profit_factor:.2f}")

    print(f"\nPERFORMANCE")
    print("-" * 40)
    print(f"Initial Equity:     ${result.initial_equity:,.2f}")
    print(f"Final Equity:       ${result.final_equity:,.2f}")
    print(f"Total Return:       {result.total_return:.2%}")
    print(f"Max Drawdown:       {result.max_drawdown:.2%}")
    print(f"Sharpe Ratio:       {result.sharpe_ratio:.2f}")
    print(f"Avg Trade Duration: {result.avg_trade_duration}")

    # Breakdown by exit reason
    if result.trades:
        print(f"\nEXIT REASONS")
        print("-" * 40)
        reasons = {}
        for t in result.trades:
            reasons[t.exit_reason] = reasons.get(t.exit_reason, 0) + 1
        for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
            pct = count / result.total_trades
            print(f"  {reason:<15}: {count:4d} ({pct:.1%})")

        # Breakdown by strength
        print(f"\nBY SIGNAL STRENGTH")
        print("-" * 40)
        strengths = {}
        for t in result.trades:
            if t.strength not in strengths:
                strengths[t.strength] = {"count": 0, "wins": 0, "pnl": 0}
            strengths[t.strength]["count"] += 1
            if t.pnl_pct > 0:
                strengths[t.strength]["wins"] += 1
            strengths[t.strength]["pnl"] += t.pnl_pct

        for strength, stats in sorted(strengths.items()):
            wr = stats["wins"] / stats["count"] if stats["count"] > 0 else 0
            print(
                f"  {strength:<15}: {stats['count']:4d} trades, "
                f"{wr:.1%} win rate, {stats['pnl']:.2%} total PnL"
            )


def main():
    parser = argparse.ArgumentParser(description="Backtest MTF system")
    parser.add_argument(
        "--data",
        type=str,
        default="data/forex/EURUSD_20200101_20251231_5min_combined.csv",
        help="Path to 5-minute data",
    )
    parser.add_argument(
        "--models",
        type=str,
        default="models/mtf_scalper",
        help="Model directory",
    )
    parser.add_argument(
        "--equity",
        type=float,
        default=10000,
        help="Initial equity",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.65,
        help="Minimum primary confidence",
    )
    parser.add_argument(
        "--spread",
        type=float,
        default=1.0,
        help="Spread in pips",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.2,
        help="Ratio of data to use for testing (from end)",
    )
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("MTF SCALPER SYSTEM BACKTEST")
    print("=" * 70)
    print(f"Data:       {args.data}")
    print(f"Models:     {args.models}")
    print(f"Equity:     ${args.equity:,.2f}")
    print(f"Min Conf:   {args.confidence:.0%}")
    print(f"Spread:     {args.spread} pips")
    print(f"Test Ratio: {args.test_ratio:.0%}")
    print("=" * 70)

    # Load data
    data_path = project_root / args.data
    df_5min = load_5min_data(data_path)
    logger.info(f"Loaded {len(df_5min)} bars")

    # Use last X% for testing
    test_start = int(len(df_5min) * (1 - args.test_ratio))
    df_test = df_5min.iloc[test_start:].copy()
    logger.info(f"Testing on {len(df_test)} bars ({args.test_ratio:.0%} of data)")

    # Load models
    model_dir = project_root / args.models
    configs = [
        TimeframeConfig.scalper_5min(),
        TimeframeConfig.scalper_15min(),
        TimeframeConfig.scalper_30min(),
    ]
    mtf_model = MultiTimeframeModel(configs=configs, model_dir=model_dir)
    mtf_model.load_all()

    # Check if models are loaded
    for name, model in mtf_model.models.items():
        if model.is_trained:
            logger.info(f"Loaded {name} model (val_acc={model.validation_accuracy:.2%})")
        else:
            logger.error(f"Model {name} not loaded!")
            return

    # Create predictor and signal generator
    predictor = MTFPredictor(
        mtf_model=mtf_model,
        primary_min_confidence=args.confidence,
        require_confirmation=True,
        require_trend_alignment=True,
    )

    signal_config = ScalperConfig(
        base_position_pct=0.02,
        risk_per_trade_pct=0.01,
    )
    signal_gen = MTFSignalGenerator(config=signal_config)

    # Create backtester
    backtester = MTFBacktester(
        mtf_model=mtf_model,
        predictor=predictor,
        signal_generator=signal_gen,
        initial_equity=args.equity,
        spread_pips=args.spread,
    )

    # Run backtest
    # Need 300+ bars for 30min (50 bars * 6) feature calculation
    logger.info("Running backtest...")
    result = backtester.run(df_test, lookback_bars=350)

    # Print results
    print_results(result, "BACKTEST RESULTS")

    # Summary comparison
    print("\n" + "=" * 70)
    print("PERFORMANCE VS TARGETS")
    print("=" * 70)
    print(f"{'Metric':<25} {'Target':>12} {'Actual':>12} {'Status':>10}")
    print("-" * 60)

    targets = [
        ("Win Rate", 0.55, result.win_rate),
        ("Profit Factor", 1.5, result.profit_factor),
        ("Max Drawdown", 0.15, result.max_drawdown),
        ("Sharpe Ratio", 1.0, result.sharpe_ratio),
    ]

    for name, target, actual in targets:
        if name == "Max Drawdown":
            status = "✓" if actual <= target else "✗"
        else:
            status = "✓" if actual >= target else "✗"
        print(f"{name:<25} {target:>11.2f} {actual:>12.2f} {status:>10}")

    print("=" * 70)


if __name__ == "__main__":
    main()
