#!/usr/bin/env python3
"""Fast multi-timeframe backtest with pre-calculated features.

This optimized version pre-calculates all features upfront instead of
calculating them for every bar, making it orders of magnitude faster.
"""

import argparse
import logging
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
import warnings

warnings.filterwarnings("ignore")

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd

from src.models.multi_timeframe import (
    MultiTimeframeModel,
    TimeframeConfig,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class Position:
    """Track an open position."""
    direction: str  # "long" or "short"
    entry_price: float
    entry_time: pd.Timestamp
    size: float
    stop_loss: float
    take_profit: float
    timeout_bars: int = 24  # 2 hours at 5min
    bars_held: int = 0


@dataclass
class Trade:
    """Completed trade record."""
    direction: str
    entry_price: float
    exit_price: float
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    pnl: float
    pnl_pct: float
    exit_reason: str
    confidence: float


class FastBacktester:
    """Fast backtester with pre-calculated features."""

    def __init__(
        self,
        model: MultiTimeframeModel,
        initial_equity: float = 10000.0,
        min_confidence: float = 0.65,
    ):
        self.model = model
        self.initial_equity = initial_equity
        self.min_confidence = min_confidence

        self.equity = initial_equity
        self.position: Optional[Position] = None
        self.trades: list[Trade] = []
        self.equity_curve: list[float] = []

    def _prepare_all_data(self, df_5min: pd.DataFrame) -> dict:
        """Pre-calculate features and prepare data for all timeframes."""
        logger.info("Pre-calculating features for all timeframes...")

        data_dict = {}
        feature_calc = self.model.feature_calculator

        for tf_name, config in self.model.configs.items():
            logger.info(f"  Processing {tf_name}...")

            # Resample data
            if config.resample_rule != "5min":
                resampled = df_5min.resample(config.resample_rule).agg({
                    "open": "first",
                    "high": "max",
                    "low": "min",
                    "close": "last",
                    "volume": "sum",
                }).dropna()
            else:
                resampled = df_5min.copy()

            # Calculate features
            features = feature_calc.calculate(resampled)
            features = features.dropna()

            # Get model's expected feature names
            tf_model = self.model.models.get(tf_name)
            if tf_model and tf_model.feature_names:
                # Only keep features that the model was trained with
                available_features = [f for f in tf_model.feature_names if f in features.columns]
                if len(available_features) < len(tf_model.feature_names):
                    missing = set(tf_model.feature_names) - set(available_features)
                    logger.warning(f"  {tf_name}: Missing features: {missing}")

                # Filter and align features
                features = features[available_features]

            data_dict[tf_name] = {
                "features": features,
                "timestamps": features.index.tolist(),
            }
            logger.info(f"    {tf_name}: {len(features)} rows, {len(features.columns)} features")

        return data_dict

    def _get_prediction(
        self,
        data_dict: dict,
        timestamp: pd.Timestamp,
    ) -> dict:
        """Get predictions for all timeframes at a given timestamp."""
        predictions = {}

        for tf_name, data in data_dict.items():
            features_df = data["features"]
            tf_model = self.model.models.get(tf_name)

            if tf_model is None or not tf_model.is_trained:
                continue

            # Find the most recent feature row at or before this timestamp
            available = features_df[features_df.index <= timestamp]
            if len(available) == 0:
                continue

            # Get the last row
            X = available.iloc[-1:].values

            try:
                # Use model's scaler and predict
                X_scaled = tf_model.scaler.transform(X)
                probs = tf_model.model.predict_proba(X_scaled)[0]
                pred = tf_model.model.predict(X_scaled)[0]
                confidence = max(probs)

                # Direction: 1 = UP, 0 = DOWN
                direction = int(pred)
                if len(probs) == 2:
                    prob_up = probs[1]
                else:
                    prob_up = probs[0] if pred == 1 else 1 - probs[0]

                predictions[tf_name] = {
                    "direction": direction,
                    "confidence": confidence,
                    "probability_up": prob_up,
                }
            except Exception as e:
                logger.debug(f"Prediction error for {tf_name}: {e}")
                continue

        return predictions

    def _check_exit(self, current_price: float) -> Optional[str]:
        """Check if position should be exited."""
        if self.position is None:
            return None

        # Check stop loss
        if self.position.direction == "long":
            if current_price <= self.position.stop_loss:
                return "stop_loss"
            if current_price >= self.position.take_profit:
                return "take_profit"
        else:  # short
            if current_price >= self.position.stop_loss:
                return "stop_loss"
            if current_price <= self.position.take_profit:
                return "take_profit"

        # Check timeout
        if self.position.bars_held >= self.position.timeout_bars:
            return "timeout"

        return None

    def _close_position(
        self,
        exit_price: float,
        exit_time: pd.Timestamp,
        exit_reason: str,
    ):
        """Close current position and record trade."""
        if self.position is None:
            return

        # Calculate PnL
        if self.position.direction == "long":
            pnl_pips = (exit_price - self.position.entry_price) * 10000
        else:
            pnl_pips = (self.position.entry_price - exit_price) * 10000

        # $1 per pip per 0.01 lot, size is fraction of equity
        pnl_dollars = pnl_pips * self.position.size * 10
        pnl_pct = pnl_dollars / self.equity

        # Record trade
        trade = Trade(
            direction=self.position.direction,
            entry_price=self.position.entry_price,
            exit_price=exit_price,
            entry_time=self.position.entry_time,
            exit_time=exit_time,
            pnl=pnl_dollars,
            pnl_pct=pnl_pct,
            exit_reason=exit_reason,
            confidence=getattr(self.position, "confidence", 0),
        )
        self.trades.append(trade)

        # Update equity
        self.equity += pnl_dollars

        # Clear position
        self.position = None

    def run(self, df_5min: pd.DataFrame, test_start_idx: int) -> dict:
        """Run backtest on test portion of data."""

        # Pre-calculate all features
        data_dict = self._prepare_all_data(df_5min)

        # Get test data
        test_data = df_5min.iloc[test_start_idx:]
        total_bars = len(test_data)

        logger.info(f"Running backtest on {total_bars} bars...")
        logger.info(f"Test period: {test_data.index[0]} to {test_data.index[-1]}")

        signals_generated = 0
        positions_opened = 0

        for i, (timestamp, row) in enumerate(test_data.iterrows()):
            current_price = row["close"]

            # Record equity
            self.equity_curve.append(self.equity)

            # Check for exit if we have a position
            if self.position is not None:
                self.position.bars_held += 1
                exit_reason = self._check_exit(current_price)
                if exit_reason:
                    self._close_position(current_price, timestamp, exit_reason)

            # Only generate new signals if no position
            if self.position is None:
                # Get predictions
                predictions = self._get_prediction(data_dict, timestamp)

                if predictions:
                    primary = predictions.get("5min")
                    confirm = predictions.get("15min")
                    trend = predictions.get("30min")

                    # Check if primary signal meets confidence threshold
                    if primary and primary["confidence"] >= self.min_confidence:
                        signals_generated += 1

                        # Check confirmation alignment
                        confirmed = True
                        if confirm:
                            if confirm["direction"] != primary["direction"]:
                                confirmed = False

                        # Check trend alignment
                        trend_aligned = True
                        if trend:
                            if trend["direction"] != primary["direction"]:
                                trend_aligned = False

                        # Only trade if confirmed and trend aligned
                        if confirmed and trend_aligned:
                            direction = "long" if primary["direction"] == 1 else "short"

                            # Fixed SL/TP
                            atr_pips = 0.0010  # 10 pips

                            if direction == "long":
                                sl = current_price - atr_pips
                                tp = current_price + (atr_pips * 2)
                            else:
                                sl = current_price + atr_pips
                                tp = current_price - (atr_pips * 2)

                            # Position size based on confidence
                            base_size = 0.02
                            conf_multiplier = min(1.0 + (primary["confidence"] - 0.65) * 2, 1.5)
                            size = base_size * conf_multiplier

                            self.position = Position(
                                direction=direction,
                                entry_price=current_price,
                                entry_time=timestamp,
                                size=size,
                                stop_loss=sl,
                                take_profit=tp,
                            )
                            self.position.confidence = primary["confidence"]
                            positions_opened += 1

            # Progress report
            if (i + 1) % 10000 == 0:
                win_rate = 0
                if self.trades:
                    wins = sum(1 for t in self.trades if t.pnl > 0)
                    win_rate = wins / len(self.trades) * 100
                logger.info(
                    f"Progress: {i+1}/{total_bars} ({100*(i+1)/total_bars:.1f}%), "
                    f"Trades: {len(self.trades)}, Win Rate: {win_rate:.1f}%, "
                    f"Equity: ${self.equity:,.2f}"
                )

        # Close any remaining position at end
        if self.position:
            self._close_position(
                test_data.iloc[-1]["close"],
                test_data.index[-1],
                "end_of_test",
            )

        # Calculate results
        results = self._calculate_results()
        results["signals_generated"] = signals_generated
        results["positions_opened"] = positions_opened

        return results

    def _calculate_results(self) -> dict:
        """Calculate backtest metrics."""
        if not self.trades:
            return {
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate": 0,
                "total_pnl": 0,
                "total_pnl_pct": 0,
                "avg_win": 0,
                "avg_loss": 0,
                "profit_factor": 0,
                "max_drawdown": 0,
                "sharpe_ratio": 0,
                "final_equity": self.initial_equity,
                "exit_reasons": {},
            }

        # Basic stats
        wins = [t for t in self.trades if t.pnl > 0]
        losses = [t for t in self.trades if t.pnl <= 0]

        total_profit = sum(t.pnl for t in wins)
        total_loss = abs(sum(t.pnl for t in losses))

        # Profit factor
        profit_factor = total_profit / total_loss if total_loss > 0 else float("inf")

        # Max drawdown
        equity_arr = np.array(self.equity_curve)
        peak = np.maximum.accumulate(equity_arr)
        drawdown = (peak - equity_arr) / peak
        max_drawdown = drawdown.max()

        # Sharpe ratio
        returns = np.diff(equity_arr) / equity_arr[:-1]
        sharpe = np.sqrt(252 * 288) * returns.mean() / returns.std() if returns.std() > 0 else 0

        # Trade breakdown by exit reason
        exit_reasons = {}
        for t in self.trades:
            exit_reasons[t.exit_reason] = exit_reasons.get(t.exit_reason, 0) + 1

        return {
            "total_trades": len(self.trades),
            "winning_trades": len(wins),
            "losing_trades": len(losses),
            "win_rate": len(wins) / len(self.trades) * 100,
            "total_pnl": sum(t.pnl for t in self.trades),
            "total_pnl_pct": (self.equity - self.initial_equity) / self.initial_equity * 100,
            "avg_win": total_profit / len(wins) if wins else 0,
            "avg_loss": total_loss / len(losses) if losses else 0,
            "profit_factor": profit_factor,
            "max_drawdown": max_drawdown * 100,
            "sharpe_ratio": sharpe,
            "final_equity": self.equity,
            "exit_reasons": exit_reasons,
        }


def main():
    parser = argparse.ArgumentParser(description="Fast MTF Backtest")
    parser.add_argument(
        "--data",
        type=str,
        default="data/forex/EURUSD_20200101_20251231_5min_combined.csv",
        help="Path to 5-minute data",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="models/mtf_scalper",
        help="Model directory",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.65,
        help="Minimum confidence threshold",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.2,
        help="Portion of data to use for testing",
    )
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("FAST MULTI-TIMEFRAME BACKTEST")
    print("=" * 70)
    print(f"Data:       {args.data}")
    print(f"Model dir:  {args.model_dir}")
    print(f"Confidence: {args.confidence:.0%}")
    print(f"Test ratio: {args.test_ratio:.0%}")
    print("=" * 70)

    # Load data
    data_path = project_root / args.data
    logger.info(f"Loading data from {data_path}")

    df = pd.read_csv(data_path)
    df.columns = [c.lower() for c in df.columns]

    # Handle timestamp
    time_col = None
    for col in ["timestamp", "time", "date", "datetime"]:
        if col in df.columns:
            time_col = col
            break

    if time_col:
        df[time_col] = pd.to_datetime(df[time_col])
        df = df.set_index(time_col)

    df = df.sort_index()
    logger.info(f"Loaded {len(df)} bars from {df.index[0]} to {df.index[-1]}")

    # Load models
    model_dir = project_root / args.model_dir
    configs = [
        TimeframeConfig.scalper_5min(),
        TimeframeConfig.scalper_15min(),
        TimeframeConfig.scalper_30min(),
    ]

    mtf_model = MultiTimeframeModel(configs=configs, model_dir=model_dir)
    mtf_model.load_all()

    # Calculate test start index
    test_start_idx = int(len(df) * (1 - args.test_ratio))

    # Create backtester
    backtester = FastBacktester(
        model=mtf_model,
        initial_equity=10000.0,
        min_confidence=args.confidence,
    )

    # Run backtest
    results = backtester.run(df, test_start_idx)

    # Print results
    print("\n" + "=" * 70)
    print("BACKTEST RESULTS")
    print("=" * 70)

    print(f"\nPerformance Summary:")
    print(f"  Total Trades:     {results['total_trades']}")
    print(f"  Winning Trades:   {results.get('winning_trades', 0)}")
    print(f"  Losing Trades:    {results.get('losing_trades', 0)}")
    print(f"  Win Rate:         {results['win_rate']:.2f}%")
    print(f"  Total PnL:        ${results['total_pnl']:,.2f}")
    print(f"  Total Return:     {results['total_pnl_pct']:.2f}%")
    print(f"  Profit Factor:    {results['profit_factor']:.2f}")
    print(f"  Max Drawdown:     {results['max_drawdown']:.2f}%")
    print(f"  Sharpe Ratio:     {results['sharpe_ratio']:.2f}")
    print(f"  Final Equity:     ${results['final_equity']:,.2f}")

    if results.get('avg_win'):
        print(f"\nTrade Analysis:")
        print(f"  Avg Win:          ${results['avg_win']:.2f}")
        print(f"  Avg Loss:         ${results['avg_loss']:.2f}")
        if results['avg_loss'] > 0:
            print(f"  Win/Loss Ratio:   {results['avg_win']/results['avg_loss']:.2f}")

    print(f"\nSignal Analysis:")
    print(f"  Signals Generated: {results.get('signals_generated', 0)}")
    print(f"  Positions Opened:  {results.get('positions_opened', 0)}")

    if results.get('exit_reasons'):
        print(f"\nExit Reasons:")
        for reason, count in results['exit_reasons'].items():
            pct = count / results['total_trades'] * 100
            print(f"  {reason}: {count} ({pct:.1f}%)")

    print("=" * 70)

    # Analysis
    print("\nANALYSIS:")
    if results['win_rate'] >= 55:
        print("  [OK] Win rate meets target (>55%)")
    else:
        print(f"  [!!] Win rate below target: {results['win_rate']:.1f}% < 55%")

    if results['profit_factor'] >= 1.5:
        print("  [OK] Profit factor meets target (>1.5)")
    else:
        print(f"  [!!] Profit factor below target: {results['profit_factor']:.2f} < 1.5")

    if results['max_drawdown'] <= 15:
        print("  [OK] Max drawdown within limit (<15%)")
    else:
        print(f"  [!!] Max drawdown exceeds limit: {results['max_drawdown']:.1f}% > 15%")

    print("=" * 70)


if __name__ == "__main__":
    main()
