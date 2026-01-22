#!/usr/bin/env python3
"""Walk-Forward Optimization for MTF Ensemble.

This script validates model robustness across multiple time periods using
rolling window training and testing:

1. Splits data into multiple train/test windows
2. Trains fresh models on each training window
3. Tests on the subsequent out-of-sample period
4. Aggregates results to validate consistency

Key Parameters:
- train_months: Duration of training window (default: 24 months)
- test_months: Duration of test window (default: 6 months)
- step_months: How much to slide forward each iteration (default: 6 months)

Example with 6 years of data (2020-2025):
  Window 1: Train 2020-01 to 2021-12, Test 2022-01 to 2022-06
  Window 2: Train 2020-07 to 2022-06, Test 2022-07 to 2022-12
  Window 3: Train 2021-01 to 2022-12, Test 2023-01 to 2023-06
  ... and so on
"""

import argparse
import json
import logging
import sys
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

warnings.filterwarnings("ignore")

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd

from src.models.multi_timeframe import MTFEnsemble, MTFEnsembleConfig, StackingConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class WFOWindow:
    """Represents a single walk-forward window."""
    window_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp

    @property
    def train_period(self) -> str:
        return f"{self.train_start.strftime('%Y-%m')} to {self.train_end.strftime('%Y-%m')}"

    @property
    def test_period(self) -> str:
        return f"{self.test_start.strftime('%Y-%m')} to {self.test_end.strftime('%Y-%m')}"


@dataclass
class WFOWindowResult:
    """Results from a single walk-forward window."""
    window: WFOWindow

    # Model accuracy metrics
    model_accuracies: Dict[str, float] = field(default_factory=dict)
    ensemble_accuracy: float = 0.0

    # Trading simulation metrics
    total_trades: int = 0
    winning_trades: int = 0
    win_rate: float = 0.0
    total_pips: float = 0.0
    profit_factor: float = 0.0
    avg_pips_per_trade: float = 0.0

    # Confidence analysis
    high_conf_trades: int = 0
    high_conf_win_rate: float = 0.0

    # Direction analysis
    long_trades: int = 0
    short_trades: int = 0
    long_win_rate: float = 0.0
    short_win_rate: float = 0.0

    # Exit analysis
    tp_hits: int = 0
    sl_hits: int = 0
    timeouts: int = 0

    # Balance tracking (with position sizing)
    initial_balance: float = 0.0
    final_balance: float = 0.0
    total_return_pct: float = 0.0
    max_drawdown_pct: float = 0.0
    total_pnl_usd: float = 0.0

    # Risk management metrics
    max_losing_streak: int = 0
    risk_reductions: int = 0


@dataclass
class WFOSummary:
    """Aggregated results from all walk-forward windows."""
    windows: List[WFOWindowResult] = field(default_factory=list)
    baseline_total_pips: float = 0.0
    baseline_profit_factor: float = 0.0
    baseline_win_rate: float = 0.0
    initial_balance: float = 10000.0  # Starting balance for position sizing

    @property
    def total_windows(self) -> int:
        return len(self.windows)

    @property
    def profitable_windows(self) -> int:
        return sum(1 for w in self.windows if w.total_pips > 0)

    @property
    def consistency_rate(self) -> float:
        if not self.windows:
            return 0.0
        return self.profitable_windows / self.total_windows

    @property
    def total_pips(self) -> float:
        return sum(w.total_pips for w in self.windows)

    @property
    def total_trades(self) -> int:
        return sum(w.total_trades for w in self.windows)

    @property
    def overall_win_rate(self) -> float:
        total_wins = sum(w.winning_trades for w in self.windows)
        total_trades = sum(w.total_trades for w in self.windows)
        return total_wins / total_trades * 100 if total_trades > 0 else 0.0

    @property
    def overall_profit_factor(self) -> float:
        total_profit = sum(max(0, w.total_pips) for w in self.windows)
        total_loss = sum(abs(min(0, w.total_pips)) for w in self.windows)
        return total_profit / total_loss if total_loss > 0 else float('inf')

    @property
    def avg_pips_per_window(self) -> float:
        return self.total_pips / self.total_windows if self.windows else 0.0

    @property
    def pips_std(self) -> float:
        if not self.windows:
            return 0.0
        pips_list = [w.total_pips for w in self.windows]
        return np.std(pips_list)

    @property
    def min_pips_window(self) -> float:
        return min(w.total_pips for w in self.windows) if self.windows else 0.0

    @property
    def max_pips_window(self) -> float:
        return max(w.total_pips for w in self.windows) if self.windows else 0.0

    @property
    def total_return_pct(self) -> float:
        """Calculate total return across all windows (compounded)."""
        if not self.windows:
            return 0.0
        # Each window starts fresh with initial_balance, so sum the returns
        total_pnl = sum(w.total_pnl_usd for w in self.windows)
        return (total_pnl / self.initial_balance) * 100

    @property
    def avg_return_per_window(self) -> float:
        """Average return per window."""
        if not self.windows:
            return 0.0
        return sum(w.total_return_pct for w in self.windows) / len(self.windows)

    @property
    def max_drawdown(self) -> float:
        """Maximum drawdown across all windows."""
        return max(w.max_drawdown_pct for w in self.windows) if self.windows else 0.0


def create_wfo_windows(
    data_start: pd.Timestamp,
    data_end: pd.Timestamp,
    train_months: int = 24,
    test_months: int = 6,
    step_months: int = 6,
) -> List[WFOWindow]:
    """Create walk-forward optimization windows.

    Args:
        data_start: Start of available data
        data_end: End of available data
        train_months: Duration of training window in months
        test_months: Duration of test window in months
        step_months: How much to slide forward each iteration

    Returns:
        List of WFOWindow objects
    """
    windows = []
    window_id = 1

    # Start from the beginning
    current_train_start = data_start

    while True:
        # Calculate window boundaries
        train_end = current_train_start + pd.DateOffset(months=train_months)
        test_start = train_end
        test_end = test_start + pd.DateOffset(months=test_months)

        # Check if we have enough data for this window
        if test_end > data_end:
            break

        windows.append(WFOWindow(
            window_id=window_id,
            train_start=current_train_start,
            train_end=train_end - pd.Timedelta(days=1),  # End of last day
            test_start=test_start,
            test_end=test_end - pd.Timedelta(days=1),
        ))

        # Slide forward
        current_train_start += pd.DateOffset(months=step_months)
        window_id += 1

    return windows


def load_data(data_path: Path) -> pd.DataFrame:
    """Load 5-minute OHLCV data."""
    logger.info(f"Loading data from {data_path}")

    if data_path.suffix == ".parquet":
        df = pd.read_parquet(data_path)
    else:
        # Try reading with index_col=0 first (for CSV with unnamed index column)
        df = pd.read_csv(data_path, index_col=0, parse_dates=True)

    df.columns = [c.lower() for c in df.columns]

    # If index is not datetime, try to find a time column
    if not isinstance(df.index, pd.DatetimeIndex):
        time_col = None
        for col in ["timestamp", "time", "date", "datetime"]:
            if col in df.columns:
                time_col = col
                break

        if time_col:
            df[time_col] = pd.to_datetime(df[time_col])
            df = df.set_index(time_col)
        else:
            # Try to convert the index to datetime
            df.index = pd.to_datetime(df.index)

    df = df.sort_index()
    logger.info(f"Loaded {len(df)} bars from {df.index[0]} to {df.index[-1]}")

    return df


def run_window_backtest(
    ensemble: MTFEnsemble,
    df_5min: pd.DataFrame,
    min_confidence: float = 0.55,
    min_agreement: float = 0.5,
    tp_pips: float = 25.0,
    sl_pips: float = 15.0,
    max_holding_bars: int = 12,
    initial_balance: float = 10000.0,
    risk_per_trade: float = 0.02,
    reduce_risk_on_losses: bool = True,
    spread_pips: float = 1.0,
    slippage_pips: float = 0.5,
) -> Dict:
    """Run backtest on test data using the ensemble with proper position sizing.

    Position sizing uses fixed fractional risk:
    - Risk amount = Balance × Risk%
    - Position size (lots) = Risk amount / (SL pips × $10 per pip per lot)
    - Profits/losses compound into subsequent trades

    Risk Management (reduce_risk_on_losses):
    - After 1 loss: reduce risk to 75% of base
    - After 2 losses: reduce risk to 50% of base
    - After 3+ losses: reduce risk to 25% of base
    - After a win: reset to 100% of base risk

    Args:
        ensemble: Trained MTF Ensemble model
        df_5min: 5-minute OHLCV data for testing
        min_confidence: Minimum confidence to enter trade (HOLD if below)
        min_agreement: Minimum agreement score between timeframes
        tp_pips: Take profit in pips
        sl_pips: Stop loss in pips
        max_holding_bars: Maximum bars to hold before timeout
        initial_balance: Starting account balance in USD
        risk_per_trade: Fraction of balance to risk per trade (e.g., 0.02 = 2%)
        reduce_risk_on_losses: If True, reduce position size after consecutive losses
        spread_pips: Spread cost in pips (realistic trading cost)
        slippage_pips: Slippage in pips (realistic execution cost)

    Returns:
        Dict with trading metrics including balance tracking
    """
    from src.features.technical.calculator import TechnicalIndicatorCalculator

    calc = TechnicalIndicatorCalculator(model_type="short_term")

    # Prepare 1H data (primary trading timeframe)
    model_1h = ensemble.models["1H"]
    df_1h = ensemble.resample_data(df_5min, "1H")
    higher_tf_data_1h = ensemble.prepare_higher_tf_data(df_5min, "1H")
    df_1h_features = calc.calculate(df_1h)
    df_1h_features = model_1h.feature_engine.add_all_features(df_1h_features, higher_tf_data_1h)
    df_1h_features = df_1h_features.dropna()

    feature_cols_1h = model_1h.feature_names
    available_cols_1h = [c for c in feature_cols_1h if c in df_1h_features.columns]
    X_1h = df_1h_features[available_cols_1h].values

    # Get 1H predictions
    preds_1h, confs_1h = model_1h.predict_batch(X_1h)

    # Prepare 4H predictions
    model_4h = ensemble.models["4H"]
    df_4h = ensemble.resample_data(df_5min, "4H")
    higher_tf_data_4h = ensemble.prepare_higher_tf_data(df_5min, "4H")
    df_4h_features = calc.calculate(df_4h)
    df_4h_features = model_4h.feature_engine.add_all_features(df_4h_features, higher_tf_data_4h)
    df_4h_features = df_4h_features.dropna()

    feature_cols_4h = model_4h.feature_names
    available_cols_4h = [c for c in feature_cols_4h if c in df_4h_features.columns]
    X_4h = df_4h_features[available_cols_4h].values
    preds_4h_all, confs_4h_all = model_4h.predict_batch(X_4h)
    pred_4h_map = dict(zip(df_4h_features.index, zip(preds_4h_all, confs_4h_all)))

    # Prepare Daily predictions
    model_d = ensemble.models["D"]
    df_d = ensemble.resample_data(df_5min, "D")
    df_d_features = calc.calculate(df_d)
    df_d_features = model_d.feature_engine.add_all_features(df_d_features, {})
    df_d_features = df_d_features.dropna()

    feature_cols_d = model_d.feature_names
    available_cols_d = [c for c in feature_cols_d if c in df_d_features.columns]
    X_d = df_d_features[available_cols_d].values
    preds_d_all, confs_d_all = model_d.predict_batch(X_d)
    pred_d_map = dict(zip(df_d_features.index.date, zip(preds_d_all, confs_d_all)))

    # Combine predictions
    weights = ensemble._normalize_weights(ensemble.config.weights)
    w_1h, w_4h, w_d = weights.get("1H", 0.6), weights.get("4H", 0.3), weights.get("D", 0.1)

    closes = df_1h_features["close"].values
    highs = df_1h_features["high"].values
    lows = df_1h_features["low"].values
    timestamps = df_1h_features.index

    test_directions = []
    test_confidences = []
    test_agreements = []

    for i, ts in enumerate(timestamps):
        p_1h, c_1h = preds_1h[i], confs_1h[i]

        # Find 4H prediction
        ts_4h = ts.floor("4H")
        if ts_4h in pred_4h_map:
            p_4h, c_4h = pred_4h_map[ts_4h]
        else:
            prev_4h = [t for t in pred_4h_map.keys() if t <= ts]
            p_4h, c_4h = pred_4h_map[max(prev_4h)] if prev_4h else (p_1h, c_1h)

        # Find Daily prediction
        day = ts.date()
        if day in pred_d_map:
            p_d, c_d = pred_d_map[day]
        else:
            prev_days = [d for d in pred_d_map.keys() if d <= day]
            p_d, c_d = pred_d_map[max(prev_days)] if prev_days else (p_1h, c_1h)

        # Weighted combination
        prob_up_1h = c_1h if p_1h == 1 else 1 - c_1h
        prob_up_4h = c_4h if p_4h == 1 else 1 - c_4h
        prob_up_d = c_d if p_d == 1 else 1 - c_d

        weighted_prob_up = w_1h * prob_up_1h + w_4h * prob_up_4h + w_d * prob_up_d

        direction = 1 if weighted_prob_up > 0.5 else 0
        base_conf = abs(weighted_prob_up - 0.5) * 2 + 0.5

        agreement_count = sum([1 for p in [p_1h, p_4h, p_d] if p == direction])
        agreement_score = agreement_count / 3.0

        if agreement_count == 3:
            conf = min(base_conf + ensemble.config.agreement_bonus, 1.0)
        else:
            conf = base_conf

        test_directions.append(direction)
        test_confidences.append(conf)
        test_agreements.append(agreement_score)

    test_directions = np.array(test_directions)
    test_confidences = np.array(test_confidences)
    test_agreements = np.array(test_agreements)

    # Simulate trading with proper position sizing
    pip_price_value = 0.0001  # Price movement per pip for EUR/USD
    pip_dollar_value = 10.0   # USD per pip per standard lot (100,000 units)

    trades = []
    balance = initial_balance
    peak_balance = initial_balance
    max_drawdown = 0.0
    equity_curve = [initial_balance]

    # Risk management: track consecutive losses
    consecutive_losses = 0
    risk_reductions = 0  # Count how many times risk was reduced

    i = 0
    n = len(test_directions)

    while i < n - max_holding_bars:
        conf = test_confidences[i]
        agreement = test_agreements[i]
        pred = test_directions[i]

        # Only trade if confidence meets threshold (otherwise HOLD)
        if conf >= min_confidence and agreement >= min_agreement:
            entry_price = closes[i]
            entry_time = timestamps[i]
            direction = "long" if pred == 1 else "short"

            # Calculate current risk based on consecutive losses
            if reduce_risk_on_losses:
                if consecutive_losses == 0:
                    current_risk = risk_per_trade  # 100% of base risk
                elif consecutive_losses == 1:
                    current_risk = risk_per_trade * 0.75  # 75% of base risk
                    risk_reductions += 1
                elif consecutive_losses == 2:
                    current_risk = risk_per_trade * 0.50  # 50% of base risk
                    risk_reductions += 1
                else:  # 3+ consecutive losses
                    current_risk = risk_per_trade * 0.25  # 25% of base risk
                    risk_reductions += 1
            else:
                current_risk = risk_per_trade

            # Calculate position size based on risk
            # Risk amount = Balance × Risk%
            risk_amount = balance * current_risk

            # Position size (lots) = Risk amount / (SL pips × $ per pip per lot)
            # This ensures we lose exactly risk_amount if stopped out
            position_lots = risk_amount / (sl_pips * pip_dollar_value)

            # Cap position size to reasonable limits (max 5 lots for realistic trading)
            position_lots = min(position_lots, 5.0)

            # Apply spread cost on entry
            if direction == "long":
                entry_price += spread_pips * pip_price_value  # Pay spread on entry
                tp_price = entry_price + tp_pips * pip_price_value
                sl_price = entry_price - sl_pips * pip_price_value
            else:
                entry_price -= spread_pips * pip_price_value
                tp_price = entry_price - tp_pips * pip_price_value
                sl_price = entry_price + sl_pips * pip_price_value

            exit_price = None
            exit_reason = None
            exit_idx = i

            for j in range(i + 1, min(i + max_holding_bars + 1, n)):
                if direction == "long":
                    if highs[j] >= tp_price:
                        # Apply slippage on exit
                        exit_price = tp_price - slippage_pips * pip_price_value
                        exit_reason = "take_profit"
                        exit_idx = j
                        break
                    if lows[j] <= sl_price:
                        exit_price = sl_price - slippage_pips * pip_price_value
                        exit_reason = "stop_loss"
                        exit_idx = j
                        break
                else:
                    if lows[j] <= tp_price:
                        exit_price = tp_price + slippage_pips * pip_price_value
                        exit_reason = "take_profit"
                        exit_idx = j
                        break
                    if highs[j] >= sl_price:
                        exit_price = sl_price + slippage_pips * pip_price_value
                        exit_reason = "stop_loss"
                        exit_idx = j
                        break

            if exit_price is None:
                exit_idx = min(i + max_holding_bars, n - 1)
                exit_price = closes[exit_idx]
                exit_reason = "timeout"

            # Calculate PnL in pips and USD
            if direction == "long":
                pnl_pips = (exit_price - entry_price) / pip_price_value
            else:
                pnl_pips = (entry_price - exit_price) / pip_price_value

            # PnL in USD = pips × position_lots × pip_dollar_value
            pnl_usd = pnl_pips * position_lots * pip_dollar_value

            # Update balance (compound profits/losses)
            balance += pnl_usd
            equity_curve.append(balance)

            # Track consecutive losses for risk management
            if pnl_pips > 0:
                consecutive_losses = 0  # Reset on win
            else:
                consecutive_losses += 1

            # Track peak and drawdown
            if balance > peak_balance:
                peak_balance = balance
            current_drawdown = (peak_balance - balance) / peak_balance if peak_balance > 0 else 0
            max_drawdown = max(max_drawdown, current_drawdown)

            trades.append({
                "direction": direction,
                "confidence": conf,
                "pnl_pips": pnl_pips,
                "pnl_usd": pnl_usd,
                "position_lots": position_lots,
                "balance_after": balance,
                "exit_reason": exit_reason,
                "risk_used": current_risk,
            })

            i = exit_idx

        i += 1

    # Calculate results
    if not trades:
        return {
            "total_trades": 0,
            "winning_trades": 0,
            "win_rate": 0,
            "total_pips": 0,
            "profit_factor": 0,
            "avg_pips": 0,
            "initial_balance": initial_balance,
            "final_balance": initial_balance,
            "total_return_pct": 0,
            "max_drawdown_pct": 0,
            "max_losing_streak": 0,
            "risk_reductions": 0,
        }

    trades_df = pd.DataFrame(trades)
    wins = trades_df[trades_df["pnl_pips"] > 0]
    losses = trades_df[trades_df["pnl_pips"] <= 0]

    total_profit_pips = wins["pnl_pips"].sum() if len(wins) > 0 else 0
    total_loss_pips = abs(losses["pnl_pips"].sum()) if len(losses) > 0 else 0
    profit_factor = total_profit_pips / total_loss_pips if total_loss_pips > 0 else float("inf")

    high_conf = trades_df[trades_df["confidence"] >= 0.60]

    # Calculate max losing streak
    max_losing_streak = 0
    current_streak = 0
    for pnl in trades_df["pnl_pips"]:
        if pnl <= 0:
            current_streak += 1
            max_losing_streak = max(max_losing_streak, current_streak)
        else:
            current_streak = 0

    # Calculate return metrics
    final_balance = balance
    total_return_pct = ((final_balance - initial_balance) / initial_balance) * 100

    return {
        "total_trades": len(trades_df),
        "winning_trades": len(wins),
        "losing_trades": len(losses),
        "win_rate": len(wins) / len(trades_df) * 100,
        "total_pips": trades_df["pnl_pips"].sum(),
        "avg_pips": trades_df["pnl_pips"].mean(),
        "profit_factor": profit_factor,
        "high_conf_trades": len(high_conf),
        "high_conf_win_rate": len(high_conf[high_conf["pnl_pips"] > 0]) / len(high_conf) * 100 if len(high_conf) > 0 else 0,
        "long_trades": len(trades_df[trades_df["direction"] == "long"]),
        "short_trades": len(trades_df[trades_df["direction"] == "short"]),
        "long_win_rate": len(trades_df[(trades_df["direction"] == "long") & (trades_df["pnl_pips"] > 0)]) / len(trades_df[trades_df["direction"] == "long"]) * 100 if len(trades_df[trades_df["direction"] == "long"]) > 0 else 0,
        "short_win_rate": len(trades_df[(trades_df["direction"] == "short") & (trades_df["pnl_pips"] > 0)]) / len(trades_df[trades_df["direction"] == "short"]) * 100 if len(trades_df[trades_df["direction"] == "short"]) > 0 else 0,
        "tp_hits": len(trades_df[trades_df["exit_reason"] == "take_profit"]),
        "sl_hits": len(trades_df[trades_df["exit_reason"] == "stop_loss"]),
        "timeouts": len(trades_df[trades_df["exit_reason"] == "timeout"]),
        # Balance tracking metrics
        "initial_balance": initial_balance,
        "final_balance": final_balance,
        "total_return_pct": total_return_pct,
        "max_drawdown_pct": max_drawdown * 100,
        "avg_position_lots": trades_df["position_lots"].mean(),
        "total_pnl_usd": trades_df["pnl_usd"].sum(),
        # Risk management metrics
        "max_losing_streak": max_losing_streak,
        "risk_reductions": risk_reductions,
    }


def run_wfo_window(
    window: WFOWindow,
    df_5min: pd.DataFrame,
    config: MTFEnsembleConfig,
    output_dir: Path,
    min_confidence: float = 0.55,
    initial_balance: float = 10000.0,
    risk_per_trade: float = 0.02,
    reduce_risk_on_losses: bool = True,
) -> WFOWindowResult:
    """Run walk-forward optimization on a single window.

    Args:
        window: The WFO window definition
        df_5min: Full 5-minute data
        config: MTF Ensemble configuration
        output_dir: Directory to save window models
        min_confidence: Minimum confidence for trading
        initial_balance: Starting account balance in USD
        risk_per_trade: Fraction of balance to risk per trade (e.g., 0.02 = 2%)
        reduce_risk_on_losses: If True, reduce position size after consecutive losses

    Returns:
        WFOWindowResult with all metrics
    """
    logger.info(f"\n{'=' * 70}")
    logger.info(f"WINDOW {window.window_id}")
    logger.info(f"Train: {window.train_period}")
    logger.info(f"Test:  {window.test_period}")
    logger.info(f"{'=' * 70}")

    # Filter data for training period
    df_train = df_5min[
        (df_5min.index >= window.train_start) &
        (df_5min.index <= window.train_end)
    ].copy()

    # Filter data for test period
    df_test = df_5min[
        (df_5min.index >= window.test_start) &
        (df_5min.index <= window.test_end)
    ].copy()

    logger.info(f"Training bars: {len(df_train)}")
    logger.info(f"Test bars: {len(df_test)}")

    if len(df_train) < 10000:
        logger.warning(f"Insufficient training data ({len(df_train)} bars), skipping window")
        return WFOWindowResult(window=window)

    if len(df_test) < 1000:
        logger.warning(f"Insufficient test data ({len(df_test)} bars), skipping window")
        return WFOWindowResult(window=window)

    # Create and train ensemble for this window
    window_model_dir = output_dir / f"window_{window.window_id}"
    ensemble = MTFEnsemble(config=config, model_dir=window_model_dir)

    # Train using 80/20 train/val split within training period
    train_results = ensemble.train(
        df_train,
        train_ratio=0.8,
        val_ratio=0.2,
        timeframes=["1H", "4H", "D"],
    )

    # Save the window's model
    ensemble.save()

    # Run backtest on test period with position sizing
    logger.info(f"Running backtest on test period...")
    logger.info(f"  Initial balance: ${initial_balance:,.0f}, Risk per trade: {risk_per_trade:.1%}")
    logger.info(f"  Risk reduction on losses: {'ON' if reduce_risk_on_losses else 'OFF'}")
    backtest_results = run_window_backtest(
        ensemble=ensemble,
        df_5min=df_test,
        min_confidence=min_confidence,
        initial_balance=initial_balance,
        risk_per_trade=risk_per_trade,
        reduce_risk_on_losses=reduce_risk_on_losses,
    )

    # Create result object
    result = WFOWindowResult(
        window=window,
        model_accuracies={
            tf: train_results.get(tf, {}).get("val_accuracy", 0)
            for tf in ["1H", "4H", "D"]
        },
        total_trades=backtest_results["total_trades"],
        winning_trades=backtest_results["winning_trades"],
        win_rate=backtest_results["win_rate"],
        total_pips=backtest_results["total_pips"],
        profit_factor=backtest_results["profit_factor"],
        avg_pips_per_trade=backtest_results["avg_pips"],
        high_conf_trades=backtest_results["high_conf_trades"],
        high_conf_win_rate=backtest_results["high_conf_win_rate"],
        long_trades=backtest_results["long_trades"],
        short_trades=backtest_results["short_trades"],
        long_win_rate=backtest_results["long_win_rate"],
        short_win_rate=backtest_results["short_win_rate"],
        tp_hits=backtest_results["tp_hits"],
        sl_hits=backtest_results["sl_hits"],
        timeouts=backtest_results["timeouts"],
        # Balance tracking metrics
        initial_balance=backtest_results["initial_balance"],
        final_balance=backtest_results["final_balance"],
        total_return_pct=backtest_results["total_return_pct"],
        max_drawdown_pct=backtest_results["max_drawdown_pct"],
        total_pnl_usd=backtest_results["total_pnl_usd"],
        # Risk management metrics
        max_losing_streak=backtest_results["max_losing_streak"],
        risk_reductions=backtest_results["risk_reductions"],
    )

    # Print window summary
    logger.info(f"\nWindow {window.window_id} Results:")
    logger.info(f"  Trades:        {result.total_trades}")
    logger.info(f"  Win Rate:      {result.win_rate:.1f}%")
    logger.info(f"  Total Pips:    {result.total_pips:+.1f}")
    logger.info(f"  Profit Factor: {result.profit_factor:.2f}")
    logger.info(f"  Balance:       ${result.initial_balance:,.0f} → ${result.final_balance:,.0f} ({result.total_return_pct:+.1f}%)")
    logger.info(f"  Max Drawdown:  {result.max_drawdown_pct:.1f}%")
    logger.info(f"  Model Accuracies: 1H={result.model_accuracies.get('1H', 0):.1%}, "
                f"4H={result.model_accuracies.get('4H', 0):.1%}, "
                f"D={result.model_accuracies.get('D', 0):.1%}")

    return result


def print_wfo_summary(summary: WFOSummary):
    """Print comprehensive WFO summary."""
    print("\n" + "=" * 100)
    print("WALK-FORWARD OPTIMIZATION SUMMARY")
    print("=" * 100)

    print(f"\n{'WINDOW RESULTS':^100}")
    print("-" * 100)
    print(f"{'Window':<8} {'Period':<20} {'Trades':>7} {'Win%':>7} {'PF':>6} {'Pips':>9} {'Return':>10} {'Max DD':>8}")
    print("-" * 100)

    for w in summary.windows:
        status = "+" if w.total_pips > 0 else "-"
        return_str = f"{w.total_return_pct:+.1f}%" if w.total_return_pct != 0 else "0.0%"
        dd_str = f"{w.max_drawdown_pct:.1f}%"
        print(f"{w.window.window_id:<8} {w.window.test_period:<20} "
              f"{w.total_trades:>7} {w.win_rate:>6.1f}% {w.profit_factor:>6.2f} "
              f"{status}{abs(w.total_pips):>8.0f} {return_str:>10} {dd_str:>8}")

    print("-" * 100)

    print(f"\n{'AGGREGATED METRICS':^80}")
    print("-" * 80)

    print(f"  Total Windows:          {summary.total_windows}")
    print(f"  Profitable Windows:     {summary.profitable_windows} ({summary.consistency_rate:.1%})")
    print(f"  Total Trades:           {summary.total_trades}")
    print(f"  Overall Win Rate:       {summary.overall_win_rate:.1f}%")
    print(f"  Overall Profit Factor:  {summary.overall_profit_factor:.2f}")

    print(f"\n{'PIPS ANALYSIS':^80}")
    print("-" * 80)
    print(f"  Total Pips (all windows):  {summary.total_pips:+.1f}")
    print(f"  Avg Pips per Window:       {summary.avg_pips_per_window:+.1f}")
    print(f"  Pips Std Dev:              {summary.pips_std:.1f}")
    print(f"  Best Window:               {summary.max_pips_window:+.1f}")
    print(f"  Worst Window:              {summary.min_pips_window:+.1f}")

    print(f"\n{'BALANCE ANALYSIS (Position Sizing)':^80}")
    print("-" * 80)
    print(f"  Initial Balance per Window: ${summary.initial_balance:,.0f}")
    print(f"  Avg Return per Window:      {summary.avg_return_per_window:+.1f}%")
    print(f"  Total Return (sum):         {summary.total_return_pct:+.1f}%")
    print(f"  Max Drawdown (worst):       {summary.max_drawdown:.1f}%")

    print(f"\n{'BASELINE COMPARISON':^80}")
    print("-" * 80)
    print(f"  {'Metric':<25} {'Baseline':>15} {'WFO':>15} {'Diff':>15}")
    print(f"  {'-' * 70}")
    print(f"  {'Total Pips':<25} {summary.baseline_total_pips:>+14.1f} "
          f"{summary.total_pips:>+14.1f} "
          f"{summary.total_pips - summary.baseline_total_pips:>+14.1f}")
    print(f"  {'Profit Factor':<25} {summary.baseline_profit_factor:>15.2f} "
          f"{summary.overall_profit_factor:>15.2f} "
          f"{summary.overall_profit_factor - summary.baseline_profit_factor:>+14.2f}")
    print(f"  {'Win Rate':<25} {summary.baseline_win_rate:>14.1f}% "
          f"{summary.overall_win_rate:>14.1f}% "
          f"{summary.overall_win_rate - summary.baseline_win_rate:>+13.1f}%")

    print(f"\n{'ROBUSTNESS ASSESSMENT':^80}")
    print("-" * 80)

    # Assess consistency
    if summary.consistency_rate >= 0.8:
        consistency_status = "[EXCELLENT]"
    elif summary.consistency_rate >= 0.6:
        consistency_status = "[GOOD]"
    elif summary.consistency_rate >= 0.5:
        consistency_status = "[ACCEPTABLE]"
    else:
        consistency_status = "[POOR]"

    # Assess stability (low variance is good)
    cv = summary.pips_std / abs(summary.avg_pips_per_window) if summary.avg_pips_per_window != 0 else float('inf')
    if cv < 1.0:
        stability_status = "[STABLE]"
    elif cv < 2.0:
        stability_status = "[MODERATE]"
    else:
        stability_status = "[UNSTABLE]"

    # Overall assessment
    if summary.consistency_rate >= 0.6 and summary.overall_profit_factor >= 1.5:
        overall_status = "ROBUST - Model performs consistently across different time periods"
    elif summary.consistency_rate >= 0.5 and summary.overall_profit_factor >= 1.0:
        overall_status = "ACCEPTABLE - Model shows reasonable consistency but may need refinement"
    else:
        overall_status = "NEEDS IMPROVEMENT - Model may be overfit to specific time periods"

    print(f"  Consistency: {summary.consistency_rate:.1%} {consistency_status}")
    print(f"  Stability (CV): {cv:.2f} {stability_status}")
    print(f"\n  OVERALL: {overall_status}")

    print("=" * 80)


def generate_api_backtest_results(
    summary: "WFOSummary",
    results_data: Dict,
    confidence_threshold: float,
) -> None:
    """Generate backtest_results.json for the What If Calculator API.

    This aggregates WFO window results into period-based data (6m, 1y, 2y, etc.)
    that can be queried by the frontend.

    The file is saved to data/backtest_results.json and is read by the API
    without requiring code changes or restarts.
    """
    api_results_path = project_root / "data" / "backtest_results.json"

    # Get window data sorted by test period end date
    windows = sorted(
        results_data["windows"],
        key=lambda w: w["test_period"].split(" to ")[1],
        reverse=True,  # Most recent first
    )

    # Calculate aggregate metrics for different periods
    # Each WFO window is ~6 months, so we combine windows for longer periods
    periods = {}

    # Helper to aggregate windows
    def aggregate_windows(window_list: List[Dict]) -> Dict:
        if not window_list:
            return None
        total_pips = sum(w["total_pips"] for w in window_list)
        total_trades = sum(w["total_trades"] for w in window_list)
        # Weighted average for win_rate and profit_factor
        total_wins = sum(w["win_rate"] * w["total_trades"] for w in window_list)
        avg_win_rate = total_wins / total_trades if total_trades > 0 else 0
        # Simple average for profit factor (weighted would need gross profit/loss)
        avg_pf = sum(w["profit_factor"] for w in window_list) / len(window_list)
        # Aggregate return metrics
        total_return = sum(w.get("total_return_pct", 0) for w in window_list)
        max_dd = max(w.get("max_drawdown_pct", 0) for w in window_list)
        total_pnl = sum(w.get("total_pnl_usd", 0) for w in window_list)
        return {
            "total_pips": round(total_pips),
            "win_rate": round(avg_win_rate, 3),
            "profit_factor": round(avg_pf, 2),
            "total_trades": total_trades,
            "total_return_pct": round(total_return, 2),
            "max_drawdown_pct": round(max_dd, 2),
            "total_pnl_usd": round(total_pnl, 2),
        }

    # Parse test period dates from windows
    def get_test_dates(window: Dict) -> Tuple[str, str]:
        parts = window["test_period"].split(" to ")
        return parts[0], parts[1]

    # Build period aggregations based on available windows
    # Most recent window (6m)
    if len(windows) >= 1:
        latest = windows[0]
        start, end = get_test_dates(latest)
        periods["6m"] = {
            "label": "Last 6 Months",
            **aggregate_windows([latest]),
            "period_start": f"{start}-01",
            "period_end": f"{end}-30",
            "period_years": 0.5,
            "period_months": 6,
        }

    # Last 2 windows (1y)
    if len(windows) >= 2:
        _, end = get_test_dates(windows[0])
        start, _ = get_test_dates(windows[1])
        periods["1y"] = {
            "label": "Last Year",
            **aggregate_windows(windows[:2]),
            "period_start": f"{start}-01",
            "period_end": f"{end}-30",
            "period_years": 1.0,
            "period_months": 12,
        }

    # Last 4 windows (2y)
    if len(windows) >= 4:
        _, end = get_test_dates(windows[0])
        start, _ = get_test_dates(windows[3])
        periods["2y"] = {
            "label": "Last 2 Years",
            **aggregate_windows(windows[:4]),
            "period_start": f"{start}-01",
            "period_end": f"{end}-30",
            "period_years": 2.0,
            "period_months": 24,
        }

    # Last 6 windows (3y)
    if len(windows) >= 6:
        _, end = get_test_dates(windows[0])
        start, _ = get_test_dates(windows[5])
        periods["3y"] = {
            "label": "Last 3 Years",
            **aggregate_windows(windows[:6]),
            "period_start": f"{start}-01",
            "period_end": f"{end}-30",
            "period_years": 3.0,
            "period_months": 36,
        }

    # All windows (5y or "All Time")
    if len(windows) >= 1:
        _, end = get_test_dates(windows[0])
        start, _ = get_test_dates(windows[-1])
        total_months = len(windows) * 6
        total_years = total_months / 12
        periods["5y"] = {
            "label": f"All Time ({total_years:.0f} Years)" if total_years >= 3 else "All Time",
            **aggregate_windows(windows),
            "period_start": f"{start}-01",
            "period_end": f"{end}-31",
            "period_years": total_years,
            "period_months": total_months,
        }

    # Get position sizing config from results_data
    config = results_data.get("config", {})
    initial_balance = config.get("initial_balance", 10000.0)
    risk_per_trade = config.get("risk_per_trade", 0.02)

    # Build the API response structure
    api_data = {
        "metadata": {
            "last_updated": datetime.now().isoformat(),
            "generated_by": "walk_forward_optimization.py",
            "confidence_threshold": confidence_threshold,
            "model_version": "mtf_ensemble_v1",
            "total_windows": len(windows),
            "position_sizing": {
                "initial_balance": initial_balance,
                "risk_per_trade": risk_per_trade,
                "method": "fixed_fractional",
            },
        },
        "periods": periods,
        "leverage_options": [
            {"value": 1, "label": "No Leverage (1:1)", "risk": "low"},
            {"value": 10, "label": "10:1", "risk": "medium"},
            {"value": 20, "label": "20:1", "risk": "high"},
            {"value": 30, "label": "30:1 (EU Retail)", "risk": "high"},
            {"value": 50, "label": "50:1", "risk": "extreme"},
        ],
        "forex_constants": {
            "standard_lot_size": 100000,
            "pip_value_per_lot": 10,
        },
    }

    # Save to file
    with open(api_results_path, "w") as f:
        json.dump(api_data, f, indent=2)

    logger.info(f"API backtest results saved to {api_results_path}")
    print(f"\n  API backtest data updated: {api_results_path}")
    print(f"  Periods available: {', '.join(periods.keys())}")


def main():
    parser = argparse.ArgumentParser(description="Walk-Forward Optimization for MTF Ensemble")
    parser.add_argument(
        "--data", type=str,
        default="data/forex/EURUSD_20200101_20251231_5min_combined.csv",
        help="Path to 5-minute data"
    )
    parser.add_argument(
        "--output", type=str,
        default="models/wfo_validation",
        help="Output directory for WFO models and results"
    )
    parser.add_argument(
        "--train-months", type=int, default=24,
        help="Training window duration in months"
    )
    parser.add_argument(
        "--test-months", type=int, default=6,
        help="Test window duration in months"
    )
    parser.add_argument(
        "--step-months", type=int, default=6,
        help="Step size in months between windows"
    )
    parser.add_argument(
        "--confidence", type=float, default=0.55,
        help="Minimum confidence threshold for trading"
    )
    parser.add_argument(
        "--sentiment", action="store_true",
        help="Include sentiment features (Daily model only)"
    )
    parser.add_argument(
        "--baseline-model-dir", type=str,
        default="models/mtf_ensemble",
        help="Directory of baseline model for comparison"
    )
    parser.add_argument(
        "--balance", type=float, default=10000.0,
        help="Initial account balance in USD for position sizing (default: 10000)"
    )
    parser.add_argument(
        "--risk", type=float, default=0.02,
        help="Risk per trade as fraction (e.g., 0.02 = 2%%, default: 2%%)"
    )
    parser.add_argument(
        "--no-risk-reduction", action="store_true",
        help="Disable reducing risk after consecutive losses (default: enabled)"
    )
    parser.add_argument(
        "--stacking", action="store_true",
        help="Enable stacking meta-learner (replaces fixed weighted averaging)"
    )
    parser.add_argument(
        "--stacking-blend", type=float, default=0.0,
        help="Blend ratio between stacking and weighted avg (0=pure stacking, 1=pure weighted, default: 0)"
    )
    parser.add_argument(
        "--enhanced-meta-features", action="store_true",
        help="Enable 12 additional meta-features for stacking meta-learner"
    )
    parser.add_argument(
        "--use-rfecv", action="store_true",
        help="Enable RFECV feature selection for all timeframes"
    )
    parser.add_argument(
        "--rfecv-min-features", type=int, default=20,
        help="Minimum features to select per model (default: 20)"
    )
    parser.add_argument(
        "--rfecv-step", type=float, default=0.1,
        help="Fraction of features to remove per RFECV iteration (default: 0.1)"
    )
    parser.add_argument(
        "--rfecv-cv-folds", type=int, default=3,
        help="Number of CV folds for RFECV (default: 3)"
    )
    parser.add_argument(
        "--calibration", action="store_true",
        help="Enable isotonic regression calibration for probability outputs (disabled by default)"
    )
    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("WALK-FORWARD OPTIMIZATION")
    print("=" * 80)
    print(f"Data:           {args.data}")
    print(f"Output:         {args.output}")
    print(f"Train Window:   {args.train_months} months")
    print(f"Test Window:    {args.test_months} months")
    print(f"Step Size:      {args.step_months} months")
    print(f"Min Confidence: {args.confidence}")
    print(f"Sentiment:      {'ON (Daily only)' if args.sentiment else 'OFF'}")
    print(f"Initial Balance: ${args.balance:,.0f}")
    print(f"Risk per Trade:  {args.risk:.1%}")
    print(f"Risk Reduction:  {'OFF' if args.no_risk_reduction else 'ON (reduce after losses)'}")
    print(f"Stacking:        {'ON' if args.stacking else 'OFF'}")
    if args.stacking:
        print(f"  Blend:         {args.stacking_blend}")
        print(f"  Enhanced:      {'ON (21 meta-features)' if args.enhanced_meta_features else 'OFF (9 meta-features)'}")
    print(f"RFECV:           {'ON' if args.use_rfecv else 'OFF'}")
    if args.use_rfecv:
        print(f"  Min Features:  {args.rfecv_min_features}")
        print(f"  Step:          {args.rfecv_step}")
        print(f"  CV Folds:      {args.rfecv_cv_folds}")
    print(f"Calibration:     {'ON (isotonic regression)' if args.calibration else 'OFF'}")
    print("=" * 80)

    # Load data
    data_path = project_root / args.data
    df_5min = load_data(data_path)

    # Create output directory
    output_dir = project_root / args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create WFO windows
    windows = create_wfo_windows(
        data_start=df_5min.index[0],
        data_end=df_5min.index[-1],
        train_months=args.train_months,
        test_months=args.test_months,
        step_months=args.step_months,
    )

    print(f"\nCreated {len(windows)} walk-forward windows:")
    for w in windows:
        print(f"  Window {w.window_id}: Train {w.train_period} | Test {w.test_period}")

    # Create RFECV config if enabled
    rfecv_config = None
    if args.use_rfecv:
        from src.models.feature_selection import RFECVConfig
        rfecv_config = RFECVConfig(
            min_features_to_select=args.rfecv_min_features,
            step=args.rfecv_step,
            cv=args.rfecv_cv_folds,
            verbose=0,  # Less verbose for WFO
        )

    # Create config
    stacking_config = None
    if args.stacking:
        # For WFO, use smaller min_train_size since daily data is limited
        stacking_config = StackingConfig(
            blend_with_weighted_avg=args.stacking_blend,
            n_folds=3,  # Reduce folds for smaller sample sizes in WFO
            min_train_size=100,  # Reduce min samples for WFO windows
            use_enhanced_meta_features=args.enhanced_meta_features,
        )

    if args.stacking and args.sentiment:
        config = MTFEnsembleConfig.with_stacking_and_sentiment(
            trading_pair="EURUSD",
            stacking_config=stacking_config,
        )
    elif args.stacking:
        config = MTFEnsembleConfig.with_stacking(stacking_config=stacking_config)
    elif args.sentiment:
        config = MTFEnsembleConfig.with_sentiment("EURUSD")
    else:
        config = MTFEnsembleConfig.default()

    # Add RFECV config if enabled
    if args.use_rfecv:
        config.use_rfecv = True
        config.rfecv_config = rfecv_config

    # Add calibration if enabled
    if args.calibration:
        config.use_calibration = True

    # Run WFO
    summary = WFOSummary(initial_balance=args.balance)

    for window in windows:
        try:
            result = run_wfo_window(
                window=window,
                df_5min=df_5min,
                config=config,
                output_dir=output_dir,
                min_confidence=args.confidence,
                initial_balance=args.balance,
                risk_per_trade=args.risk,
                reduce_risk_on_losses=not args.no_risk_reduction,
            )
            summary.windows.append(result)
        except Exception as e:
            logger.error(f"Error processing window {window.window_id}: {e}")
            import traceback
            traceback.print_exc()

    # Load baseline results for comparison
    baseline_metadata_path = project_root / args.baseline_model_dir / "training_metadata.json"
    if baseline_metadata_path.exists():
        with open(baseline_metadata_path) as f:
            baseline_metadata = json.load(f)
        # Use the documented baseline results
        summary.baseline_total_pips = 7987.0  # From CLAUDE.md
        summary.baseline_profit_factor = 2.22
        summary.baseline_win_rate = 57.8
    else:
        logger.warning("Baseline model metadata not found, using defaults")
        summary.baseline_total_pips = 7987.0
        summary.baseline_profit_factor = 2.22
        summary.baseline_win_rate = 57.8

    # Print summary
    print_wfo_summary(summary)

    # Save results
    results_data = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "train_months": args.train_months,
            "test_months": args.test_months,
            "step_months": args.step_months,
            "min_confidence": args.confidence,
            "sentiment": args.sentiment,
            "stacking": args.stacking,
            "stacking_blend": args.stacking_blend if args.stacking else None,
            "initial_balance": args.balance,
            "risk_per_trade": args.risk,
            "use_calibration": args.calibration,
        },
        "windows": [
            {
                "window_id": w.window.window_id,
                "train_period": w.window.train_period,
                "test_period": w.window.test_period,
                "total_trades": w.total_trades,
                "win_rate": w.win_rate,
                "total_pips": w.total_pips,
                "profit_factor": w.profit_factor,
                "model_accuracies": w.model_accuracies,
                "initial_balance": w.initial_balance,
                "final_balance": w.final_balance,
                "total_return_pct": w.total_return_pct,
                "max_drawdown_pct": w.max_drawdown_pct,
                "total_pnl_usd": w.total_pnl_usd,
            }
            for w in summary.windows
        ],
        "summary": {
            "total_windows": summary.total_windows,
            "profitable_windows": summary.profitable_windows,
            "consistency_rate": summary.consistency_rate,
            "total_trades": summary.total_trades,
            "overall_win_rate": summary.overall_win_rate,
            "overall_profit_factor": summary.overall_profit_factor,
            "total_pips": summary.total_pips,
            "avg_pips_per_window": summary.avg_pips_per_window,
            "pips_std": summary.pips_std,
            "min_pips_window": summary.min_pips_window,
            "max_pips_window": summary.max_pips_window,
            "initial_balance": summary.initial_balance,
            "total_return_pct": summary.total_return_pct,
            "avg_return_per_window": summary.avg_return_per_window,
            "max_drawdown": summary.max_drawdown,
        },
        "baseline_comparison": {
            "baseline_total_pips": summary.baseline_total_pips,
            "baseline_profit_factor": summary.baseline_profit_factor,
            "baseline_win_rate": summary.baseline_win_rate,
            "pips_diff": summary.total_pips - summary.baseline_total_pips,
            "pf_diff": summary.overall_profit_factor - summary.baseline_profit_factor,
        },
    }

    results_path = output_dir / "wfo_results.json"
    with open(results_path, "w") as f:
        json.dump(results_data, f, indent=2)

    logger.info(f"\nResults saved to {results_path}")

    # Generate API backtest results file for What If Calculator
    generate_api_backtest_results(summary, results_data, args.confidence)

    # Final recommendation
    print("\n" + "=" * 80)
    print("RECOMMENDATION")
    print("=" * 80)

    if summary.consistency_rate >= 0.7 and summary.overall_profit_factor >= 1.8:
        print("  The model demonstrates STRONG robustness across time periods.")
        print("  Recommendation: PROCEED with current model configuration.")
        print("  No retraining needed.")
    elif summary.consistency_rate >= 0.5 and summary.overall_profit_factor >= 1.2:
        print("  The model shows ACCEPTABLE robustness with some variance.")
        print("  Recommendation: Model is usable but consider:")
        print("    - More conservative position sizing")
        print("    - Higher confidence thresholds in live trading")
    else:
        print("  The model shows INCONSISTENT performance across periods.")
        print("  Recommendation: RETRAIN with the following adjustments:")
        print("    - Use walk-forward windows for training")
        print("    - Consider regime-specific models")
        print("    - Review feature engineering for overfitting")

    print("=" * 80)


if __name__ == "__main__":
    main()
