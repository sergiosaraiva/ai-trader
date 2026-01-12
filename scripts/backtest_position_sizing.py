#!/usr/bin/env python3
"""Backtest Position Sizing Strategies.

This script compares different position sizing strategies on the MTF Ensemble:
1. Fixed sizing (baseline) - 2% risk per trade
2. Full Kelly - Optimal growth but high variance
3. Half Kelly - Balance of growth and stability
4. Quarter Kelly - Conservative approach
5. Confidence-Adjusted Kelly - Scale by model confidence

The backtest uses the actual MTF Ensemble model and simulates account growth
over the test period with each sizing strategy.
"""

import argparse
import json
import logging
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

warnings.filterwarnings("ignore")

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd

from src.models.multi_timeframe import MTFEnsemble, MTFEnsembleConfig
from src.trading.position_sizing import (
    KellyPositionSizer,
    KellyParameters,
    PositionSizingConfig,
    SizingStrategy,
    calculate_kelly_from_stats,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class SizingBacktestResult:
    """Results from position sizing backtest."""
    strategy: str
    initial_balance: float
    final_balance: float
    total_return_pct: float
    total_pips: float
    total_trades: int
    winning_trades: int
    win_rate: float
    profit_factor: float
    max_drawdown_pct: float
    max_drawdown_amount: float
    sharpe_ratio: float
    avg_position_size: float
    max_position_size: float
    risk_adjusted_return: float  # Return / Max DD


def run_sizing_backtest(
    ensemble: MTFEnsemble,
    df_5min: pd.DataFrame,
    sizing_strategy: SizingStrategy,
    kelly_params: KellyParameters,
    initial_balance: float = 100000.0,
    min_confidence: float = 0.55,
    min_agreement: float = 0.5,
    tp_pips: float = 25.0,
    sl_pips: float = 15.0,
    max_holding_bars: int = 12,
    pip_value: float = 10.0,
) -> Tuple[SizingBacktestResult, List[Dict]]:
    """Run backtest with specific position sizing strategy.

    Args:
        ensemble: Trained MTF Ensemble
        df_5min: 5-minute price data
        sizing_strategy: Position sizing strategy to use
        kelly_params: Kelly criterion parameters
        initial_balance: Starting account balance
        min_confidence: Minimum confidence to trade
        min_agreement: Minimum agreement score
        tp_pips: Take profit in pips
        sl_pips: Stop loss in pips
        max_holding_bars: Maximum bars to hold position
        pip_value: Value per pip per lot

    Returns:
        Tuple of (SizingBacktestResult, list of trades)
    """
    from src.features.technical.calculator import TechnicalIndicatorCalculator

    calc = TechnicalIndicatorCalculator(model_type="short_term")

    # Configure position sizer with realistic limits
    # Use lower max_position_pct to see differentiation between strategies
    # and realistic growth patterns
    kelly_frac = {
        SizingStrategy.HALF_KELLY: 0.5,
        SizingStrategy.QUARTER_KELLY: 0.25,
        SizingStrategy.CONFIDENCE_KELLY: 0.5,
    }.get(sizing_strategy, 1.0)

    config = PositionSizingConfig(
        strategy=sizing_strategy,
        kelly_params=kelly_params,
        kelly_fraction=kelly_frac,
        fixed_risk_pct=0.01,     # 1% fixed risk per trade (conservative)
        max_position_pct=0.03,   # Max 3% per position
        min_position_pct=0.002,
        max_total_exposure=0.15,
        max_lot_size=2.0,        # Realistic max: 2 standard lots for $100k account
        min_lot_size=0.01,       # Micro lot
    )

    sizer = KellyPositionSizer(
        account_balance=initial_balance,
        config=config,
    )

    # Prepare data for all timeframes
    model_1h = ensemble.models["1H"]
    df_1h = ensemble.resample_data(df_5min, "1H")
    higher_tf_data_1h = ensemble.prepare_higher_tf_data(df_5min, "1H")
    df_1h_features = calc.calculate(df_1h)
    df_1h_features = model_1h.feature_engine.add_all_features(df_1h_features, higher_tf_data_1h)
    df_1h_features = df_1h_features.dropna()

    feature_cols_1h = model_1h.feature_names
    available_cols_1h = [c for c in feature_cols_1h if c in df_1h_features.columns]
    X_1h = df_1h_features[available_cols_1h].values

    # Split - use test portion only
    n_total = len(X_1h)
    n_train = int(n_total * 0.6)
    n_val = int(n_total * 0.2)
    test_start = n_train + n_val

    X_1h_test = X_1h[test_start:]
    df_test = df_1h_features.iloc[test_start:]

    preds_1h, confs_1h = model_1h.predict_batch(X_1h_test)

    # 4H predictions
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

    # Daily predictions
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

    # Ensemble weights
    weights = ensemble._normalize_weights(ensemble.config.weights)
    w_1h, w_4h, w_d = weights.get("1H", 0.6), weights.get("4H", 0.3), weights.get("D", 0.1)

    closes = df_test["close"].values
    highs = df_test["high"].values
    lows = df_test["low"].values
    timestamps = df_test.index

    # Combine predictions
    test_directions = []
    test_confidences = []
    test_agreements = []

    for i, ts in enumerate(timestamps):
        p_1h, c_1h = preds_1h[i], confs_1h[i]

        ts_4h = ts.floor("4H")
        if ts_4h in pred_4h_map:
            p_4h, c_4h = pred_4h_map[ts_4h]
        else:
            prev_4h = [t for t in pred_4h_map.keys() if t <= ts]
            p_4h, c_4h = pred_4h_map[max(prev_4h)] if prev_4h else (p_1h, c_1h)

        day = ts.date()
        if day in pred_d_map:
            p_d, c_d = pred_d_map[day]
        else:
            prev_days = [d for d in pred_d_map.keys() if d <= day]
            p_d, c_d = pred_d_map[max(prev_days)] if prev_days else (p_1h, c_1h)

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

    # Simulate trading with position sizing
    balance = initial_balance
    peak_balance = balance
    max_drawdown = 0.0
    max_drawdown_amount = 0.0
    trades = []
    balance_history = [balance]
    position_sizes = []

    pip_multiplier = 0.0001  # For EURUSD

    i = 0
    n = len(test_directions)

    while i < n - max_holding_bars:
        conf = test_confidences[i]
        agreement = test_agreements[i]
        pred = test_directions[i]

        if conf >= min_confidence and agreement >= min_agreement:
            # Calculate position size using the sizer
            position_size, sizing_details = sizer.calculate_position_size(
                confidence=conf,
                stop_loss_pips=sl_pips,
                pip_value=pip_value,
            )

            if position_size <= 0:
                i += 1
                continue

            position_sizes.append(position_size)

            entry_price = closes[i]
            entry_time = timestamps[i]
            direction = "long" if pred == 1 else "short"

            if direction == "long":
                tp_price = entry_price + tp_pips * pip_multiplier
                sl_price = entry_price - sl_pips * pip_multiplier
            else:
                tp_price = entry_price - tp_pips * pip_multiplier
                sl_price = entry_price + sl_pips * pip_multiplier

            exit_price = None
            exit_reason = None
            exit_idx = i

            for j in range(i + 1, min(i + max_holding_bars + 1, n)):
                if direction == "long":
                    if highs[j] >= tp_price:
                        exit_price, exit_reason = tp_price, "take_profit"
                        exit_idx = j
                        break
                    if lows[j] <= sl_price:
                        exit_price, exit_reason = sl_price, "stop_loss"
                        exit_idx = j
                        break
                else:
                    if lows[j] <= tp_price:
                        exit_price, exit_reason = tp_price, "take_profit"
                        exit_idx = j
                        break
                    if highs[j] >= sl_price:
                        exit_price, exit_reason = sl_price, "stop_loss"
                        exit_idx = j
                        break

            if exit_price is None:
                exit_idx = min(i + max_holding_bars, n - 1)
                exit_price = closes[exit_idx]
                exit_reason = "timeout"

            # Calculate P&L
            if direction == "long":
                pnl_pips = (exit_price - entry_price) / pip_multiplier
            else:
                pnl_pips = (entry_price - exit_price) / pip_multiplier

            # P&L in dollars = Position Size * Pips * Pip Value
            pnl_dollars = position_size * pnl_pips * pip_value

            # Update balance
            balance += pnl_dollars
            sizer.update_balance(balance)
            balance_history.append(balance)

            # Track drawdown
            if balance > peak_balance:
                peak_balance = balance
            current_dd = (peak_balance - balance) / peak_balance
            current_dd_amount = peak_balance - balance
            if current_dd > max_drawdown:
                max_drawdown = current_dd
                max_drawdown_amount = current_dd_amount

            trades.append({
                "entry_time": entry_time,
                "exit_time": timestamps[exit_idx],
                "direction": direction,
                "confidence": conf,
                "position_size": position_size,
                "pnl_pips": pnl_pips,
                "pnl_dollars": pnl_dollars,
                "exit_reason": exit_reason,
                "balance_after": balance,
            })

            i = exit_idx

        i += 1

    # Calculate results
    if not trades:
        return SizingBacktestResult(
            strategy=sizing_strategy.value,
            initial_balance=initial_balance,
            final_balance=initial_balance,
            total_return_pct=0,
            total_pips=0,
            total_trades=0,
            winning_trades=0,
            win_rate=0,
            profit_factor=0,
            max_drawdown_pct=0,
            max_drawdown_amount=0,
            sharpe_ratio=0,
            avg_position_size=0,
            max_position_size=0,
            risk_adjusted_return=0,
        ), []

    trades_df = pd.DataFrame(trades)
    wins = trades_df[trades_df["pnl_pips"] > 0]
    losses = trades_df[trades_df["pnl_pips"] <= 0]

    total_profit = wins["pnl_dollars"].sum() if len(wins) > 0 else 0
    total_loss = abs(losses["pnl_dollars"].sum()) if len(losses) > 0 else 0
    profit_factor = total_profit / total_loss if total_loss > 0 else float("inf")

    # Calculate Sharpe ratio (simplified)
    returns = np.diff(balance_history) / np.array(balance_history[:-1])
    sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252) if len(returns) > 1 and np.std(returns) > 0 else 0

    total_return = (balance - initial_balance) / initial_balance

    result = SizingBacktestResult(
        strategy=sizing_strategy.value,
        initial_balance=initial_balance,
        final_balance=balance,
        total_return_pct=total_return * 100,
        total_pips=trades_df["pnl_pips"].sum(),
        total_trades=len(trades_df),
        winning_trades=len(wins),
        win_rate=len(wins) / len(trades_df) * 100,
        profit_factor=profit_factor if profit_factor != float("inf") else 999.99,
        max_drawdown_pct=max_drawdown * 100,
        max_drawdown_amount=max_drawdown_amount,
        sharpe_ratio=sharpe,
        avg_position_size=np.mean(position_sizes) if position_sizes else 0,
        max_position_size=max(position_sizes) if position_sizes else 0,
        risk_adjusted_return=total_return / max_drawdown if max_drawdown > 0 else 0,
    )

    return result, trades


def main():
    parser = argparse.ArgumentParser(description="Backtest Position Sizing Strategies")
    parser.add_argument("--data", type=str, default="data/forex/EURUSD_20200101_20251231_5min_combined.csv")
    parser.add_argument("--model-dir", type=str, default="models/mtf_ensemble")
    parser.add_argument("--balance", type=float, default=100000, help="Initial account balance")
    parser.add_argument("--confidence", type=float, default=0.55, help="Minimum confidence")
    parser.add_argument("--wfo-results", type=str, default="models/wfo_validation/wfo_results.json")
    parser.add_argument("--output", type=str, default="results/position_sizing")
    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("POSITION SIZING STRATEGY COMPARISON")
    print("=" * 80)
    print(f"Data:            {args.data}")
    print(f"Model:           {args.model_dir}")
    print(f"Initial Balance: ${args.balance:,.2f}")
    print(f"Min Confidence:  {args.confidence}")
    print("=" * 80)

    # Load data
    data_path = project_root / args.data
    df = pd.read_csv(data_path)
    df.columns = [c.lower() for c in df.columns]
    time_col = next((c for c in ["timestamp", "time", "date", "datetime"] if c in df.columns), None)
    if time_col:
        df[time_col] = pd.to_datetime(df[time_col])
        df = df.set_index(time_col)
    df = df.sort_index()
    logger.info(f"Loaded {len(df)} bars")

    # Load model configuration
    model_dir = project_root / args.model_dir
    metadata_path = model_dir / "training_metadata.json"

    weights = {"1H": 0.6, "4H": 0.3, "D": 0.1}
    include_sentiment = False
    sentiment_by_timeframe = {"1H": False, "4H": False, "D": False}
    sentiment_source = "epu"

    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)
            include_sentiment = metadata.get("include_sentiment", False)
            weights = metadata.get("weights", weights)
            sentiment_by_timeframe = metadata.get("sentiment_by_timeframe", sentiment_by_timeframe)
            sentiment_source = metadata.get("sentiment_source", "epu")

    config = MTFEnsembleConfig(
        weights=weights,
        include_sentiment=include_sentiment,
        sentiment_source=sentiment_source,
        sentiment_by_timeframe=sentiment_by_timeframe,
    )

    ensemble = MTFEnsemble(config=config, model_dir=model_dir)
    ensemble.load()

    # Calculate Kelly parameters from WFO results or baseline
    wfo_path = project_root / args.wfo_results
    if wfo_path.exists():
        kelly_params = KellyParameters.from_wfo_results(wfo_path)
        print(f"\nKelly Parameters (from WFO):")
    else:
        # Use baseline stats from CLAUDE.md
        kelly_params = KellyParameters(
            win_rate=0.578,
            profit_factor=2.22,
            avg_win_pips=22.8,
            avg_loss_pips=14.1,
        )
        print(f"\nKelly Parameters (from baseline):")

    print(f"  Win Rate:       {kelly_params.win_rate:.2%}")
    print(f"  Profit Factor:  {kelly_params.profit_factor:.2f}")
    print(f"  Win/Loss Ratio: {kelly_params.win_loss_ratio:.2f}")
    print(f"  Full Kelly:     {kelly_params.full_kelly:.2%}")

    # Define strategies to test
    strategies = [
        SizingStrategy.FIXED,
        SizingStrategy.QUARTER_KELLY,
        SizingStrategy.HALF_KELLY,
        SizingStrategy.FULL_KELLY,
        SizingStrategy.CONFIDENCE_KELLY,
    ]

    # Run backtests
    results = []
    all_trades = {}

    for strategy in strategies:
        print(f"\nRunning backtest: {strategy.value}...")
        result, trades = run_sizing_backtest(
            ensemble=ensemble,
            df_5min=df,
            sizing_strategy=strategy,
            kelly_params=kelly_params,
            initial_balance=args.balance,
            min_confidence=args.confidence,
        )
        results.append(result)
        all_trades[strategy.value] = trades
        print(f"  Final Balance: ${result.final_balance:,.2f} ({result.total_return_pct:+.1f}%)")

    # Print comparison table
    print("\n" + "=" * 100)
    print("RESULTS COMPARISON")
    print("=" * 100)

    headers = ["Strategy", "Final Balance", "Return %", "Pips", "Trades", "Win%", "PF", "MaxDD%", "Sharpe", "RAR"]
    print(f"{'Strategy':<20} {'Final Balance':>15} {'Return%':>10} {'Pips':>10} {'Trades':>8} "
          f"{'Win%':>8} {'PF':>8} {'MaxDD%':>8} {'Sharpe':>8} {'RAR':>8}")
    print("-" * 110)

    for r in results:
        print(f"{r.strategy:<20} ${r.final_balance:>13,.0f} {r.total_return_pct:>+9.1f}% "
              f"{r.total_pips:>+9.0f} {r.total_trades:>8} {r.win_rate:>7.1f}% "
              f"{r.profit_factor:>7.2f} {r.max_drawdown_pct:>7.1f}% "
              f"{r.sharpe_ratio:>7.2f} {r.risk_adjusted_return:>7.2f}")

    # Find best strategy by risk-adjusted return
    best_rar = max(results, key=lambda x: x.risk_adjusted_return)
    best_return = max(results, key=lambda x: x.total_return_pct)
    lowest_dd = min(results, key=lambda x: x.max_drawdown_pct)

    print("\n" + "=" * 80)
    print("ANALYSIS")
    print("=" * 80)

    print(f"\nBest Risk-Adjusted Return: {best_rar.strategy}")
    print(f"  Return: {best_rar.total_return_pct:+.1f}%, MaxDD: {best_rar.max_drawdown_pct:.1f}%, RAR: {best_rar.risk_adjusted_return:.2f}")

    print(f"\nHighest Return: {best_return.strategy}")
    print(f"  Return: {best_return.total_return_pct:+.1f}%, MaxDD: {best_return.max_drawdown_pct:.1f}%")

    print(f"\nLowest Drawdown: {lowest_dd.strategy}")
    print(f"  Return: {lowest_dd.total_return_pct:+.1f}%, MaxDD: {lowest_dd.max_drawdown_pct:.1f}%")

    # Compare to baseline (fixed sizing)
    fixed_result = next(r for r in results if r.strategy == "fixed")
    print(f"\n{'COMPARISON TO FIXED SIZING':^80}")
    print("-" * 80)
    print(f"{'Strategy':<20} {'Return Diff':>15} {'DD Diff':>15} {'RAR Diff':>15}")
    print("-" * 80)

    for r in results:
        if r.strategy != "fixed":
            return_diff = r.total_return_pct - fixed_result.total_return_pct
            dd_diff = r.max_drawdown_pct - fixed_result.max_drawdown_pct
            rar_diff = r.risk_adjusted_return - fixed_result.risk_adjusted_return
            print(f"{r.strategy:<20} {return_diff:>+14.1f}% {dd_diff:>+14.1f}% {rar_diff:>+14.2f}")

    # Save results
    output_dir = project_root / args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    results_data = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "initial_balance": args.balance,
            "min_confidence": args.confidence,
            "data_file": args.data,
        },
        "kelly_params": {
            "win_rate": kelly_params.win_rate,
            "profit_factor": kelly_params.profit_factor,
            "win_loss_ratio": kelly_params.win_loss_ratio,
            "full_kelly": kelly_params.full_kelly,
        },
        "results": [
            {
                "strategy": r.strategy,
                "final_balance": r.final_balance,
                "total_return_pct": r.total_return_pct,
                "total_pips": r.total_pips,
                "total_trades": r.total_trades,
                "win_rate": r.win_rate,
                "profit_factor": r.profit_factor,
                "max_drawdown_pct": r.max_drawdown_pct,
                "sharpe_ratio": r.sharpe_ratio,
                "risk_adjusted_return": r.risk_adjusted_return,
                "avg_position_size": r.avg_position_size,
            }
            for r in results
        ],
        "recommendation": best_rar.strategy,
    }

    results_path = output_dir / "position_sizing_comparison.json"
    with open(results_path, "w") as f:
        json.dump(results_data, f, indent=2)

    print(f"\nResults saved to {results_path}")

    # Recommendation
    print("\n" + "=" * 80)
    print("RECOMMENDATION")
    print("=" * 80)

    if best_rar.strategy == "half_kelly":
        print("  Half Kelly (50%) provides the best risk-adjusted returns.")
        print("  This balances growth potential with acceptable drawdowns.")
    elif best_rar.strategy == "quarter_kelly":
        print("  Quarter Kelly (25%) provides the best risk-adjusted returns.")
        print("  This is a conservative approach suitable for risk-averse traders.")
    elif best_rar.strategy == "confidence_kelly":
        print("  Confidence-Adjusted Kelly provides the best risk-adjusted returns.")
        print("  Position size scales with model confidence for optimal capital allocation.")
    else:
        print(f"  {best_rar.strategy} provides the best risk-adjusted returns.")

    print(f"\n  Recommended Kelly Fraction: {kelly_params.full_kelly * 0.5:.2%} (Half Kelly)")
    print(f"  For $100,000 account, risk ~${args.balance * kelly_params.full_kelly * 0.5:,.0f} per trade")

    print("=" * 80)


if __name__ == "__main__":
    main()
