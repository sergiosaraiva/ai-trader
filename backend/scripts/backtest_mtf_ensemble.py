#!/usr/bin/env python3
"""Backtest Multi-Timeframe Ensemble model.

This script simulates trading using the MTF Ensemble:
- Uses triple barrier method for trade outcomes
- Tracks win rate, profit factor, and other metrics
- Compares ensemble to individual models
"""

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict

import warnings
warnings.filterwarnings("ignore")

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd

from src.models.multi_timeframe import MTFEnsemble, MTFEnsembleConfig, MTFPrediction

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class Trade:
    """Represents a single trade."""
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    direction: str  # "long" or "short"
    entry_price: float
    exit_price: float
    confidence: float
    agreement_score: float
    pnl_pips: float
    exit_reason: str  # "take_profit", "stop_loss", "timeout"


class MTFEnsembleBacktester:
    """Backtester for MTF Ensemble."""

    def __init__(
        self,
        ensemble: MTFEnsemble,
        min_confidence: float = 0.55,
        min_agreement: float = 0.5,
        tp_pips: float = 25.0,
        sl_pips: float = 15.0,
        max_holding_bars: int = 12,
        filter_mode: bool = False,
        strict_filter: bool = False,
        use_dynamic_weights: bool = False,
    ):
        self.ensemble = ensemble
        self.min_confidence = min_confidence
        self.min_agreement = min_agreement
        self.tp_pips = tp_pips
        self.sl_pips = sl_pips
        self.max_holding_bars = max_holding_bars
        self.filter_mode = filter_mode
        self.strict_filter = strict_filter
        self.use_dynamic_weights = use_dynamic_weights
        self.trades: List[Trade] = []

    def run(self, df_5min: pd.DataFrame, test_start_idx: int) -> Dict:
        """Run backtest on test data.

        This backtest uses the 1H timeframe for trade simulation, with ensemble
        predictions combining 1H, 4H, and Daily model outputs.

        Args:
            df_5min: 5-minute OHLCV data
            test_start_idx: Index where test period starts

        Returns:
            Dict of performance metrics
        """
        from src.features.technical.calculator import TechnicalIndicatorCalculator

        logger.info("Preparing data for backtest...")

        calc = TechnicalIndicatorCalculator(model_type="short_term")

        # Prepare 1H data (primary trading timeframe)
        model_1h = self.ensemble.models["1H"]
        df_1h = self.ensemble.resample_data(df_5min, "1H")
        higher_tf_data_1h = self.ensemble.prepare_higher_tf_data(df_5min, "1H")
        df_1h_features = calc.calculate(df_1h)
        df_1h_features = model_1h.feature_engine.add_all_features(df_1h_features, higher_tf_data_1h)
        df_1h_features = df_1h_features.dropna()

        feature_cols_1h = model_1h.feature_names
        available_cols_1h = [c for c in feature_cols_1h if c in df_1h_features.columns]
        X_1h = df_1h_features[available_cols_1h].values

        logger.info(f"1H: {len(X_1h)} samples")

        # Split 1H data
        n_total = len(X_1h)
        n_train = int(n_total * 0.6)
        n_val = int(n_total * 0.2)
        test_start = n_train + n_val

        X_1h_test = X_1h[test_start:]
        df_test = df_1h_features.iloc[test_start:]

        logger.info(f"Test period: bar {test_start} to {n_total}")
        logger.info(f"Test bars: {len(X_1h_test)}")

        # Get 1H predictions
        preds_1h, confs_1h = model_1h.predict_batch(X_1h_test)

        # Prepare 4H data and get predictions
        model_4h = self.ensemble.models["4H"]
        df_4h = self.ensemble.resample_data(df_5min, "4H")
        higher_tf_data_4h = self.ensemble.prepare_higher_tf_data(df_5min, "4H")
        df_4h_features = calc.calculate(df_4h)
        df_4h_features = model_4h.feature_engine.add_all_features(df_4h_features, higher_tf_data_4h)
        df_4h_features = df_4h_features.dropna()

        feature_cols_4h = model_4h.feature_names
        available_cols_4h = [c for c in feature_cols_4h if c in df_4h_features.columns]
        X_4h = df_4h_features[available_cols_4h].values
        preds_4h_all, confs_4h_all = model_4h.predict_batch(X_4h)

        # Create a mapping from timestamp to 4H prediction
        pred_4h_map = dict(zip(df_4h_features.index, zip(preds_4h_all, confs_4h_all)))

        # Prepare Daily data and get predictions
        model_d = self.ensemble.models["D"]
        df_d = self.ensemble.resample_data(df_5min, "D")
        higher_tf_data_d = {}  # Daily doesn't need higher TF data
        df_d_features = calc.calculate(df_d)
        df_d_features = model_d.feature_engine.add_all_features(df_d_features, higher_tf_data_d)
        df_d_features = df_d_features.dropna()

        feature_cols_d = model_d.feature_names
        available_cols_d = [c for c in feature_cols_d if c in df_d_features.columns]
        X_d = df_d_features[available_cols_d].values
        preds_d_all, confs_d_all = model_d.predict_batch(X_d)

        # Create a mapping from date to Daily prediction
        pred_d_map = dict(zip(df_d_features.index.date, zip(preds_d_all, confs_d_all)))

        logger.info(f"4H: {len(pred_4h_map)} predictions, D: {len(pred_d_map)} predictions")

        # Now combine predictions for each test bar
        if self.strict_filter:
            logger.info("Combining predictions using STRICT FILTER MODE (1H primary, 4H must agree)...")
        elif self.filter_mode:
            logger.info("Combining predictions using FILTER MODE (1H primary, 4H/D filters)...")
        elif self.use_dynamic_weights:
            logger.info("Combining predictions with DYNAMIC WEIGHTS (adapting based on recent accuracy)...")
        else:
            logger.info("Combining predictions for ensemble...")

        # Get initial weights (will be updated dynamically if enabled)
        weights = self.ensemble._normalize_weights(self.ensemble.config.weights)
        w_1h = weights.get("1H", 0.6)
        w_4h = weights.get("4H", 0.3)
        w_d = weights.get("D", 0.1)

        # Get price data
        closes = df_test["close"].values
        highs = df_test["high"].values
        lows = df_test["low"].values
        timestamps = df_test.index

        # Pre-compute ensemble predictions for all test bars
        test_directions = []
        test_confidences = []
        test_agreements = []

        for i, ts in enumerate(timestamps):
            p_1h, c_1h = preds_1h[i], confs_1h[i]

            # Find corresponding 4H prediction (most recent at or before this timestamp)
            ts_4h = ts.floor("4h")
            if ts_4h in pred_4h_map:
                p_4h, c_4h = pred_4h_map[ts_4h]
            else:
                # Fall back to previous 4H bar
                prev_4h_times = [t for t in pred_4h_map.keys() if t <= ts]
                if prev_4h_times:
                    p_4h, c_4h = pred_4h_map[max(prev_4h_times)]
                else:
                    p_4h, c_4h = p_1h, c_1h  # fallback to 1H

            # Find corresponding Daily prediction
            day = ts.date()
            if day in pred_d_map:
                p_d, c_d = pred_d_map[day]
            else:
                # Fall back to previous day
                prev_days = [d for d in pred_d_map.keys() if d <= day]
                if prev_days:
                    p_d, c_d = pred_d_map[max(prev_days)]
                else:
                    p_d, c_d = p_1h, c_1h  # fallback to 1H

            if self.strict_filter:
                # STRICT FILTER MODE: 1H is primary, 4H must agree to trade
                # Direction comes from 1H only
                direction = p_1h

                # Agreement calculation
                agreement_count = sum([1 for p in [p_1h, p_4h, p_d] if p == direction])
                agreement_score = agreement_count / 3.0

                # In strict mode, if 4H disagrees, set confidence to 0 (skip trade)
                if p_4h != direction:
                    conf = 0.0  # Will be filtered out by min_confidence
                else:
                    # 4H agrees - use 1H confidence with small D adjustment
                    base_conf = c_1h
                    if p_d == direction:
                        conf = min(base_conf + 0.05, 1.0)  # +5% for full agreement
                    elif c_d >= 0.65:  # D strongly disagrees
                        conf = base_conf * 0.95  # Small penalty
                    else:
                        conf = base_conf

            elif self.filter_mode:
                # FILTER MODE: 1H is primary signal, 4H/D are filters
                # Direction comes from 1H only
                direction = p_1h

                # Agreement calculation
                agreement_count = sum([1 for p in [p_1h, p_4h, p_d] if p == direction])
                agreement_score = agreement_count / 3.0

                # Confidence logic for filter mode:
                # - Start with 1H confidence
                # - Boost if 4H agrees
                # - Boost more if D agrees
                # - Penalize if 4H or D strongly disagrees

                base_conf = c_1h

                # 4H filter: if agrees, boost; if disagrees with high confidence, reduce
                if p_4h == direction:
                    base_conf = min(base_conf + 0.05, 1.0)  # +5% for 4H agreement
                elif c_4h >= 0.65:  # 4H strongly disagrees
                    base_conf = base_conf * 0.85  # -15% penalty

                # Daily filter: smaller effect
                if p_d == direction:
                    base_conf = min(base_conf + 0.02, 1.0)  # +2% for D agreement
                elif c_d >= 0.65:  # D strongly disagrees
                    base_conf = base_conf * 0.95  # -5% penalty

                # Full agreement bonus
                if agreement_count == 3:
                    conf = min(base_conf + 0.05, 1.0)
                else:
                    conf = base_conf

            else:
                # WEIGHTED MODE: Original weighted combination (with optional dynamic weights)
                # Update weights dynamically if enabled and we have enough history
                if self.use_dynamic_weights and len(self.ensemble.prediction_history) >= 10:
                    dyn_weights = self.ensemble._calculate_dynamic_weights()
                    w_1h = dyn_weights.get("1H", 0.6)
                    w_4h = dyn_weights.get("4H", 0.3)
                    w_d = dyn_weights.get("D", 0.1)

                prob_up_1h = c_1h if p_1h == 1 else 1 - c_1h
                prob_up_4h = c_4h if p_4h == 1 else 1 - c_4h
                prob_up_d = c_d if p_d == 1 else 1 - c_d

                weighted_prob_up = w_1h * prob_up_1h + w_4h * prob_up_4h + w_d * prob_up_d

                direction = 1 if weighted_prob_up > 0.5 else 0
                base_conf = abs(weighted_prob_up - 0.5) * 2 + 0.5

                # Agreement
                agreement_count = sum([1 for p in [p_1h, p_4h, p_d] if p == direction])
                agreement_score = agreement_count / 3.0

                # Agreement bonus
                if agreement_count == 3:
                    conf = min(base_conf + self.ensemble.config.agreement_bonus, 1.0)
                else:
                    conf = base_conf

            test_directions.append(direction)
            test_confidences.append(conf)
            test_agreements.append(agreement_score)

        test_directions = np.array(test_directions)
        test_confidences = np.array(test_confidences)
        test_agreements = np.array(test_agreements)

        logger.info(f"Simulating trades on {len(test_directions)} bars...")

        # Simulate trading
        pip_value = 0.0001
        n = len(test_directions)
        i = 0

        # Track dynamic weights over time for logging
        weight_history = []

        while i < n - self.max_holding_bars:
            # For dynamic weights, recompute prediction at trade time
            if self.use_dynamic_weights and len(self.ensemble.prediction_history) >= 10:
                dyn_weights = self.ensemble._calculate_dynamic_weights()
                w_1h = dyn_weights.get("1H", 0.6)
                w_4h = dyn_weights.get("4H", 0.3)
                w_d = dyn_weights.get("D", 0.1)

                # Recompute weighted combination with updated weights
                ts = timestamps[i]
                p_1h, c_1h = preds_1h[i], confs_1h[i]

                ts_4h = ts.floor("4h")
                if ts_4h in pred_4h_map:
                    p_4h, c_4h = pred_4h_map[ts_4h]
                else:
                    prev_4h_times = [t for t in pred_4h_map.keys() if t <= ts]
                    if prev_4h_times:
                        p_4h, c_4h = pred_4h_map[max(prev_4h_times)]
                    else:
                        p_4h, c_4h = p_1h, c_1h

                day = ts.date()
                if day in pred_d_map:
                    p_d, c_d = pred_d_map[day]
                else:
                    prev_days = [d for d in pred_d_map.keys() if d <= day]
                    if prev_days:
                        p_d, c_d = pred_d_map[max(prev_days)]
                    else:
                        p_d, c_d = p_1h, c_1h

                prob_up_1h = c_1h if p_1h == 1 else 1 - c_1h
                prob_up_4h = c_4h if p_4h == 1 else 1 - c_4h
                prob_up_d = c_d if p_d == 1 else 1 - c_d

                weighted_prob_up = w_1h * prob_up_1h + w_4h * prob_up_4h + w_d * prob_up_d
                pred = 1 if weighted_prob_up > 0.5 else 0
                base_conf = abs(weighted_prob_up - 0.5) * 2 + 0.5

                agreement_count = sum([1 for p in [p_1h, p_4h, p_d] if p == pred])
                agreement = agreement_count / 3.0

                if agreement_count == 3:
                    conf = min(base_conf + self.ensemble.config.agreement_bonus, 1.0)
                else:
                    conf = base_conf

                # Log weight changes periodically
                if len(weight_history) == 0 or len(self.trades) % 50 == 0:
                    weight_history.append((len(self.trades), w_1h, w_4h, w_d))
            else:
                conf = test_confidences[i]
                agreement = test_agreements[i]
                pred = test_directions[i]

            # Check entry conditions
            if conf >= self.min_confidence and agreement >= self.min_agreement:
                entry_price = closes[i]
                entry_time = timestamps[i]
                direction = "long" if pred == 1 else "short"

                # Set TP/SL levels
                if direction == "long":
                    tp_price = entry_price + self.tp_pips * pip_value
                    sl_price = entry_price - self.sl_pips * pip_value
                else:
                    tp_price = entry_price - self.tp_pips * pip_value
                    sl_price = entry_price + self.sl_pips * pip_value

                exit_price = None
                exit_reason = None
                exit_idx = i

                # Simulate trade
                for j in range(i + 1, min(i + self.max_holding_bars + 1, n)):
                    if direction == "long":
                        if highs[j] >= tp_price:
                            exit_price, exit_reason = tp_price, "take_profit"
                            exit_idx = j
                            break
                        if lows[j] <= sl_price:
                            exit_price, exit_reason = sl_price, "stop_loss"
                            exit_idx = j
                            break
                    else:  # short
                        if lows[j] <= tp_price:
                            exit_price, exit_reason = tp_price, "take_profit"
                            exit_idx = j
                            break
                        if highs[j] >= sl_price:
                            exit_price, exit_reason = sl_price, "stop_loss"
                            exit_idx = j
                            break

                # Timeout
                if exit_price is None:
                    exit_idx = min(i + self.max_holding_bars, n - 1)
                    exit_price = closes[exit_idx]
                    exit_reason = "timeout"

                # Calculate P&L
                if direction == "long":
                    pnl_pips = (exit_price - entry_price) / pip_value
                else:
                    pnl_pips = (entry_price - exit_price) / pip_value

                self.trades.append(Trade(
                    entry_time=entry_time,
                    exit_time=timestamps[exit_idx],
                    direction=direction,
                    entry_price=entry_price,
                    exit_price=exit_price,
                    confidence=conf,
                    agreement_score=agreement,
                    pnl_pips=pnl_pips,
                    exit_reason=exit_reason,
                ))

                # Record outcome for dynamic weights if enabled
                if self.use_dynamic_weights:
                    # Determine actual direction (1=up, 0=down) based on price movement
                    actual_direction = 1 if exit_price > entry_price else 0

                    # Get individual model predictions at entry time
                    entry_idx = i
                    ts = timestamps[entry_idx]
                    p_1h_entry, c_1h_entry = preds_1h[entry_idx], confs_1h[entry_idx]

                    ts_4h = ts.floor("4h")
                    if ts_4h in pred_4h_map:
                        p_4h_entry, c_4h_entry = pred_4h_map[ts_4h]
                    else:
                        prev_4h = [t for t in pred_4h_map.keys() if t <= ts]
                        p_4h_entry, c_4h_entry = pred_4h_map[max(prev_4h)] if prev_4h else (p_1h_entry, c_1h_entry)

                    day = ts.date()
                    if day in pred_d_map:
                        p_d_entry, c_d_entry = pred_d_map[day]
                    else:
                        prev_days = [d for d in pred_d_map.keys() if d <= day]
                        p_d_entry, c_d_entry = pred_d_map[max(prev_days)] if prev_days else (p_1h_entry, c_1h_entry)

                    # Create MTFPrediction object for record_outcome
                    prob_up = conf if pred == 1 else 1 - conf
                    mock_prediction = MTFPrediction(
                        direction=pred,
                        confidence=conf,
                        prob_up=prob_up,
                        prob_down=1 - prob_up,
                        agreement_score=agreement,
                        component_directions={"1H": p_1h_entry, "4H": p_4h_entry, "D": p_d_entry},
                        component_confidences={"1H": c_1h_entry, "4H": c_4h_entry, "D": c_d_entry},
                        component_weights={"1H": w_1h, "4H": w_4h, "D": w_d},
                    )
                    self.ensemble.record_outcome(mock_prediction, actual_direction)

                # Skip to after exit
                i = exit_idx

            i += 1

        # Log dynamic weight evolution if enabled
        if self.use_dynamic_weights and weight_history:
            logger.info("Dynamic weight evolution:")
            for trade_num, w1h, w4h, wd in weight_history:
                logger.info(f"  After {trade_num} trades: 1H={w1h:.3f}, 4H={w4h:.3f}, D={wd:.3f}")
            # Log final weights
            if len(self.ensemble.prediction_history) >= 10:
                final_weights = self.ensemble._calculate_dynamic_weights()
                logger.info(f"  Final: 1H={final_weights['1H']:.3f}, 4H={final_weights['4H']:.3f}, D={final_weights['D']:.3f}")

        return self._calculate_results()

    def _calculate_results(self) -> Dict:
        """Calculate performance metrics."""
        if not self.trades:
            return {
                "total_trades": 0,
                "win_rate": 0,
                "total_pips": 0,
                "profit_factor": 0,
            }

        trades_df = pd.DataFrame([{
            "direction": t.direction,
            "confidence": t.confidence,
            "agreement_score": t.agreement_score,
            "pnl_pips": t.pnl_pips,
            "exit_reason": t.exit_reason,
        } for t in self.trades])

        wins = trades_df[trades_df["pnl_pips"] > 0]
        losses = trades_df[trades_df["pnl_pips"] <= 0]

        total_profit = wins["pnl_pips"].sum() if len(wins) > 0 else 0
        total_loss = abs(losses["pnl_pips"].sum()) if len(losses) > 0 else 0
        profit_factor = total_profit / total_loss if total_loss > 0 else float("inf")

        # Analyze by confidence level
        high_conf = trades_df[trades_df["confidence"] >= 0.60]
        very_high_conf = trades_df[trades_df["confidence"] >= 0.65]

        # Analyze by agreement
        full_agree = trades_df[trades_df["agreement_score"] == 1.0]

        return {
            "total_trades": len(trades_df),
            "winning_trades": len(wins),
            "losing_trades": len(losses),
            "win_rate": len(wins) / len(trades_df) * 100,
            "total_pips": trades_df["pnl_pips"].sum(),
            "avg_pips": trades_df["pnl_pips"].mean(),
            "profit_factor": profit_factor,
            "avg_win": total_profit / len(wins) if len(wins) > 0 else 0,
            "avg_loss": total_loss / len(losses) if len(losses) > 0 else 0,
            "tp_hits": len(trades_df[trades_df["exit_reason"] == "take_profit"]),
            "sl_hits": len(trades_df[trades_df["exit_reason"] == "stop_loss"]),
            "timeouts": len(trades_df[trades_df["exit_reason"] == "timeout"]),
            # By confidence
            "high_conf_trades": len(high_conf),
            "high_conf_win_rate": len(high_conf[high_conf["pnl_pips"] > 0]) / len(high_conf) * 100 if len(high_conf) > 0 else 0,
            "very_high_conf_trades": len(very_high_conf),
            "very_high_conf_win_rate": len(very_high_conf[very_high_conf["pnl_pips"] > 0]) / len(very_high_conf) * 100 if len(very_high_conf) > 0 else 0,
            # By agreement
            "full_agree_trades": len(full_agree),
            "full_agree_win_rate": len(full_agree[full_agree["pnl_pips"] > 0]) / len(full_agree) * 100 if len(full_agree) > 0 else 0,
            # By direction
            "long_trades": len(trades_df[trades_df["direction"] == "long"]),
            "short_trades": len(trades_df[trades_df["direction"] == "short"]),
            "long_win_rate": len(trades_df[(trades_df["direction"] == "long") & (trades_df["pnl_pips"] > 0)]) / len(trades_df[trades_df["direction"] == "long"]) * 100 if len(trades_df[trades_df["direction"] == "long"]) > 0 else 0,
            "short_win_rate": len(trades_df[(trades_df["direction"] == "short") & (trades_df["pnl_pips"] > 0)]) / len(trades_df[trades_df["direction"] == "short"]) * 100 if len(trades_df[trades_df["direction"] == "short"]) > 0 else 0,
        }


def run_individual_backtest(
    ensemble: MTFEnsemble,
    df_5min: pd.DataFrame,
    timeframe: str,
    min_confidence: float = 0.55,
) -> Dict:
    """Run backtest using a single model for comparison."""
    from src.features.technical.calculator import TechnicalIndicatorCalculator

    model = ensemble.models[timeframe]
    config = ensemble.model_configs[timeframe]

    # Resample
    df_tf = ensemble.resample_data(df_5min, config.base_timeframe)

    # Prepare higher TF data
    higher_tf_data = ensemble.prepare_higher_tf_data(df_5min, config.base_timeframe)

    # Calculate features
    calc = TechnicalIndicatorCalculator(model_type="short_term")
    df_features = calc.calculate(df_tf)
    df_features = model.feature_engine.add_all_features(df_features, higher_tf_data)
    df_features = df_features.dropna()

    # Get predictions
    feature_cols = model.feature_names
    available_cols = [c for c in feature_cols if c in df_features.columns]
    X = df_features[available_cols].values

    # Split
    n_total = len(X)
    n_train = int(n_total * 0.6)
    n_val = int(n_total * 0.2)
    test_start = n_train + n_val

    X_test = X[test_start:]
    df_test = df_features.iloc[test_start:]

    predictions, confidences = model.predict_batch(X_test)

    # Simulate trades
    pip_value = 0.0001
    tp_pips = config.tp_pips
    sl_pips = config.sl_pips
    max_holding = config.max_holding_bars

    closes = df_test["close"].values
    highs = df_test["high"].values
    lows = df_test["low"].values

    trades = []
    i = 0
    n = len(predictions)

    while i < n - max_holding:
        conf = confidences[i]
        pred = predictions[i]

        if conf >= min_confidence:
            entry_price = closes[i]
            direction = "long" if pred == 1 else "short"

            if direction == "long":
                tp_price = entry_price + tp_pips * pip_value
                sl_price = entry_price - sl_pips * pip_value
            else:
                tp_price = entry_price - tp_pips * pip_value
                sl_price = entry_price + sl_pips * pip_value

            exit_price = None
            exit_reason = None

            for j in range(i + 1, min(i + max_holding + 1, n)):
                if direction == "long":
                    if highs[j] >= tp_price:
                        exit_price, exit_reason = tp_price, "take_profit"
                        break
                    if lows[j] <= sl_price:
                        exit_price, exit_reason = sl_price, "stop_loss"
                        break
                else:
                    if lows[j] <= tp_price:
                        exit_price, exit_reason = tp_price, "take_profit"
                        break
                    if highs[j] >= sl_price:
                        exit_price, exit_reason = sl_price, "stop_loss"
                        break

            if exit_price is None:
                exit_idx = min(i + max_holding, n - 1)
                exit_price = closes[exit_idx]
                exit_reason = "timeout"
                j = exit_idx

            if direction == "long":
                pnl = (exit_price - entry_price) / pip_value
            else:
                pnl = (entry_price - exit_price) / pip_value

            trades.append({"pnl": pnl, "conf": conf})
            i = j

        i += 1

    # Calculate results
    if not trades:
        return {"trades": 0, "win_rate": 0, "total_pips": 0, "profit_factor": 0}

    pnls = [t["pnl"] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    total_profit = sum(wins) if wins else 0
    total_loss = abs(sum(losses)) if losses else 0

    return {
        "trades": len(trades),
        "win_rate": len(wins) / len(trades) * 100,
        "total_pips": sum(pnls),
        "profit_factor": total_profit / total_loss if total_loss > 0 else float("inf"),
    }


def main():
    parser = argparse.ArgumentParser(description="Backtest MTF Ensemble")
    parser.add_argument("--data", type=str, default="data/forex/EURUSD_20200101_20251231_5min_combined.csv")
    parser.add_argument("--model-dir", type=str, default="models/mtf_ensemble")
    parser.add_argument("--confidence", type=float, default=0.60)
    parser.add_argument("--agreement", type=float, default=0.5)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--compare", action="store_true", help="Compare to individual models")
    parser.add_argument("--filter-mode", action="store_true", help="Use 1H as primary signal with 4H/D as filters (instead of weighted averaging)")
    parser.add_argument("--strict-filter", action="store_true", help="Require 4H agreement to trade (stricter than --filter-mode)")
    parser.add_argument("--dynamic-weights", action="store_true", help="Enable dynamic weight adjustment based on recent accuracy")
    args = parser.parse_args()

    print("\n" + "=" * 70)
    if args.strict_filter:
        print("MTF ENSEMBLE BACKTEST (STRICT FILTER MODE)")
        print("1H = Primary signal | 4H must agree to trade")
    elif args.filter_mode:
        print("MTF ENSEMBLE BACKTEST (FILTER MODE)")
        print("1H = Primary signal | 4H/D = Confirmation filters")
    elif args.dynamic_weights:
        print("MTF ENSEMBLE BACKTEST (DYNAMIC WEIGHTS)")
        print("Weights adapt based on recent model accuracy")
    else:
        print("MTF ENSEMBLE BACKTEST")
    print("=" * 70)

    # Load data
    data_path = project_root / args.data
    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    df.columns = [c.lower() for c in df.columns]
    # If index is not datetime, look for a timestamp column
    if not isinstance(df.index, pd.DatetimeIndex):
        time_col = next((c for c in ["timestamp", "time", "date", "datetime"] if c in df.columns), None)
        if time_col:
            df[time_col] = pd.to_datetime(df[time_col])
            df = df.set_index(time_col)
        else:
            df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    logger.info(f"Loaded {len(df)} bars")

    # Load ensemble - first check for training metadata to get sentiment settings
    model_dir = project_root / args.model_dir
    metadata_path = model_dir / "training_metadata.json"

    # Check if model was trained with sentiment
    include_sentiment = False
    trading_pair = "EURUSD"
    weights = {"1H": 0.6, "4H": 0.3, "D": 0.1}
    sentiment_by_timeframe = {"1H": False, "4H": False, "D": False}
    sentiment_source = "epu"  # Default to EPU

    use_stacking = False
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)
            include_sentiment = metadata.get("include_sentiment", False)
            trading_pair = metadata.get("trading_pair", "EURUSD")
            weights = metadata.get("weights", weights)
            # Get per-timeframe sentiment settings
            sentiment_by_timeframe = metadata.get("sentiment_by_timeframe", sentiment_by_timeframe)
            sentiment_mode = metadata.get("sentiment_mode", "disabled")
            sentiment_source = metadata.get("sentiment_source", "epu")
            use_stacking = metadata.get("use_stacking", False)
            if include_sentiment:
                logger.info(f"Model was trained with sentiment ({sentiment_mode}, source={sentiment_source}) for {trading_pair}")
                logger.info(f"Sentiment by TF: {sentiment_by_timeframe}")
            if use_stacking:
                logger.info("Model was trained with stacking meta-learner")

    # Create config with correct settings
    config = MTFEnsembleConfig(
        weights=weights,
        include_sentiment=include_sentiment,
        trading_pair=trading_pair,
        sentiment_source=sentiment_source,
        sentiment_by_timeframe=sentiment_by_timeframe,
        use_stacking=use_stacking,
        use_dynamic_weights=args.dynamic_weights,
    )

    if args.dynamic_weights:
        logger.info("Dynamic weights ENABLED - weights will adapt based on recent accuracy")

    ensemble = MTFEnsemble(config=config, model_dir=model_dir)
    ensemble.load()

    print(ensemble.summary())

    # Run backtest
    test_start_idx = int(len(df) * (1 - args.test_ratio))

    backtester = MTFEnsembleBacktester(
        ensemble=ensemble,
        min_confidence=args.confidence,
        min_agreement=args.agreement,
        filter_mode=args.filter_mode,
        strict_filter=args.strict_filter,
        use_dynamic_weights=args.dynamic_weights,
    )
    results = backtester.run(df, test_start_idx)

    # Print results
    print("\n" + "=" * 70)
    print("ENSEMBLE BACKTEST RESULTS")
    print("=" * 70)

    print(f"\nTrade Summary:")
    print(f"  Total Trades:     {results['total_trades']}")
    print(f"  Winning:          {results['winning_trades']}")
    print(f"  Losing:           {results['losing_trades']}")
    print(f"  Win Rate:         {results['win_rate']:.1f}%")

    print(f"\nP&L Summary:")
    print(f"  Total Pips:       {results['total_pips']:+.1f}")
    print(f"  Avg Pips/Trade:   {results['avg_pips']:+.1f}")
    print(f"  Profit Factor:    {results['profit_factor']:.2f}")
    print(f"  Avg Win:          {results['avg_win']:.1f} pips")
    print(f"  Avg Loss:         {results['avg_loss']:.1f} pips")

    print(f"\nExit Analysis:")
    if results['total_trades'] > 0:
        print(f"  Take Profit:      {results['tp_hits']} ({results['tp_hits']/results['total_trades']*100:.1f}%)")
        print(f"  Stop Loss:        {results['sl_hits']} ({results['sl_hits']/results['total_trades']*100:.1f}%)")
        print(f"  Timeout:          {results['timeouts']} ({results['timeouts']/results['total_trades']*100:.1f}%)")

    print(f"\nConfidence Analysis:")
    print(f"  Conf >= 60%:      {results['high_conf_trades']} trades, {results['high_conf_win_rate']:.1f}% win rate")
    print(f"  Conf >= 65%:      {results['very_high_conf_trades']} trades, {results['very_high_conf_win_rate']:.1f}% win rate")

    print(f"\nAgreement Analysis:")
    print(f"  Full Agreement:   {results['full_agree_trades']} trades, {results['full_agree_win_rate']:.1f}% win rate")

    print(f"\nDirection Analysis:")
    print(f"  Long:             {results['long_trades']} trades, {results['long_win_rate']:.1f}% win rate")
    print(f"  Short:            {results['short_trades']} trades, {results['short_win_rate']:.1f}% win rate")

    # Compare to individual models
    if args.compare:
        print("\n" + "=" * 70)
        print("COMPARISON TO INDIVIDUAL MODELS")
        print("=" * 70)

        individual_results = {}
        for tf in ["1H", "4H", "D"]:
            try:
                tf_results = run_individual_backtest(
                    ensemble, df, tf, min_confidence=args.confidence
                )
                individual_results[tf] = tf_results
                print(f"\n{tf} Model:")
                print(f"  Trades:        {tf_results['trades']}")
                print(f"  Win Rate:      {tf_results['win_rate']:.1f}%")
                print(f"  Total Pips:    {tf_results['total_pips']:+.1f}")
                print(f"  Profit Factor: {tf_results['profit_factor']:.2f}")
            except Exception as e:
                logger.warning(f"Could not backtest {tf}: {e}")

        print("\n" + "-" * 50)
        print("Summary Comparison:")
        print(f"{'Model':<12} {'Trades':>8} {'Win Rate':>10} {'PF':>8} {'Pips':>10}")
        print("-" * 50)

        for tf, r in individual_results.items():
            print(f"{tf:<12} {r['trades']:>8} {r['win_rate']:>9.1f}% {r['profit_factor']:>7.2f} {r['total_pips']:>+9.1f}")

        print("-" * 50)
        print(f"{'ENSEMBLE':<12} {results['total_trades']:>8} {results['win_rate']:>9.1f}% {results['profit_factor']:>7.2f} {results['total_pips']:>+9.1f}")

    # Target analysis
    print("\n" + "=" * 70)
    print("TARGET ANALYSIS")
    print("=" * 70)

    targets = [
        ("Win Rate >= 59%", results['win_rate'] >= 59),
        ("Profit Factor >= 2.0", results['profit_factor'] >= 2.0),
        ("High-Conf Win Rate >= 65%", results['high_conf_win_rate'] >= 65),
        ("Full Agreement Win Rate >= 70%", results['full_agree_win_rate'] >= 70),
    ]

    for target, achieved in targets:
        status = "[OK]" if achieved else "[!!]"
        print(f"  {status} {target}")

    print("=" * 70)


if __name__ == "__main__":
    main()
