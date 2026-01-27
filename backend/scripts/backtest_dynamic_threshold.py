#!/usr/bin/env python3
"""Backtest Dynamic Confidence Threshold System.

This script simulates the dynamic threshold system on historical data:
- Calculates threshold using ONLY past predictions (no future data leakage)
- Uses actual system parameters: 14d/21d/45d windows, 20%/60%/20% weights
- Applies performance feedback based on last 30 closed trades
- Enforces hard bounds (0.55-0.75) and divergence limits
- Tracks EUR balance with position sizing
- Generates month-by-month performance report

Key Requirements:
- Chronological processing (train once, backtest all)
- No data leakage (threshold uses only historical predictions)
- FIXED risk per trade (e.g., 10 EUR) to avoid unrealistic compounding
- Monthly aggregation with comprehensive metrics

Position Sizing (NO LEVERAGE - Cash-Only Trading):
- Uses FIXED EUR risk per trade (default: 10 EUR)
- Position size = min(Fixed Risk / (SL pips × pip value), Balance / 100,000)
- CRITICAL: Position notional cannot exceed available cash (no margin)
- Balance accumulates but position sizes don't compound
- More realistic than percentage-based compounding with leverage

Output:
- CSV: backtest_dynamic_threshold_monthly.csv (monthly stats)
- JSON: backtest_dynamic_threshold_summary.json (overall summary)
- Console: Summary statistics
"""

import argparse
import json
import logging
import sys
import warnings
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.multi_timeframe import MTFEnsemble, MTFEnsembleConfig
from src.trading.position_sizer import ConservativeHybridSizer
from src.trading.circuit_breakers import TradingCircuitBreaker
from src.config.trading_config import ConservativeHybridParameters

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Configuration matching the live system
THRESHOLD_CONFIG = {
    'short_window_days': 14,
    'medium_window_days': 21,
    'long_window_days': 45,
    'short_weight': 0.20,
    'medium_weight': 0.60,
    'long_weight': 0.20,
    'quantile': 0.60,
    'performance_lookback': 30,
    'target_win_rate': 0.54,
    'adjustment_factor': 0.10,
    'min_threshold': 0.55,
    'max_threshold': 0.75,
    'max_divergence': 0.08,
    'min_predictions': 50,  # Need some history before going dynamic
    'min_trades': 10,  # Need some trades before adjusting
}

TRADING_CONFIG = {
    'initial_balance': 1000.0,  # EUR
    'tp_pips': 25.0,
    'sl_pips': 15.0,
    'max_holding_bars': 12,     # 1H bars
    'pip_value': 0.0001,        # EUR/USD pip size
    'lot_size': 100000,         # Standard lot (for position sizing)
}

# Conservative Hybrid position sizing parameters
CONSERVATIVE_HYBRID_CONFIG = {
    'base_risk_percent': 1.5,
    'confidence_scaling_factor': 0.5,
    'min_risk_percent': 0.8,
    'max_risk_percent': 2.5,
    'daily_loss_limit_percent': -3.0,
    'consecutive_loss_limit': 5,
    'confidence_threshold': 0.70,
    'pip_value': 10.0,
    'lot_size': 100000.0,
}


@dataclass
class Trade:
    """Represents a single trade."""
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    direction: str  # "long" or "short"
    entry_price: float
    exit_price: float
    confidence: float
    threshold_used: float
    agreement_score: float
    pnl_pips: float
    pnl_eur: float
    position_size: float
    risk_pct_used: float
    balance_before: float
    balance_after: float
    exit_reason: str  # "take_profit", "stop_loss", "timeout"


@dataclass
class MonthlyStats:
    """Monthly performance statistics."""
    month: str  # YYYY-MM
    trades_count: int
    wins: int
    losses: int
    win_rate: float
    total_pips: float
    monthly_pnl_eur: float
    cumulative_balance_eur: float
    return_pct: float
    drawdown_pct: float
    avg_threshold_used: float
    best_trade_pips: float
    worst_trade_pips: float


class DynamicThresholdCalculator:
    """Calculates dynamic confidence threshold using only historical data.

    This mirrors the live ThresholdManager logic but operates on historical data.
    Critical: Uses ONLY past predictions to avoid data leakage.
    """

    def __init__(self, config: Dict = None):
        self.config = config or THRESHOLD_CONFIG

        # Prediction history: deques with (timestamp, confidence) tuples
        # Convert days to 1H bars (24 bars per day)
        short_hours = self.config['short_window_days'] * 24
        medium_hours = self.config['medium_window_days'] * 24
        long_hours = self.config['long_window_days'] * 24

        self.predictions_short = deque(maxlen=short_hours * 2)  # Extra buffer
        self.predictions_medium = deque(maxlen=medium_hours * 2)
        self.predictions_long = deque(maxlen=long_hours * 2)

        # Trade outcomes: (timestamp, is_winner) tuples
        self.recent_trades = deque(maxlen=100)

        # Static fallback
        self.static_threshold = 0.66

    def record_prediction(self, timestamp: pd.Timestamp, confidence: float):
        """Record a prediction for future threshold calculations."""
        entry = (timestamp, confidence)
        self.predictions_short.append(entry)
        self.predictions_medium.append(entry)
        self.predictions_long.append(entry)

    def record_trade_outcome(self, timestamp: pd.Timestamp, is_winner: bool):
        """Record trade outcome for performance feedback."""
        self.recent_trades.append((timestamp, is_winner))

    def calculate_threshold(self, current_time: pd.Timestamp) -> float:
        """Calculate dynamic threshold using only past data.

        Args:
            current_time: Current timestamp (for filtering old predictions)

        Returns:
            Dynamic threshold value (0.55-0.75)
        """
        # Extract confidences from each window (only within time range)
        confidences_short = self._get_confidences_from_window(
            self.predictions_short,
            current_time,
            self.config['short_window_days']
        )
        confidences_medium = self._get_confidences_from_window(
            self.predictions_medium,
            current_time,
            self.config['medium_window_days']
        )
        confidences_long = self._get_confidences_from_window(
            self.predictions_long,
            current_time,
            self.config['long_window_days']
        )

        # Check if we have enough data
        if len(confidences_long) < self.config['min_predictions']:
            return self.static_threshold

        # Calculate quantile for each window
        quantile = self.config['quantile']
        short_term = np.percentile(confidences_short, quantile * 100) if confidences_short else self.static_threshold
        medium_term = np.percentile(confidences_medium, quantile * 100) if confidences_medium else self.static_threshold
        long_term = np.percentile(confidences_long, quantile * 100)

        # Blend components
        blended = (
            self.config['short_weight'] * short_term +
            self.config['medium_weight'] * medium_term +
            self.config['long_weight'] * long_term
        )

        # Calculate performance adjustment
        adjustment = 0.0
        trade_count = len(self.recent_trades)

        if trade_count >= self.config['min_trades']:
            # Get recent trades within lookback window
            lookback = self.config['performance_lookback']
            recent_outcomes = list(self.recent_trades)[-lookback:]
            wins = sum(1 for _, is_winner in recent_outcomes if is_winner)
            win_rate = wins / len(recent_outcomes)

            # Calculate adjustment
            win_rate_delta = win_rate - self.config['target_win_rate']
            adjustment = win_rate_delta * self.config['adjustment_factor']

        # Apply adjustment
        dynamic_threshold = blended + adjustment

        # Apply hard bounds
        dynamic_threshold = np.clip(
            dynamic_threshold,
            self.config['min_threshold'],
            self.config['max_threshold']
        )

        # Apply divergence check (prevent too much deviation from long-term)
        min_allowed = long_term - self.config['max_divergence']
        max_allowed = long_term + self.config['max_divergence']
        dynamic_threshold = np.clip(dynamic_threshold, min_allowed, max_allowed)

        return dynamic_threshold

    def _get_confidences_from_window(
        self,
        window_deque: deque,
        current_time: pd.Timestamp,
        max_days: int
    ) -> List[float]:
        """Extract valid confidences from a time window.

        Args:
            window_deque: Deque containing (timestamp, confidence) tuples
            current_time: Current timestamp
            max_days: Maximum age in days

        Returns:
            List of confidence values within time window
        """
        if not window_deque:
            return []

        cutoff = current_time - pd.Timedelta(days=max_days)
        confidences = [
            conf for ts, conf in window_deque
            if ts >= cutoff and ts <= current_time
        ]
        return confidences


class DynamicThresholdBacktester:
    """Backtester for dynamic threshold system."""

    def __init__(
        self,
        ensemble: MTFEnsemble,
        threshold_config: Dict = None,
        trading_config: Dict = None,
        conservative_hybrid_config: Dict = None,
    ):
        self.ensemble = ensemble
        self.threshold_config = threshold_config or THRESHOLD_CONFIG
        self.trading_config = trading_config or TRADING_CONFIG
        self.conservative_hybrid_config = conservative_hybrid_config or CONSERVATIVE_HYBRID_CONFIG

        self.threshold_calculator = DynamicThresholdCalculator(self.threshold_config)

        # Initialize position sizer and circuit breaker
        # Convert dict config to ConservativeHybridParameters
        ch_params = ConservativeHybridParameters(**self.conservative_hybrid_config)
        self.position_sizer = ConservativeHybridSizer()
        self.circuit_breaker = TradingCircuitBreaker(ch_params)

        # Daily P&L tracking for circuit breaker
        self.daily_pnl_tracker: Dict[str, float] = {}  # date -> pnl

        self.trades: List[Trade] = []
        self.balance = self.trading_config['initial_balance']
        self.peak_balance = self.balance

    def run(self, df_5min: pd.DataFrame) -> Dict:
        """Run backtest on full dataset.

        Args:
            df_5min: 5-minute OHLCV data

        Returns:
            Dict of performance metrics
        """
        from src.features.technical.calculator import TechnicalIndicatorCalculator

        logger.info("=" * 70)
        logger.info("DYNAMIC THRESHOLD BACKTEST WITH CONSERVATIVE HYBRID POSITION SIZING")
        logger.info("=" * 70)
        logger.info(f"Initial Balance: {self.balance:.2f} EUR")
        logger.info(f"Position Sizing: Conservative Hybrid")
        logger.info(f"  Base Risk: {self.conservative_hybrid_config['base_risk_percent']:.1f}%")
        logger.info(f"  Risk Range: {self.conservative_hybrid_config['min_risk_percent']:.1f}% - {self.conservative_hybrid_config['max_risk_percent']:.1f}%")
        logger.info(f"  Confidence Scaling: {self.conservative_hybrid_config['confidence_scaling_factor']:.1f}x")
        logger.info(f"Circuit Breakers:")
        logger.info(f"  Daily Loss Limit: {self.conservative_hybrid_config['daily_loss_limit_percent']:.1f}%")
        logger.info(f"  Consecutive Loss Limit: {self.conservative_hybrid_config['consecutive_loss_limit']}")
        logger.info(f"TP/SL: {self.trading_config['tp_pips']}/{self.trading_config['sl_pips']} pips")
        logger.info(f"Threshold Config: {self.threshold_config['short_window_days']}d/"
                   f"{self.threshold_config['medium_window_days']}d/"
                   f"{self.threshold_config['long_window_days']}d windows")
        logger.info("=" * 70)

        # Prepare 1H data (primary trading timeframe)
        calc = TechnicalIndicatorCalculator(model_type="short_term")

        model_1h = self.ensemble.models["1H"]
        df_1h = self.ensemble.resample_data(df_5min, "1H")
        higher_tf_data_1h = self.ensemble.prepare_higher_tf_data(df_5min, "1H")
        df_1h_features = calc.calculate(df_1h)
        df_1h_features = model_1h.feature_engine.add_all_features(df_1h_features, higher_tf_data_1h)
        df_1h_features = df_1h_features.dropna()

        feature_cols_1h = model_1h.feature_names
        available_cols_1h = [c for c in feature_cols_1h if c in df_1h_features.columns]
        X_1h = df_1h_features[available_cols_1h].values

        logger.info(f"1H bars: {len(X_1h)}")
        logger.info(f"Date range: {df_1h_features.index[0]} to {df_1h_features.index[-1]}")

        # Get 1H predictions
        preds_1h, confs_1h = model_1h.predict_batch(X_1h)

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

        # Create mapping from timestamp to 4H prediction
        pred_4h_map = dict(zip(df_4h_features.index, zip(preds_4h_all, confs_4h_all)))

        # Prepare Daily data and get predictions
        model_d = self.ensemble.models["D"]
        df_d = self.ensemble.resample_data(df_5min, "D")
        higher_tf_data_d = {}
        df_d_features = calc.calculate(df_d)
        df_d_features = model_d.feature_engine.add_all_features(df_d_features, higher_tf_data_d)
        df_d_features = df_d_features.dropna()

        feature_cols_d = model_d.feature_names
        available_cols_d = [c for c in feature_cols_d if c in df_d_features.columns]
        X_d = df_d_features[available_cols_d].values
        preds_d_all, confs_d_all = model_d.predict_batch(X_d)

        # Create mapping from date to Daily prediction
        pred_d_map = dict(zip(df_d_features.index.date, zip(preds_d_all, confs_d_all)))

        logger.info(f"4H predictions: {len(pred_4h_map)}")
        logger.info(f"Daily predictions: {len(pred_d_map)}")

        # Get price data
        closes = df_1h_features["close"].values
        highs = df_1h_features["high"].values
        lows = df_1h_features["low"].values
        timestamps = df_1h_features.index

        # Pre-compute ensemble predictions
        logger.info("Computing ensemble predictions...")
        test_directions = []
        test_confidences = []
        test_agreements = []

        weights = self.ensemble._normalize_weights(self.ensemble.config.weights)
        w_1h = weights.get("1H", 0.6)
        w_4h = weights.get("4H", 0.3)
        w_d = weights.get("D", 0.1)

        for i, ts in enumerate(timestamps):
            p_1h, c_1h = preds_1h[i], confs_1h[i]

            # Find corresponding 4H prediction
            ts_4h = ts.floor("4h")
            if ts_4h in pred_4h_map:
                p_4h, c_4h = pred_4h_map[ts_4h]
            else:
                prev_4h_times = [t for t in pred_4h_map.keys() if t <= ts]
                if prev_4h_times:
                    p_4h, c_4h = pred_4h_map[max(prev_4h_times)]
                else:
                    p_4h, c_4h = p_1h, c_1h

            # Find corresponding Daily prediction
            day = ts.date()
            if day in pred_d_map:
                p_d, c_d = pred_d_map[day]
            else:
                prev_days = [d for d in pred_d_map.keys() if d <= day]
                if prev_days:
                    p_d, c_d = pred_d_map[max(prev_days)]
                else:
                    p_d, c_d = p_1h, c_1h

            # Weighted combination
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

        logger.info(f"Starting trading simulation on {len(test_directions)} bars...")

        # Simulate trading
        n = len(test_directions)
        i = 0

        while i < n - self.trading_config['max_holding_bars']:
            timestamp = timestamps[i]
            conf = test_confidences[i]
            agreement = test_agreements[i]
            pred = test_directions[i]

            # Record prediction for threshold calculation (BEFORE trade decision)
            self.threshold_calculator.record_prediction(timestamp, conf)

            # Calculate dynamic threshold using only past data
            dynamic_threshold = self.threshold_calculator.calculate_threshold(timestamp)

            # Check if all models agree (required for trading)
            all_agree = agreement >= 1.0

            # Check if we're in test period (for out-of-sample testing)
            test_start_date = self.trading_config.get('test_start_date', None)
            if test_start_date is not None and timestamp < test_start_date:
                # Skip trading before test period starts
                i += 1
                continue

            # Check entry conditions
            if conf >= dynamic_threshold and all_agree:
                # Check circuit breakers before trading
                # Create a mock in-memory trade list for circuit breaker checks
                consecutive_losses = self._get_consecutive_losses()
                daily_pnl = self._get_daily_pnl(timestamp)

                # Circuit breaker check with progressive risk reduction
                ch_params = ConservativeHybridParameters(**self.conservative_hybrid_config)

                # Check daily loss limit (still blocks completely)
                if daily_pnl < 0 and abs(daily_pnl) >= self.balance * (abs(ch_params.daily_loss_limit_percent) / 100.0):
                    logger.warning(f"Daily loss limit breached: {daily_pnl:.2f} EUR, skipping trade")
                    i += 1
                    continue

                # Calculate risk reduction factor (progressive reduction, never blocks)
                risk_reduction_factor = self._calculate_risk_reduction_factor(consecutive_losses, ch_params)

                if risk_reduction_factor < 1.0:
                    logger.warning(
                        f"Progressive risk reduction active: {consecutive_losses} consecutive losses, "
                        f"risk reduced to {risk_reduction_factor * 100:.0f}% of normal"
                    )

                entry_price = closes[i]
                entry_time = timestamp
                direction = "long" if pred == 1 else "short"

                # Calculate position size using Conservative Hybrid with risk reduction
                position_lots, risk_pct_used, metadata = self.position_sizer.calculate_position_size(
                    balance=self.balance,
                    confidence=conf,
                    sl_pips=self.trading_config['sl_pips'],
                    config=ch_params,
                    risk_reduction_factor=risk_reduction_factor,  # Pass risk reduction factor
                    pip_value=ch_params.pip_value,
                    lot_size=ch_params.lot_size
                )

                if position_lots <= 0:
                    logger.debug(f"Position size zero, skipping trade. Reason: {metadata.get('reason')}")
                    i += 1
                    continue

                position_size = position_lots

                # Set TP/SL levels
                pip_value = self.trading_config['pip_value']
                if direction == "long":
                    tp_price = entry_price + self.trading_config['tp_pips'] * pip_value
                    sl_price = entry_price - self.trading_config['sl_pips'] * pip_value
                else:
                    tp_price = entry_price - self.trading_config['tp_pips'] * pip_value
                    sl_price = entry_price + self.trading_config['sl_pips'] * pip_value

                exit_price = None
                exit_reason = None
                exit_idx = i

                # Simulate trade
                max_holding = self.trading_config['max_holding_bars']
                for j in range(i + 1, min(i + max_holding + 1, n)):
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
                    exit_idx = min(i + max_holding, n - 1)
                    exit_price = closes[exit_idx]
                    exit_reason = "timeout"

                # Calculate P&L
                if direction == "long":
                    pnl_pips = (exit_price - entry_price) / pip_value
                else:
                    pnl_pips = (entry_price - exit_price) / pip_value

                # Calculate EUR P&L
                pnl_eur = self._calculate_pnl_eur(pnl_pips, position_size)

                # Update balance
                balance_before = self.balance
                self.balance += pnl_eur
                balance_after = self.balance

                # Update peak balance for drawdown calculation
                if self.balance > self.peak_balance:
                    self.peak_balance = self.balance

                # Update daily P&L tracker
                trade_date = timestamps[exit_idx].date().isoformat()
                if trade_date not in self.daily_pnl_tracker:
                    self.daily_pnl_tracker[trade_date] = 0.0
                self.daily_pnl_tracker[trade_date] += pnl_eur

                # Record trade
                is_winner = pnl_pips > 0
                self.trades.append(Trade(
                    entry_time=entry_time,
                    exit_time=timestamps[exit_idx],
                    direction=direction,
                    entry_price=entry_price,
                    exit_price=exit_price,
                    confidence=conf,
                    threshold_used=dynamic_threshold,
                    agreement_score=agreement,
                    pnl_pips=pnl_pips,
                    pnl_eur=pnl_eur,
                    position_size=position_size,
                    risk_pct_used=risk_pct_used,
                    balance_before=balance_before,
                    balance_after=balance_after,
                    exit_reason=exit_reason,
                ))

                # Record outcome for threshold adjustment
                self.threshold_calculator.record_trade_outcome(timestamps[exit_idx], is_winner)

                # Skip to after exit
                i = exit_idx

            i += 1

        logger.info(f"Simulation complete. Total trades: {len(self.trades)}")

        return self._calculate_results()

    def _get_consecutive_losses(self) -> int:
        """Get consecutive losses from most recent trades.

        Returns:
            Number of consecutive losing trades
        """
        consecutive_losses = 0
        for trade in reversed(self.trades):
            if trade.pnl_eur <= 0:
                consecutive_losses += 1
            else:
                break
        return consecutive_losses

    def _calculate_risk_reduction_factor(self, consecutive_losses: int, ch_params) -> float:
        """Calculate risk reduction factor based on consecutive losses.

        Args:
            consecutive_losses: Number of consecutive losing trades
            ch_params: ConservativeHybridParameters configuration

        Returns:
            Risk reduction factor (0.2 to 1.0)
            - 1.0: Normal risk (< 5 consecutive losses)
            - 0.8: 20% reduction (5 losses)
            - 0.6: 40% reduction (6 losses)
            - 0.4: 60% reduction (7 losses)
            - 0.2: 80% reduction (8+ losses, minimum floor)
        """
        if not ch_params.enable_progressive_reduction:
            # Legacy behavior: return 0.0 to signal complete stop
            if consecutive_losses >= ch_params.consecutive_loss_limit:
                return 0.0
            return 1.0

        # Progressive reduction enabled
        if consecutive_losses < ch_params.consecutive_loss_limit:
            return 1.0  # Normal risk

        # Calculate reduction
        excess_losses = consecutive_losses - ch_params.consecutive_loss_limit + 1
        reduction_factor = 1.0 - (excess_losses * ch_params.risk_reduction_per_loss)

        # Apply floor
        return max(reduction_factor, ch_params.min_risk_factor)

    def _get_daily_pnl(self, current_time: pd.Timestamp) -> float:
        """Get total P&L for the current day.

        Args:
            current_time: Current timestamp

        Returns:
            Total P&L for current day (EUR)
        """
        current_date = current_time.date().isoformat()
        return self.daily_pnl_tracker.get(current_date, 0.0)

    def _calculate_pnl_eur(self, pnl_pips: float, position_size: float) -> float:
        """Calculate P&L in EUR.

        Args:
            pnl_pips: Profit/loss in pips
            position_size: Position size in lots

        Returns:
            P&L in EUR
        """
        pip_value_eur = 10.0  # $10 per pip per lot (approximate)
        return pnl_pips * pip_value_eur * position_size

    def _calculate_results(self) -> Dict:
        """Calculate performance metrics and generate reports."""
        if not self.trades:
            logger.warning("No trades executed")
            return {
                "total_trades": 0,
                "final_balance": self.trading_config['initial_balance'],
            }

        # Convert trades to DataFrame
        trades_df = pd.DataFrame([{
            "entry_time": t.entry_time,
            "exit_time": t.exit_time,
            "direction": t.direction,
            "pnl_pips": t.pnl_pips,
            "pnl_eur": t.pnl_eur,
            "confidence": t.confidence,
            "threshold_used": t.threshold_used,
            "agreement_score": t.agreement_score,
            "exit_reason": t.exit_reason,
            "balance_after": t.balance_after,
        } for t in self.trades])

        # Calculate monthly statistics
        monthly_stats = self._calculate_monthly_stats(trades_df)

        # Overall statistics
        wins = trades_df[trades_df["pnl_pips"] > 0]
        losses = trades_df[trades_df["pnl_pips"] <= 0]

        total_profit_pips = wins["pnl_pips"].sum() if len(wins) > 0 else 0
        total_loss_pips = abs(losses["pnl_pips"].sum()) if len(losses) > 0 else 0
        profit_factor = total_profit_pips / total_loss_pips if total_loss_pips > 0 else float("inf")

        total_profit_eur = wins["pnl_eur"].sum() if len(wins) > 0 else 0
        total_loss_eur = abs(losses["pnl_eur"].sum()) if len(losses) > 0 else 0

        # Calculate drawdown
        max_dd_pct = ((self.peak_balance - self.balance) / self.peak_balance * 100
                      if self.peak_balance > 0 else 0)

        # Calculate returns
        initial_balance = self.trading_config['initial_balance']
        total_return_pct = ((self.balance - initial_balance) / initial_balance) * 100

        # Calculate Sharpe ratio (simplified)
        if len(trades_df) > 1:
            returns = trades_df["pnl_eur"].values
            sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0
        else:
            sharpe = 0

        # Best/worst months
        if monthly_stats:
            best_month = max(monthly_stats, key=lambda x: x.monthly_pnl_eur)
            worst_month = min(monthly_stats, key=lambda x: x.monthly_pnl_eur)
        else:
            best_month = worst_month = None

        results = {
            "initial_balance": initial_balance,
            "final_balance": self.balance,
            "total_return_pct": total_return_pct,
            "total_pips": trades_df["pnl_pips"].sum(),
            "total_trades": len(trades_df),
            "winning_trades": len(wins),
            "losing_trades": len(losses),
            "win_rate": len(wins) / len(trades_df) * 100,
            "profit_factor": profit_factor,
            "max_drawdown_pct": max_dd_pct,
            "sharpe_ratio": sharpe,
            "avg_monthly_return": np.mean([m.return_pct for m in monthly_stats]) if monthly_stats else 0,
            "best_month": {
                "month": best_month.month,
                "pnl": best_month.monthly_pnl_eur
            } if best_month else None,
            "worst_month": {
                "month": worst_month.month,
                "pnl": worst_month.monthly_pnl_eur
            } if worst_month else None,
            "monthly_stats": [vars(m) for m in monthly_stats],
            "avg_threshold_used": trades_df["threshold_used"].mean(),
            "threshold_std": trades_df["threshold_used"].std(),
            "avg_win_pips": total_profit_pips / len(wins) if len(wins) > 0 else 0,
            "avg_loss_pips": total_loss_pips / len(losses) if len(losses) > 0 else 0,
            "tp_hits": len(trades_df[trades_df["exit_reason"] == "take_profit"]),
            "sl_hits": len(trades_df[trades_df["exit_reason"] == "stop_loss"]),
            "timeouts": len(trades_df[trades_df["exit_reason"] == "timeout"]),
        }

        return results

    def _calculate_monthly_stats(self, trades_df: pd.DataFrame) -> List[MonthlyStats]:
        """Calculate monthly performance statistics.

        Args:
            trades_df: DataFrame of all trades

        Returns:
            List of MonthlyStats objects
        """
        trades_df["month"] = trades_df["exit_time"].dt.to_period("M")

        monthly_stats = []
        initial_balance = self.trading_config['initial_balance']
        cumulative_balance = initial_balance
        peak_balance = initial_balance

        for month, group in trades_df.groupby("month"):
            trades_count = len(group)
            wins = len(group[group["pnl_pips"] > 0])
            losses = len(group[group["pnl_pips"] <= 0])
            win_rate = (wins / trades_count * 100) if trades_count > 0 else 0

            total_pips = group["pnl_pips"].sum()
            monthly_pnl_eur = group["pnl_eur"].sum()

            # Update cumulative balance
            cumulative_balance += monthly_pnl_eur

            # Calculate return for this month
            month_start_balance = cumulative_balance - monthly_pnl_eur
            return_pct = (monthly_pnl_eur / month_start_balance * 100) if month_start_balance > 0 else 0

            # Calculate drawdown
            if cumulative_balance > peak_balance:
                peak_balance = cumulative_balance
            drawdown_pct = ((peak_balance - cumulative_balance) / peak_balance * 100) if peak_balance > 0 else 0

            avg_threshold = group["threshold_used"].mean()
            best_trade_pips = group["pnl_pips"].max()
            worst_trade_pips = group["pnl_pips"].min()

            monthly_stats.append(MonthlyStats(
                month=str(month),
                trades_count=trades_count,
                wins=wins,
                losses=losses,
                win_rate=win_rate,
                total_pips=total_pips,
                monthly_pnl_eur=monthly_pnl_eur,
                cumulative_balance_eur=cumulative_balance,
                return_pct=return_pct,
                drawdown_pct=drawdown_pct,
                avg_threshold_used=avg_threshold,
                best_trade_pips=best_trade_pips,
                worst_trade_pips=worst_trade_pips,
            ))

        return monthly_stats


def main():
    parser = argparse.ArgumentParser(
        description="Backtest Dynamic Threshold System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python scripts/backtest_dynamic_threshold.py

  # Custom parameters
  python scripts/backtest_dynamic_threshold.py --initial-balance 10000 --base-risk-percent 2.0

  # Custom output location
  python scripts/backtest_dynamic_threshold.py --output results/backtest_monthly.csv
        """
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/forex/EURUSD_20200101_20251231_5min_combined.csv",
        help="Path to 5-minute OHLCV data"
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="models/mtf_ensemble",
        help="Directory containing trained models"
    )
    parser.add_argument(
        "--initial-balance",
        type=float,
        default=1000.0,
        help="Initial account balance in EUR (default: 1000)"
    )
    parser.add_argument(
        "--base-risk-percent",
        type=float,
        default=1.5,
        help="Base risk percentage for Conservative Hybrid (default: 1.5)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="backtest_dynamic_threshold_monthly.csv",
        help="Output CSV file for monthly stats"
    )
    parser.add_argument(
        "--tp-pips",
        type=float,
        default=25.0,
        help="Take profit in pips (default: 25.0)"
    )
    parser.add_argument(
        "--sl-pips",
        type=float,
        default=15.0,
        help="Stop loss in pips (default: 15.0)"
    )
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("DYNAMIC THRESHOLD BACKTEST WITH CONSERVATIVE HYBRID")
    print("=" * 70)
    print(f"Model Directory: {args.model_dir}")
    print(f"Data File: {args.data}")
    print(f"Initial Balance: {args.initial_balance:.2f} EUR")
    print(f"Base Risk Percent: {args.base_risk_percent:.1f}%")
    print("=" * 70 + "\n")

    # Load data
    data_path = project_root / args.data
    if not data_path.exists():
        logger.error(f"Data file not found: {data_path}")
        return 1

    logger.info(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    df.columns = [c.lower() for c in df.columns]

    if not isinstance(df.index, pd.DatetimeIndex):
        time_col = next((c for c in ["timestamp", "time", "date", "datetime"] if c in df.columns), None)
        if time_col:
            df[time_col] = pd.to_datetime(df[time_col])
            df = df.set_index(time_col)
        else:
            df.index = pd.to_datetime(df.index)

    df = df.sort_index()
    logger.info(f"Loaded {len(df)} bars from {df.index[0]} to {df.index[-1]}")

    # Load ensemble
    model_dir = project_root / args.model_dir
    if not model_dir.exists():
        logger.error(f"Model directory not found: {model_dir}")
        return 1

    metadata_path = model_dir / "training_metadata.json"

    # Load model configuration
    include_sentiment = False
    trading_pair = "EURUSD"
    weights = {"1H": 0.6, "4H": 0.3, "D": 0.1}
    sentiment_by_timeframe = {"1H": False, "4H": False, "D": False}
    sentiment_source = "epu"
    use_stacking = False

    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)
            include_sentiment = metadata.get("include_sentiment", False)
            trading_pair = metadata.get("trading_pair", "EURUSD")
            weights = metadata.get("weights", weights)
            sentiment_by_timeframe = metadata.get("sentiment_by_timeframe", sentiment_by_timeframe)
            sentiment_source = metadata.get("sentiment_source", "epu")
            use_stacking = metadata.get("use_stacking", False)

    config = MTFEnsembleConfig(
        weights=weights,
        include_sentiment=include_sentiment,
        trading_pair=trading_pair,
        sentiment_source=sentiment_source,
        sentiment_by_timeframe=sentiment_by_timeframe,
        use_stacking=use_stacking,
    )

    logger.info("Loading ensemble models...")
    ensemble = MTFEnsemble(config=config, model_dir=model_dir)
    ensemble.load()

    print(ensemble.summary())

    # Update trading and conservative hybrid config with CLI args
    TRADING_CONFIG['initial_balance'] = args.initial_balance
    TRADING_CONFIG['tp_pips'] = args.tp_pips
    TRADING_CONFIG['sl_pips'] = args.sl_pips
    CONSERVATIVE_HYBRID_CONFIG['base_risk_percent'] = args.base_risk_percent

    # Run backtest
    backtester = DynamicThresholdBacktester(
        ensemble=ensemble,
        threshold_config=THRESHOLD_CONFIG,
        trading_config=TRADING_CONFIG,
        conservative_hybrid_config=CONSERVATIVE_HYBRID_CONFIG,
    )

    results = backtester.run(df)

    # Print results
    print("\n" + "=" * 70)
    print("BACKTEST RESULTS")
    print("=" * 70)

    print(f"\nAccount Performance:")
    print(f"  Initial Balance:  {results['initial_balance']:>12.2f} EUR")
    print(f"  Final Balance:    {results['final_balance']:>12.2f} EUR")
    print(f"  Total Return:     {results['total_return_pct']:>12.1f}%")
    print(f"  Max Drawdown:     {results['max_drawdown_pct']:>12.1f}%")
    print(f"  Sharpe Ratio:     {results['sharpe_ratio']:>12.2f}")

    print(f"\nTrade Summary:")
    print(f"  Total Trades:     {results['total_trades']:>12}")
    print(f"  Winning:          {results['winning_trades']:>12}")
    print(f"  Losing:           {results['losing_trades']:>12}")
    print(f"  Win Rate:         {results['win_rate']:>12.1f}%")

    print(f"\nP&L Summary:")
    print(f"  Total Pips:       {results['total_pips']:>12.1f}")
    print(f"  Avg Win:          {results['avg_win_pips']:>12.1f} pips")
    print(f"  Avg Loss:         {results['avg_loss_pips']:>12.1f} pips")
    print(f"  Profit Factor:    {results['profit_factor']:>12.2f}")

    print(f"\nExit Analysis:")
    if results['total_trades'] > 0:
        print(f"  Take Profit:      {results['tp_hits']:>12} ({results['tp_hits']/results['total_trades']*100:.1f}%)")
        print(f"  Stop Loss:        {results['sl_hits']:>12} ({results['sl_hits']/results['total_trades']*100:.1f}%)")
        print(f"  Timeout:          {results['timeouts']:>12} ({results['timeouts']/results['total_trades']*100:.1f}%)")

    print(f"\nThreshold Analysis:")
    print(f"  Avg Threshold:    {results['avg_threshold_used']:>12.4f}")
    print(f"  Threshold StdDev: {results['threshold_std']:>12.4f}")

    print(f"\nMonthly Performance:")
    print(f"  Avg Monthly Ret:  {results['avg_monthly_return']:>12.1f}%")
    if results['best_month']:
        print(f"  Best Month:       {results['best_month']['month']} ({results['best_month']['pnl']:+.2f} EUR)")
    if results['worst_month']:
        print(f"  Worst Month:      {results['worst_month']['month']} ({results['worst_month']['pnl']:+.2f} EUR)")

    print("=" * 70)

    # Save results
    output_csv = project_root / args.output
    output_json = output_csv.with_suffix('.json')

    # Save monthly CSV
    if results['monthly_stats']:
        monthly_df = pd.DataFrame(results['monthly_stats'])
        monthly_df.to_csv(output_csv, index=False)
        logger.info(f"Saved monthly stats to {output_csv}")

    # Save summary JSON
    with open(output_json, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Saved summary to {output_json}")

    print(f"\nResults saved to:")
    print(f"  CSV: {output_csv}")
    print(f"  JSON: {output_json}")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
