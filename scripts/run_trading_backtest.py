#!/usr/bin/env python3
"""Run a trading backtest with regime filtering.

This script:
1. Loads historical EURUSD data
2. Trains a simple model (or loads existing)
3. Simulates trading with regime filtering
4. Reports performance metrics
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

from src.trading.filters import RegimeFilter, MarketRegime
from src.trading.risk.profiles import load_risk_profile

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class Trade:
    """Represents a completed trade."""
    entry_time: datetime
    exit_time: datetime
    direction: str  # 'long' or 'short'
    entry_price: float
    exit_price: float
    size: float  # Position size as fraction of equity
    pnl_pct: float
    regime: str
    confidence: float

    @property
    def is_win(self) -> bool:
        return self.pnl_pct > 0


@dataclass
class BacktestResult:
    """Results from backtest."""
    trades: List[Trade] = field(default_factory=list)
    initial_equity: float = 10000.0
    final_equity: float = 10000.0

    @property
    def total_trades(self) -> int:
        return len(self.trades)

    @property
    def winning_trades(self) -> int:
        return sum(1 for t in self.trades if t.is_win)

    @property
    def losing_trades(self) -> int:
        return sum(1 for t in self.trades if not t.is_win)

    @property
    def win_rate(self) -> float:
        return self.winning_trades / self.total_trades if self.total_trades > 0 else 0

    @property
    def total_return(self) -> float:
        return (self.final_equity - self.initial_equity) / self.initial_equity

    @property
    def avg_win(self) -> float:
        wins = [t.pnl_pct for t in self.trades if t.is_win]
        return np.mean(wins) if wins else 0

    @property
    def avg_loss(self) -> float:
        losses = [t.pnl_pct for t in self.trades if not t.is_win]
        return np.mean(losses) if losses else 0

    @property
    def profit_factor(self) -> float:
        gross_profit = sum(t.pnl_pct for t in self.trades if t.is_win)
        gross_loss = abs(sum(t.pnl_pct for t in self.trades if not t.is_win))
        return gross_profit / gross_loss if gross_loss > 0 else float('inf')

    @property
    def max_drawdown(self) -> float:
        if not self.trades:
            return 0
        equity_curve = [self.initial_equity]
        for t in self.trades:
            equity_curve.append(equity_curve[-1] * (1 + t.pnl_pct * t.size))
        peak = equity_curve[0]
        max_dd = 0
        for eq in equity_curve:
            if eq > peak:
                peak = eq
            dd = (peak - eq) / peak
            if dd > max_dd:
                max_dd = dd
        return max_dd


def load_data(timeframe: str) -> pd.DataFrame:
    """Load OHLCV data."""
    data_path = project_root / f"data/forex/derived_proper/{timeframe}/EURUSD_{timeframe}.parquet"
    df = pd.read_parquet(data_path)
    df.columns = [c.lower() for c in df.columns]

    if "timestamp" in df.columns:
        df = df.set_index("timestamp")
        df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    return df


def calculate_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate technical features."""
    from src.features.technical.calculator import TechnicalIndicatorCalculator

    calc = TechnicalIndicatorCalculator(model_type="short_term")
    df_features = calc.calculate(df)

    return df_features


def create_labels(df: pd.DataFrame, threshold: float = 0.0005) -> pd.Series:
    """Create binary labels for direction prediction."""
    returns = df["close"].pct_change(1).shift(-1)

    # Filter significant moves
    up_mask = returns > threshold
    down_mask = returns < -threshold

    # 1 for up, 0 for down, NaN for insignificant
    labels = pd.Series(index=df.index, dtype=float)
    labels[up_mask] = 1
    labels[down_mask] = 0

    return labels


def train_model(X_train: np.ndarray, y_train: np.ndarray):
    """Train a gradient boosting classifier."""
    model = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=4,
        min_samples_split=20,
        min_samples_leaf=10,
        subsample=0.8,
        random_state=42,
    )
    model.fit(X_train, y_train)
    return model


def run_backtest(
    df: pd.DataFrame,
    model,
    scaler: StandardScaler,
    feature_cols: List[str],
    regime_filter: RegimeFilter,
    initial_equity: float = 10000.0,
    base_position_size: float = 0.02,  # 2% per trade
    stop_loss_pct: float = 0.01,  # 1% stop loss
    take_profit_pct: float = 0.02,  # 2% take profit
    min_confidence: float = 0.60,
    use_regime_filter: bool = True,
) -> BacktestResult:
    """Run backtest simulation."""

    result = BacktestResult(initial_equity=initial_equity)
    equity = initial_equity

    # We need at least 50 bars for regime detection
    lookback = 50

    trades = []
    position = None  # None, 'long', or 'short'
    entry_price = 0
    entry_time = None
    entry_confidence = 0
    entry_regime = ""
    position_size = 0

    # Iterate through data (skip first lookback bars)
    for i in range(lookback, len(df) - 1):
        current_time = df.index[i]
        current_price = df['close'].iloc[i]
        next_price = df['close'].iloc[i + 1]

        # Get features for prediction
        X = df[feature_cols].iloc[i:i+1].values
        if np.isnan(X).any():
            continue
        X_scaled = scaler.transform(X)

        # Get prediction
        prob = model.predict_proba(X_scaled)[0]
        pred_class = model.predict(X_scaled)[0]
        confidence = max(prob)

        # Get regime
        market_data = df.iloc[i-lookback:i+1].copy()
        regime_analysis = regime_filter.analyze(market_data)
        regime = regime_analysis.regime.value

        # Check if we should trade
        should_trade = True
        if use_regime_filter and not regime_analysis.should_trade:
            should_trade = False

        if confidence < min_confidence:
            should_trade = False

        # Position management
        if position is not None:
            # Check exit conditions
            if position == 'long':
                pnl_pct = (current_price - entry_price) / entry_price
                if pnl_pct >= take_profit_pct or pnl_pct <= -stop_loss_pct:
                    # Exit position
                    trade = Trade(
                        entry_time=entry_time,
                        exit_time=current_time,
                        direction='long',
                        entry_price=entry_price,
                        exit_price=current_price,
                        size=position_size,
                        pnl_pct=pnl_pct,
                        regime=entry_regime,
                        confidence=entry_confidence,
                    )
                    trades.append(trade)
                    equity *= (1 + pnl_pct * position_size)
                    position = None

            elif position == 'short':
                pnl_pct = (entry_price - current_price) / entry_price
                if pnl_pct >= take_profit_pct or pnl_pct <= -stop_loss_pct:
                    # Exit position
                    trade = Trade(
                        entry_time=entry_time,
                        exit_time=current_time,
                        direction='short',
                        entry_price=entry_price,
                        exit_price=current_price,
                        size=position_size,
                        pnl_pct=pnl_pct,
                        regime=entry_regime,
                        confidence=entry_confidence,
                    )
                    trades.append(trade)
                    equity *= (1 + pnl_pct * position_size)
                    position = None

        # Enter new position if no position and should trade
        if position is None and should_trade:
            # Apply regime modifier to position size
            adjusted_size = base_position_size * regime_analysis.confidence_modifier

            if pred_class == 1:  # Predict UP
                position = 'long'
            else:  # Predict DOWN
                position = 'short'

            entry_price = current_price
            entry_time = current_time
            entry_confidence = confidence
            entry_regime = regime
            position_size = adjusted_size

    # Close any remaining position at end
    if position is not None:
        current_price = df['close'].iloc[-1]
        if position == 'long':
            pnl_pct = (current_price - entry_price) / entry_price
        else:
            pnl_pct = (entry_price - current_price) / entry_price

        trade = Trade(
            entry_time=entry_time,
            exit_time=df.index[-1],
            direction=position,
            entry_price=entry_price,
            exit_price=current_price,
            size=position_size,
            pnl_pct=pnl_pct,
            regime=entry_regime,
            confidence=entry_confidence,
        )
        trades.append(trade)
        equity *= (1 + pnl_pct * position_size)

    result.trades = trades
    result.final_equity = equity

    return result


def analyze_regime_distribution(df: pd.DataFrame, regime_filter: RegimeFilter, lookback: int = 50):
    """Analyze regime distribution in the data."""
    regimes = []
    for i in range(lookback, len(df)):
        market_data = df.iloc[i-lookback:i+1].copy()
        analysis = regime_filter.analyze(market_data)
        regimes.append(analysis.regime.value)

    regime_counts = pd.Series(regimes).value_counts()
    print("\nRegime Distribution in Test Data:")
    print("-" * 40)
    for regime, count in regime_counts.items():
        pct = count / len(regimes) * 100
        print(f"  {regime:15s}: {count:5d} bars ({pct:5.1f}%)")
    return regimes


def print_results(result: BacktestResult, title: str):
    """Print backtest results."""
    print(f"\n{'=' * 60}")
    print(f"{title}")
    print('=' * 60)
    print(f"Total Trades:      {result.total_trades}")
    print(f"Winning Trades:    {result.winning_trades}")
    print(f"Losing Trades:     {result.losing_trades}")
    print(f"Win Rate:          {result.win_rate:.2%}")
    print(f"")
    print(f"Initial Equity:    ${result.initial_equity:,.2f}")
    print(f"Final Equity:      ${result.final_equity:,.2f}")
    print(f"Total Return:      {result.total_return:.2%}")
    print(f"")
    print(f"Avg Win:           {result.avg_win:.4%}")
    print(f"Avg Loss:          {result.avg_loss:.4%}")
    print(f"Profit Factor:     {result.profit_factor:.2f}")
    print(f"Max Drawdown:      {result.max_drawdown:.2%}")
    print('=' * 60)

    # Breakdown by regime
    if result.trades:
        print("\nBreakdown by Regime:")
        print("-" * 40)
        regimes = set(t.regime for t in result.trades)
        for regime in sorted(regimes):
            regime_trades = [t for t in result.trades if t.regime == regime]
            wins = sum(1 for t in regime_trades if t.is_win)
            total = len(regime_trades)
            win_rate = wins / total if total > 0 else 0
            avg_pnl = np.mean([t.pnl_pct for t in regime_trades])
            print(f"  {regime:15s}: {total:4d} trades, {win_rate:6.2%} win rate, {avg_pnl:+.4%} avg PnL")


def main():
    parser = argparse.ArgumentParser(description="Run trading backtest")
    parser.add_argument("--timeframe", type=str, default="1H", help="Timeframe")
    parser.add_argument("--initial-equity", type=float, default=10000, help="Initial equity")
    parser.add_argument("--position-size", type=float, default=0.02, help="Base position size (fraction)")
    parser.add_argument("--stop-loss", type=float, default=0.01, help="Stop loss percentage")
    parser.add_argument("--take-profit", type=float, default=0.02, help="Take profit percentage")
    parser.add_argument("--min-confidence", type=float, default=0.60, help="Minimum confidence to trade")
    parser.add_argument("--threshold", type=float, default=0.0005, help="Label threshold")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("TRADING BACKTEST WITH REGIME FILTERING")
    print("=" * 60)
    print(f"Timeframe:       {args.timeframe}")
    print(f"Initial Equity:  ${args.initial_equity:,.2f}")
    print(f"Position Size:   {args.position_size:.1%}")
    print(f"Stop Loss:       {args.stop_loss:.1%}")
    print(f"Take Profit:     {args.take_profit:.1%}")
    print(f"Min Confidence:  {args.min_confidence:.1%}")
    print("=" * 60)

    # Load and prepare data
    logger.info("Loading data...")
    df = load_data(args.timeframe)
    logger.info(f"Loaded {len(df)} bars")

    # Calculate features
    logger.info("Calculating features...")
    df_features = calculate_features(df)

    # Create labels
    labels = create_labels(df_features, args.threshold)

    # Get feature columns
    feature_cols = [c for c in df_features.columns if c not in ['open', 'high', 'low', 'close', 'volume']]

    # Prepare data for training
    X = df_features[feature_cols].values
    y = labels.values

    # Remove NaN rows
    valid_mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
    X = X[valid_mask]
    y = y[valid_mask]
    df_valid = df_features[valid_mask].copy()

    logger.info(f"Valid samples: {len(X)}")

    # Split data: 60% train, 20% validation, 20% test
    n_train = int(len(X) * 0.6)
    n_val = int(len(X) * 0.2)

    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train:n_train+n_val], y[n_train:n_train+n_val]
    X_test, y_test = X[n_train+n_val:], y[n_train+n_val:]
    df_test = df_valid.iloc[n_train+n_val:].copy()

    logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # Normalize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Train model
    logger.info("Training model...")
    model = train_model(X_train_scaled, y_train)

    # Evaluate on validation set
    val_pred = model.predict(X_val_scaled)
    val_acc = (val_pred == y_val).mean()
    logger.info(f"Validation accuracy: {val_acc:.2%}")

    # Create regime filter
    regime_filter = RegimeFilter(timeframe=args.timeframe)

    # Analyze regime distribution in test data
    analyze_regime_distribution(df_test, regime_filter)

    # Store scaled features in df for backtest
    df_test_with_features = df_test.copy()
    df_test_scaled = pd.DataFrame(
        scaler.transform(df_test[feature_cols].values),
        index=df_test.index,
        columns=feature_cols
    )
    for col in feature_cols:
        df_test_with_features[col] = df_test_scaled[col].values

    # Run backtest WITHOUT regime filter
    logger.info("Running backtest WITHOUT regime filter...")
    result_no_filter = run_backtest(
        df=df_test,
        model=model,
        scaler=scaler,
        feature_cols=feature_cols,
        regime_filter=regime_filter,
        initial_equity=args.initial_equity,
        base_position_size=args.position_size,
        stop_loss_pct=args.stop_loss,
        take_profit_pct=args.take_profit,
        min_confidence=args.min_confidence,
        use_regime_filter=False,
    )

    print_results(result_no_filter, "RESULTS WITHOUT REGIME FILTER")

    # Run backtest WITH regime filter
    logger.info("Running backtest WITH regime filter...")
    result_with_filter = run_backtest(
        df=df_test,
        model=model,
        scaler=scaler,
        feature_cols=feature_cols,
        regime_filter=regime_filter,
        initial_equity=args.initial_equity,
        base_position_size=args.position_size,
        stop_loss_pct=args.stop_loss,
        take_profit_pct=args.take_profit,
        min_confidence=args.min_confidence,
        use_regime_filter=True,
    )

    print_results(result_with_filter, "RESULTS WITH REGIME FILTER")

    # Comparison
    print("\n" + "=" * 60)
    print("COMPARISON: REGIME FILTER IMPACT")
    print("=" * 60)
    print(f"{'Metric':<25} {'No Filter':>15} {'With Filter':>15} {'Diff':>12}")
    print("-" * 67)
    print(f"{'Total Trades':<25} {result_no_filter.total_trades:>15} {result_with_filter.total_trades:>15} {result_with_filter.total_trades - result_no_filter.total_trades:>+12}")
    print(f"{'Win Rate':<25} {result_no_filter.win_rate:>14.2%} {result_with_filter.win_rate:>14.2%} {(result_with_filter.win_rate - result_no_filter.win_rate)*100:>+11.2f}pp")
    print(f"{'Total Return':<25} {result_no_filter.total_return:>14.2%} {result_with_filter.total_return:>14.2%} {(result_with_filter.total_return - result_no_filter.total_return)*100:>+11.2f}pp")
    print(f"{'Profit Factor':<25} {result_no_filter.profit_factor:>15.2f} {result_with_filter.profit_factor:>15.2f} {result_with_filter.profit_factor - result_no_filter.profit_factor:>+12.2f}")
    print(f"{'Max Drawdown':<25} {result_no_filter.max_drawdown:>14.2%} {result_with_filter.max_drawdown:>14.2%} {(result_with_filter.max_drawdown - result_no_filter.max_drawdown)*100:>+11.2f}pp")
    print("=" * 60)

    # Final verdict
    print("\n" + "=" * 60)
    print("VERDICT")
    print("=" * 60)
    if result_with_filter.total_return > result_no_filter.total_return:
        improvement = result_with_filter.total_return - result_no_filter.total_return
        print(f"Regime filtering IMPROVED returns by {improvement:.2%}")
    else:
        degradation = result_no_filter.total_return - result_with_filter.total_return
        print(f"Regime filtering REDUCED returns by {degradation:.2%}")

    if result_with_filter.win_rate > result_no_filter.win_rate:
        print(f"Win rate improved from {result_no_filter.win_rate:.2%} to {result_with_filter.win_rate:.2%}")

    if result_with_filter.max_drawdown < result_no_filter.max_drawdown:
        print(f"Max drawdown reduced from {result_no_filter.max_drawdown:.2%} to {result_with_filter.max_drawdown:.2%}")

    print("=" * 60)


if __name__ == "__main__":
    main()
