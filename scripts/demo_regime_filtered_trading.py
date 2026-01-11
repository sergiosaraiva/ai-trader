#!/usr/bin/env python3
"""Demo: Regime-filtered trading signal generation.

This script demonstrates how the regime filter improves trading decisions
by only allowing trades in favorable market conditions.

Based on backtesting results:
- 1H: Best in trending_down (54.55% accuracy), avoid ranging (46.29%)
- 4H: Best in ranging (56.00%) and trending_up (55.00%)
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np

from src.trading.filters import RegimeFilter, MarketRegime
from src.trading.signals.generator import SignalGenerator, EnsemblePrediction
from src.trading.risk.profiles import load_risk_profile
from src.trading.circuit_breakers.base import CircuitBreakerState, TradingState


def create_sample_market_data() -> pd.DataFrame:
    """Create sample OHLCV data for demonstration."""
    np.random.seed(42)
    n_bars = 100

    # Simulate trending down market (optimal for 1H)
    price = 1.0850
    prices = [price]
    for _ in range(n_bars - 1):
        change = np.random.randn() * 0.001 - 0.0002  # Slight downward bias
        price = price * (1 + change)
        prices.append(price)

    df = pd.DataFrame({
        'open': prices,
        'high': [p * (1 + abs(np.random.randn() * 0.001)) for p in prices],
        'low': [p * (1 - abs(np.random.randn() * 0.001)) for p in prices],
        'close': prices,
        'volume': [np.random.randint(1000, 5000) for _ in range(n_bars)],
    })

    return df


def create_sample_prediction(bullish: bool = True, confidence: float = 0.70) -> EnsemblePrediction:
    """Create a sample prediction for demonstration."""
    if bullish:
        direction_prob = 0.5 + (confidence - 0.5) * 0.3
    else:
        direction_prob = 0.5 - (confidence - 0.5) * 0.3

    return EnsemblePrediction(
        direction_probability=direction_prob,
        confidence=confidence,
        alpha=confidence * 10,
        beta=(1 - confidence) * 10,
        short_term_signal=0.3 if bullish else -0.3,
        medium_term_signal=0.2 if bullish else -0.2,
        long_term_signal=0.1 if bullish else -0.1,
        ensemble_agreement=0.8,
    )


def main():
    print("=" * 70)
    print("REGIME-FILTERED TRADING DEMO")
    print("=" * 70)
    print()

    # Initialize components
    risk_profile = load_risk_profile("moderate")

    # Create regime filter for 1H timeframe
    regime_filter = RegimeFilter(timeframe="1H")

    # Create signal generator with regime filter
    signal_gen = SignalGenerator(
        risk_profile=risk_profile,
        regime_filter=regime_filter,
        timeframe="1H",
    )

    # Create sample market data
    market_data = create_sample_market_data()

    # Analyze current regime
    analysis = regime_filter.analyze(market_data)

    print("MARKET REGIME ANALYSIS")
    print("-" * 40)
    print(f"Regime:          {analysis.regime.value}")
    print(f"ADX:             {analysis.adx:.1f}")
    print(f"Trend Strength:  {analysis.trend_strength:.4f}")
    print(f"Volatility:      {analysis.volatility_ratio:.2f}x normal")
    print(f"Should Trade:    {analysis.should_trade}")
    print(f"Position Mod:    {analysis.confidence_modifier:.1%}")
    print(f"Reason:          {analysis.reason}")
    print()

    # Create a breaker state (no breakers triggered)
    breaker_state = CircuitBreakerState(
        overall_state=TradingState.ACTIVE,
        active_breakers=[],
        reasons=[],
        size_multiplier=1.0,
        min_confidence_override=None,
    )

    # Test different scenarios
    scenarios = [
        ("High confidence SELL", create_sample_prediction(bullish=False, confidence=0.85)),
        ("High confidence BUY", create_sample_prediction(bullish=True, confidence=0.85)),
        ("Low confidence SELL", create_sample_prediction(bullish=False, confidence=0.55)),
        ("Medium confidence SELL", create_sample_prediction(bullish=False, confidence=0.70)),
    ]

    print("SIGNAL GENERATION RESULTS")
    print("-" * 40)

    for scenario_name, prediction in scenarios:
        signal = signal_gen.generate_signal(
            prediction=prediction,
            symbol="EURUSD",
            current_price=market_data['close'].iloc[-1],
            breaker_state=breaker_state,
            market_data=market_data,
            atr=0.0010,
        )

        print(f"\n{scenario_name}:")
        print(f"  Action:        {signal.action.value}")
        print(f"  Confidence:    {signal.confidence:.1%}")
        print(f"  Position Size: {signal.position_size_pct:.2%}")
        print(f"  Reason:        {signal.reason}")

    print()
    print("=" * 70)
    print("KEY INSIGHTS")
    print("=" * 70)
    print("""
1. REGIME FILTERING improves accuracy by only trading in favorable conditions
   - 1H optimal in trending_down: 54.55% vs 51.09% baseline
   - 4H optimal in ranging: 56.00% vs 51.95% baseline

2. POSITION SIZING is reduced in non-optimal regimes (confidence_modifier)

3. COUNTER-TREND signals get reduced confidence in trending markets

4. AVOID ranging markets on 1H (46.29% - worse than random!)

5. The regime filter is now integrated into SignalGenerator and will
   automatically filter signals based on market conditions.
""")


if __name__ == "__main__":
    main()
