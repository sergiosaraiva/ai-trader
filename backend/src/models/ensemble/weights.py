"""Dynamic weight calculation for ensemble models.

Provides weight adjustment based on:
- Recent model performance
- Market regime
- Volatility conditions
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    """Market regime classification."""

    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    VOLATILE = "volatile"
    UNKNOWN = "unknown"


class VolatilityLevel(Enum):
    """Volatility level classification."""

    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    EXTREME = "extreme"


@dataclass
class TradeResult:
    """Result of a single trade for performance tracking."""

    model_name: str
    direction_correct: bool
    profit_pct: float
    confidence: float
    timestamp: float


@dataclass
class WeightConfig:
    """Configuration for dynamic weight calculation.

    Attributes:
        base_weights: Default weights for each model.
        lookback_trades: Number of recent trades for performance calculation.
        performance_blend: How much to weight recent performance vs base.
        min_weight: Minimum weight any model can have.
        max_weight: Maximum weight any model can have.
        regime_adjustments: Weight multipliers per market regime.
        volatility_adjustments: Weight multipliers per volatility level.
    """

    base_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "short_term": 0.35,
            "medium_term": 0.35,
            "long_term": 0.30,
        }
    )
    lookback_trades: int = 50
    performance_blend: float = 0.3
    min_weight: float = 0.1
    max_weight: float = 0.6
    regime_adjustments: Dict[str, Dict[str, float]] = field(default_factory=dict)
    volatility_adjustments: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def __post_init__(self):
        """Initialize default adjustments if not provided."""
        if not self.regime_adjustments:
            self.regime_adjustments = {
                MarketRegime.TRENDING_UP.value: {
                    "short_term": 0.8,
                    "medium_term": 1.2,
                    "long_term": 1.3,
                },
                MarketRegime.TRENDING_DOWN.value: {
                    "short_term": 0.8,
                    "medium_term": 1.2,
                    "long_term": 1.3,
                },
                MarketRegime.RANGING.value: {
                    "short_term": 1.4,
                    "medium_term": 1.0,
                    "long_term": 0.6,
                },
                MarketRegime.VOLATILE.value: {
                    "short_term": 1.1,
                    "medium_term": 0.9,
                    "long_term": 0.5,
                },
                MarketRegime.UNKNOWN.value: {
                    "short_term": 1.0,
                    "medium_term": 1.0,
                    "long_term": 1.0,
                },
            }

        if not self.volatility_adjustments:
            self.volatility_adjustments = {
                VolatilityLevel.LOW.value: {
                    "short_term": 0.9,
                    "medium_term": 1.0,
                    "long_term": 1.1,
                },
                VolatilityLevel.NORMAL.value: {
                    "short_term": 1.0,
                    "medium_term": 1.0,
                    "long_term": 1.0,
                },
                VolatilityLevel.HIGH.value: {
                    "short_term": 1.1,
                    "medium_term": 1.0,
                    "long_term": 0.8,
                },
                VolatilityLevel.EXTREME.value: {
                    "short_term": 1.2,
                    "medium_term": 0.9,
                    "long_term": 0.5,
                },
            }


class DynamicWeightCalculator:
    """Calculator for dynamic ensemble model weights.

    Adjusts weights based on:
    - Recent model performance (accuracy, profit)
    - Market regime (trending vs ranging)
    - Volatility conditions

    Example:
        ```python
        calculator = DynamicWeightCalculator()

        # Update with recent performance
        weights = calculator.calculate_weights(
            recent_performance={
                "short_term": [TradeResult(...), ...],
                "medium_term": [TradeResult(...), ...],
                "long_term": [TradeResult(...), ...],
            },
            market_regime=MarketRegime.TRENDING_UP,
            volatility_level=VolatilityLevel.NORMAL,
        )
        # weights = {"short_term": 0.25, "medium_term": 0.40, "long_term": 0.35}
        ```
    """

    def __init__(self, config: Optional[WeightConfig] = None):
        """Initialize weight calculator.

        Args:
            config: Weight configuration.
        """
        self.config = config or WeightConfig()
        self._trade_history: Dict[str, List[TradeResult]] = {
            name: [] for name in self.config.base_weights
        }

    def add_trade_result(self, result: TradeResult) -> None:
        """Add a trade result to history.

        Args:
            result: Trade result to add.
        """
        if result.model_name in self._trade_history:
            self._trade_history[result.model_name].append(result)

            # Keep only recent trades
            if len(self._trade_history[result.model_name]) > self.config.lookback_trades:
                self._trade_history[result.model_name] = self._trade_history[
                    result.model_name
                ][-self.config.lookback_trades :]

    def calculate_weights(
        self,
        recent_performance: Optional[Dict[str, List[TradeResult]]] = None,
        market_regime: MarketRegime = MarketRegime.UNKNOWN,
        volatility_level: VolatilityLevel = VolatilityLevel.NORMAL,
    ) -> Dict[str, float]:
        """Calculate dynamic weights for ensemble models.

        Args:
            recent_performance: Dict mapping model name to trade results.
                If None, uses internal trade history.
            market_regime: Current market regime.
            volatility_level: Current volatility level.

        Returns:
            Dictionary of model name to weight.
        """
        # Use provided performance or internal history
        performance = recent_performance or self._trade_history

        # Start with base weights
        weights = self.config.base_weights.copy()

        # Apply regime adjustments
        weights = self._apply_regime_adjustment(weights, market_regime)

        # Apply volatility adjustments
        weights = self._apply_volatility_adjustment(weights, volatility_level)

        # Apply performance adjustments
        if performance and any(len(trades) > 0 for trades in performance.values()):
            weights = self._apply_performance_adjustment(weights, performance)

        # Normalize and clamp
        weights = self._normalize_and_clamp(weights)

        return weights

    def _apply_regime_adjustment(
        self,
        weights: Dict[str, float],
        regime: MarketRegime,
    ) -> Dict[str, float]:
        """Apply market regime weight adjustments."""
        adjustments = self.config.regime_adjustments.get(
            regime.value, {name: 1.0 for name in weights}
        )

        return {
            name: weight * adjustments.get(name, 1.0)
            for name, weight in weights.items()
        }

    def _apply_volatility_adjustment(
        self,
        weights: Dict[str, float],
        volatility: VolatilityLevel,
    ) -> Dict[str, float]:
        """Apply volatility weight adjustments."""
        adjustments = self.config.volatility_adjustments.get(
            volatility.value, {name: 1.0 for name in weights}
        )

        return {
            name: weight * adjustments.get(name, 1.0)
            for name, weight in weights.items()
        }

    def _apply_performance_adjustment(
        self,
        weights: Dict[str, float],
        performance: Dict[str, List[TradeResult]],
    ) -> Dict[str, float]:
        """Apply performance-based weight adjustments."""
        # Calculate performance scores
        performance_scores = {}

        for name, trades in performance.items():
            if not trades:
                performance_scores[name] = 0.5  # Neutral score
                continue

            # Calculate accuracy and profit-weighted accuracy
            accuracy = sum(1 for t in trades if t.direction_correct) / len(trades)
            avg_profit = np.mean([t.profit_pct for t in trades])

            # Combine metrics (accuracy more important than profit)
            score = 0.7 * accuracy + 0.3 * (0.5 + np.clip(avg_profit * 10, -0.5, 0.5))
            performance_scores[name] = max(0.1, min(0.9, score))

        # Normalize scores
        total_score = sum(performance_scores.values())
        if total_score > 0:
            normalized_scores = {
                name: score / total_score for name, score in performance_scores.items()
            }
        else:
            normalized_scores = {name: 1.0 / len(weights) for name in weights}

        # Blend with current weights
        blend = self.config.performance_blend
        return {
            name: (1 - blend) * weights.get(name, 0) + blend * normalized_scores.get(name, 0)
            for name in set(weights) | set(normalized_scores)
        }

    def _normalize_and_clamp(self, weights: Dict[str, float]) -> Dict[str, float]:
        """Normalize weights to sum to 1 and apply min/max constraints."""
        # Clamp to min/max
        clamped = {
            name: max(self.config.min_weight, min(self.config.max_weight, weight))
            for name, weight in weights.items()
        }

        # Normalize
        total = sum(clamped.values())
        if total > 0:
            return {name: weight / total for name, weight in clamped.items()}
        else:
            n = len(clamped)
            return {name: 1.0 / n for name in clamped}

    def get_weight_explanation(
        self,
        weights: Dict[str, float],
        market_regime: MarketRegime,
        volatility_level: VolatilityLevel,
    ) -> str:
        """Generate human-readable explanation of weight calculation.

        Args:
            weights: Calculated weights.
            market_regime: Current market regime.
            volatility_level: Current volatility level.

        Returns:
            Explanation string.
        """
        lines = ["Weight Calculation Explanation:", "-" * 40]

        lines.append(f"\nMarket Regime: {market_regime.value}")
        lines.append(f"Volatility Level: {volatility_level.value}")

        lines.append("\nBase Weights:")
        for name, weight in self.config.base_weights.items():
            lines.append(f"  {name}: {weight:.2%}")

        lines.append("\nFinal Weights:")
        for name, weight in weights.items():
            change = weight - self.config.base_weights.get(name, 0)
            change_str = f"({change:+.2%})" if change != 0 else ""
            lines.append(f"  {name}: {weight:.2%} {change_str}")

        # Performance summary
        if any(len(trades) > 0 for trades in self._trade_history.values()):
            lines.append("\nRecent Performance:")
            for name, trades in self._trade_history.items():
                if trades:
                    accuracy = sum(1 for t in trades if t.direction_correct) / len(trades)
                    lines.append(f"  {name}: {accuracy:.1%} accuracy ({len(trades)} trades)")

        return "\n".join(lines)


def detect_market_regime(
    prices: np.ndarray,
    lookback: int = 20,
    trend_threshold: float = 0.02,
    volatility_threshold: float = 0.03,
) -> Tuple[MarketRegime, VolatilityLevel]:
    """Detect market regime and volatility from price data.

    Args:
        prices: Array of prices.
        lookback: Lookback period for calculations.
        trend_threshold: Threshold for trend detection.
        volatility_threshold: Threshold for high volatility.

    Returns:
        Tuple of (MarketRegime, VolatilityLevel).
    """
    if len(prices) < lookback + 1:
        return MarketRegime.UNKNOWN, VolatilityLevel.NORMAL

    recent_prices = prices[-lookback:]

    # Calculate returns
    returns = np.diff(recent_prices) / recent_prices[:-1]

    # Trend detection
    cumulative_return = (recent_prices[-1] / recent_prices[0]) - 1
    avg_return = np.mean(returns)

    # Volatility calculation
    volatility = np.std(returns) * np.sqrt(252)  # Annualized

    # Determine regime
    if cumulative_return > trend_threshold and avg_return > 0:
        regime = MarketRegime.TRENDING_UP
    elif cumulative_return < -trend_threshold and avg_return < 0:
        regime = MarketRegime.TRENDING_DOWN
    elif volatility > volatility_threshold * 2:
        regime = MarketRegime.VOLATILE
    else:
        regime = MarketRegime.RANGING

    # Determine volatility level
    if volatility < 0.10:
        vol_level = VolatilityLevel.LOW
    elif volatility < 0.20:
        vol_level = VolatilityLevel.NORMAL
    elif volatility < 0.35:
        vol_level = VolatilityLevel.HIGH
    else:
        vol_level = VolatilityLevel.EXTREME

    return regime, vol_level
