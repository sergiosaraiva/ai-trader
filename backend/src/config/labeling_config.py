"""Labeling configuration.

This module defines parameters for different labeling methods:
- Triple barrier labeling (primary method)
- Multi-bar lookahead labeling (alternative)
- Volatility-adjusted labeling (alternative)
"""

from dataclasses import dataclass, field
from typing import Dict, Any


@dataclass
class TripleBarrierParameters:
    """Triple barrier labeling (primary method).

    Note: TP/SL/max_holding are in TradingConfig.timeframes.
    This just documents alternative approaches if needed.
    """

    pip_value: float = 0.0001  # EUR/USD standard

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "pip_value": self.pip_value,
        }


@dataclass
class MultiBarParameters:
    """Multi-bar lookahead labeling (alternative)."""

    forward_bars: int = 12
    threshold_pips: float = 10.0
    pip_value: float = 0.0001

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "forward_bars": self.forward_bars,
            "threshold_pips": self.threshold_pips,
            "pip_value": self.pip_value,
        }


@dataclass
class VolatilityAdjustedParameters:
    """Volatility-adjusted labeling (alternative)."""

    atr_multiplier: float = 2.0
    use_dynamic_barriers: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "atr_multiplier": self.atr_multiplier,
            "use_dynamic_barriers": self.use_dynamic_barriers,
        }


@dataclass
class LabelingParameters:
    """Complete labeling configuration."""

    # Primary method (triple_barrier uses TradingConfig.timeframes)
    primary_method: str = "triple_barrier"  # or "multi_bar", "volatility"

    # Alternative methods
    triple_barrier: TripleBarrierParameters = field(
        default_factory=TripleBarrierParameters
    )
    multi_bar: MultiBarParameters = field(default_factory=MultiBarParameters)
    volatility: VolatilityAdjustedParameters = field(
        default_factory=VolatilityAdjustedParameters
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "primary_method": self.primary_method,
            "triple_barrier": self.triple_barrier.to_dict(),
            "multi_bar": self.multi_bar.to_dict(),
            "volatility": self.volatility.to_dict(),
        }
