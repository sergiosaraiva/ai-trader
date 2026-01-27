"""Technical indicator configuration.

This module defines all parameters for technical indicators used in feature engineering:
- Trend indicators (SMA, EMA, ADX, Aroon, etc.)
- Momentum indicators (RSI, MACD, Stochastic, CCI, etc.)
- Volatility indicators (ATR, Bollinger, Keltner, etc.)
- Volume indicators (CMF, Volume SMA, etc.)
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any


@dataclass
class TrendIndicators:
    """Trend indicator configuration."""

    # Moving Averages
    sma_periods: List[int] = field(default_factory=lambda: [5, 10, 20, 50, 100, 200])
    ema_periods: List[int] = field(default_factory=lambda: [5, 10, 20, 50, 100, 200])
    wma_periods: List[int] = field(default_factory=lambda: [10, 20, 50])

    # Directional Indicators
    adx_period: int = 14
    aroon_period: int = 25

    # Supertrend
    supertrend_period: int = 10
    supertrend_multiplier: float = 3.0

    # MA Crossovers
    sma_crossover_pairs: List[Tuple[int, int]] = field(
        default_factory=lambda: [(5, 20), (20, 50), (50, 200)]
    )
    ema_crossover_pairs: List[Tuple[int, int]] = field(
        default_factory=lambda: [(5, 20), (12, 26)]
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "sma_periods": self.sma_periods,
            "ema_periods": self.ema_periods,
            "wma_periods": self.wma_periods,
            "adx_period": self.adx_period,
            "aroon_period": self.aroon_period,
            "supertrend_period": self.supertrend_period,
            "supertrend_multiplier": self.supertrend_multiplier,
            "sma_crossover_pairs": self.sma_crossover_pairs,
            "ema_crossover_pairs": self.ema_crossover_pairs,
        }


@dataclass
class MomentumIndicators:
    """Momentum indicator configuration."""

    # RSI
    rsi_periods: List[int] = field(default_factory=lambda: [7, 14, 21])

    # Stochastic
    stochastic_k_period: int = 14
    stochastic_d_period: int = 3

    # MACD
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9

    # CCI
    cci_periods: List[int] = field(default_factory=lambda: [14, 20])
    cci_constant: float = 0.015

    # Momentum & ROC
    momentum_periods: List[int] = field(default_factory=lambda: [10, 14])
    roc_periods: List[int] = field(default_factory=lambda: [10, 14])

    # Williams %R
    williams_period: int = 14

    # MFI
    mfi_period: int = 14

    # TSI
    tsi_long: int = 25
    tsi_short: int = 13

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "rsi_periods": self.rsi_periods,
            "stochastic_k_period": self.stochastic_k_period,
            "stochastic_d_period": self.stochastic_d_period,
            "macd_fast": self.macd_fast,
            "macd_slow": self.macd_slow,
            "macd_signal": self.macd_signal,
            "cci_periods": self.cci_periods,
            "cci_constant": self.cci_constant,
            "momentum_periods": self.momentum_periods,
            "roc_periods": self.roc_periods,
            "williams_period": self.williams_period,
            "mfi_period": self.mfi_period,
            "tsi_long": self.tsi_long,
            "tsi_short": self.tsi_short,
        }


@dataclass
class VolatilityIndicators:
    """Volatility indicator configuration."""

    # ATR
    atr_period: int = 14
    natr_period: int = 14

    # Bollinger Bands
    bollinger_period: int = 20
    bollinger_std: float = 2.0

    # Keltner Channel
    keltner_period: int = 20
    keltner_multiplier: float = 2.0

    # Donchian Channel
    donchian_period: int = 20

    # Standard Deviation
    std_periods: List[int] = field(default_factory=lambda: [10, 20])

    # Historical Volatility
    hvol_periods: List[int] = field(default_factory=lambda: [10, 20, 30])
    hvol_annualization_factor: int = 252

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "atr_period": self.atr_period,
            "natr_period": self.natr_period,
            "bollinger_period": self.bollinger_period,
            "bollinger_std": self.bollinger_std,
            "keltner_period": self.keltner_period,
            "keltner_multiplier": self.keltner_multiplier,
            "donchian_period": self.donchian_period,
            "std_periods": self.std_periods,
            "hvol_periods": self.hvol_periods,
            "hvol_annualization_factor": self.hvol_annualization_factor,
        }


@dataclass
class VolumeIndicators:
    """Volume indicator configuration."""

    # Chaikin Money Flow
    cmf_period: int = 20

    # Ease of Movement
    emv_period: int = 14
    emv_scaling_factor: float = 1e8

    # Force Index
    force_index_period: int = 13

    # A/D Oscillator
    adosc_fast: int = 3
    adosc_slow: int = 10

    # Volume SMA
    volume_sma_periods: List[int] = field(default_factory=lambda: [10, 20])

    # Volume Ratio
    volume_ratio_period: int = 14

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "cmf_period": self.cmf_period,
            "emv_period": self.emv_period,
            "emv_scaling_factor": self.emv_scaling_factor,
            "force_index_period": self.force_index_period,
            "adosc_fast": self.adosc_fast,
            "adosc_slow": self.adosc_slow,
            "volume_sma_periods": self.volume_sma_periods,
            "volume_ratio_period": self.volume_ratio_period,
        }


@dataclass
class IndicatorParameters:
    """Complete indicator configuration wrapper."""

    trend: TrendIndicators = field(default_factory=TrendIndicators)
    momentum: MomentumIndicators = field(default_factory=MomentumIndicators)
    volatility: VolatilityIndicators = field(default_factory=VolatilityIndicators)
    volume: VolumeIndicators = field(default_factory=VolumeIndicators)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "trend": self.trend.to_dict(),
            "momentum": self.momentum.to_dict(),
            "volatility": self.volatility.to_dict(),
            "volume": self.volume.to_dict(),
        }
