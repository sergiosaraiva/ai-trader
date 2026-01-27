"""Feature engineering configuration.

This module defines parameters for feature engineering:
- Lag parameters (standard lags, ROC periods)
- Session parameters (trading session times)
- Cyclical encoding (time cycles)
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any


@dataclass
class LagParameters:
    """Lag feature configuration."""

    standard_lags: List[int] = field(default_factory=lambda: [1, 2, 3, 6, 12])
    rsi_roc_periods: List[int] = field(default_factory=lambda: [3, 6])
    macd_roc_periods: List[int] = field(default_factory=lambda: [3])
    adx_roc_periods: List[int] = field(default_factory=lambda: [3])
    atr_roc_periods: List[int] = field(default_factory=lambda: [3, 6])
    price_roc_periods: List[int] = field(default_factory=lambda: [1, 3, 6, 12])
    volume_roc_periods: List[int] = field(default_factory=lambda: [3, 6])

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "standard_lags": self.standard_lags,
            "rsi_roc_periods": self.rsi_roc_periods,
            "macd_roc_periods": self.macd_roc_periods,
            "adx_roc_periods": self.adx_roc_periods,
            "atr_roc_periods": self.atr_roc_periods,
            "price_roc_periods": self.price_roc_periods,
            "volume_roc_periods": self.volume_roc_periods,
        }


@dataclass
class SessionParameters:
    """Trading session configuration (UTC hours)."""

    asian_session: Tuple[int, int] = (0, 8)
    london_session: Tuple[int, int] = (8, 16)
    ny_session: Tuple[int, int] = (13, 22)
    overlap_session: Tuple[int, int] = (13, 16)

    # Timezone offset (if needed for non-UTC deployment)
    timezone_offset_hours: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "asian_session": self.asian_session,
            "london_session": self.london_session,
            "ny_session": self.ny_session,
            "overlap_session": self.overlap_session,
            "timezone_offset_hours": self.timezone_offset_hours,
        }


@dataclass
class CyclicalEncoding:
    """Cyclical feature encoding parameters."""

    hour_encoding_cycles: int = 24  # 24-hour cycle
    day_of_week_cycles: int = 7  # 7-day cycle
    day_of_month_cycles: int = 31  # Month cycle

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "hour_encoding_cycles": self.hour_encoding_cycles,
            "day_of_week_cycles": self.day_of_week_cycles,
            "day_of_month_cycles": self.day_of_month_cycles,
        }


@dataclass
class FeatureParameters:
    """Complete feature engineering configuration."""

    lags: LagParameters = field(default_factory=LagParameters)
    sessions: SessionParameters = field(default_factory=SessionParameters)
    cyclical: CyclicalEncoding = field(default_factory=CyclicalEncoding)

    # Normalization windows
    percentile_window: int = 50
    zscore_window: int = 50

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "lags": self.lags.to_dict(),
            "sessions": self.sessions.to_dict(),
            "cyclical": self.cyclical.to_dict(),
            "percentile_window": self.percentile_window,
            "zscore_window": self.zscore_window,
        }
