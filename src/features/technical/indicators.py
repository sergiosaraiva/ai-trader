"""Main technical indicators class combining all indicator types."""

from typing import List, Optional
import pandas as pd

from .trend import TrendIndicators
from .momentum import MomentumIndicators
from .volatility import VolatilityIndicators
from .volume import VolumeIndicators


class TechnicalIndicators:
    """
    Comprehensive technical indicators calculator.

    Combines trend, momentum, volatility, and volume indicators.
    """

    def __init__(self):
        """Initialize technical indicators calculator."""
        self.trend = TrendIndicators()
        self.momentum = MomentumIndicators()
        self.volatility = VolatilityIndicators()
        self.volume = VolumeIndicators()

    def calculate_all(
        self,
        df: pd.DataFrame,
        include_groups: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Calculate all technical indicators.

        Args:
            df: OHLCV dataframe with columns [open, high, low, close, volume]
            include_groups: List of indicator groups to include
                           ('trend', 'momentum', 'volatility', 'volume')
                           If None, includes all groups.

        Returns:
            DataFrame with all calculated indicators
        """
        result = df.copy()
        include_groups = include_groups or ["trend", "momentum", "volatility", "volume"]

        if "trend" in include_groups:
            result = self.trend.calculate_all(result)

        if "momentum" in include_groups:
            result = self.momentum.calculate_all(result)

        if "volatility" in include_groups:
            result = self.volatility.calculate_all(result)

        if "volume" in include_groups:
            result = self.volume.calculate_all(result)

        return result

    def get_feature_names(self, include_groups: Optional[List[str]] = None) -> List[str]:
        """Get list of all feature names that will be generated."""
        features = []
        include_groups = include_groups or ["trend", "momentum", "volatility", "volume"]

        if "trend" in include_groups:
            features.extend(self.trend.get_feature_names())
        if "momentum" in include_groups:
            features.extend(self.momentum.get_feature_names())
        if "volatility" in include_groups:
            features.extend(self.volatility.get_feature_names())
        if "volume" in include_groups:
            features.extend(self.volume.get_feature_names())

        return features

    @staticmethod
    def get_default_periods() -> dict:
        """Get default periods for each indicator type."""
        return {
            "sma": [5, 10, 20, 50, 100, 200],
            "ema": [5, 10, 20, 50, 100, 200],
            "rsi": [7, 14, 21],
            "macd": {"fast": 12, "slow": 26, "signal": 9},
            "bb": {"period": 20, "std": 2},
            "atr": [14],
            "stoch": {"k": 14, "d": 3},
            "adx": [14],
        }
