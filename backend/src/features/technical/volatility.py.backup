"""Volatility indicators for technical analysis."""

from typing import List
import numpy as np
import pandas as pd


class VolatilityIndicators:
    """Calculate volatility-based technical indicators."""

    def __init__(self):
        """Initialize volatility indicators."""
        self._feature_names: List[str] = []

    def get_feature_names(self) -> List[str]:
        """Get list of generated feature names."""
        return self._feature_names.copy()

    def calculate_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all volatility indicators."""
        df = df.copy()
        self._feature_names = []

        df = self.atr(df, periods=[14])
        df = self.natr(df, period=14)
        df = self.bollinger_bands(df, period=20, std_dev=2)
        df = self.keltner_channel(df, period=20, multiplier=2)
        df = self.donchian_channel(df, period=20)
        df = self.stddev(df, periods=[10, 20])
        df = self.historical_volatility(df, periods=[10, 20, 30])

        return df

    def true_range(self, df: pd.DataFrame) -> pd.Series:
        """Calculate True Range."""
        high = df["high"]
        low = df["low"]
        prev_close = df["close"].shift(1)

        tr1 = high - low
        tr2 = abs(high - prev_close)
        tr3 = abs(low - prev_close)

        return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    def atr(self, df: pd.DataFrame, periods: List[int]) -> pd.DataFrame:
        """Average True Range."""
        tr = self.true_range(df)

        for period in periods:
            col_name = f"atr_{period}"
            df[col_name] = tr.ewm(span=period, adjust=False).mean()
            self._feature_names.append(col_name)

        return df

    def natr(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Normalized Average True Range."""
        tr = self.true_range(df)
        atr = tr.ewm(span=period, adjust=False).mean()
        natr = 100 * atr / df["close"]

        col_name = f"natr_{period}"
        df[col_name] = natr
        self._feature_names.append(col_name)
        return df

    def bollinger_bands(
        self, df: pd.DataFrame, period: int = 20, std_dev: float = 2, column: str = "close"
    ) -> pd.DataFrame:
        """Bollinger Bands."""
        sma = df[column].rolling(window=period).mean()
        std = df[column].rolling(window=period).std()

        df[f"bb_upper_{period}"] = sma + std_dev * std
        df[f"bb_middle_{period}"] = sma
        df[f"bb_lower_{period}"] = sma - std_dev * std
        df[f"bb_width_{period}"] = (df[f"bb_upper_{period}"] - df[f"bb_lower_{period}"]) / sma
        df[f"bb_pct_{period}"] = (df[column] - df[f"bb_lower_{period}"]) / (
            df[f"bb_upper_{period}"] - df[f"bb_lower_{period}"] + 1e-10
        )

        self._feature_names.extend([
            f"bb_upper_{period}",
            f"bb_middle_{period}",
            f"bb_lower_{period}",
            f"bb_width_{period}",
            f"bb_pct_{period}",
        ])
        return df

    def keltner_channel(
        self, df: pd.DataFrame, period: int = 20, multiplier: float = 2
    ) -> pd.DataFrame:
        """Keltner Channel."""
        tp = (df["high"] + df["low"] + df["close"]) / 3
        ema = tp.ewm(span=period, adjust=False).mean()
        atr = self.true_range(df).ewm(span=period, adjust=False).mean()

        df[f"kc_upper_{period}"] = ema + multiplier * atr
        df[f"kc_middle_{period}"] = ema
        df[f"kc_lower_{period}"] = ema - multiplier * atr

        self._feature_names.extend([
            f"kc_upper_{period}",
            f"kc_middle_{period}",
            f"kc_lower_{period}",
        ])
        return df

    def donchian_channel(self, df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """Donchian Channel."""
        df[f"dc_upper_{period}"] = df["high"].rolling(window=period).max()
        df[f"dc_lower_{period}"] = df["low"].rolling(window=period).min()
        df[f"dc_middle_{period}"] = (
            df[f"dc_upper_{period}"] + df[f"dc_lower_{period}"]
        ) / 2

        self._feature_names.extend([
            f"dc_upper_{period}",
            f"dc_middle_{period}",
            f"dc_lower_{period}",
        ])
        return df

    def stddev(
        self, df: pd.DataFrame, periods: List[int], column: str = "close"
    ) -> pd.DataFrame:
        """Rolling Standard Deviation."""
        for period in periods:
            col_name = f"stddev_{period}"
            df[col_name] = df[column].rolling(window=period).std()
            self._feature_names.append(col_name)
        return df

    def historical_volatility(
        self, df: pd.DataFrame, periods: List[int], column: str = "close"
    ) -> pd.DataFrame:
        """Historical Volatility (annualized)."""
        log_returns = np.log(df[column] / df[column].shift(1))

        for period in periods:
            col_name = f"hvol_{period}"
            # Annualized volatility (assuming 252 trading days)
            df[col_name] = log_returns.rolling(window=period).std() * np.sqrt(252)
            self._feature_names.append(col_name)

        return df

    def atr_percent(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """ATR as percentage of price."""
        tr = self.true_range(df)
        atr = tr.ewm(span=period, adjust=False).mean()

        col_name = f"atr_pct_{period}"
        df[col_name] = 100 * atr / df["close"]
        self._feature_names.append(col_name)
        return df
