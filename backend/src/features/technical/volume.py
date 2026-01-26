"""Volume indicators for technical analysis."""

from typing import List
import numpy as np
import pandas as pd


class VolumeIndicators:
    """Calculate volume-based technical indicators."""

    def __init__(self):
        """Initialize volume indicators."""
        self._feature_names: List[str] = []

    def get_feature_names(self) -> List[str]:
        """Get list of generated feature names."""
        return self._feature_names.copy()

    def calculate_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all volume indicators."""
        df = df.copy()
        self._feature_names = []

        # Check if volume data exists and is meaningful
        if "volume" not in df.columns or df["volume"].sum() == 0:
            return df

        df = self.obv(df)
        df = self.ad_line(df)
        df = self.adosc(df)
        df = self.cmf(df, period=20)
        df = self.vpt(df)
        df = self.emv(df, period=14)
        df = self.force_index(df, period=13)
        df = self.volume_sma(df, periods=[10, 20])
        df = self.volume_ratio(df)

        return df

    def obv(self, df: pd.DataFrame) -> pd.DataFrame:
        """On-Balance Volume."""
        obv = pd.Series(index=df.index, dtype=float)
        obv.iloc[0] = df["volume"].iloc[0]

        for i in range(1, len(df)):
            if df["close"].iloc[i] > df["close"].iloc[i - 1]:
                obv.iloc[i] = obv.iloc[i - 1] + df["volume"].iloc[i]
            elif df["close"].iloc[i] < df["close"].iloc[i - 1]:
                obv.iloc[i] = obv.iloc[i - 1] - df["volume"].iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i - 1]

        df["obv"] = obv
        self._feature_names.append("obv")
        return df

    def ad_line(self, df: pd.DataFrame) -> pd.DataFrame:
        """Accumulation/Distribution Line."""
        high = df["high"]
        low = df["low"]
        close = df["close"]
        volume = df["volume"]

        # Money Flow Multiplier
        mfm = ((close - low) - (high - close)) / (high - low + 1e-10)

        # Money Flow Volume
        mfv = mfm * volume

        # A/D Line
        df["ad"] = mfv.cumsum()
        self._feature_names.append("ad")
        return df

    def adosc(self, df: pd.DataFrame, fast: int = 3, slow: int = 10) -> pd.DataFrame:
        """Accumulation/Distribution Oscillator (Chaikin Oscillator)."""
        if "ad" not in df.columns:
            df = self.ad_line(df)

        ad = df["ad"]
        df["adosc"] = ad.ewm(span=fast, adjust=False).mean() - ad.ewm(span=slow, adjust=False).mean()
        self._feature_names.append("adosc")
        return df

    def cmf(self, df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """Chaikin Money Flow."""
        high = df["high"]
        low = df["low"]
        close = df["close"]
        volume = df["volume"]

        mfm = ((close - low) - (high - close)) / (high - low + 1e-10)
        mfv = mfm * volume

        col_name = f"cmf_{period}"
        df[col_name] = mfv.rolling(window=period).sum() / (
            volume.rolling(window=period).sum() + 1e-10
        )
        self._feature_names.append(col_name)
        return df

    def vpt(self, df: pd.DataFrame) -> pd.DataFrame:
        """Volume Price Trend."""
        roc = (df["close"] - df["close"].shift(1)) / (df["close"].shift(1) + 1e-10)
        vpt = (roc * df["volume"]).cumsum()

        df["vpt"] = vpt
        self._feature_names.append("vpt")
        return df

    def emv(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Ease of Movement."""
        high = df["high"]
        low = df["low"]
        volume = df["volume"]

        distance = ((high + low) / 2) - ((high.shift(1) + low.shift(1)) / 2)
        box_ratio = (volume / 1e8) / (high - low + 1e-10)
        emv = distance / (box_ratio + 1e-10)

        col_name = f"emv_{period}"
        df[col_name] = emv.rolling(window=period).mean()
        self._feature_names.append(col_name)
        return df

    def force_index(self, df: pd.DataFrame, period: int = 13) -> pd.DataFrame:
        """Force Index."""
        fi = (df["close"] - df["close"].shift(1)) * df["volume"]

        col_name = f"fi_{period}"
        df[col_name] = fi.ewm(span=period, adjust=False).mean()
        self._feature_names.append(col_name)
        return df

    def volume_sma(self, df: pd.DataFrame, periods: List[int]) -> pd.DataFrame:
        """Volume Simple Moving Average."""
        for period in periods:
            col_name = f"vol_sma_{period}"
            df[col_name] = df["volume"].rolling(window=period).mean()
            self._feature_names.append(col_name)
        return df

    def volume_ratio(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Volume ratio compared to average."""
        avg_vol = df["volume"].rolling(window=period).mean()

        col_name = f"vol_ratio_{period}"
        df[col_name] = df["volume"] / (avg_vol + 1e-10)
        self._feature_names.append(col_name)
        return df

    def nvi(self, df: pd.DataFrame) -> pd.DataFrame:
        """Negative Volume Index."""
        nvi = pd.Series(index=df.index, dtype=float)
        nvi.iloc[0] = 1000

        for i in range(1, len(df)):
            if df["volume"].iloc[i] < df["volume"].iloc[i - 1]:
                pct_change = (
                    df["close"].iloc[i] - df["close"].iloc[i - 1]
                ) / df["close"].iloc[i - 1]
                nvi.iloc[i] = nvi.iloc[i - 1] * (1 + pct_change)
            else:
                nvi.iloc[i] = nvi.iloc[i - 1]

        df["nvi"] = nvi
        self._feature_names.append("nvi")
        return df

    def pvi(self, df: pd.DataFrame) -> pd.DataFrame:
        """Positive Volume Index."""
        pvi = pd.Series(index=df.index, dtype=float)
        pvi.iloc[0] = 1000

        for i in range(1, len(df)):
            if df["volume"].iloc[i] > df["volume"].iloc[i - 1]:
                pct_change = (
                    df["close"].iloc[i] - df["close"].iloc[i - 1]
                ) / df["close"].iloc[i - 1]
                pvi.iloc[i] = pvi.iloc[i - 1] * (1 + pct_change)
            else:
                pvi.iloc[i] = pvi.iloc[i - 1]

        df["pvi"] = pvi
        self._feature_names.append("pvi")
        return df
