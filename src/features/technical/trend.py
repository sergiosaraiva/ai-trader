"""Trend indicators for technical analysis."""

from typing import List
import numpy as np
import pandas as pd


class TrendIndicators:
    """Calculate trend-based technical indicators."""

    def __init__(self):
        """Initialize trend indicators."""
        self._feature_names: List[str] = []

    def get_feature_names(self) -> List[str]:
        """Get list of generated feature names."""
        return self._feature_names.copy()

    def calculate_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all trend indicators."""
        df = df.copy()
        self._feature_names = []

        # Moving averages
        df = self.sma(df, periods=[5, 10, 20, 50, 100, 200])
        df = self.ema(df, periods=[5, 10, 20, 50, 100, 200])
        df = self.wma(df, periods=[10, 20, 50])

        # Trend direction
        df = self.adx(df, period=14)
        df = self.aroon(df, period=25)

        # Price relative to MAs
        df = self.price_to_ma(df)

        # MA crossovers
        df = self.ma_crossovers(df)

        return df

    def sma(self, df: pd.DataFrame, periods: List[int], column: str = "close") -> pd.DataFrame:
        """Simple Moving Average."""
        for period in periods:
            col_name = f"sma_{period}"
            df[col_name] = df[column].rolling(window=period).mean()
            self._feature_names.append(col_name)
        return df

    def ema(self, df: pd.DataFrame, periods: List[int], column: str = "close") -> pd.DataFrame:
        """Exponential Moving Average."""
        for period in periods:
            col_name = f"ema_{period}"
            df[col_name] = df[column].ewm(span=period, adjust=False).mean()
            self._feature_names.append(col_name)
        return df

    def wma(self, df: pd.DataFrame, periods: List[int], column: str = "close") -> pd.DataFrame:
        """Weighted Moving Average."""
        for period in periods:
            col_name = f"wma_{period}"
            weights = np.arange(1, period + 1)
            df[col_name] = (
                df[column]
                .rolling(window=period)
                .apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)
            )
            self._feature_names.append(col_name)
        return df

    def dema(self, df: pd.DataFrame, periods: List[int], column: str = "close") -> pd.DataFrame:
        """Double Exponential Moving Average."""
        for period in periods:
            col_name = f"dema_{period}"
            ema1 = df[column].ewm(span=period, adjust=False).mean()
            ema2 = ema1.ewm(span=period, adjust=False).mean()
            df[col_name] = 2 * ema1 - ema2
            self._feature_names.append(col_name)
        return df

    def tema(self, df: pd.DataFrame, periods: List[int], column: str = "close") -> pd.DataFrame:
        """Triple Exponential Moving Average."""
        for period in periods:
            col_name = f"tema_{period}"
            ema1 = df[column].ewm(span=period, adjust=False).mean()
            ema2 = ema1.ewm(span=period, adjust=False).mean()
            ema3 = ema2.ewm(span=period, adjust=False).mean()
            df[col_name] = 3 * ema1 - 3 * ema2 + ema3
            self._feature_names.append(col_name)
        return df

    def adx(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Average Directional Index."""
        high = df["high"]
        low = df["low"]
        close = df["close"]

        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # Directional Movement
        up_move = high - high.shift(1)
        down_move = low.shift(1) - low

        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

        # Smoothed values
        atr = tr.ewm(span=period, adjust=False).mean()
        plus_di = 100 * pd.Series(plus_dm).ewm(span=period, adjust=False).mean() / atr
        minus_di = 100 * pd.Series(minus_dm).ewm(span=period, adjust=False).mean() / atr

        # ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx = dx.ewm(span=period, adjust=False).mean()

        df[f"adx_{period}"] = adx.values
        df[f"plus_di_{period}"] = plus_di.values
        df[f"minus_di_{period}"] = minus_di.values

        self._feature_names.extend([f"adx_{period}", f"plus_di_{period}", f"minus_di_{period}"])
        return df

    def aroon(self, df: pd.DataFrame, period: int = 25) -> pd.DataFrame:
        """Aroon Indicator."""
        aroon_up = (
            df["high"]
            .rolling(window=period + 1)
            .apply(lambda x: x.argmax(), raw=True)
            / period
            * 100
        )
        aroon_down = (
            df["low"]
            .rolling(window=period + 1)
            .apply(lambda x: x.argmin(), raw=True)
            / period
            * 100
        )

        df[f"aroon_up_{period}"] = aroon_up
        df[f"aroon_down_{period}"] = aroon_down
        df[f"aroon_osc_{period}"] = aroon_up - aroon_down

        self._feature_names.extend(
            [f"aroon_up_{period}", f"aroon_down_{period}", f"aroon_osc_{period}"]
        )
        return df

    def price_to_ma(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate price position relative to moving averages."""
        close = df["close"]

        for period in [20, 50, 200]:
            sma_col = f"sma_{period}"
            if sma_col in df.columns:
                col_name = f"price_to_sma_{period}"
                df[col_name] = (close - df[sma_col]) / df[sma_col]
                self._feature_names.append(col_name)

        return df

    def ma_crossovers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate moving average crossover signals."""
        crossovers = [
            ("sma_5", "sma_20"),
            ("sma_20", "sma_50"),
            ("sma_50", "sma_200"),
            ("ema_5", "ema_20"),
            ("ema_12", "ema_26"),
        ]

        for fast, slow in crossovers:
            if fast in df.columns and slow in df.columns:
                col_name = f"{fast}_{slow}_cross"
                df[col_name] = np.where(df[fast] > df[slow], 1, -1)
                self._feature_names.append(col_name)

        return df

    def supertrend(
        self, df: pd.DataFrame, period: int = 10, multiplier: float = 3.0
    ) -> pd.DataFrame:
        """Supertrend indicator."""
        high = df["high"]
        low = df["low"]
        close = df["close"]

        # ATR
        tr = pd.concat(
            [high - low, abs(high - close.shift(1)), abs(low - close.shift(1))], axis=1
        ).max(axis=1)
        atr = tr.ewm(span=period, adjust=False).mean()

        # Basic bands
        hl2 = (high + low) / 2
        upper_band = hl2 + multiplier * atr
        lower_band = hl2 - multiplier * atr

        # Supertrend calculation
        supertrend = pd.Series(index=df.index, dtype=float)
        direction = pd.Series(index=df.index, dtype=int)

        for i in range(period, len(df)):
            if close.iloc[i] > upper_band.iloc[i - 1]:
                direction.iloc[i] = 1
            elif close.iloc[i] < lower_band.iloc[i - 1]:
                direction.iloc[i] = -1
            else:
                direction.iloc[i] = direction.iloc[i - 1]

            if direction.iloc[i] == 1:
                supertrend.iloc[i] = lower_band.iloc[i]
            else:
                supertrend.iloc[i] = upper_band.iloc[i]

        df[f"supertrend_{period}"] = supertrend
        df[f"supertrend_dir_{period}"] = direction

        self._feature_names.extend([f"supertrend_{period}", f"supertrend_dir_{period}"])
        return df
