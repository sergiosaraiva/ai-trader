"""Momentum indicators for technical analysis."""

from typing import List, Optional
import numpy as np
import pandas as pd

from src.config.trading_config import TradingConfig
from src.config.indicator_config import MomentumIndicators as MomentumConfig


class MomentumIndicators:
    """Calculate momentum-based technical indicators."""

    def __init__(self):
        """Initialize momentum indicators."""
        self._feature_names: List[str] = []

    def get_feature_names(self) -> List[str]:
        """Get list of generated feature names."""
        return self._feature_names.copy()

    def calculate_all(self, df: pd.DataFrame, config: Optional[TradingConfig] = None) -> pd.DataFrame:
        """Calculate all momentum indicators.

        Args:
            df: DataFrame with OHLCV data
            config: Optional TradingConfig instance. If None, uses fresh defaults.

        Returns:
            DataFrame with momentum indicators added
        """
        # Use fresh default config if none provided (avoid singleton)
        if config is None:
            momentum_config = MomentumConfig()
        else:
            momentum_config = config.indicators.momentum

        df = df.copy()
        self._feature_names = []

        df = self.rsi(df, periods=momentum_config.rsi_periods)
        df = self.stochastic(
            df,
            k_period=momentum_config.stochastic_k_period,
            d_period=momentum_config.stochastic_d_period,
        )
        df = self.macd(
            df,
            fast=momentum_config.macd_fast,
            slow=momentum_config.macd_slow,
            signal=momentum_config.macd_signal,
        )
        df = self.cci(
            df,
            periods=momentum_config.cci_periods,
            constant=momentum_config.cci_constant,
        )
        df = self.momentum(df, periods=momentum_config.momentum_periods)
        df = self.roc(df, periods=momentum_config.roc_periods)
        df = self.williams_r(df, period=momentum_config.williams_period)
        df = self.mfi(df, period=momentum_config.mfi_period)
        df = self.tsi(
            df,
            long=momentum_config.tsi_long,
            short=momentum_config.tsi_short,
        )
        df = self.ultimate_oscillator(df)

        return df

    def rsi(self, df: pd.DataFrame, periods: List[int], column: str = "close") -> pd.DataFrame:
        """Relative Strength Index."""
        for period in periods:
            delta = df[column].diff()
            gain = delta.where(delta > 0, 0)
            loss = (-delta).where(delta < 0, 0)

            avg_gain = gain.ewm(span=period, adjust=False).mean()
            avg_loss = loss.ewm(span=period, adjust=False).mean()

            rs = avg_gain / (avg_loss + 1e-10)
            rsi = 100 - (100 / (1 + rs))

            col_name = f"rsi_{period}"
            df[col_name] = rsi
            self._feature_names.append(col_name)

        return df

    def stochastic(
        self, df: pd.DataFrame, k_period: int = 14, d_period: int = 3
    ) -> pd.DataFrame:
        """Stochastic Oscillator."""
        low_min = df["low"].rolling(window=k_period).min()
        high_max = df["high"].rolling(window=k_period).max()

        stoch_k = 100 * (df["close"] - low_min) / (high_max - low_min + 1e-10)
        stoch_d = stoch_k.rolling(window=d_period).mean()

        df[f"stoch_k_{k_period}"] = stoch_k
        df[f"stoch_d_{k_period}"] = stoch_d

        self._feature_names.extend([f"stoch_k_{k_period}", f"stoch_d_{k_period}"])
        return df

    def macd(
        self,
        df: pd.DataFrame,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
        column: str = "close",
    ) -> pd.DataFrame:
        """Moving Average Convergence Divergence."""
        ema_fast = df[column].ewm(span=fast, adjust=False).mean()
        ema_slow = df[column].ewm(span=slow, adjust=False).mean()

        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line

        df["macd"] = macd_line
        df["macd_signal"] = signal_line
        df["macd_hist"] = histogram

        self._feature_names.extend(["macd", "macd_signal", "macd_hist"])
        return df

    def cci(self, df: pd.DataFrame, periods: List[int], constant: float = 0.015) -> pd.DataFrame:
        """Commodity Channel Index.

        Args:
            df: DataFrame with OHLC data
            periods: List of periods to calculate
            constant: CCI constant (default 0.015)

        Returns:
            DataFrame with CCI columns added
        """
        tp = (df["high"] + df["low"] + df["close"]) / 3

        for period in periods:
            sma = tp.rolling(window=period).mean()
            mad = tp.rolling(window=period).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
            cci = (tp - sma) / (constant * mad + 1e-10)

            col_name = f"cci_{period}"
            df[col_name] = cci
            self._feature_names.append(col_name)

        return df

    def momentum(
        self, df: pd.DataFrame, periods: List[int], column: str = "close"
    ) -> pd.DataFrame:
        """Momentum indicator."""
        for period in periods:
            col_name = f"mom_{period}"
            df[col_name] = df[column] - df[column].shift(period)
            self._feature_names.append(col_name)
        return df

    def roc(self, df: pd.DataFrame, periods: List[int], column: str = "close") -> pd.DataFrame:
        """Rate of Change."""
        for period in periods:
            col_name = f"roc_{period}"
            df[col_name] = (df[column] - df[column].shift(period)) / (
                df[column].shift(period) + 1e-10
            ) * 100
            self._feature_names.append(col_name)
        return df

    def williams_r(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Williams %R."""
        high_max = df["high"].rolling(window=period).max()
        low_min = df["low"].rolling(window=period).min()

        willr = -100 * (high_max - df["close"]) / (high_max - low_min + 1e-10)

        col_name = f"willr_{period}"
        df[col_name] = willr
        self._feature_names.append(col_name)
        return df

    def mfi(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Money Flow Index."""
        tp = (df["high"] + df["low"] + df["close"]) / 3
        mf = tp * df["volume"]

        pos_mf = mf.where(tp > tp.shift(1), 0)
        neg_mf = mf.where(tp < tp.shift(1), 0)

        pos_mf_sum = pos_mf.rolling(window=period).sum()
        neg_mf_sum = neg_mf.rolling(window=period).sum()

        mfi = 100 - (100 / (1 + pos_mf_sum / (neg_mf_sum + 1e-10)))

        col_name = f"mfi_{period}"
        df[col_name] = mfi
        self._feature_names.append(col_name)
        return df

    def tsi(
        self, df: pd.DataFrame, long: int = 25, short: int = 13, column: str = "close"
    ) -> pd.DataFrame:
        """True Strength Index."""
        diff = df[column].diff()

        double_smoothed = diff.ewm(span=long, adjust=False).mean().ewm(span=short, adjust=False).mean()
        double_smoothed_abs = (
            diff.abs().ewm(span=long, adjust=False).mean().ewm(span=short, adjust=False).mean()
        )

        tsi = 100 * double_smoothed / (double_smoothed_abs + 1e-10)

        df["tsi"] = tsi
        self._feature_names.append("tsi")
        return df

    def ultimate_oscillator(
        self, df: pd.DataFrame, period1: int = 7, period2: int = 14, period3: int = 28
    ) -> pd.DataFrame:
        """Ultimate Oscillator."""
        high = df["high"]
        low = df["low"]
        close = df["close"]
        prev_close = close.shift(1)

        bp = close - pd.concat([low, prev_close], axis=1).min(axis=1)
        tr = pd.concat([high, prev_close], axis=1).max(axis=1) - pd.concat(
            [low, prev_close], axis=1
        ).min(axis=1)

        avg1 = bp.rolling(period1).sum() / (tr.rolling(period1).sum() + 1e-10)
        avg2 = bp.rolling(period2).sum() / (tr.rolling(period2).sum() + 1e-10)
        avg3 = bp.rolling(period3).sum() / (tr.rolling(period3).sum() + 1e-10)

        uo = 100 * (4 * avg1 + 2 * avg2 + avg3) / 7

        df["uo"] = uo
        self._feature_names.append("uo")
        return df
