"""OHLCV data processor."""

from typing import Optional, Tuple, List
import numpy as np
import pandas as pd


class OHLCVProcessor:
    """Process and transform OHLCV data."""

    def __init__(self):
        """Initialize OHLCV processor."""
        self.scalers = {}

    def validate(self, df: pd.DataFrame) -> bool:
        """
        Validate OHLCV dataframe.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            True if valid, raises ValueError otherwise
        """
        required_columns = ["open", "high", "low", "close", "volume"]
        missing = [col for col in required_columns if col not in df.columns]

        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("Index must be DatetimeIndex")

        if df.isnull().any().any():
            null_counts = df.isnull().sum()
            raise ValueError(f"DataFrame contains null values: {null_counts}")

        # Validate OHLC relationships
        invalid_high = df["high"] < df[["open", "close"]].max(axis=1)
        invalid_low = df["low"] > df[["open", "close"]].min(axis=1)

        if invalid_high.any():
            raise ValueError(f"Invalid high values at: {df.index[invalid_high].tolist()[:5]}")
        if invalid_low.any():
            raise ValueError(f"Invalid low values at: {df.index[invalid_low].tolist()[:5]}")

        return True

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean OHLCV data.

        Args:
            df: Raw OHLCV dataframe

        Returns:
            Cleaned dataframe
        """
        df = df.copy()

        # Sort by index
        df = df.sort_index()

        # Remove duplicates
        df = df[~df.index.duplicated(keep="last")]

        # Forward fill missing values (common in forex)
        df = df.ffill()

        # Remove rows with zero volume (market closed)
        if "volume" in df.columns:
            df = df[df["volume"] > 0]

        return df

    def add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add derived price features.

        Args:
            df: OHLCV dataframe

        Returns:
            DataFrame with additional features
        """
        df = df.copy()

        # Returns
        df["returns"] = df["close"].pct_change()
        df["log_returns"] = np.log(df["close"] / df["close"].shift(1))

        # Price range features
        df["range"] = df["high"] - df["low"]
        df["body"] = df["close"] - df["open"]
        df["body_pct"] = df["body"] / df["open"]

        # Shadows
        df["upper_shadow"] = df["high"] - df[["open", "close"]].max(axis=1)
        df["lower_shadow"] = df[["open", "close"]].min(axis=1) - df["low"]

        # Relative position
        df["close_position"] = (df["close"] - df["low"]) / (df["high"] - df["low"])

        # Gap
        df["gap"] = df["open"] - df["close"].shift(1)
        df["gap_pct"] = df["gap"] / df["close"].shift(1)

        return df

    def resample(self, df: pd.DataFrame, target_timeframe: str) -> pd.DataFrame:
        """
        Resample OHLCV data to different timeframe.

        Args:
            df: OHLCV dataframe
            target_timeframe: Target timeframe (e.g., '4H', '1D')

        Returns:
            Resampled dataframe
        """
        # Convert timeframe string to pandas offset
        tf_map = {
            "1M": "1min",
            "5M": "5min",
            "15M": "15min",
            "30M": "30min",
            "1H": "1h",
            "4H": "4h",
            "1D": "1D",
            "1W": "1W",
        }

        offset = tf_map.get(target_timeframe.upper())
        if offset is None:
            raise ValueError(f"Invalid timeframe: {target_timeframe}")

        resampled = df.resample(offset).agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        })

        return resampled.dropna()

    def normalize(
        self,
        df: pd.DataFrame,
        method: str = "zscore",
        columns: Optional[List[str]] = None,
    ) -> Tuple[pd.DataFrame, dict]:
        """
        Normalize dataframe columns.

        Args:
            df: Input dataframe
            method: Normalization method ('zscore', 'minmax', 'robust')
            columns: Columns to normalize (default: all numeric)

        Returns:
            Tuple of (normalized dataframe, scaler parameters)
        """
        df = df.copy()
        columns = columns or df.select_dtypes(include=[np.number]).columns.tolist()
        scalers = {}

        for col in columns:
            if col not in df.columns:
                continue

            if method == "zscore":
                mean = df[col].mean()
                std = df[col].std()
                df[col] = (df[col] - mean) / (std + 1e-8)
                scalers[col] = {"method": "zscore", "mean": mean, "std": std}

            elif method == "minmax":
                min_val = df[col].min()
                max_val = df[col].max()
                df[col] = (df[col] - min_val) / (max_val - min_val + 1e-8)
                scalers[col] = {"method": "minmax", "min": min_val, "max": max_val}

            elif method == "robust":
                median = df[col].median()
                q75, q25 = df[col].quantile([0.75, 0.25])
                iqr = q75 - q25
                df[col] = (df[col] - median) / (iqr + 1e-8)
                scalers[col] = {"method": "robust", "median": median, "iqr": iqr}

        self.scalers = scalers
        return df, scalers

    def denormalize(
        self,
        df: pd.DataFrame,
        scalers: dict,
        columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Reverse normalization.

        Args:
            df: Normalized dataframe
            scalers: Scaler parameters from normalize()
            columns: Columns to denormalize

        Returns:
            Denormalized dataframe
        """
        df = df.copy()
        columns = columns or list(scalers.keys())

        for col in columns:
            if col not in df.columns or col not in scalers:
                continue

            params = scalers[col]
            method = params["method"]

            if method == "zscore":
                df[col] = df[col] * params["std"] + params["mean"]
            elif method == "minmax":
                df[col] = df[col] * (params["max"] - params["min"]) + params["min"]
            elif method == "robust":
                df[col] = df[col] * params["iqr"] + params["median"]

        return df

    def create_sequences(
        self,
        df: pd.DataFrame,
        sequence_length: int,
        target_column: str = "close",
        prediction_horizon: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for time series modeling.

        Args:
            df: Input dataframe
            sequence_length: Length of input sequences
            target_column: Column to predict
            prediction_horizon: Steps ahead to predict

        Returns:
            Tuple of (X sequences, y targets)
        """
        data = df.values
        target_idx = df.columns.tolist().index(target_column)

        X, y = [], []
        for i in range(len(data) - sequence_length - prediction_horizon + 1):
            X.append(data[i : i + sequence_length])
            y.append(data[i + sequence_length + prediction_horizon - 1, target_idx])

        return np.array(X), np.array(y)
