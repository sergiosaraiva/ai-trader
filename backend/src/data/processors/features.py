"""Feature processor for creating model inputs."""

from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union
import numpy as np
import pandas as pd


class FeatureProcessor:
    """Process and prepare features for model training."""

    def __init__(
        self,
        include_sentiment: bool = False,
        sentiment_path: Optional[Union[str, Path]] = None,
    ):
        """
        Initialize feature processor.

        Args:
            include_sentiment: Whether to include sentiment features
            sentiment_path: Path to sentiment data file
        """
        self.feature_names: List[str] = []
        self.scalers: Dict = {}
        self.include_sentiment = include_sentiment
        self.sentiment_path = sentiment_path
        self._sentiment_features = None

        if include_sentiment:
            self._init_sentiment()

    def _init_sentiment(self) -> None:
        """Initialize sentiment features module."""
        try:
            from src.features.sentiment import SentimentFeatures
            default_path = 'data/sentiment/sentiment_epu_20200101_20251231_daily.csv'
            self._sentiment_features = SentimentFeatures(
                sentiment_path=self.sentiment_path or default_path
            )
        except ImportError:
            print("Warning: Sentiment features module not available")
            self.include_sentiment = False

    def add_sentiment_features(
        self,
        df: pd.DataFrame,
        pair: str,
        shift_days: int = 1,
    ) -> pd.DataFrame:
        """
        Add sentiment features to price data.

        CRITICAL: Uses shift_days to avoid look-ahead bias. Default is 1 day,
        meaning we use yesterday's sentiment for today's prediction.

        Args:
            df: Price DataFrame with DatetimeIndex
            pair: Trading pair (e.g., 'EURUSD', 'BTCUSDT')
            shift_days: Days to shift sentiment (default 1 for no look-ahead)

        Returns:
            DataFrame with sentiment features added
        """
        if not self.include_sentiment or self._sentiment_features is None:
            return df

        return self._sentiment_features.calculate_all(
            df=df,
            pair=pair,
            shift_days=shift_days,
        )

    def add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add cyclical temporal features.

        Args:
            df: DataFrame with DatetimeIndex

        Returns:
            DataFrame with temporal features
        """
        df = df.copy()
        idx = df.index

        # Hour of day (cyclical)
        if hasattr(idx, "hour"):
            df["hour_sin"] = np.sin(2 * np.pi * idx.hour / 24)
            df["hour_cos"] = np.cos(2 * np.pi * idx.hour / 24)

        # Day of week (cyclical)
        df["dow_sin"] = np.sin(2 * np.pi * idx.dayofweek / 7)
        df["dow_cos"] = np.cos(2 * np.pi * idx.dayofweek / 7)

        # Day of month (cyclical)
        df["dom_sin"] = np.sin(2 * np.pi * idx.day / 31)
        df["dom_cos"] = np.cos(2 * np.pi * idx.day / 31)

        # Month (cyclical)
        df["month_sin"] = np.sin(2 * np.pi * idx.month / 12)
        df["month_cos"] = np.cos(2 * np.pi * idx.month / 12)

        # Week of year (cyclical)
        week = idx.isocalendar().week
        df["week_sin"] = np.sin(2 * np.pi * week / 52)
        df["week_cos"] = np.cos(2 * np.pi * week / 52)

        return df

    def add_trading_session_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add forex trading session indicators.

        Args:
            df: DataFrame with DatetimeIndex (assumed UTC)

        Returns:
            DataFrame with session features
        """
        df = df.copy()

        if not hasattr(df.index, "hour"):
            return df

        hour = df.index.hour

        # Tokyo session: 00:00 - 09:00 UTC
        df["is_tokyo"] = ((hour >= 0) & (hour < 9)).astype(int)

        # London session: 08:00 - 17:00 UTC
        df["is_london"] = ((hour >= 8) & (hour < 17)).astype(int)

        # New York session: 13:00 - 22:00 UTC
        df["is_newyork"] = ((hour >= 13) & (hour < 22)).astype(int)

        # Session overlaps (high volatility periods)
        df["is_london_tokyo_overlap"] = ((hour >= 8) & (hour < 9)).astype(int)
        df["is_london_newyork_overlap"] = ((hour >= 13) & (hour < 17)).astype(int)

        # Any overlap
        df["is_overlap"] = (
            df["is_london_tokyo_overlap"] | df["is_london_newyork_overlap"]
        ).astype(int)

        return df

    def select_features(
        self,
        df: pd.DataFrame,
        feature_groups: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Select feature groups to include.

        Args:
            df: DataFrame with all features
            feature_groups: List of groups ('price', 'returns', 'temporal', 'session', 'trend', 'momentum', 'volatility', 'volume')

        Returns:
            DataFrame with selected features
        """
        if feature_groups is None:
            return df

        selected_cols = []

        feature_patterns = {
            "price": ["open", "high", "low", "close", "volume"],
            "returns": ["returns", "log_returns"],
            "derived": ["range", "body", "shadow", "gap", "position"],
            "temporal": ["hour_", "dow_", "dom_", "month_", "week_"],
            "session": ["is_tokyo", "is_london", "is_newyork", "is_overlap"],
            "trend": ["sma", "ema", "wma", "dema", "tema", "adx", "aroon", "psar", "supertrend"],
            "momentum": ["rsi", "stoch", "macd", "cci", "mom", "roc", "willr", "mfi", "tsi", "uo"],
            "volatility": ["atr", "natr", "trange", "bb_", "kc_", "dc_", "stddev"],
            "volume": ["obv", "ad", "adosc", "cmf", "vwap", "vpt", "emv", "fi", "nvi", "pvi"],
            "sentiment": ["sentiment_", "sent_country_", "cross_sent_", "epu_"],
        }

        for group in feature_groups:
            patterns = feature_patterns.get(group.lower(), [])
            for col in df.columns:
                col_lower = col.lower()
                if any(pattern in col_lower for pattern in patterns):
                    if col not in selected_cols:
                        selected_cols.append(col)

        return df[selected_cols] if selected_cols else df

    def handle_missing_values(
        self,
        df: pd.DataFrame,
        method: str = "ffill",
        fill_value: Optional[float] = None,
    ) -> pd.DataFrame:
        """
        Handle missing values in features.

        Args:
            df: DataFrame with potential missing values
            method: Method to handle missing ('ffill', 'bfill', 'interpolate', 'fill', 'drop')
            fill_value: Value to fill if method is 'fill'

        Returns:
            DataFrame without missing values
        """
        df = df.copy()

        if method == "ffill":
            df = df.ffill().bfill()
        elif method == "bfill":
            df = df.bfill().ffill()
        elif method == "interpolate":
            df = df.interpolate(method="linear").ffill().bfill()
        elif method == "fill":
            df = df.fillna(fill_value if fill_value is not None else 0)
        elif method == "drop":
            df = df.dropna()

        return df

    def remove_infinite_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Replace infinite values with NaN then forward fill."""
        df = df.copy()
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.ffill().bfill()
        return df

    def prepare_for_training(
        self,
        df: pd.DataFrame,
        target_column: str = "close",
        sequence_length: int = 100,
        prediction_horizon: int = 1,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
    ) -> Dict:
        """
        Prepare data for model training.

        Args:
            df: Feature dataframe
            target_column: Column to predict
            sequence_length: Input sequence length
            prediction_horizon: Steps ahead to predict
            train_ratio: Training data ratio
            val_ratio: Validation data ratio

        Returns:
            Dictionary with train/val/test splits and metadata
        """
        # Handle missing and infinite values
        df = self.handle_missing_values(df)
        df = self.remove_infinite_values(df)

        # Store feature names
        self.feature_names = df.columns.tolist()

        # Create sequences
        data = df.values
        target_idx = df.columns.tolist().index(target_column)

        X, y = [], []
        for i in range(len(data) - sequence_length - prediction_horizon + 1):
            X.append(data[i : i + sequence_length])
            y.append(data[i + sequence_length + prediction_horizon - 1, target_idx])

        X = np.array(X)
        y = np.array(y)

        # Time series split
        n = len(X)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        return {
            "X_train": X[:train_end],
            "y_train": y[:train_end],
            "X_val": X[train_end:val_end],
            "y_val": y[train_end:val_end],
            "X_test": X[val_end:],
            "y_test": y[val_end:],
            "feature_names": self.feature_names,
            "target_column": target_column,
            "sequence_length": sequence_length,
            "prediction_horizon": prediction_horizon,
        }

    def prepare_all_features(
        self,
        price_df: pd.DataFrame,
        pair: str,
        include_technical: bool = True,
        include_temporal: bool = True,
        include_session: bool = True,
        feature_groups: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Prepare all features for model training (technical, temporal, session, sentiment).

        This method combines all feature engineering steps into a single call.

        Args:
            price_df: OHLCV DataFrame with DatetimeIndex
            pair: Trading pair (e.g., 'EURUSD', 'BTCUSDT')
            include_technical: Include technical indicators
            include_temporal: Include cyclical temporal features
            include_session: Include trading session features
            feature_groups: Feature groups to select (None = all)

        Returns:
            DataFrame with all requested features
        """
        result = price_df.copy()

        # Step 1: Technical indicators
        if include_technical:
            try:
                from src.features.technical import TechnicalIndicators
                tech = TechnicalIndicators()
                result = tech.calculate_all(result)
            except ImportError:
                print("Warning: Technical indicators module not available")

        # Step 2: Temporal features
        if include_temporal:
            result = self.add_temporal_features(result)

        # Step 3: Trading session features
        if include_session:
            result = self.add_trading_session_features(result)

        # Step 4: Sentiment features (if enabled)
        if self.include_sentiment:
            result = self.add_sentiment_features(result, pair)

        # Step 5: Select feature groups
        if feature_groups:
            result = self.select_features(result, feature_groups)

        # Step 6: Handle missing and infinite values
        result = self.handle_missing_values(result)
        result = self.remove_infinite_values(result)

        # Store feature names
        self.feature_names = result.columns.tolist()

        return result

    def get_sentiment_feature_names(self) -> List[str]:
        """
        Get list of sentiment feature names.

        Returns:
            List of sentiment feature column names
        """
        if self._sentiment_features is not None:
            return self._sentiment_features.get_feature_names()
        return []

    def validate_sentiment_coverage(self, df: pd.DataFrame) -> Dict:
        """
        Validate sentiment data coverage for price data.

        Args:
            df: Price DataFrame

        Returns:
            Dictionary with validation results
        """
        if self._sentiment_features is not None:
            return self._sentiment_features.validate_data(df)
        return {"error": "Sentiment features not initialized"}
