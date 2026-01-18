"""Derived sentiment feature calculations."""

from typing import Dict, List, Optional
import pandas as pd
import numpy as np


class SentimentFeatureCalculator:
    """
    Calculate derived sentiment features from raw sentiment data.

    Features include:
    - Moving averages (smoothed trends)
    - Momentum (sentiment changes)
    - Volatility (sentiment stability)
    - Z-scores (deviation from normal)
    - Regime classification (bullish/neutral/bearish)
    - Lag features (historical sentiment)
    - Sentiment differentials (cross-country)

    Note: Since sentiment data is daily but is aligned to intraday price data,
    the derived features will also have daily granularity (constant within a day).
    """

    DEFAULT_CONFIG = {
        'ma_periods': [3, 7, 14, 30],
        'std_periods': [7, 14, 30],
        'momentum_periods': [3, 7, 14],
        'lag_days': [1, 2, 3],
        'zscore_lookback': 30,
        'regime_thresholds': {'bearish': -0.05, 'bullish': 0.05},
    }

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize feature calculator.

        Args:
            config: Configuration overrides
        """
        self.config = {**self.DEFAULT_CONFIG, **(config or {})}
        self._feature_names: List[str] = []

    def calculate_all(
        self,
        df: pd.DataFrame,
        sentiment_col: str = 'sentiment_raw',
        prefix: str = 'sentiment',
    ) -> pd.DataFrame:
        """
        Calculate all derived sentiment features.

        Args:
            df: DataFrame with sentiment_raw column
            sentiment_col: Name of the raw sentiment column
            prefix: Prefix for feature names

        Returns:
            DataFrame with all sentiment features added
        """
        result = df.copy()
        self._feature_names = []

        if sentiment_col not in result.columns:
            raise ValueError(f"Column '{sentiment_col}' not found in DataFrame")

        sent = result[sentiment_col]

        # Track the raw column
        self._feature_names.append(sentiment_col)

        # Moving averages
        result = self._add_moving_averages(result, sent, prefix)

        # Standard deviations (volatility)
        result = self._add_std_features(result, sent, prefix)

        # Momentum features
        result = self._add_momentum_features(result, sent, prefix)

        # Rate of change
        result = self._add_roc_features(result, sent, prefix)

        # Z-score
        result = self._add_zscore(result, sent, prefix)

        # Regime classification
        result = self._add_regime(result, sent, prefix)

        # Lag features
        result = self._add_lag_features(result, sent, prefix)

        # Handle any remaining NaN values
        sentiment_cols = [c for c in result.columns if prefix in c.lower()]
        for col in sentiment_cols:
            result[col] = result[col].ffill().bfill()
            # Fill remaining NaN with 0 (for start of series)
            result[col] = result[col].fillna(0)

        return result

    def _add_moving_averages(
        self,
        df: pd.DataFrame,
        sent: pd.Series,
        prefix: str,
    ) -> pd.DataFrame:
        """Add moving average features."""
        for period in self.config['ma_periods']:
            col_name = f'{prefix}_ma_{period}'
            df[col_name] = sent.rolling(
                window=period,
                min_periods=1,
            ).mean()
            self._feature_names.append(col_name)
        return df

    def _add_std_features(
        self,
        df: pd.DataFrame,
        sent: pd.Series,
        prefix: str,
    ) -> pd.DataFrame:
        """Add standard deviation (volatility) features."""
        for period in self.config['std_periods']:
            col_name = f'{prefix}_std_{period}'
            df[col_name] = sent.rolling(
                window=period,
                min_periods=2,
            ).std().fillna(0)
            self._feature_names.append(col_name)
        return df

    def _add_momentum_features(
        self,
        df: pd.DataFrame,
        sent: pd.Series,
        prefix: str,
    ) -> pd.DataFrame:
        """Add momentum features (deviation from moving average)."""
        for period in self.config['momentum_periods']:
            ma_col = f'{prefix}_ma_{period}'
            if ma_col in df.columns:
                col_name = f'{prefix}_momentum_{period}'
                df[col_name] = sent - df[ma_col]
                self._feature_names.append(col_name)
        return df

    def _add_roc_features(
        self,
        df: pd.DataFrame,
        sent: pd.Series,
        prefix: str,
    ) -> pd.DataFrame:
        """Add rate of change features."""
        for period in [7, 14]:  # 1-week and 2-week ROC
            col_name = f'{prefix}_roc_{period}'
            # Use diff instead of pct_change to avoid inf values
            df[col_name] = sent.diff(period).fillna(0)
            self._feature_names.append(col_name)
        return df

    def _add_zscore(
        self,
        df: pd.DataFrame,
        sent: pd.Series,
        prefix: str,
    ) -> pd.DataFrame:
        """Add z-score feature (deviation from rolling mean in std units)."""
        lookback = self.config['zscore_lookback']

        rolling_mean = sent.rolling(window=lookback, min_periods=1).mean()
        rolling_std = sent.rolling(window=lookback, min_periods=2).std().fillna(1e-6)

        # Avoid division by zero
        rolling_std = rolling_std.replace(0, 1e-6)

        col_name = f'{prefix}_zscore'
        df[col_name] = ((sent - rolling_mean) / rolling_std).clip(-3, 3)
        self._feature_names.append(col_name)

        return df

    def _add_regime(
        self,
        df: pd.DataFrame,
        sent: pd.Series,
        prefix: str,
    ) -> pd.DataFrame:
        """Add regime classification feature."""
        thresholds = self.config['regime_thresholds']

        col_name = f'{prefix}_regime'

        # Classify into regimes: -1 (bearish), 0 (neutral), 1 (bullish)
        conditions = [
            sent < thresholds['bearish'],
            sent > thresholds['bullish'],
        ]
        choices = [-1, 1]
        df[col_name] = np.select(conditions, choices, default=0).astype(float)
        self._feature_names.append(col_name)

        return df

    def _add_lag_features(
        self,
        df: pd.DataFrame,
        sent: pd.Series,
        prefix: str,
    ) -> pd.DataFrame:
        """Add lag features (historical sentiment values)."""
        for lag in self.config['lag_days']:
            col_name = f'{prefix}_lag_{lag}'
            df[col_name] = sent.shift(lag)
            self._feature_names.append(col_name)
        return df

    def calculate_differential(
        self,
        df: pd.DataFrame,
        base_col: str,
        quote_col: str,
        output_col: str = 'sentiment_differential',
    ) -> pd.DataFrame:
        """
        Calculate sentiment differential between two countries/regions.

        Useful for forex pairs to capture relative sentiment.

        Args:
            df: DataFrame with country sentiment columns
            base_col: Base currency country sentiment column
            quote_col: Quote currency country sentiment column
            output_col: Name for output column

        Returns:
            DataFrame with differential column added
        """
        result = df.copy()

        if base_col in result.columns and quote_col in result.columns:
            result[output_col] = result[base_col] - result[quote_col]
            self._feature_names.append(output_col)

        return result

    def calculate_cross_sentiment_features(
        self,
        df: pd.DataFrame,
        sentiment_cols: List[str],
        prefix: str = 'cross_sent',
    ) -> pd.DataFrame:
        """
        Calculate cross-sentiment features from multiple country sentiments.

        Features include:
        - Mean sentiment across countries
        - Std of sentiments (disagreement)
        - Max and min sentiment

        Args:
            df: DataFrame with country sentiment columns
            sentiment_cols: List of sentiment column names
            prefix: Prefix for output columns

        Returns:
            DataFrame with cross-sentiment features added
        """
        result = df.copy()

        # Filter to existing columns
        existing_cols = [c for c in sentiment_cols if c in result.columns]

        if len(existing_cols) >= 2:
            sent_matrix = result[existing_cols]

            # Mean across countries
            col_mean = f'{prefix}_mean'
            result[col_mean] = sent_matrix.mean(axis=1)
            self._feature_names.append(col_mean)

            # Std (disagreement)
            col_std = f'{prefix}_std'
            result[col_std] = sent_matrix.std(axis=1).fillna(0)
            self._feature_names.append(col_std)

            # Range (max - min)
            col_range = f'{prefix}_range'
            result[col_range] = sent_matrix.max(axis=1) - sent_matrix.min(axis=1)
            self._feature_names.append(col_range)

        return result

    def get_feature_names(self) -> List[str]:
        """
        Get list of all generated feature names.

        Returns:
            List of feature column names
        """
        return self._feature_names.copy()

    @staticmethod
    def get_default_feature_list() -> List[str]:
        """
        Get list of expected default feature names.

        Returns:
            List of default feature names (without calculating)
        """
        features = ['sentiment_raw']

        # MA features
        for period in [3, 7, 14, 30]:
            features.append(f'sentiment_ma_{period}')

        # Std features
        for period in [7, 14, 30]:
            features.append(f'sentiment_std_{period}')

        # Momentum features
        for period in [3, 7, 14]:
            features.append(f'sentiment_momentum_{period}')

        # ROC features
        for period in [7, 14]:
            features.append(f'sentiment_roc_{period}')

        # Other features
        features.extend([
            'sentiment_zscore',
            'sentiment_regime',
            'sentiment_lag_1',
            'sentiment_lag_2',
            'sentiment_lag_3',
        ])

        return features


class SentimentFeatureConfig:
    """Configuration class for sentiment features."""

    def __init__(
        self,
        enabled: bool = True,
        include_raw: bool = True,
        include_ma: bool = True,
        include_std: bool = True,
        include_momentum: bool = True,
        include_roc: bool = True,
        include_zscore: bool = True,
        include_regime: bool = True,
        include_lags: bool = True,
        include_country: bool = True,
        include_epu: bool = False,
        include_differential: bool = True,
        include_cross_features: bool = True,
        ma_periods: Optional[List[int]] = None,
        std_periods: Optional[List[int]] = None,
        momentum_periods: Optional[List[int]] = None,
        lag_days: Optional[List[int]] = None,
    ):
        """
        Initialize feature configuration.

        Args:
            enabled: Whether sentiment features are enabled
            include_raw: Include raw sentiment score
            include_ma: Include moving average features
            include_std: Include standard deviation features
            include_momentum: Include momentum features
            include_roc: Include rate of change features
            include_zscore: Include z-score feature
            include_regime: Include regime classification
            include_lags: Include lag features
            include_country: Include country-level sentiments
            include_epu: Include raw EPU values
            include_differential: Include sentiment differential
            include_cross_features: Include cross-country features
            ma_periods: Custom MA periods
            std_periods: Custom STD periods
            momentum_periods: Custom momentum periods
            lag_days: Custom lag days
        """
        self.enabled = enabled
        self.include_raw = include_raw
        self.include_ma = include_ma
        self.include_std = include_std
        self.include_momentum = include_momentum
        self.include_roc = include_roc
        self.include_zscore = include_zscore
        self.include_regime = include_regime
        self.include_lags = include_lags
        self.include_country = include_country
        self.include_epu = include_epu
        self.include_differential = include_differential
        self.include_cross_features = include_cross_features
        self.ma_periods = ma_periods or [3, 7, 14, 30]
        self.std_periods = std_periods or [7, 14, 30]
        self.momentum_periods = momentum_periods or [3, 7, 14]
        self.lag_days = lag_days or [1, 2, 3]

    def to_dict(self) -> Dict:
        """Convert config to dictionary."""
        return {
            'enabled': self.enabled,
            'include_raw': self.include_raw,
            'include_ma': self.include_ma,
            'include_std': self.include_std,
            'include_momentum': self.include_momentum,
            'include_roc': self.include_roc,
            'include_zscore': self.include_zscore,
            'include_regime': self.include_regime,
            'include_lags': self.include_lags,
            'include_country': self.include_country,
            'include_epu': self.include_epu,
            'include_differential': self.include_differential,
            'include_cross_features': self.include_cross_features,
            'ma_periods': self.ma_periods,
            'std_periods': self.std_periods,
            'momentum_periods': self.momentum_periods,
            'lag_days': self.lag_days,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'SentimentFeatureConfig':
        """Create config from dictionary."""
        return cls(**data)

    def get_feature_filter(self) -> List[str]:
        """
        Get list of feature patterns to include based on config.

        Returns:
            List of feature name patterns
        """
        patterns = []

        if self.include_raw:
            patterns.append('sentiment_raw')

        if self.include_ma:
            patterns.append('sentiment_ma_')

        if self.include_std:
            patterns.append('sentiment_std_')

        if self.include_momentum:
            patterns.append('sentiment_momentum_')

        if self.include_roc:
            patterns.append('sentiment_roc_')

        if self.include_zscore:
            patterns.append('sentiment_zscore')

        if self.include_regime:
            patterns.append('sentiment_regime')

        if self.include_lags:
            patterns.append('sentiment_lag_')

        if self.include_country:
            patterns.append('sent_country_')

        if self.include_epu:
            patterns.append('epu_')

        if self.include_differential:
            patterns.append('sentiment_differential')

        if self.include_cross_features:
            patterns.append('cross_sent_')

        return patterns
