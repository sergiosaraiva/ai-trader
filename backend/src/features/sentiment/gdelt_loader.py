"""GDELT hourly sentiment data loader with timeframe-aware aggregation.

GDELT provides hourly news sentiment that can be properly aggregated for any timeframe:
- 1H model: Use raw hourly values (1:1 match)
- 4H model: Aggregate 4 hours → avg, std, trend
- Daily model: Aggregate 24 hours → avg, std, trend

This solves the resolution mismatch problem that VIX/EPU (daily) had with intraday models.
"""

import logging
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Default GDELT data path
DEFAULT_GDELT_PATH = Path(__file__).parent.parent.parent.parent / 'data' / 'sentiment' / 'gdelt_sentiment_20200101_20251231_hourly.csv'


class GDELTSentimentLoader:
    """
    Load and aggregate GDELT hourly sentiment data for trading models.

    Supports proper aggregation for any timeframe with multiple features:
    - avg: Mean sentiment over the period
    - std: Sentiment volatility/uncertainty
    - trend: Direction of sentiment change (last - first)
    """

    # Available sentiment columns
    SENTIMENT_COLUMNS = [
        'sentiment_us',
        'sentiment_europe',
        'sentiment_global',
        'sentiment_eurusd',
    ]

    # Article count columns (for weighting if needed)
    COUNT_COLUMNS = [
        'article_count_us',
        'article_count_europe',
    ]

    def __init__(self, gdelt_path: Optional[Union[str, Path]] = None):
        """
        Initialize GDELT loader.

        Args:
            gdelt_path: Path to GDELT hourly CSV file
        """
        self.gdelt_path = Path(gdelt_path) if gdelt_path else DEFAULT_GDELT_PATH
        self.data: Optional[pd.DataFrame] = None
        self._loaded = False

    def load(self) -> pd.DataFrame:
        """
        Load GDELT hourly sentiment data.

        Returns:
            DataFrame with DatetimeIndex and sentiment columns
        """
        if not self.gdelt_path.exists():
            raise FileNotFoundError(f"GDELT file not found: {self.gdelt_path}")

        logger.info(f"Loading GDELT sentiment from {self.gdelt_path}")

        self.data = pd.read_csv(
            self.gdelt_path,
            parse_dates=['timestamp'],
            index_col='timestamp',
        )

        # Ensure timezone-naive for easier merging
        if self.data.index.tz is not None:
            self.data.index = self.data.index.tz_localize(None)

        self.data = self.data.sort_index()
        self._loaded = True

        logger.info(f"Loaded {len(self.data):,} hourly sentiment records")
        logger.info(f"Date range: {self.data.index.min()} to {self.data.index.max()}")

        return self.data

    def get_hourly_sentiment(self) -> pd.DataFrame:
        """Get raw hourly sentiment (for 1H model)."""
        if not self._loaded:
            self.load()
        return self.data[self.SENTIMENT_COLUMNS].copy()

    def aggregate_to_timeframe(
        self,
        timeframe: str,
        include_std: bool = True,
        include_trend: bool = True,
    ) -> pd.DataFrame:
        """
        Aggregate hourly sentiment to a specific timeframe.

        Args:
            timeframe: Target timeframe ('1H', '4H', 'D', 'W')
            include_std: Include standard deviation feature
            include_trend: Include trend (last - first) feature

        Returns:
            DataFrame with aggregated sentiment features
        """
        if not self._loaded:
            self.load()

        # Map timeframe to pandas frequency
        freq_map = {
            '1H': '1H',
            '4H': '4H',
            'D': '1D',
            'W': '1W',
        }

        freq = freq_map.get(timeframe.upper(), timeframe)

        logger.info(f"Aggregating GDELT sentiment to {timeframe} timeframe")

        result_frames = []

        for col in self.SENTIMENT_COLUMNS:
            series = self.data[col]

            # Resample and aggregate
            resampled = series.resample(freq)

            # Average (central tendency)
            avg = resampled.mean()
            avg.name = f"gdelt_{col}_avg"
            result_frames.append(avg)

            # Standard deviation (uncertainty/volatility)
            if include_std:
                std = resampled.std()
                std.name = f"gdelt_{col}_std"
                result_frames.append(std)

            # Trend (direction of change within period)
            if include_trend:
                trend = resampled.last() - resampled.first()
                trend.name = f"gdelt_{col}_trend"
                result_frames.append(trend)

        # Combine all features
        result = pd.concat(result_frames, axis=1)

        # Forward fill any gaps (weekends, holidays)
        result = result.ffill().bfill()

        logger.info(f"Aggregated to {len(result):,} {timeframe} records with {len(result.columns)} features")

        return result

    def align_to_price_data(
        self,
        price_df: pd.DataFrame,
        timeframe: str,
        shift_periods: int = 1,
        include_std: bool = True,
        include_trend: bool = True,
    ) -> pd.DataFrame:
        """
        Align GDELT sentiment to price data with proper aggregation.

        CRITICAL: Shifts sentiment to avoid look-ahead bias.

        Args:
            price_df: DataFrame with DatetimeIndex (price data)
            timeframe: Timeframe of the price data ('1H', '4H', 'D')
            shift_periods: Periods to shift (1 = use previous period's sentiment)
            include_std: Include std feature
            include_trend: Include trend feature

        Returns:
            Price DataFrame with sentiment columns added
        """
        if not self._loaded:
            self.load()

        result = price_df.copy()

        # Get appropriately aggregated sentiment
        if timeframe.upper() == '1H':
            # For 1H, use raw hourly data (just avg, which is the raw value)
            sentiment = self.data[self.SENTIMENT_COLUMNS].copy()
            sentiment.columns = [f"gdelt_{col}" for col in sentiment.columns]
        else:
            # For 4H, D, etc., aggregate
            sentiment = self.aggregate_to_timeframe(
                timeframe,
                include_std=include_std,
                include_trend=include_trend,
            )

        # Shift to avoid look-ahead bias
        sentiment_shifted = sentiment.shift(shift_periods)

        # Merge with price data
        # Handle timezone differences
        if result.index.tz is not None:
            result.index = result.index.tz_localize(None)

        # Join on index
        result = result.join(sentiment_shifted, how='left')

        # Forward fill missing values
        sentiment_cols = [c for c in result.columns if c.startswith('gdelt_')]
        for col in sentiment_cols:
            result[col] = result[col].ffill().bfill()

        n_features = len(sentiment_cols)
        logger.info(f"Added {n_features} GDELT sentiment features for {timeframe} timeframe")

        return result

    def get_date_range(self) -> tuple:
        """Get the date range of available data."""
        if not self._loaded:
            self.load()
        return (self.data.index.min(), self.data.index.max())

    def get_feature_names(self, timeframe: str) -> list:
        """
        Get the list of feature names that will be created for a timeframe.

        Args:
            timeframe: Target timeframe

        Returns:
            List of feature column names
        """
        features = []
        for col in self.SENTIMENT_COLUMNS:
            if timeframe.upper() == '1H':
                features.append(f"gdelt_{col}")
            else:
                features.append(f"gdelt_{col}_avg")
                features.append(f"gdelt_{col}_std")
                features.append(f"gdelt_{col}_trend")
        return features


def load_gdelt_sentiment(
    price_df: pd.DataFrame,
    timeframe: str,
    gdelt_path: Optional[Path] = None,
    shift_periods: int = 1,
) -> pd.DataFrame:
    """
    Convenience function to load and align GDELT sentiment.

    Args:
        price_df: Price DataFrame with DatetimeIndex
        timeframe: Timeframe of price data ('1H', '4H', 'D')
        gdelt_path: Optional path to GDELT CSV
        shift_periods: Periods to shift for look-ahead bias prevention

    Returns:
        Price DataFrame with GDELT sentiment features added
    """
    loader = GDELTSentimentLoader(gdelt_path)
    return loader.align_to_price_data(
        price_df,
        timeframe=timeframe,
        shift_periods=shift_periods,
    )
