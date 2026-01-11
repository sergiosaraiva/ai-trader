"""High-level sentiment features interface."""

from pathlib import Path
from typing import Dict, List, Optional, Union
from datetime import datetime
import pandas as pd

from .sentiment_loader import SentimentLoader
from .sentiment_features import SentimentFeatureCalculator, SentimentFeatureConfig


class SentimentFeatures:
    """
    High-level interface for sentiment feature extraction.

    Combines SentimentLoader and SentimentFeatureCalculator to provide
    a simple interface for adding sentiment features to price data.

    Example:
        sentiment = SentimentFeatures('data/sentiment/sentiment_epu.csv')
        df_with_features = sentiment.calculate_all(price_df, 'EURUSD')
    """

    DEFAULT_SENTIMENT_PATH = 'data/sentiment/sentiment_epu_20200101_20251231_daily.csv'

    def __init__(
        self,
        sentiment_path: Optional[Union[str, Path]] = None,
        config: Optional[SentimentFeatureConfig] = None,
    ):
        """
        Initialize sentiment features extractor.

        Args:
            sentiment_path: Path to sentiment CSV file (uses default if not provided)
            config: Configuration for feature generation
        """
        self.sentiment_path = Path(sentiment_path or self.DEFAULT_SENTIMENT_PATH)
        self.config = config or SentimentFeatureConfig()
        self._feature_names: List[str] = []

        # Initialize components
        self.loader = SentimentLoader(self.sentiment_path)
        self.calculator = SentimentFeatureCalculator(
            config={
                'ma_periods': self.config.ma_periods,
                'std_periods': self.config.std_periods,
                'momentum_periods': self.config.momentum_periods,
                'lag_days': self.config.lag_days,
            }
        )

    def calculate_all(
        self,
        df: pd.DataFrame,
        pair: str,
        include_country_sentiments: bool = None,
        include_epu: bool = None,
        shift_days: int = 1,
    ) -> pd.DataFrame:
        """
        Calculate all sentiment features for price data.

        CRITICAL: Uses shift_days to avoid look-ahead bias. Default is 1 day,
        meaning we use yesterday's sentiment for today's prediction.

        Args:
            df: Price dataframe with DatetimeIndex
            pair: Trading pair (e.g., 'EURUSD', 'BTCUSDT')
            include_country_sentiments: Include country-level sentiments (default from config)
            include_epu: Include raw EPU values (default from config)
            shift_days: Days to shift sentiment (default 1 for no look-ahead)

        Returns:
            DataFrame with sentiment features added
        """
        if not self.config.enabled:
            return df

        # Use config defaults if not specified
        if include_country_sentiments is None:
            include_country_sentiments = self.config.include_country
        if include_epu is None:
            include_epu = self.config.include_epu

        self._feature_names = []

        # Step 1: Load and align sentiment data
        result = self.loader.align_to_price_data(
            price_df=df,
            pair=pair,
            shift_days=shift_days,
            include_country_sentiments=include_country_sentiments,
            include_epu=include_epu,
        )

        # Step 2: Calculate derived sentiment features from raw sentiment
        result = self.calculator.calculate_all(
            df=result,
            sentiment_col='sentiment_raw',
            prefix='sentiment',
        )

        # Step 3: Calculate differential for forex pairs
        if self.config.include_differential:
            result = self._add_differential(result, pair)

        # Step 4: Calculate cross-country features
        if self.config.include_cross_features and include_country_sentiments:
            country_cols = [c for c in result.columns if 'sent_country_' in c]
            if len(country_cols) >= 2:
                result = self.calculator.calculate_cross_sentiment_features(
                    df=result,
                    sentiment_cols=country_cols,
                    prefix='cross_sent',
                )

        # Update feature names
        self._feature_names = self.calculator.get_feature_names()

        # Filter features based on config
        result = self._filter_features(result)

        return result

    def _add_differential(self, df: pd.DataFrame, pair: str) -> pd.DataFrame:
        """Add sentiment differential for forex pairs."""
        pair_upper = pair.upper().replace('/', '')

        # Mapping of currencies to country sentiment columns
        currency_to_col = {
            'EUR': 'sent_country_europe',
            'USD': 'sent_country_us',
            'GBP': 'sent_country_uk',
            'JPY': 'sent_country_japan',
            'AUD': 'sent_country_australia',
        }

        if len(pair_upper) >= 6:
            base = pair_upper[:3]
            quote = pair_upper[3:6]

            base_col = currency_to_col.get(base)
            quote_col = currency_to_col.get(quote)

            if base_col and quote_col:
                if base_col in df.columns and quote_col in df.columns:
                    df = self.calculator.calculate_differential(
                        df=df,
                        base_col=base_col,
                        quote_col=quote_col,
                        output_col='sentiment_differential',
                    )

        return df

    def _filter_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter features based on configuration."""
        patterns = self.config.get_feature_filter()

        # Get all sentiment-related columns
        all_sentiment_cols = [
            c for c in df.columns
            if any(p in c.lower() for p in ['sentiment', 'sent_', 'epu_', 'cross_sent'])
        ]

        # Filter to configured patterns
        keep_cols = []
        for col in all_sentiment_cols:
            for pattern in patterns:
                if pattern in col.lower():
                    keep_cols.append(col)
                    break

        # Return original non-sentiment columns plus filtered sentiment columns
        non_sentiment_cols = [c for c in df.columns if c not in all_sentiment_cols]

        return df[non_sentiment_cols + keep_cols]

    def get_feature_names(self) -> List[str]:
        """
        Get list of generated feature names.

        Returns:
            List of feature column names
        """
        return self._feature_names.copy()

    def validate_data(self, df: pd.DataFrame) -> Dict:
        """
        Validate sentiment data coverage for price data.

        Args:
            df: Price DataFrame

        Returns:
            Dictionary with validation results
        """
        return self.loader.validate_coverage(df)

    def get_sentiment_date_range(self) -> tuple:
        """
        Get the date range of available sentiment data.

        Returns:
            Tuple of (start_date, end_date)
        """
        return self.loader.get_sentiment_date_range()

    @staticmethod
    def get_default_feature_count() -> int:
        """
        Get the expected number of default sentiment features.

        Returns:
            Number of features (approximately)
        """
        return len(SentimentFeatureCalculator.get_default_feature_list())

    def analyze_text(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment of text (placeholder for future NLP).

        This method is kept for backwards compatibility.
        Future implementation will use FinBERT or similar.

        Args:
            text: Text to analyze

        Returns:
            Dictionary with sentiment scores
        """
        return {
            'positive': 0.0,
            'negative': 0.0,
            'neutral': 1.0,
            'compound': 0.0,
        }

    def get_fear_greed_index(self, date: Optional[datetime] = None) -> float:
        """
        Get market fear/greed index (placeholder).

        Args:
            date: Date to get index for

        Returns:
            Fear/Greed index value (0-100)
        """
        return 50.0

    def get_cot_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
    ) -> pd.DataFrame:
        """
        Get Commitment of Traders data (placeholder).

        Args:
            symbol: Trading symbol
            start_date: Start date
            end_date: End date

        Returns:
            DataFrame with COT positioning data
        """
        return pd.DataFrame()
