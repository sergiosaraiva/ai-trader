"""Sentiment data loading and alignment with price data."""

from pathlib import Path
from typing import Dict, List, Optional, Union
import pandas as pd
import numpy as np


class SentimentLoader:
    """
    Load and align daily sentiment data with price data.

    Sentiment data has daily resolution (same value for all intraday candles).
    Critical: Shifts sentiment by 1 day to avoid look-ahead bias.

    Attributes:
        sentiment_path: Path to the sentiment CSV file
        sentiment_data: Loaded sentiment DataFrame
    """

    # Mapping from trading pair to sentiment column
    FOREX_SENTIMENT_MAPPING = {
        'EURUSD': 'Sentiment_EURUSD',
        'GBPUSD': 'Sentiment_GBPUSD',
        'USDJPY': 'Sentiment_USDJPY',
        'AUDUSD': 'Sentiment_AUDUSD',
        'EURGBP': 'Sentiment_EURGBP',
    }

    # Country sentiment columns for additional features
    COUNTRY_SENTIMENT_COLUMNS = [
        'Sentiment_US',
        'Sentiment_UK',
        'Sentiment_Europe',
        'Sentiment_Germany',
        'Sentiment_Japan',
        'Sentiment_Australia',
        'Sentiment_Global',
    ]

    # EPU columns for raw uncertainty values
    EPU_COLUMNS = [
        'EPU_US',
        'EPU_UK',
        'EPU_Europe',
        'EPU_Germany',
        'EPU_Japan',
        'EPU_Australia',
        'EPU_China',
        'EPU_Global',
    ]

    def __init__(self, sentiment_path: Union[str, Path]):
        """
        Initialize sentiment loader.

        Args:
            sentiment_path: Path to sentiment CSV file
        """
        self.sentiment_path = Path(sentiment_path)
        self.sentiment_data: Optional[pd.DataFrame] = None
        self._loaded = False

    def load(self) -> pd.DataFrame:
        """
        Load sentiment data from CSV.

        Returns:
            DataFrame with DatetimeIndex and sentiment columns

        Raises:
            FileNotFoundError: If sentiment file doesn't exist
        """
        if not self.sentiment_path.exists():
            raise FileNotFoundError(f"Sentiment file not found: {self.sentiment_path}")

        self.sentiment_data = pd.read_csv(
            self.sentiment_path,
            parse_dates=['Date'],
            index_col='Date',
        )

        # Ensure index is DatetimeIndex
        if not isinstance(self.sentiment_data.index, pd.DatetimeIndex):
            self.sentiment_data.index = pd.to_datetime(self.sentiment_data.index)

        # Sort by date
        self.sentiment_data = self.sentiment_data.sort_index()

        self._loaded = True
        return self.sentiment_data

    def get_pair_sentiment(self, pair: str) -> pd.Series:
        """
        Get sentiment column for a specific trading pair.

        Args:
            pair: Trading pair (e.g., 'EURUSD', 'BTCUSDT')

        Returns:
            Series with daily sentiment scores

        Raises:
            ValueError: If pair sentiment not found
        """
        if not self._loaded:
            self.load()

        pair_upper = pair.upper().replace('/', '')

        # Forex pairs - use pair-specific sentiment
        if pair_upper in self.FOREX_SENTIMENT_MAPPING:
            col = self.FOREX_SENTIMENT_MAPPING[pair_upper]
            if col in self.sentiment_data.columns:
                return self.sentiment_data[col]

        # Crypto pairs - use Sentiment_Crypto or Global
        if 'USDT' in pair_upper or 'BTC' in pair_upper or 'ETH' in pair_upper:
            if 'Sentiment_Crypto' in self.sentiment_data.columns:
                return self.sentiment_data['Sentiment_Crypto']
            elif 'Sentiment_Global' in self.sentiment_data.columns:
                return self.sentiment_data['Sentiment_Global']

        # Fallback to global sentiment
        if 'Sentiment_Global' in self.sentiment_data.columns:
            return self.sentiment_data['Sentiment_Global']

        raise ValueError(f"No sentiment data available for pair: {pair}")

    def get_country_sentiments(self, pair: str) -> pd.DataFrame:
        """
        Get relevant country sentiment columns for a trading pair.

        For forex pairs, returns base and quote currency country sentiments.
        For crypto, returns US and Global sentiments.

        Args:
            pair: Trading pair

        Returns:
            DataFrame with relevant country sentiments
        """
        if not self._loaded:
            self.load()

        pair_upper = pair.upper().replace('/', '')

        # Mapping of currencies to countries
        currency_country = {
            'EUR': ['Sentiment_Europe', 'Sentiment_Germany'],
            'USD': ['Sentiment_US'],
            'GBP': ['Sentiment_UK'],
            'JPY': ['Sentiment_Japan'],
            'AUD': ['Sentiment_Australia'],
            'CHF': ['Sentiment_Europe'],  # Swiss franc uses Europe
            'CAD': ['Sentiment_US'],  # Canada closely tied to US
            'NZD': ['Sentiment_Australia'],  # NZ closely tied to AU
        }

        # Extract base and quote currencies
        base_currency = pair_upper[:3]
        quote_currency = pair_upper[3:6] if len(pair_upper) >= 6 else 'USD'

        columns = set()

        # Add base currency country sentiments
        if base_currency in currency_country:
            columns.update(currency_country[base_currency])

        # Add quote currency country sentiments
        if quote_currency in currency_country:
            columns.update(currency_country[quote_currency])

        # Always include global
        columns.add('Sentiment_Global')

        # Filter to existing columns
        existing_cols = [c for c in columns if c in self.sentiment_data.columns]

        return self.sentiment_data[existing_cols]

    def get_epu_values(self) -> pd.DataFrame:
        """
        Get raw EPU (Economic Policy Uncertainty) values.

        Returns:
            DataFrame with EPU columns
        """
        if not self._loaded:
            self.load()

        existing_cols = [c for c in self.EPU_COLUMNS if c in self.sentiment_data.columns]
        return self.sentiment_data[existing_cols]

    def align_to_price_data(
        self,
        price_df: pd.DataFrame,
        pair: str,
        shift_days: int = 1,
        include_country_sentiments: bool = True,
        include_epu: bool = False,
    ) -> pd.DataFrame:
        """
        Align daily sentiment to price data (any timeframe).

        CRITICAL: Shifts sentiment by shift_days to avoid look-ahead bias.
        Daily sentiment will be the same for all intraday candles of that day.

        Args:
            price_df: DataFrame with DatetimeIndex (any timeframe)
            pair: Trading pair name
            shift_days: Days to shift sentiment (default 1 for no look-ahead)
            include_country_sentiments: Include relevant country sentiments
            include_epu: Include raw EPU values

        Returns:
            Price DataFrame with sentiment columns added
        """
        if not self._loaded:
            self.load()

        result = price_df.copy()

        # Extract date from price index for merging
        if hasattr(result.index, 'date'):
            result['_merge_date'] = pd.to_datetime(result.index.date)
        else:
            result['_merge_date'] = pd.to_datetime(result.index)

        # Prepare sentiment data
        sentiment_to_merge = pd.DataFrame(index=self.sentiment_data.index)

        # Add pair-specific sentiment
        pair_sentiment = self.get_pair_sentiment(pair)
        sentiment_to_merge['sentiment_raw'] = pair_sentiment

        # Add country sentiments if requested
        if include_country_sentiments:
            country_sent = self.get_country_sentiments(pair)
            for col in country_sent.columns:
                # Rename to lowercase with prefix
                new_name = col.lower().replace('sentiment_', 'sent_country_')
                sentiment_to_merge[new_name] = country_sent[col]

        # Add EPU values if requested
        if include_epu:
            epu_df = self.get_epu_values()
            for col in epu_df.columns:
                new_name = col.lower()
                sentiment_to_merge[new_name] = epu_df[col]

        # Shift sentiment to avoid look-ahead bias
        # shift_days=1 means we use yesterday's sentiment for today
        sentiment_shifted = sentiment_to_merge.shift(shift_days)

        # Prepare for merge - set index to date
        sentiment_shifted['_merge_date'] = sentiment_shifted.index
        sentiment_shifted = sentiment_shifted.reset_index(drop=True)

        # Merge on date
        result = result.reset_index()
        original_index_name = result.columns[0]

        result = result.merge(
            sentiment_shifted,
            on='_merge_date',
            how='left',
        )

        # Restore index
        result = result.set_index(original_index_name)
        result = result.drop('_merge_date', axis=1)

        # Forward fill missing values (weekends, holidays)
        sentiment_cols = [c for c in result.columns if 'sentiment' in c.lower() or 'sent_' in c.lower() or 'epu_' in c.lower()]
        for col in sentiment_cols:
            result[col] = result[col].ffill().bfill()

        return result

    def get_sentiment_date_range(self) -> tuple:
        """
        Get the date range of available sentiment data.

        Returns:
            Tuple of (start_date, end_date)
        """
        if not self._loaded:
            self.load()

        return (
            self.sentiment_data.index.min(),
            self.sentiment_data.index.max(),
        )

    def get_available_columns(self) -> List[str]:
        """
        Get list of all available sentiment columns.

        Returns:
            List of column names
        """
        if not self._loaded:
            self.load()

        return self.sentiment_data.columns.tolist()

    def validate_coverage(
        self,
        price_df: pd.DataFrame,
    ) -> Dict[str, any]:
        """
        Validate sentiment coverage for price data.

        Args:
            price_df: Price DataFrame to check coverage for

        Returns:
            Dictionary with coverage statistics
        """
        if not self._loaded:
            self.load()

        # Get date ranges
        price_start = price_df.index.min()
        price_end = price_df.index.max()
        sent_start, sent_end = self.get_sentiment_date_range()

        # Calculate coverage
        if hasattr(price_df.index, 'date'):
            price_dates = set(price_df.index.date)
        else:
            price_dates = set(price_df.index.normalize())

        sent_dates = set(self.sentiment_data.index.date)

        covered_dates = price_dates.intersection(sent_dates)
        missing_dates = price_dates - sent_dates

        return {
            'price_date_range': (price_start, price_end),
            'sentiment_date_range': (sent_start, sent_end),
            'total_price_days': len(price_dates),
            'covered_days': len(covered_dates),
            'missing_days': len(missing_dates),
            'coverage_pct': len(covered_dates) / len(price_dates) * 100 if price_dates else 0,
            'missing_dates_sample': sorted(list(missing_dates))[:10],  # First 10 missing
        }
