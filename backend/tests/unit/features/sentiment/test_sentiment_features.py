"""Unit tests for sentiment features module."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))


# Fixtures
@pytest.fixture
def sample_sentiment_data():
    """Create sample sentiment data for testing."""
    dates = pd.date_range('2024-01-01', periods=365, freq='D')
    np.random.seed(42)

    return pd.DataFrame({
        'Date': dates,
        'EPU_US': np.random.uniform(50, 200, len(dates)),
        'EPU_UK': np.random.uniform(50, 200, len(dates)),
        'EPU_Europe': np.random.uniform(50, 200, len(dates)),
        'EPU_Germany': np.random.uniform(50, 200, len(dates)),
        'EPU_Japan': np.random.uniform(50, 200, len(dates)),
        'EPU_Australia': np.random.uniform(50, 200, len(dates)),
        'EPU_Global': np.random.uniform(50, 200, len(dates)),
        'Sentiment_US': np.random.uniform(-0.2, 0.2, len(dates)),
        'Sentiment_UK': np.random.uniform(-0.2, 0.2, len(dates)),
        'Sentiment_Europe': np.random.uniform(-0.2, 0.2, len(dates)),
        'Sentiment_Germany': np.random.uniform(-0.2, 0.2, len(dates)),
        'Sentiment_Japan': np.random.uniform(-0.2, 0.2, len(dates)),
        'Sentiment_Australia': np.random.uniform(-0.2, 0.2, len(dates)),
        'Sentiment_Global': np.random.uniform(-0.2, 0.2, len(dates)),
        'Sentiment_EURUSD': np.random.uniform(-0.2, 0.2, len(dates)),
        'Sentiment_GBPUSD': np.random.uniform(-0.2, 0.2, len(dates)),
        'Sentiment_USDJPY': np.random.uniform(-0.2, 0.2, len(dates)),
        'Sentiment_AUDUSD': np.random.uniform(-0.2, 0.2, len(dates)),
        'Sentiment_EURGBP': np.random.uniform(-0.2, 0.2, len(dates)),
        'Sentiment_Crypto': np.random.uniform(-0.2, 0.2, len(dates)),
    }).set_index('Date')


@pytest.fixture
def sample_price_data():
    """Create sample intraday price data for testing."""
    # Create 30 days of 1-hour data
    dates = pd.date_range('2024-01-15', periods=720, freq='1h')
    np.random.seed(42)

    base_price = 1.1000
    prices = base_price + np.cumsum(np.random.randn(len(dates)) * 0.0001)

    return pd.DataFrame({
        'open': prices - np.random.uniform(0.0001, 0.0005, len(dates)),
        'high': prices + np.random.uniform(0.0001, 0.0010, len(dates)),
        'low': prices - np.random.uniform(0.0001, 0.0010, len(dates)),
        'close': prices,
        'volume': np.random.randint(1000, 10000, len(dates)),
    }, index=dates)


@pytest.fixture
def sentiment_csv_file(tmp_path, sample_sentiment_data):
    """Create temporary sentiment CSV file."""
    file_path = tmp_path / 'sentiment_test.csv'
    sample_sentiment_data.reset_index().to_csv(file_path, index=False)
    return file_path


class TestSentimentLoader:
    """Tests for SentimentLoader class."""

    def test_load_sentiment_data(self, sentiment_csv_file):
        """Test loading sentiment data from CSV."""
        from src.features.sentiment import SentimentLoader

        loader = SentimentLoader(sentiment_csv_file)
        df = loader.load()

        assert df is not None
        assert len(df) == 365
        assert 'Sentiment_EURUSD' in df.columns
        assert isinstance(df.index, pd.DatetimeIndex)

    def test_get_pair_sentiment_forex(self, sentiment_csv_file):
        """Test getting sentiment for forex pair."""
        from src.features.sentiment import SentimentLoader

        loader = SentimentLoader(sentiment_csv_file)
        loader.load()

        eurusd_sent = loader.get_pair_sentiment('EURUSD')

        assert eurusd_sent is not None
        assert len(eurusd_sent) == 365
        assert eurusd_sent.min() >= -0.25
        assert eurusd_sent.max() <= 0.25

    def test_get_pair_sentiment_crypto(self, sentiment_csv_file):
        """Test getting sentiment for crypto pair."""
        from src.features.sentiment import SentimentLoader

        loader = SentimentLoader(sentiment_csv_file)
        loader.load()

        btc_sent = loader.get_pair_sentiment('BTCUSDT')

        assert btc_sent is not None
        assert len(btc_sent) == 365

    def test_get_country_sentiments(self, sentiment_csv_file):
        """Test getting country sentiments for a pair."""
        from src.features.sentiment import SentimentLoader

        loader = SentimentLoader(sentiment_csv_file)
        loader.load()

        country_sent = loader.get_country_sentiments('EURUSD')

        assert isinstance(country_sent, pd.DataFrame)
        assert len(country_sent.columns) >= 2  # At least Europe and US

    def test_align_to_price_data(self, sentiment_csv_file, sample_price_data):
        """Test aligning sentiment to intraday price data."""
        from src.features.sentiment import SentimentLoader

        loader = SentimentLoader(sentiment_csv_file)
        result = loader.align_to_price_data(
            price_df=sample_price_data,
            pair='EURUSD',
            shift_days=1,
        )

        assert 'sentiment_raw' in result.columns
        assert len(result) == len(sample_price_data)
        assert result['sentiment_raw'].isna().sum() == 0

    def test_align_with_shift_prevents_lookahead(self, sentiment_csv_file, sample_price_data):
        """Test that shift_days prevents look-ahead bias."""
        from src.features.sentiment import SentimentLoader

        loader = SentimentLoader(sentiment_csv_file)

        # Align with shift=1 (using yesterday's sentiment)
        result_shifted = loader.align_to_price_data(
            price_df=sample_price_data,
            pair='EURUSD',
            shift_days=1,
        )

        # Align with shift=0 (same-day sentiment - has look-ahead bias)
        result_no_shift = loader.align_to_price_data(
            price_df=sample_price_data,
            pair='EURUSD',
            shift_days=0,
        )

        # Values should be different due to shift
        # Note: Some values might be the same if sentiment didn't change day-to-day
        # but overall there should be differences
        assert not result_shifted['sentiment_raw'].equals(result_no_shift['sentiment_raw'])

    def test_validate_coverage(self, sentiment_csv_file, sample_price_data):
        """Test sentiment coverage validation."""
        from src.features.sentiment import SentimentLoader

        loader = SentimentLoader(sentiment_csv_file)
        coverage = loader.validate_coverage(sample_price_data)

        assert 'coverage_pct' in coverage
        assert coverage['coverage_pct'] > 0
        assert 'missing_days' in coverage


class TestSentimentFeatureCalculator:
    """Tests for SentimentFeatureCalculator class."""

    @pytest.fixture
    def df_with_sentiment(self):
        """Create DataFrame with raw sentiment."""
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        np.random.seed(42)

        return pd.DataFrame({
            'close': np.random.uniform(1.0, 1.2, len(dates)),
            'sentiment_raw': np.random.uniform(-0.2, 0.2, len(dates)),
        }, index=dates)

    def test_calculate_all_features(self, df_with_sentiment):
        """Test calculating all sentiment features."""
        from src.features.sentiment import SentimentFeatureCalculator

        calculator = SentimentFeatureCalculator()
        result = calculator.calculate_all(df_with_sentiment)

        # Check expected features exist
        assert 'sentiment_raw' in result.columns
        assert 'sentiment_ma_7' in result.columns
        assert 'sentiment_ma_14' in result.columns
        assert 'sentiment_momentum_7' in result.columns
        assert 'sentiment_std_7' in result.columns
        assert 'sentiment_zscore' in result.columns
        assert 'sentiment_regime' in result.columns
        assert 'sentiment_lag_1' in result.columns

    def test_no_nan_values(self, df_with_sentiment):
        """Test that output has no NaN values."""
        from src.features.sentiment import SentimentFeatureCalculator

        calculator = SentimentFeatureCalculator()
        result = calculator.calculate_all(df_with_sentiment)

        # Check no NaN in sentiment columns
        sentiment_cols = [c for c in result.columns if 'sentiment' in c.lower()]
        for col in sentiment_cols:
            assert result[col].isna().sum() == 0, f"Column {col} has NaN values"

    def test_regime_classification(self, df_with_sentiment):
        """Test regime classification values."""
        from src.features.sentiment import SentimentFeatureCalculator

        calculator = SentimentFeatureCalculator()
        result = calculator.calculate_all(df_with_sentiment)

        # Regime should be -1, 0, or 1
        unique_regimes = result['sentiment_regime'].unique()
        assert all(r in [-1, 0, 1] for r in unique_regimes)

    def test_zscore_clipping(self, df_with_sentiment):
        """Test z-score is clipped to [-3, 3]."""
        from src.features.sentiment import SentimentFeatureCalculator

        calculator = SentimentFeatureCalculator()
        result = calculator.calculate_all(df_with_sentiment)

        assert result['sentiment_zscore'].min() >= -3
        assert result['sentiment_zscore'].max() <= 3

    def test_feature_names(self, df_with_sentiment):
        """Test get_feature_names returns correct list."""
        from src.features.sentiment import SentimentFeatureCalculator

        calculator = SentimentFeatureCalculator()
        calculator.calculate_all(df_with_sentiment)

        names = calculator.get_feature_names()

        assert isinstance(names, list)
        assert len(names) > 10  # Should have many features
        assert 'sentiment_raw' in names

    def test_custom_config(self, df_with_sentiment):
        """Test with custom configuration."""
        from src.features.sentiment import SentimentFeatureCalculator

        custom_config = {
            'ma_periods': [5, 10],
            'std_periods': [5],
            'momentum_periods': [5],
            'lag_days': [1],
        }

        calculator = SentimentFeatureCalculator(config=custom_config)
        result = calculator.calculate_all(df_with_sentiment)

        assert 'sentiment_ma_5' in result.columns
        assert 'sentiment_ma_10' in result.columns
        assert 'sentiment_ma_7' not in result.columns  # Default not present


class TestSentimentFeatures:
    """Tests for high-level SentimentFeatures class."""

    def test_calculate_all_integration(self, sentiment_csv_file, sample_price_data):
        """Test full feature calculation integration."""
        from src.features.sentiment import SentimentFeatures

        sentiment = SentimentFeatures(sentiment_path=sentiment_csv_file)
        result = sentiment.calculate_all(
            df=sample_price_data,
            pair='EURUSD',
            shift_days=1,
        )

        # Check original columns preserved
        assert 'open' in result.columns
        assert 'high' in result.columns
        assert 'low' in result.columns
        assert 'close' in result.columns
        assert 'volume' in result.columns

        # Check sentiment features added
        sentiment_cols = [c for c in result.columns if 'sentiment' in c.lower()]
        assert len(sentiment_cols) > 5

    def test_disabled_sentiment(self, sentiment_csv_file, sample_price_data):
        """Test with sentiment disabled."""
        from src.features.sentiment import SentimentFeatures, SentimentFeatureConfig

        config = SentimentFeatureConfig(enabled=False)
        sentiment = SentimentFeatures(
            sentiment_path=sentiment_csv_file,
            config=config,
        )

        result = sentiment.calculate_all(
            df=sample_price_data,
            pair='EURUSD',
        )

        # Should return original DataFrame unchanged
        assert len(result.columns) == len(sample_price_data.columns)

    def test_get_feature_names(self, sentiment_csv_file, sample_price_data):
        """Test getting feature names after calculation."""
        from src.features.sentiment import SentimentFeatures

        sentiment = SentimentFeatures(sentiment_path=sentiment_csv_file)
        sentiment.calculate_all(
            df=sample_price_data,
            pair='EURUSD',
        )

        names = sentiment.get_feature_names()
        assert isinstance(names, list)


class TestSentimentIntegration:
    """Integration tests for sentiment with existing pipeline."""

    @pytest.fixture
    def real_sentiment_file(self):
        """Get path to real sentiment file if available."""
        path = Path('data/sentiment/sentiment_epu_20200101_20251231_daily.csv')
        if not path.exists():
            pytest.skip("Real sentiment data not available")
        return path

    def test_with_real_sentiment_data(self, real_sentiment_file):
        """Test with real sentiment data file."""
        from src.features.sentiment import SentimentLoader

        loader = SentimentLoader(real_sentiment_file)
        df = loader.load()

        assert df is not None
        assert len(df) > 1000  # Should have several years of data
        assert 'Sentiment_EURUSD' in df.columns

    def test_feature_processor_integration(self, sentiment_csv_file, sample_price_data):
        """Test integration with FeatureProcessor."""
        from src.data.processors import FeatureProcessor

        processor = FeatureProcessor(
            include_sentiment=True,
            sentiment_path=sentiment_csv_file,
        )

        result = processor.add_sentiment_features(
            df=sample_price_data,
            pair='EURUSD',
        )

        sentiment_cols = [c for c in result.columns if 'sentiment' in c.lower()]
        assert len(sentiment_cols) > 0

    def test_full_feature_pipeline(self, sentiment_csv_file, sample_price_data):
        """Test full feature preparation pipeline with sentiment."""
        from src.data.processors import FeatureProcessor

        processor = FeatureProcessor(
            include_sentiment=True,
            sentiment_path=sentiment_csv_file,
        )

        # Add temporal features first (since we don't have technical indicators in test)
        result = processor.add_temporal_features(sample_price_data)
        result = processor.add_trading_session_features(result)
        result = processor.add_sentiment_features(result, pair='EURUSD')

        # Should have temporal, session, and sentiment features
        assert 'hour_sin' in result.columns or 'dow_sin' in result.columns
        assert 'is_london' in result.columns
        assert any('sentiment' in c.lower() for c in result.columns)

    def test_select_features_includes_sentiment(self, sentiment_csv_file, sample_price_data):
        """Test that select_features includes sentiment group."""
        from src.data.processors import FeatureProcessor

        processor = FeatureProcessor(
            include_sentiment=True,
            sentiment_path=sentiment_csv_file,
        )

        # Add sentiment features
        df = processor.add_sentiment_features(sample_price_data, pair='EURUSD')

        # Select only sentiment features
        result = processor.select_features(df, feature_groups=['sentiment'])

        # All columns should be sentiment-related
        for col in result.columns:
            assert any(p in col.lower() for p in ['sentiment', 'sent_', 'epu_'])


class TestSentimentDataAlignment:
    """Tests specific to data alignment behavior."""

    def test_weekend_handling(self, sentiment_csv_file):
        """Test that weekends are properly forward-filled."""
        from src.features.sentiment import SentimentLoader

        # Create price data spanning a weekend
        dates = pd.date_range('2024-01-12 00:00', '2024-01-15 23:00', freq='1h')  # Fri-Mon
        price_df = pd.DataFrame({
            'open': np.random.randn(len(dates)),
            'high': np.random.randn(len(dates)),
            'low': np.random.randn(len(dates)),
            'close': np.random.randn(len(dates)),
            'volume': np.random.randint(100, 1000, len(dates)),
        }, index=dates)

        loader = SentimentLoader(sentiment_csv_file)
        result = loader.align_to_price_data(price_df, 'EURUSD')

        # Weekend should have sentiment values (forward-filled)
        saturday_data = result[result.index.dayofweek == 5]
        sunday_data = result[result.index.dayofweek == 6]

        if len(saturday_data) > 0:
            assert saturday_data['sentiment_raw'].isna().sum() == 0
        if len(sunday_data) > 0:
            assert sunday_data['sentiment_raw'].isna().sum() == 0

    def test_intraday_same_value(self, sentiment_csv_file):
        """Test that all intraday candles have same sentiment value."""
        from src.features.sentiment import SentimentLoader

        # Create hourly data for one day
        dates = pd.date_range('2024-01-15 00:00', '2024-01-15 23:00', freq='1h')
        price_df = pd.DataFrame({
            'open': np.random.randn(len(dates)),
            'high': np.random.randn(len(dates)),
            'low': np.random.randn(len(dates)),
            'close': np.random.randn(len(dates)),
            'volume': np.random.randint(100, 1000, len(dates)),
        }, index=dates)

        loader = SentimentLoader(sentiment_csv_file)
        result = loader.align_to_price_data(price_df, 'EURUSD', shift_days=0)

        # All values for the same day should be identical
        sentiment_values = result['sentiment_raw'].unique()
        assert len(sentiment_values) == 1, "All intraday candles should have same sentiment"
