"""
Sentiment analysis features module.

This module provides sentiment data loading, alignment, and feature engineering
for integrating Economic Policy Uncertainty (EPU) data with price data.

Key Components:
    - SentimentLoader: Load and align daily sentiment data with price data
    - SentimentFeatureCalculator: Calculate derived sentiment features
    - SentimentFeatureConfig: Configuration for sentiment features
    - SentimentFeatures: High-level interface combining loader and calculator

Usage:
    from src.features.sentiment import (
        SentimentLoader,
        SentimentFeatureCalculator,
        SentimentFeatures,
    )

    # Option 1: Low-level control
    loader = SentimentLoader('data/sentiment/sentiment_epu.csv')
    df_with_sent = loader.align_to_price_data(price_df, 'EURUSD')

    calculator = SentimentFeatureCalculator()
    df_with_features = calculator.calculate_all(df_with_sent)

    # Option 2: High-level interface
    sentiment = SentimentFeatures('data/sentiment/sentiment_epu.csv')
    df_with_all = sentiment.calculate_all(price_df, 'EURUSD')
"""

from .sentiment_loader import SentimentLoader
from .sentiment_features import (
    SentimentFeatureCalculator,
    SentimentFeatureConfig,
)
from .features import SentimentFeatures

__all__ = [
    'SentimentLoader',
    'SentimentFeatureCalculator',
    'SentimentFeatureConfig',
    'SentimentFeatures',
]
