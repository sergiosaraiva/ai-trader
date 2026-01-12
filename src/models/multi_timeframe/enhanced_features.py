"""Enhanced feature engineering for improved predictions.

This module adds sophisticated features that capture:
- Cross-timeframe alignment
- Time-of-day patterns
- Rate of change / momentum of indicators
- Normalized/relative features
- Pattern recognition
- Sentiment features (optional) - supports EPU/VIX (daily) or GDELT (hourly)
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Default sentiment data paths
DEFAULT_SENTIMENT_PATH = Path(__file__).parent.parent.parent.parent / 'data' / 'sentiment' / 'sentiment_scores_20200101_20251231_daily.csv'
DEFAULT_GDELT_PATH = Path(__file__).parent.parent.parent.parent / 'data' / 'sentiment' / 'gdelt_sentiment_20200101_20251231_hourly.csv'


class EnhancedFeatureEngine:
    """Generates enhanced features for trading models."""

    def __init__(
        self,
        base_timeframe: str = "5min",
        include_time_features: bool = True,
        include_roc_features: bool = True,
        include_normalized_features: bool = True,
        include_pattern_features: bool = True,
        include_lag_features: bool = True,
        include_sentiment_features: bool = False,
        sentiment_path: Optional[Path] = None,
        trading_pair: str = "EURUSD",
        lag_periods: List[int] = None,
        us_only_sentiment: bool = True,  # Use US-only sentiment (for EPU/VIX)
        sentiment_source: str = "epu",  # 'epu' (daily VIX/EPU), 'gdelt' (hourly), or 'both'
        gdelt_path: Optional[Path] = None,
    ):
        self.base_timeframe = base_timeframe
        self.include_time_features = include_time_features
        self.include_roc_features = include_roc_features
        self.include_normalized_features = include_normalized_features
        self.include_pattern_features = include_pattern_features
        self.include_lag_features = include_lag_features
        self.include_sentiment_features = include_sentiment_features
        self.sentiment_path = sentiment_path or DEFAULT_SENTIMENT_PATH
        self.trading_pair = trading_pair
        self.lag_periods = lag_periods or [1, 2, 3, 6, 12]
        self.us_only_sentiment = us_only_sentiment  # US-only is recommended for EPU
        self.sentiment_source = sentiment_source.lower()  # 'epu', 'gdelt', or 'both'
        self.gdelt_path = gdelt_path or DEFAULT_GDELT_PATH

        # Sentiment components (lazy loaded)
        self._sentiment_loader = None
        self._sentiment_calculator = None
        self._gdelt_loader = None

    def add_all_features(
        self,
        df: pd.DataFrame,
        higher_tf_data: Optional[Dict[str, pd.DataFrame]] = None,
    ) -> pd.DataFrame:
        """Add all enhanced features to dataframe.

        Args:
            df: Base OHLCV dataframe with technical indicators
            higher_tf_data: Dict of higher timeframe dataframes for cross-TF features

        Returns:
            DataFrame with enhanced features added
        """
        result = df.copy()

        if self.include_time_features:
            result = self._add_time_features(result)

        if self.include_roc_features:
            result = self._add_roc_features(result)

        if self.include_normalized_features:
            result = self._add_normalized_features(result)

        if self.include_pattern_features:
            result = self._add_pattern_features(result)

        if self.include_lag_features:
            result = self._add_lag_features(result)

        if higher_tf_data:
            result = self._add_cross_tf_features(result, higher_tf_data)

        if self.include_sentiment_features:
            if self.sentiment_source == 'gdelt':
                result = self._add_gdelt_sentiment_features(result)
            elif self.sentiment_source == 'both':
                result = self._add_sentiment_features(result)
                result = self._add_gdelt_sentiment_features(result)
            else:  # 'epu' (default)
                result = self._add_sentiment_features(result)

        logger.info(f"Added enhanced features: {len(result.columns)} total columns")

        return result

    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-of-day and day-of-week features.

        Forex markets behave differently during different sessions.
        """
        if not isinstance(df.index, pd.DatetimeIndex):
            logger.warning("DataFrame index is not DatetimeIndex, skipping time features")
            return df

        # Hour of day (cyclical encoding)
        hour = df.index.hour
        df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
        df["hour_cos"] = np.cos(2 * np.pi * hour / 24)

        # Day of week (cyclical encoding)
        dow = df.index.dayofweek
        df["dow_sin"] = np.sin(2 * np.pi * dow / 7)
        df["dow_cos"] = np.cos(2 * np.pi * dow / 7)

        # Trading sessions (UTC times, adjust for your data timezone)
        df["is_asian"] = ((hour >= 0) & (hour < 8)).astype(int)
        df["is_london"] = ((hour >= 8) & (hour < 16)).astype(int)
        df["is_newyork"] = ((hour >= 13) & (hour < 22)).astype(int)
        df["is_overlap"] = ((hour >= 13) & (hour < 16)).astype(int)  # London-NY overlap

        # Session open indicators (first bar of session)
        df["london_open"] = ((hour == 8) & (df.index.minute == 0)).astype(int)
        df["ny_open"] = ((hour == 13) & (df.index.minute == 0)).astype(int)

        # Week position (Monday=0, Friday=4)
        df["week_position"] = dow / 4.0  # 0 to 1 scale

        # Month position (cyclical)
        day_of_month = df.index.day
        df["month_sin"] = np.sin(2 * np.pi * day_of_month / 31)
        df["month_cos"] = np.cos(2 * np.pi * day_of_month / 31)

        return df

    def _add_roc_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add rate-of-change features for key indicators.

        Captures momentum of indicators, not just their values.
        """
        # Key indicators to compute ROC for
        roc_columns = []

        # Find RSI columns
        rsi_cols = [c for c in df.columns if c.startswith("rsi_")]
        for col in rsi_cols:
            df[f"{col}_roc3"] = df[col].diff(3)
            df[f"{col}_roc6"] = df[col].diff(6)
            roc_columns.extend([f"{col}_roc3", f"{col}_roc6"])

        # Find MACD columns
        if "macd" in df.columns:
            df["macd_roc3"] = df["macd"].diff(3)
            df["macd_hist_roc3"] = df["macd_hist"].diff(3) if "macd_hist" in df.columns else 0
            roc_columns.extend(["macd_roc3", "macd_hist_roc3"])

        # ADX rate of change (trend strength change)
        adx_cols = [c for c in df.columns if c.startswith("adx_")]
        for col in adx_cols:
            df[f"{col}_roc3"] = df[col].diff(3)
            roc_columns.append(f"{col}_roc3")

        # ATR rate of change (volatility change)
        atr_cols = [c for c in df.columns if c.startswith("atr_")]
        for col in atr_cols:
            df[f"{col}_roc3"] = df[col].diff(3)
            df[f"{col}_roc6"] = df[col].diff(6)
            roc_columns.extend([f"{col}_roc3", f"{col}_roc6"])

        # Price momentum (acceleration)
        df["price_roc1"] = df["close"].pct_change(1)
        df["price_roc3"] = df["close"].pct_change(3)
        df["price_roc6"] = df["close"].pct_change(6)
        df["price_roc12"] = df["close"].pct_change(12)

        # Acceleration (second derivative)
        df["price_accel"] = df["price_roc3"].diff(3)

        # Volume momentum
        if "volume" in df.columns:
            df["volume_roc3"] = df["volume"].pct_change(3)
            df["volume_roc6"] = df["volume"].pct_change(6)

        return df

    def _add_normalized_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add normalized/relative features.

        Converts absolute indicator values to relative positions.
        """
        # RSI percentile (where is current RSI vs recent history)
        rsi_cols = [c for c in df.columns if c.startswith("rsi_") and not c.endswith("_roc3") and not c.endswith("_roc6")]
        for col in rsi_cols:
            df[f"{col}_pctl"] = df[col].rolling(100).apply(
                lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min() + 1e-10)
                if len(x) > 0 else 0.5,
                raw=False
            )

        # Price position in recent range
        df["price_pctl_20"] = (df["close"] - df["close"].rolling(20).min()) / \
                              (df["close"].rolling(20).max() - df["close"].rolling(20).min() + 1e-10)
        df["price_pctl_50"] = (df["close"] - df["close"].rolling(50).min()) / \
                              (df["close"].rolling(50).max() - df["close"].rolling(50).min() + 1e-10)

        # ATR percentile (volatility regime)
        atr_cols = [c for c in df.columns if c.startswith("atr_") and not c.endswith("_roc")]
        for col in atr_cols:
            df[f"{col}_pctl"] = df[col].rolling(100).apply(
                lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min() + 1e-10)
                if len(x) > 0 else 0.5,
                raw=False
            )

        # Distance from moving averages (normalized by ATR)
        ema_cols = [c for c in df.columns if c.startswith("ema_")]
        atr_col = next((c for c in df.columns if c.startswith("atr_")), None)

        if atr_col:
            for col in ema_cols:
                df[f"dist_{col}"] = (df["close"] - df[col]) / (df[atr_col] + 1e-10)

        # Bollinger Band position (already have bb_pctb but ensure it exists)
        if "bb_pctb_20" not in df.columns and "bb_upper_20" in df.columns:
            df["bb_pctb_20"] = (df["close"] - df["bb_lower_20"]) / \
                               (df["bb_upper_20"] - df["bb_lower_20"] + 1e-10)

        # Z-score of returns
        df["returns_zscore"] = (df["close"].pct_change() - df["close"].pct_change().rolling(50).mean()) / \
                               (df["close"].pct_change().rolling(50).std() + 1e-10)

        return df

    def _add_pattern_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add price action pattern features."""
        high = df["high"]
        low = df["low"]
        close = df["close"]
        open_ = df["open"]

        # Higher highs and lower lows
        df["higher_high"] = ((high > high.shift(1)) & (high.shift(1) > high.shift(2))).astype(int)
        df["lower_low"] = ((low < low.shift(1)) & (low.shift(1) < low.shift(2))).astype(int)
        df["higher_low"] = ((low > low.shift(1)) & (low.shift(1) > low.shift(2))).astype(int)
        df["lower_high"] = ((high < high.shift(1)) & (high.shift(1) < high.shift(2))).astype(int)

        # Trend structure
        df["uptrend_structure"] = (df["higher_high"] & df["higher_low"]).astype(int)
        df["downtrend_structure"] = (df["lower_low"] & df["lower_high"]).astype(int)

        # Inside/outside bars
        df["inside_bar"] = ((high < high.shift(1)) & (low > low.shift(1))).astype(int)
        df["outside_bar"] = ((high > high.shift(1)) & (low < low.shift(1))).astype(int)

        # Candle patterns
        body = close - open_
        range_ = high - low
        upper_shadow = high - pd.concat([open_, close], axis=1).max(axis=1)
        lower_shadow = pd.concat([open_, close], axis=1).min(axis=1) - low

        # Doji (small body relative to range)
        df["doji"] = (abs(body) < range_ * 0.1).astype(int)

        # Hammer/Hanging man (small body, long lower shadow)
        df["hammer"] = ((abs(body) < range_ * 0.3) & (lower_shadow > abs(body) * 2)).astype(int)

        # Shooting star/Inverted hammer (small body, long upper shadow)
        df["shooting_star"] = ((abs(body) < range_ * 0.3) & (upper_shadow > abs(body) * 2)).astype(int)

        # Engulfing patterns
        df["bullish_engulf"] = ((body > 0) & (body.shift(1) < 0) &
                                (close > open_.shift(1)) & (open_ < close.shift(1))).astype(int)
        df["bearish_engulf"] = ((body < 0) & (body.shift(1) > 0) &
                                (close < open_.shift(1)) & (open_ > close.shift(1))).astype(int)

        # Momentum bars (large body relative to recent average)
        avg_body = abs(body).rolling(20).mean()
        df["strong_bullish"] = ((body > 0) & (body > avg_body * 2)).astype(int)
        df["strong_bearish"] = ((body < 0) & (abs(body) > avg_body * 2)).astype(int)

        # Consecutive direction
        df["consec_up"] = (body > 0).astype(int).rolling(3).sum()
        df["consec_down"] = (body < 0).astype(int).rolling(3).sum()

        return df

    def _add_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add lagged versions of key features.

        This helps tree-based models capture temporal patterns.
        """
        # Key features to lag
        lag_cols = []

        # Lag returns
        if "returns" in df.columns:
            lag_cols.append("returns")
        else:
            df["returns"] = df["close"].pct_change()
            lag_cols.append("returns")

        # Lag RSI
        rsi_cols = [c for c in df.columns if c.startswith("rsi_") and "_roc" not in c and "_pctl" not in c]
        lag_cols.extend(rsi_cols[:1])  # Just first RSI to avoid too many features

        # Lag MACD histogram
        if "macd_hist" in df.columns:
            lag_cols.append("macd_hist")

        # Create lags
        for col in lag_cols:
            if col in df.columns:
                for lag in self.lag_periods:
                    df[f"{col}_lag{lag}"] = df[col].shift(lag)

        return df

    def _add_cross_tf_features(
        self,
        df: pd.DataFrame,
        higher_tf_data: Dict[str, pd.DataFrame],
    ) -> pd.DataFrame:
        """Add cross-timeframe alignment features.

        Args:
            df: Base timeframe data
            higher_tf_data: Dict mapping timeframe name to DataFrame
        """
        for tf_name, tf_df in higher_tf_data.items():
            prefix = f"htf_{tf_name}"

            # Resample higher TF data to base TF (forward fill)
            # Get trend direction from higher timeframe
            if "ema_20" in tf_df.columns and "ema_50" in tf_df.columns:
                tf_df[f"{prefix}_trend"] = (tf_df["ema_20"] > tf_df["ema_50"]).astype(int)
            elif any(c.startswith("ema_") for c in tf_df.columns):
                ema_cols = sorted([c for c in tf_df.columns if c.startswith("ema_")])
                if len(ema_cols) >= 2:
                    tf_df[f"{prefix}_trend"] = (tf_df[ema_cols[0]] > tf_df[ema_cols[1]]).astype(int)

            # RSI from higher timeframe
            rsi_col = next((c for c in tf_df.columns if c.startswith("rsi_")), None)
            if rsi_col:
                tf_df[f"{prefix}_rsi"] = tf_df[rsi_col]

            # Forward fill to base timeframe
            cols_to_join = [c for c in tf_df.columns if c.startswith(prefix)]
            if cols_to_join:
                htf_features = tf_df[cols_to_join].copy()
                df = df.join(htf_features, how="left")
                for col in cols_to_join:
                    df[col] = df[col].ffill()

        # Calculate alignment scores
        trend_cols = [c for c in df.columns if c.endswith("_trend")]
        if trend_cols and "ema_8" in df.columns and "ema_21" in df.columns:
            base_trend = (df["ema_8"] > df["ema_21"]).astype(int)
            df["trend_alignment"] = base_trend
            for col in trend_cols:
                df["trend_alignment"] += df[col].fillna(0)
            df["trend_alignment"] = df["trend_alignment"] / (len(trend_cols) + 1)

        return df

    def _add_sentiment_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add sentiment-based features.

        When us_only_sentiment=True (recommended):
        - Uses only US sentiment (EPU + VIX) which has daily resolution
        - VIX captures market fear, EPU captures policy uncertainty
        - Both are daily (unlike European EPU which is monthly)

        When us_only_sentiment=False (legacy):
        - Uses pair-specific sentiment
        - Includes country sentiments which may be monthly resolution
        """
        if not isinstance(df.index, pd.DatetimeIndex):
            logger.warning("DataFrame index is not DatetimeIndex, skipping sentiment features")
            return df

        # Check if sentiment file exists
        if not self.sentiment_path.exists():
            logger.warning(f"Sentiment file not found: {self.sentiment_path}, skipping sentiment features")
            return df

        try:
            # Lazy load sentiment components
            if self._sentiment_loader is None:
                from src.features.sentiment.sentiment_loader import SentimentLoader
                from src.features.sentiment.sentiment_features import SentimentFeatureCalculator

                self._sentiment_loader = SentimentLoader(self.sentiment_path)
                self._sentiment_calculator = SentimentFeatureCalculator()
                self._sentiment_loader.load()

            # Align sentiment to price data
            # us_only=True uses daily US sentiment (EPU + VIX) - RECOMMENDED
            df_with_sent = self._sentiment_loader.align_to_price_data(
                df,
                pair=self.trading_pair,
                shift_days=1,  # Avoid look-ahead bias
                include_country_sentiments=not self.us_only_sentiment,
                include_epu=False,
                us_only=self.us_only_sentiment,
            )

            # Find the raw sentiment column
            sent_col = 'sentiment_raw'
            if sent_col not in df_with_sent.columns:
                sent_cols = [c for c in df_with_sent.columns if 'sentiment' in c.lower()]
                if sent_cols:
                    sent_col = sent_cols[0]
                else:
                    logger.warning("No sentiment column found after alignment")
                    return df

            # Calculate derived sentiment features
            df_with_sent = self._sentiment_calculator.calculate_all(
                df_with_sent,
                sentiment_col=sent_col,
                prefix='sentiment'
            )

            if self.us_only_sentiment:
                # US-only mode: Add VIX-specific features if available
                if 'vix_raw' in df_with_sent.columns:
                    # VIX momentum features
                    df_with_sent['vix_roc_3'] = df_with_sent['vix_raw'].pct_change(3)
                    df_with_sent['vix_roc_5'] = df_with_sent['vix_raw'].pct_change(5)
                    df_with_sent['vix_ma_5'] = df_with_sent['vix_raw'].rolling(5).mean()
                    df_with_sent['vix_ma_20'] = df_with_sent['vix_raw'].rolling(20).mean()
                    # VIX regime (high vs low volatility)
                    df_with_sent['vix_regime'] = (df_with_sent['vix_raw'] > df_with_sent['vix_ma_20']).astype(int)
                    # VIX z-score
                    vix_std = df_with_sent['vix_raw'].rolling(50).std()
                    df_with_sent['vix_zscore'] = (df_with_sent['vix_raw'] - df_with_sent['vix_ma_20']) / (vix_std + 1e-10)

                # Cross-sentiment features for US indicators
                us_sent_cols = [c for c in df_with_sent.columns if c.startswith('sent_us') or c.startswith('sent_vix')]
                if len(us_sent_cols) >= 2:
                    df_with_sent = self._sentiment_calculator.calculate_cross_sentiment_features(
                        df_with_sent,
                        sentiment_cols=us_sent_cols,
                        prefix='cross_sent'
                    )
            else:
                # Legacy mode: Add cross-country sentiment features
                country_cols = [c for c in df_with_sent.columns if c.startswith('sent_country_')]
                if len(country_cols) >= 2:
                    df_with_sent = self._sentiment_calculator.calculate_cross_sentiment_features(
                        df_with_sent,
                        sentiment_cols=country_cols,
                        prefix='cross_sent'
                    )

            # Count sentiment features added
            sent_feature_cols = [c for c in df_with_sent.columns
                               if 'sentiment' in c.lower()
                               or 'sent_' in c.lower()
                               or 'cross_sent' in c.lower()
                               or 'vix' in c.lower()]
            n_sent_features = len(sent_feature_cols)

            mode_str = "US-only (EPU+VIX)" if self.us_only_sentiment else "legacy (pair-specific)"
            logger.info(f"Added {n_sent_features} sentiment features ({mode_str})")

            return df_with_sent

        except Exception as e:
            logger.warning(f"Error adding sentiment features: {e}")
            import traceback
            traceback.print_exc()
            return df

    def _add_gdelt_sentiment_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add GDELT hourly news sentiment features.

        GDELT provides hourly sentiment data which is properly aggregated based on timeframe:
        - 1H model: Raw hourly values (perfect match)
        - 4H model: Aggregated with avg, std, trend (4 hours → 3 features per region)
        - Daily model: Aggregated with avg, std, trend (24 hours → 3 features per region)

        This solves the resolution mismatch problem that daily VIX/EPU had with intraday models.
        """
        if not isinstance(df.index, pd.DatetimeIndex):
            logger.warning("DataFrame index is not DatetimeIndex, skipping GDELT features")
            return df

        # Check if GDELT file exists
        if not self.gdelt_path.exists():
            logger.warning(f"GDELT file not found: {self.gdelt_path}, skipping GDELT features")
            return df

        try:
            # Lazy load GDELT loader
            if self._gdelt_loader is None:
                from src.features.sentiment.gdelt_loader import GDELTSentimentLoader
                self._gdelt_loader = GDELTSentimentLoader(self.gdelt_path)
                self._gdelt_loader.load()

            # Align GDELT sentiment to price data with proper aggregation
            df_with_gdelt = self._gdelt_loader.align_to_price_data(
                df,
                timeframe=self.base_timeframe,
                shift_periods=1,  # Avoid look-ahead bias
                include_std=True,
                include_trend=True,
            )

            # Count GDELT features added
            gdelt_cols = [c for c in df_with_gdelt.columns if c.startswith('gdelt_')]
            n_gdelt_features = len(gdelt_cols)

            logger.info(f"Added {n_gdelt_features} GDELT sentiment features (hourly, {self.base_timeframe} aggregation)")

            return df_with_gdelt

        except Exception as e:
            logger.warning(f"Error adding GDELT sentiment features: {e}")
            import traceback
            traceback.print_exc()
            return df


def add_enhanced_features(
    df: pd.DataFrame,
    higher_tf_data: Optional[Dict[str, pd.DataFrame]] = None,
) -> pd.DataFrame:
    """Convenience function to add all enhanced features.

    Args:
        df: Base OHLCV dataframe with technical indicators
        higher_tf_data: Optional dict of higher timeframe data

    Returns:
        DataFrame with enhanced features
    """
    engine = EnhancedFeatureEngine()
    return engine.add_all_features(df, higher_tf_data)
