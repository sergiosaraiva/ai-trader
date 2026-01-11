"""Multi-timeframe data transformation for technical analysis.

This module provides advanced timeframe transformation capabilities,
including proper OHLCV aggregation, multi-timeframe alignment, and
cross-timeframe feature generation.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class TimeframeTransformError(Exception):
    """Exception raised for timeframe transformation errors."""

    pass


@dataclass
class TimeframeConfig:
    """Configuration for a single timeframe.

    Attributes:
        name: Human-readable name (e.g., '1H', '4H', '1D').
        minutes: Number of minutes per candle.
        pandas_freq: Pandas frequency string for resampling.
        input_window: Number of candles for model input.
        prediction_horizons: List of prediction horizons in candles.
    """

    name: str
    minutes: int
    pandas_freq: str
    input_window: int = 100
    prediction_horizons: List[int] = field(default_factory=lambda: [1, 4, 12, 24])


# Standard timeframe configurations
STANDARD_TIMEFRAMES: Dict[str, TimeframeConfig] = {
    "1M": TimeframeConfig("1M", 1, "1min", 60, [1, 5, 15, 30]),
    "5M": TimeframeConfig("5M", 5, "5min", 60, [1, 3, 6, 12]),
    "15M": TimeframeConfig("15M", 15, "15min", 96, [1, 4, 8, 16]),
    "30M": TimeframeConfig("30M", 30, "30min", 96, [1, 2, 4, 8]),
    "1H": TimeframeConfig("1H", 60, "1h", 168, [1, 4, 12, 24]),
    "4H": TimeframeConfig("4H", 240, "4h", 180, [1, 3, 6, 12]),
    "1D": TimeframeConfig("1D", 1440, "1D", 90, [1, 3, 5, 7]),
    "1W": TimeframeConfig("1W", 10080, "1W", 52, [1, 2, 4]),
    "1MO": TimeframeConfig("1MO", 43200, "1ME", 36, [1, 2, 3]),
}


class TimeframeTransformer:
    """Transform OHLCV data between timeframes with proper aggregation.

    Supports:
    - Downsampling (e.g., 5M → 1H)
    - Multi-timeframe alignment for ML features
    - Cross-timeframe feature generation
    - Sliding window with overlap

    Example:
        ```python
        transformer = TimeframeTransformer()

        # Simple resampling
        df_1h = transformer.resample(df_5m, "5M", "1H")

        # Multi-timeframe features
        features = transformer.create_multi_timeframe_features(
            df_5m,
            source_tf="5M",
            target_timeframes=["15M", "1H", "4H"]
        )
        ```
    """

    def __init__(self, timeframes: Optional[Dict[str, TimeframeConfig]] = None):
        """Initialize transformer.

        Args:
            timeframes: Custom timeframe configurations (uses standard if None).
        """
        self.timeframes = timeframes or STANDARD_TIMEFRAMES

    def get_timeframe_config(self, timeframe: str) -> TimeframeConfig:
        """Get configuration for a timeframe.

        Args:
            timeframe: Timeframe string (e.g., '1H', '4H').

        Returns:
            TimeframeConfig for the timeframe.

        Raises:
            TimeframeTransformError: If timeframe not recognized.
        """
        tf_upper = timeframe.upper()
        if tf_upper not in self.timeframes:
            available = ", ".join(self.timeframes.keys())
            raise TimeframeTransformError(
                f"Unknown timeframe: {timeframe}. Available: {available}"
            )
        return self.timeframes[tf_upper]

    def resample(
        self,
        df: pd.DataFrame,
        source_timeframe: str,
        target_timeframe: str,
        *,
        label: str = "right",
        closed: str = "right",
    ) -> pd.DataFrame:
        """Resample OHLCV data to a different timeframe.

        Args:
            df: Source DataFrame with OHLCV data and DatetimeIndex.
            source_timeframe: Source timeframe (e.g., '5M').
            target_timeframe: Target timeframe (e.g., '1H').
            label: Which side of bin interval to label.
            closed: Which side of bin interval is closed.

        Returns:
            Resampled DataFrame.

        Raises:
            TimeframeTransformError: If upsampling attempted or invalid data.
        """
        source_config = self.get_timeframe_config(source_timeframe)
        target_config = self.get_timeframe_config(target_timeframe)

        if target_config.minutes < source_config.minutes:
            raise TimeframeTransformError(
                f"Cannot upsample from {source_timeframe} to {target_timeframe}. "
                "Only downsampling (to larger timeframes) is supported."
            )

        if not isinstance(df.index, pd.DatetimeIndex):
            raise TimeframeTransformError("DataFrame must have DatetimeIndex")

        # Define aggregation rules
        agg_rules = {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }

        # Add spread aggregation if present
        if "spread" in df.columns:
            agg_rules["spread"] = "mean"

        # Filter to only existing columns
        agg_rules = {k: v for k, v in agg_rules.items() if k in df.columns}

        # Resample
        resampled = df.resample(
            target_config.pandas_freq,
            label=label,
            closed=closed,
        ).agg(agg_rules)

        # Remove rows with all NaN (periods with no data)
        resampled = resampled.dropna(how="all")

        logger.debug(
            f"Resampled {len(df)} rows from {source_timeframe} to "
            f"{len(resampled)} rows at {target_timeframe}"
        )

        return resampled

    def align_timeframes(
        self,
        dfs: Dict[str, pd.DataFrame],
        base_timeframe: str,
    ) -> Dict[str, pd.DataFrame]:
        """Align multiple timeframes to a base timeframe's index.

        Higher timeframes are forward-filled to match base timeframe timestamps.
        This ensures no lookahead bias (each timestamp only sees past data).

        Args:
            dfs: Dictionary mapping timeframe names to DataFrames.
            base_timeframe: The smallest timeframe to align to.

        Returns:
            Dictionary of aligned DataFrames.
        """
        if base_timeframe not in dfs:
            raise TimeframeTransformError(
                f"Base timeframe {base_timeframe} not in provided DataFrames"
            )

        base_df = dfs[base_timeframe]
        base_index = base_df.index
        aligned = {base_timeframe: base_df}

        for tf_name, df in dfs.items():
            if tf_name == base_timeframe:
                continue

            # Reindex to base timeframe with forward fill
            # This ensures we use the most recent completed candle
            aligned_df = df.reindex(base_index, method="ffill")
            aligned[tf_name] = aligned_df

        return aligned

    def create_multi_timeframe_features(
        self,
        df: pd.DataFrame,
        source_timeframe: str,
        target_timeframes: List[str],
        *,
        include_original: bool = True,
        prefix_columns: bool = True,
    ) -> pd.DataFrame:
        """Create features from multiple timeframes aligned to source.

        Args:
            df: Source OHLCV DataFrame.
            source_timeframe: Source timeframe.
            target_timeframes: List of larger timeframes to create.
            include_original: Include source timeframe columns.
            prefix_columns: Add timeframe prefix to column names.

        Returns:
            DataFrame with aligned multi-timeframe features.

        Example:
            ```python
            features = transformer.create_multi_timeframe_features(
                df_5m,
                source_tf="5M",
                target_timeframes=["15M", "1H", "4H"]
            )
            # Result has columns like: 5M_close, 15M_close, 1H_close, etc.
            ```
        """
        result_dfs = {}

        # Source timeframe
        source_df = df.copy()
        if prefix_columns:
            source_df.columns = [f"{source_timeframe}_{col}" for col in source_df.columns]
        if include_original:
            result_dfs[source_timeframe] = source_df

        # Create and align higher timeframes
        for target_tf in target_timeframes:
            target_config = self.get_timeframe_config(target_tf)
            source_config = self.get_timeframe_config(source_timeframe)

            if target_config.minutes <= source_config.minutes:
                logger.warning(
                    f"Skipping {target_tf} (not larger than {source_timeframe})"
                )
                continue

            # Resample to target timeframe
            resampled = self.resample(df, source_timeframe, target_tf)

            # Forward fill to source index (no lookahead)
            aligned = resampled.reindex(df.index, method="ffill")

            # Add prefix
            if prefix_columns:
                aligned.columns = [f"{target_tf}_{col}" for col in aligned.columns]

            result_dfs[target_tf] = aligned

        # Combine all timeframes
        result = pd.concat(result_dfs.values(), axis=1)

        # Drop rows where higher timeframes don't have data yet
        result = result.dropna()

        logger.info(
            f"Created multi-timeframe features: {list(result_dfs.keys())}, "
            f"{len(result.columns)} columns, {len(result)} rows"
        )

        return result

    def create_cross_timeframe_features(
        self,
        df: pd.DataFrame,
        source_timeframe: str,
        higher_timeframes: List[str],
    ) -> pd.DataFrame:
        """Create cross-timeframe comparison features.

        Generates features like:
        - Price relative to higher timeframe close
        - Higher timeframe trend direction
        - Timeframe alignment signals

        Args:
            df: Source OHLCV DataFrame.
            source_timeframe: Source timeframe.
            higher_timeframes: List of higher timeframes.

        Returns:
            DataFrame with cross-timeframe features.
        """
        # First create aligned multi-timeframe data
        mtf_df = self.create_multi_timeframe_features(
            df,
            source_timeframe,
            higher_timeframes,
            include_original=True,
            prefix_columns=True,
        )

        result = pd.DataFrame(index=mtf_df.index)
        source_prefix = source_timeframe

        for htf in higher_timeframes:
            htf_prefix = htf

            # Price relative to higher TF close
            if f"{source_prefix}_close" in mtf_df.columns and f"{htf_prefix}_close" in mtf_df.columns:
                result[f"rel_close_{htf}"] = (
                    mtf_df[f"{source_prefix}_close"] / mtf_df[f"{htf_prefix}_close"] - 1
                )

            # Higher TF candle direction
            if f"{htf_prefix}_close" in mtf_df.columns and f"{htf_prefix}_open" in mtf_df.columns:
                result[f"htf_direction_{htf}"] = np.sign(
                    mtf_df[f"{htf_prefix}_close"] - mtf_df[f"{htf_prefix}_open"]
                )

            # Higher TF range (volatility proxy)
            if f"{htf_prefix}_high" in mtf_df.columns and f"{htf_prefix}_low" in mtf_df.columns:
                result[f"htf_range_{htf}"] = (
                    mtf_df[f"{htf_prefix}_high"] - mtf_df[f"{htf_prefix}_low"]
                ) / mtf_df[f"{htf_prefix}_close"]

            # Position within higher TF range
            if all(f"{htf_prefix}_{c}" in mtf_df.columns for c in ["high", "low", "close"]):
                htf_range = mtf_df[f"{htf_prefix}_high"] - mtf_df[f"{htf_prefix}_low"]
                result[f"pos_in_htf_{htf}"] = (
                    mtf_df[f"{source_prefix}_close"] - mtf_df[f"{htf_prefix}_low"]
                ) / htf_range.replace(0, np.nan)

        # Trend alignment (all timeframes moving same direction)
        direction_cols = [col for col in result.columns if col.startswith("htf_direction_")]
        if direction_cols:
            result["trend_alignment"] = result[direction_cols].mean(axis=1)
            result["trend_aligned"] = (result["trend_alignment"].abs() > 0.5).astype(int)

        return result

    def sliding_window_transform(
        self,
        df: pd.DataFrame,
        source_timeframe: str,
        target_timeframe: str,
        *,
        step_size: Optional[int] = None,
    ) -> pd.DataFrame:
        """Transform with sliding window for maximum data utilization.

        Instead of standard resampling which creates non-overlapping windows,
        this creates overlapping windows stepped by source timeframe.

        Args:
            df: Source OHLCV DataFrame.
            source_timeframe: Source timeframe.
            target_timeframe: Target timeframe.
            step_size: Number of source candles between windows (default: 1).

        Returns:
            DataFrame with sliding window aggregates.
        """
        source_config = self.get_timeframe_config(source_timeframe)
        target_config = self.get_timeframe_config(target_timeframe)

        window_size = target_config.minutes // source_config.minutes
        step_size = step_size or 1

        if window_size < 2:
            raise TimeframeTransformError(
                f"Target timeframe must be larger than source for sliding window"
            )

        # Pre-calculate rolling aggregates
        result = pd.DataFrame(index=df.index)

        result["sw_open"] = df["open"].shift(window_size - 1)
        result["sw_high"] = df["high"].rolling(window=window_size).max()
        result["sw_low"] = df["low"].rolling(window=window_size).min()
        result["sw_close"] = df["close"]
        result["sw_volume"] = df["volume"].rolling(window=window_size).sum()

        # Apply step size if > 1
        if step_size > 1:
            result = result.iloc[::step_size]

        return result.dropna()

    def get_required_history(
        self,
        target_timeframe: str,
        num_candles: int,
        source_timeframe: str,
    ) -> int:
        """Calculate required source candles for target timeframe output.

        Args:
            target_timeframe: Desired output timeframe.
            num_candles: Number of target candles needed.
            source_timeframe: Source data timeframe.

        Returns:
            Number of source candles required.
        """
        source_config = self.get_timeframe_config(source_timeframe)
        target_config = self.get_timeframe_config(target_timeframe)

        ratio = target_config.minutes // source_config.minutes
        # Add buffer for incomplete candles
        return num_candles * ratio + ratio

    def validate_timeframe_data(
        self,
        df: pd.DataFrame,
        expected_timeframe: str,
        tolerance: float = 0.1,
    ) -> Tuple[bool, List[str]]:
        """Validate DataFrame matches expected timeframe.

        Args:
            df: DataFrame to validate.
            expected_timeframe: Expected timeframe.
            tolerance: Allowed deviation from expected frequency.

        Returns:
            Tuple of (is_valid, list of issues).
        """
        issues = []
        config = self.get_timeframe_config(expected_timeframe)

        if not isinstance(df.index, pd.DatetimeIndex):
            issues.append("Index is not DatetimeIndex")
            return False, issues

        if len(df) < 2:
            issues.append("Not enough data points")
            return False, issues

        # Check time differences
        time_diffs = df.index.to_series().diff().dropna()
        expected_delta = timedelta(minutes=config.minutes)

        median_diff = time_diffs.median()
        expected_seconds = expected_delta.total_seconds()
        actual_seconds = median_diff.total_seconds()

        deviation = abs(actual_seconds - expected_seconds) / expected_seconds

        if deviation > tolerance:
            issues.append(
                f"Median time difference ({actual_seconds}s) deviates "
                f"{deviation:.1%} from expected ({expected_seconds}s)"
            )

        # Check for gaps
        large_gaps = time_diffs[time_diffs > expected_delta * 2]
        if len(large_gaps) > 0:
            issues.append(f"Found {len(large_gaps)} gaps larger than 2x expected interval")

        return len(issues) == 0, issues


def resample_ohlcv(
    df: pd.DataFrame,
    target_timeframe: str,
    source_timeframe: Optional[str] = None,
) -> pd.DataFrame:
    """Convenience function for simple resampling.

    Args:
        df: Source OHLCV DataFrame.
        target_timeframe: Target timeframe string.
        source_timeframe: Source timeframe (auto-detected if None).

    Returns:
        Resampled DataFrame.
    """
    transformer = TimeframeTransformer()

    # Auto-detect source timeframe if not provided
    if source_timeframe is None:
        if len(df) < 2:
            raise TimeframeTransformError("Need at least 2 rows to detect timeframe")

        time_diff = (df.index[1] - df.index[0]).total_seconds() / 60
        # Find closest standard timeframe
        for tf_name, tf_config in STANDARD_TIMEFRAMES.items():
            if abs(tf_config.minutes - time_diff) < tf_config.minutes * 0.1:
                source_timeframe = tf_name
                break

        if source_timeframe is None:
            raise TimeframeTransformError(
                f"Could not auto-detect source timeframe (interval: {time_diff} minutes)"
            )

    return transformer.resample(df, source_timeframe, target_timeframe)
