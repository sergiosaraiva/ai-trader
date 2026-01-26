"""Technical Indicator Calculator with configuration-driven computation.

This module provides a high-level interface for calculating technical indicators
based on trading profile configurations. It integrates with the indicator registry
and supports multi-timeframe indicator generation.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd
import yaml

from .registry import ConfigurableIndicatorEngine, IndicatorRegistry
from .indicators import TechnicalIndicators


logger = logging.getLogger(__name__)


@dataclass
class CalculatorConfig:
    """Configuration for the indicator calculator.

    Attributes:
        model_type: Model type ('short_term', 'medium_term', 'long_term').
        config_path: Path to indicator configuration YAML.
        enabled_categories: List of categories to calculate.
        max_priority: Maximum priority level to include (0=P0 only, 3=all).
        drop_na: Whether to drop rows with NaN after calculation.
        fill_method: Method to fill NaN values ('ffill', 'bfill', 'interpolate', None).
    """

    model_type: str = "medium_term"
    config_path: Optional[Path] = None
    enabled_categories: List[str] = field(
        default_factory=lambda: ["trend", "momentum", "volatility", "volume"]
    )
    max_priority: int = 2  # Include P0, P1, P2 by default
    drop_na: bool = True
    fill_method: Optional[str] = "ffill"


class TechnicalIndicatorCalculator:
    """High-level calculator for technical indicators.

    This class provides a unified interface for:
    - Loading indicator configurations from YAML files
    - Calculating indicators based on model type
    - Multi-timeframe indicator generation
    - Automatic NaN handling

    Example:
        ```python
        # Simple usage with defaults
        calculator = TechnicalIndicatorCalculator(model_type="medium_term")
        df_features = calculator.calculate(df_ohlcv)

        # Custom configuration
        calculator = TechnicalIndicatorCalculator(
            config_path="configs/indicators/short_term_indicators.yaml"
        )
        df_features = calculator.calculate(df_ohlcv)

        # Get feature names for model input
        feature_names = calculator.get_feature_names()
        ```
    """

    # Default configuration paths
    CONFIG_PATHS = {
        "short_term": "configs/indicators/short_term_indicators.yaml",
        "medium_term": "configs/indicators/medium_term_indicators.yaml",
        "long_term": "configs/indicators/long_term_indicators.yaml",
    }

    def __init__(
        self,
        config: Optional[CalculatorConfig] = None,
        model_type: Optional[str] = None,
        config_path: Optional[Union[str, Path]] = None,
        indicator_config: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the calculator.

        Args:
            config: CalculatorConfig instance.
            model_type: Model type (overrides config).
            config_path: Path to YAML config (overrides config).
            indicator_config: Direct indicator configuration dict (overrides file).
        """
        self.config = config or CalculatorConfig()

        # Override from direct parameters
        if model_type:
            self.config.model_type = model_type
        if config_path:
            self.config.config_path = Path(config_path)

        # Load indicator configuration
        self._indicator_config = indicator_config or self._load_config()

        # Initialize engines
        self._registry_engine = ConfigurableIndicatorEngine(self._indicator_config)
        self._legacy_indicators = TechnicalIndicators()

        # Track calculated features
        self._feature_names: List[str] = []
        self._original_columns: List[str] = []

    def _load_config(self) -> Dict[str, Any]:
        """Load indicator configuration from file."""
        # Determine config path
        if self.config.config_path:
            config_path = self.config.config_path
        else:
            config_path = Path(self.CONFIG_PATHS.get(
                self.config.model_type,
                self.CONFIG_PATHS["medium_term"]
            ))

        if not config_path.exists():
            logger.warning(f"Config file not found: {config_path}, using defaults")
            return self._get_default_config()

        try:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)

            # Extract indicators section if present
            if "indicators" in config:
                return config["indicators"]
            return config

        except Exception as e:
            logger.error(f"Failed to load config {config_path}: {e}")
            return self._get_default_config()

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default indicator configuration."""
        return {
            "trend": {
                "sma": {"enabled": True, "periods": [20, 50]},
                "ema": {"enabled": True, "periods": [12, 26]},
            },
            "momentum": {
                "rsi": {"enabled": True, "periods": [14]},
                "macd": {"enabled": True},
            },
            "volatility": {
                "atr": {"enabled": True, "period": 14},
                "bollinger": {"enabled": True, "period": 20},
            },
            "volume": {
                "obv": {"enabled": True},
            },
        }

    def calculate(
        self,
        df: pd.DataFrame,
        *,
        use_registry: bool = True,
        include_derived: bool = True,
        config_override: Optional[Dict[str, Any]] = None,
    ) -> pd.DataFrame:
        """Calculate technical indicators for the DataFrame.

        Args:
            df: OHLCV DataFrame with columns [open, high, low, close, volume].
            use_registry: Use registry-based indicators (vs legacy).
            include_derived: Include derived price features (returns, range, etc.).
            config_override: Override indicator configuration for this call.

        Returns:
            DataFrame with calculated indicators.

        Raises:
            ValueError: If required columns are missing.
        """
        # Validate input
        self._validate_input(df)

        # Store original columns
        self._original_columns = df.columns.tolist()

        # Work with a copy
        result = df.copy()

        # Add derived features first (they may be used by indicators)
        if include_derived:
            result = self._add_derived_features(result)

        # Calculate indicators
        config = config_override or self._indicator_config

        if use_registry:
            result = self._calculate_with_registry(result, config)
        else:
            result = self._calculate_with_legacy(result)

        # Handle NaN values
        result = self._handle_nan(result)

        # Store feature names (exclude original OHLCV)
        self._feature_names = [
            col for col in result.columns
            if col not in ["open", "high", "low", "close", "volume"]
        ]

        logger.info(
            f"Calculated {len(self._feature_names)} features "
            f"({len(result)} rows after NaN handling)"
        )

        return result

    def _validate_input(self, df: pd.DataFrame) -> None:
        """Validate input DataFrame."""
        required = {"open", "high", "low", "close", "volume"}
        columns = set(df.columns.str.lower())

        missing = required - columns
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame must have DatetimeIndex")

        if df.empty:
            raise ValueError("DataFrame is empty")

    def _add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add derived price features."""
        # Returns
        df["returns"] = df["close"].pct_change()
        df["log_returns"] = (df["close"] / df["close"].shift(1)).apply(
            lambda x: np.log(x) if x > 0 else 0
        )

        # Price range features
        df["range"] = df["high"] - df["low"]
        df["body"] = df["close"] - df["open"]
        df["body_pct"] = df["body"] / df["open"]

        # Shadows
        df["upper_shadow"] = df["high"] - df[["open", "close"]].max(axis=1)
        df["lower_shadow"] = df[["open", "close"]].min(axis=1) - df["low"]

        # Relative position
        range_safe = df["range"].replace(0, np.nan)
        df["close_position"] = (df["close"] - df["low"]) / range_safe

        # Gap
        df["gap"] = df["open"] - df["close"].shift(1)
        df["gap_pct"] = df["gap"] / df["close"].shift(1)

        return df

    def _calculate_with_registry(
        self,
        df: pd.DataFrame,
        config: Dict[str, Any],
    ) -> pd.DataFrame:
        """Calculate indicators using registry engine."""
        try:
            result = self._registry_engine.calculate(df, config)
            return result
        except Exception as e:
            logger.error(f"Registry calculation failed: {e}, falling back to legacy")
            return self._calculate_with_legacy(df)

    def _calculate_with_legacy(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate indicators using legacy static methods."""
        try:
            return self._legacy_indicators.calculate_all(df)
        except Exception as e:
            logger.error(f"Legacy calculation failed: {e}")
            return df

    def _handle_nan(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle NaN values according to configuration."""
        if self.config.fill_method:
            if self.config.fill_method == "ffill":
                df = df.ffill().bfill()
            elif self.config.fill_method == "bfill":
                df = df.bfill().ffill()
            elif self.config.fill_method == "interpolate":
                df = df.interpolate(method="linear").ffill().bfill()

        if self.config.drop_na:
            df = df.dropna()

        return df

    def calculate_for_model(
        self,
        df: pd.DataFrame,
        model_type: str,
    ) -> pd.DataFrame:
        """Calculate indicators for a specific model type.

        Args:
            df: OHLCV DataFrame.
            model_type: One of 'short_term', 'medium_term', 'long_term'.

        Returns:
            DataFrame with indicators appropriate for the model.
        """
        config_path = self.CONFIG_PATHS.get(model_type)
        if config_path and Path(config_path).exists():
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
                if "indicators" in config:
                    config = config["indicators"]
        else:
            config = self._get_default_config()

        return self.calculate(df, config_override=config)

    def get_feature_names(self) -> List[str]:
        """Get list of calculated feature names.

        Returns:
            List of feature column names (excluding original OHLCV).
        """
        return self._feature_names.copy()

    def get_feature_count(self) -> int:
        """Get number of calculated features."""
        return len(self._feature_names)

    def get_feature_groups(self) -> Dict[str, List[str]]:
        """Get features organized by group/category.

        Returns:
            Dictionary mapping group names to feature lists.
        """
        groups: Dict[str, List[str]] = {
            "price": [],
            "returns": [],
            "derived": [],
            "trend": [],
            "momentum": [],
            "volatility": [],
            "volume": [],
            "other": [],
        }

        patterns = {
            "price": ["open", "high", "low", "close"],
            "returns": ["returns", "log_returns"],
            "derived": ["range", "body", "shadow", "gap", "position"],
            "trend": ["sma", "ema", "wma", "dema", "tema", "adx", "aroon", "psar", "supertrend", "ichimoku"],
            "momentum": ["rsi", "stoch", "macd", "cci", "mom", "roc", "willr", "mfi", "tsi", "uo"],
            "volatility": ["atr", "natr", "bb_", "kc_", "dc_", "stddev", "volatility"],
            "volume": ["obv", "ad", "cmf", "vwap", "vpt", "emv", "fi", "nvi", "pvi", "volume"],
        }

        for feature in self._feature_names:
            feature_lower = feature.lower()
            matched = False

            for group, group_patterns in patterns.items():
                if any(p in feature_lower for p in group_patterns):
                    groups[group].append(feature)
                    matched = True
                    break

            if not matched:
                groups["other"].append(feature)

        # Remove empty groups
        return {k: v for k, v in groups.items() if v}

    def validate_configuration(self) -> Dict[str, List[str]]:
        """Validate current indicator configuration.

        Returns:
            Dictionary with 'errors' and 'warnings' lists.
        """
        return self._registry_engine.validate_config(self._indicator_config)

    def list_available_indicators(self) -> List[Dict[str, Any]]:
        """List all available indicators in the registry.

        Returns:
            List of indicator metadata dictionaries.
        """
        return self._registry_engine.get_available_indicators()

    def get_config(self) -> Dict[str, Any]:
        """Get current indicator configuration.

        Returns:
            Current configuration dictionary.
        """
        return self._indicator_config.copy()

    def set_config(self, config: Dict[str, Any]) -> None:
        """Update indicator configuration.

        Args:
            config: New configuration dictionary.
        """
        self._indicator_config = config
        self._registry_engine = ConfigurableIndicatorEngine(config)


def calculate_indicators(
    df: pd.DataFrame,
    model_type: str = "medium_term",
) -> pd.DataFrame:
    """Convenience function to calculate indicators.

    Args:
        df: OHLCV DataFrame.
        model_type: Model type ('short_term', 'medium_term', 'long_term').

    Returns:
        DataFrame with calculated indicators.
    """
    calculator = TechnicalIndicatorCalculator(model_type=model_type)
    return calculator.calculate(df)


def get_feature_names(
    model_type: str = "medium_term",
    df: Optional[pd.DataFrame] = None,
) -> List[str]:
    """Get feature names for a model type.

    Args:
        model_type: Model type.
        df: Optional sample DataFrame to calculate features and get exact names.

    Returns:
        List of feature names.
    """
    calculator = TechnicalIndicatorCalculator(model_type=model_type)

    if df is not None:
        calculator.calculate(df)
        return calculator.get_feature_names()

    # Return expected features from config
    return calculator.list_available_indicators()
