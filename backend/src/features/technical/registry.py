"""
Indicator Registry - Dynamic indicator registration and discovery.

This module provides a registry-based architecture for technical indicators:
1. Indicators self-register using the @indicator decorator
2. Configuration drives which indicators are calculated
3. Adding new indicators requires NO changes to existing code

Usage:
    from src.features.technical.registry import indicator, IndicatorRegistry

    @indicator(
        name="rsi",
        category="momentum",
        description="Relative Strength Index",
        params={
            "period": {"type": int, "default": 14, "min": 2, "max": 100},
            "overbought": {"type": float, "default": 70},
            "oversold": {"type": float, "default": 30},
        }
    )
    def calculate_rsi(df, period=14, **kwargs):
        # ... calculation logic ...
        return df, ["rsi_14"]  # Returns df and list of added column names
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union
import importlib
import inspect
import logging
import pkgutil
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class IndicatorParam:
    """Definition of an indicator parameter."""
    name: str
    type: Type
    default: Any
    description: str = ""
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    choices: Optional[List[Any]] = None

    def validate(self, value: Any) -> Any:
        """Validate and coerce parameter value."""
        # Type coercion
        if not isinstance(value, self.type):
            try:
                value = self.type(value)
            except (ValueError, TypeError) as e:
                raise ValueError(f"Parameter '{self.name}' must be {self.type.__name__}: {e}")

        # Range validation
        if self.min_value is not None and value < self.min_value:
            raise ValueError(f"Parameter '{self.name}' must be >= {self.min_value}")
        if self.max_value is not None and value > self.max_value:
            raise ValueError(f"Parameter '{self.name}' must be <= {self.max_value}")

        # Choices validation
        if self.choices is not None and value not in self.choices:
            raise ValueError(f"Parameter '{self.name}' must be one of {self.choices}")

        return value


@dataclass
class IndicatorDefinition:
    """Complete definition of a registered indicator."""
    name: str
    category: str
    description: str
    calculate_fn: Callable
    params: Dict[str, IndicatorParam] = field(default_factory=dict)
    priority: int = 1  # P0=0 (critical), P1=1, P2=2, P3=3
    requires: List[str] = field(default_factory=list)  # Dependencies on other indicators
    output_columns: Optional[List[str]] = None  # Expected output columns (for documentation)

    def get_default_params(self) -> Dict[str, Any]:
        """Get dictionary of default parameter values."""
        return {name: param.default for name, param in self.params.items()}

    def validate_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and fill in missing parameters with defaults."""
        validated = self.get_default_params()
        for name, value in params.items():
            if name in self.params:
                validated[name] = self.params[name].validate(value)
            else:
                # Allow extra params to pass through (for flexibility)
                validated[name] = value
        return validated


class IndicatorRegistry:
    """
    Central registry for all technical indicators.

    This is a singleton that stores all registered indicators and provides
    methods for discovery, validation, and calculation.
    """
    _instance = None
    _indicators: Dict[str, IndicatorDefinition] = {}
    _categories: Dict[str, Set[str]] = {}  # category -> set of indicator names
    _initialized: bool = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def register(cls, definition: IndicatorDefinition) -> None:
        """Register an indicator definition."""
        if definition.name in cls._indicators:
            logger.warning(f"Indicator '{definition.name}' already registered, overwriting")

        cls._indicators[definition.name] = definition

        # Track by category
        if definition.category not in cls._categories:
            cls._categories[definition.category] = set()
        cls._categories[definition.category].add(definition.name)

        logger.debug(f"Registered indicator: {definition.name} ({definition.category})")

    @classmethod
    def get(cls, name: str) -> Optional[IndicatorDefinition]:
        """Get an indicator by name."""
        return cls._indicators.get(name)

    @classmethod
    def get_all(cls) -> Dict[str, IndicatorDefinition]:
        """Get all registered indicators."""
        return cls._indicators.copy()

    @classmethod
    def get_by_category(cls, category: str) -> Dict[str, IndicatorDefinition]:
        """Get all indicators in a category."""
        names = cls._categories.get(category, set())
        return {name: cls._indicators[name] for name in names if name in cls._indicators}

    @classmethod
    def get_categories(cls) -> List[str]:
        """Get list of all categories."""
        return list(cls._categories.keys())

    @classmethod
    def list_indicators(cls) -> List[Dict[str, Any]]:
        """List all indicators with their metadata."""
        return [
            {
                "name": ind.name,
                "category": ind.category,
                "description": ind.description,
                "priority": ind.priority,
                "params": {
                    name: {
                        "type": param.type.__name__,
                        "default": param.default,
                        "description": param.description,
                    }
                    for name, param in ind.params.items()
                },
            }
            for ind in cls._indicators.values()
        ]

    @classmethod
    def discover_indicators(cls, package_path: Optional[str] = None) -> None:
        """
        Auto-discover and import all indicator modules.

        This will import all Python modules in the registered package,
        triggering their @indicator decorators to register them.
        """
        if cls._initialized:
            return

        if package_path is None:
            # Default to the registered directory
            package_path = str(Path(__file__).parent / "registered")

        if not Path(package_path).exists():
            logger.warning(f"Indicators path does not exist: {package_path}")
            return

        # Import all modules in the registered directory
        for module_info in pkgutil.iter_modules([package_path]):
            if not module_info.name.startswith("_"):
                try:
                    module_name = f"src.features.technical.registered.{module_info.name}"
                    importlib.import_module(module_name)
                    logger.debug(f"Loaded indicator module: {module_name}")
                except Exception as e:
                    logger.error(f"Failed to load indicator module {module_info.name}: {e}")

        cls._initialized = True

    @classmethod
    def clear(cls) -> None:
        """Clear all registered indicators (mainly for testing)."""
        cls._indicators.clear()
        cls._categories.clear()
        cls._initialized = False


def indicator(
    name: str,
    category: str,
    description: str = "",
    params: Optional[Dict[str, Dict[str, Any]]] = None,
    priority: int = 1,
    requires: Optional[List[str]] = None,
    output_columns: Optional[List[str]] = None,
) -> Callable:
    """
    Decorator to register a function as a technical indicator.

    The decorated function should have signature:
        def calculate_xxx(df: pd.DataFrame, **params) -> Tuple[pd.DataFrame, List[str]]

    It should return:
        - The DataFrame with new columns added
        - List of column names that were added

    Args:
        name: Unique identifier for the indicator (e.g., "rsi", "macd")
        category: Category (e.g., "momentum", "trend", "volatility", "volume")
        description: Human-readable description
        params: Parameter definitions as dict of dicts:
            {
                "period": {"type": int, "default": 14, "min": 2, "max": 100, "description": "..."},
                "column": {"type": str, "default": "close", "choices": ["open", "high", "low", "close"]},
            }
        priority: P0=0 (critical), P1=1 (important), P2=2 (useful), P3=3 (optional)
        requires: List of indicator names this depends on
        output_columns: List of column names this indicator produces (for documentation)

    Example:
        @indicator(
            name="rsi",
            category="momentum",
            description="Relative Strength Index - momentum oscillator",
            params={
                "period": {"type": int, "default": 14, "min": 2, "max": 100},
            },
            priority=0,
        )
        def calculate_rsi(df, period=14, column="close"):
            # ... calculation ...
            return df, [f"rsi_{period}"]
    """
    def decorator(fn: Callable) -> Callable:
        # Parse parameter definitions
        parsed_params = {}
        if params:
            for param_name, param_def in params.items():
                parsed_params[param_name] = IndicatorParam(
                    name=param_name,
                    type=param_def.get("type", type(param_def.get("default", 0))),
                    default=param_def.get("default"),
                    description=param_def.get("description", ""),
                    min_value=param_def.get("min"),
                    max_value=param_def.get("max"),
                    choices=param_def.get("choices"),
                )

        # Create indicator definition
        definition = IndicatorDefinition(
            name=name,
            category=category,
            description=description,
            calculate_fn=fn,
            params=parsed_params,
            priority=priority,
            requires=requires or [],
            output_columns=output_columns,
        )

        # Register the indicator
        IndicatorRegistry.register(definition)

        return fn

    return decorator


class ConfigurableIndicatorEngine:
    """
    Configuration-driven indicator calculation engine.

    This engine reads indicator configuration from profiles and calculates
    only the enabled indicators with their configured parameters.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the engine with optional configuration.

        Args:
            config: Indicator configuration from profile, structured as:
                {
                    "trend": {
                        "ema": {"enabled": True, "periods": [8, 13, 21]},
                        "sma": {"enabled": False},
                        ...
                    },
                    "momentum": {
                        "rsi": {"enabled": True, "periods": [7, 14]},
                        ...
                    },
                    ...
                }
        """
        self.config = config or {}
        self._calculated_columns: List[str] = []

        # Ensure indicators are discovered
        IndicatorRegistry.discover_indicators()

    def calculate(
        self,
        df: pd.DataFrame,
        config: Optional[Dict[str, Any]] = None,
    ) -> pd.DataFrame:
        """
        Calculate indicators based on configuration.

        Args:
            df: OHLCV DataFrame
            config: Optional config override (uses self.config if not provided)

        Returns:
            DataFrame with calculated indicators added
        """
        config = config or self.config
        result = df.copy()
        self._calculated_columns = []

        # Process each category
        for category, indicators in config.items():
            if not isinstance(indicators, dict):
                continue

            for indicator_name, indicator_config in indicators.items():
                if not isinstance(indicator_config, dict):
                    continue

                # Check if enabled
                if not indicator_config.get("enabled", True):
                    continue

                # Get indicator definition
                indicator_def = IndicatorRegistry.get(indicator_name)
                if indicator_def is None:
                    logger.warning(f"Unknown indicator: {indicator_name}")
                    continue

                # Calculate with config params
                result, columns = self._calculate_indicator(
                    result, indicator_def, indicator_config
                )
                self._calculated_columns.extend(columns)

        return result

    def _calculate_indicator(
        self,
        df: pd.DataFrame,
        indicator_def: IndicatorDefinition,
        config: Dict[str, Any],
    ) -> Tuple[pd.DataFrame, List[str]]:
        """Calculate a single indicator with its configuration."""
        # Handle 'periods' as multiple calls with different 'period' values
        if "periods" in config:
            periods = config.pop("periods")
            all_columns = []
            for period in periods:
                params = indicator_def.validate_params({**config, "period": period})
                try:
                    df, columns = indicator_def.calculate_fn(df, **params)
                    all_columns.extend(columns)
                except Exception as e:
                    logger.error(f"Error calculating {indicator_def.name}: {e}")
            config["periods"] = periods  # Restore
            return df, all_columns

        # Standard single calculation
        params = indicator_def.validate_params(config)
        try:
            df, columns = indicator_def.calculate_fn(df, **params)
            return df, columns
        except Exception as e:
            logger.error(f"Error calculating {indicator_def.name}: {e}")
            return df, []

    def get_calculated_columns(self) -> List[str]:
        """Get list of columns added by last calculation."""
        return self._calculated_columns.copy()

    def get_available_indicators(self) -> List[Dict[str, Any]]:
        """Get list of all available indicators."""
        return IndicatorRegistry.list_indicators()

    def validate_config(self, config: Dict[str, Any]) -> Dict[str, List[str]]:
        """
        Validate a configuration and return any errors/warnings.

        Returns:
            Dict with "errors" and "warnings" lists
        """
        errors = []
        warnings = []

        for category, indicators in config.items():
            if not isinstance(indicators, dict):
                warnings.append(f"Category '{category}' is not a dict")
                continue

            for indicator_name, indicator_config in indicators.items():
                if not isinstance(indicator_config, dict):
                    continue

                indicator_def = IndicatorRegistry.get(indicator_name)
                if indicator_def is None:
                    errors.append(f"Unknown indicator: {indicator_name}")
                    continue

                # Validate parameters
                try:
                    # Check for unknown params
                    known_params = set(indicator_def.params.keys()) | {"enabled", "periods", "priority"}
                    for param in indicator_config:
                        if param not in known_params:
                            warnings.append(f"{indicator_name}: unknown param '{param}'")

                    # Validate known params
                    indicator_def.validate_params(indicator_config)
                except ValueError as e:
                    errors.append(f"{indicator_name}: {e}")

        return {"errors": errors, "warnings": warnings}


def generate_config_template() -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
    Generate a configuration template with all available indicators.

    Returns:
        Nested dict structure ready to be saved as YAML
    """
    template = {}

    for indicator in IndicatorRegistry.get_all().values():
        if indicator.category not in template:
            template[indicator.category] = {}

        indicator_config = {
            "enabled": True,
            "priority": f"P{indicator.priority}",
        }

        # Add default parameters
        for param_name, param in indicator.params.items():
            indicator_config[param_name] = param.default

        # Add description as comment (will need YAML processing)
        indicator_config["_description"] = indicator.description

        template[indicator.category][indicator.name] = indicator_config

    return template
