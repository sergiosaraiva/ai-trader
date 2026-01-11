"""Technical analysis indicators.

This module provides two approaches to technical indicators:

1. **Legacy Static Approach** (TechnicalIndicators class):
   - Fixed indicator set with hardcoded parameters
   - Simple to use: `indicators.calculate_all(df)`

2. **Dynamic Registry Approach** (ConfigurableIndicatorEngine):
   - Configuration-driven indicator selection
   - New indicators auto-register via @indicator decorator
   - Flexible parameters via YAML config
   - Usage: `engine.calculate(df, config)`

For new development, prefer the ConfigurableIndicatorEngine.
"""

# Legacy static indicators (backward compatibility)
from .indicators import TechnicalIndicators
from .trend import TrendIndicators
from .momentum import MomentumIndicators
from .volatility import VolatilityIndicators
from .volume import VolumeIndicators

# New dynamic registry system
from .registry import (
    IndicatorRegistry,
    ConfigurableIndicatorEngine,
    indicator,
    IndicatorDefinition,
    IndicatorParam,
    generate_config_template,
)

# High-level calculator
from .calculator import (
    TechnicalIndicatorCalculator,
    CalculatorConfig,
    calculate_indicators,
    get_feature_names,
)

__all__ = [
    # Legacy
    "TechnicalIndicators",
    "TrendIndicators",
    "MomentumIndicators",
    "VolatilityIndicators",
    "VolumeIndicators",
    # New registry system
    "IndicatorRegistry",
    "ConfigurableIndicatorEngine",
    "indicator",
    "IndicatorDefinition",
    "IndicatorParam",
    "generate_config_template",
    # High-level calculator
    "TechnicalIndicatorCalculator",
    "CalculatorConfig",
    "calculate_indicators",
    "get_feature_names",
]
