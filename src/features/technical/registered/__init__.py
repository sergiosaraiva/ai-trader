"""
Technical Indicators Package - Auto-registered indicators.

All indicators in this package are automatically registered with the
IndicatorRegistry when imported. Each indicator is a decorated function
that self-registers.

To add a new indicator:
1. Create a new file or add to existing category file
2. Use the @indicator decorator with proper metadata
3. The indicator becomes immediately available in configuration

Example:
    from src.features.technical.registry import indicator

    @indicator(
        name="my_indicator",
        category="momentum",
        description="My custom indicator",
        params={"period": {"type": int, "default": 14}},
    )
    def calculate_my_indicator(df, period=14):
        # ... calculation ...
        return df, ["my_indicator_14"]
"""

# Import all indicator modules to trigger registration
from . import momentum
from . import trend
from . import volatility
from . import volume
