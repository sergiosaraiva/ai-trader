"""Pydantic schemas for asset metadata."""

from typing import Optional

from pydantic import BaseModel, Field


class AssetMetadata(BaseModel):
    """Metadata describing an asset's characteristics.

    This schema provides asset-specific formatting information for
    displaying prices, profit/loss, and symbols across different asset types.
    """

    symbol: str = Field(..., description="Trading symbol (e.g., EURUSD, BTC-USD)")
    asset_type: str = Field(
        ...,
        description="Asset type: forex, crypto, stock, commodity, index"
    )
    price_precision: int = Field(
        ...,
        ge=0,
        le=8,
        description="Decimal places for price display (e.g., 5 for forex, 2 for stocks)"
    )
    profit_unit: str = Field(
        ...,
        description="Unit for P&L display: pips, points, ticks, dollars"
    )
    profit_multiplier: float = Field(
        ...,
        gt=0,
        description="Multiplier to convert price delta to profit units (e.g., 10000 for pips)"
    )
    formatted_symbol: str = Field(
        ...,
        description="Human-readable format (e.g., EUR/USD, BTC/USD)"
    )
    base_currency: Optional[str] = Field(
        None,
        description="Base currency for pairs (e.g., EUR in EURUSD)"
    )
    quote_currency: Optional[str] = Field(
        None,
        description="Quote currency for pairs (e.g., USD in EURUSD)"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "symbol": "EURUSD",
                    "asset_type": "forex",
                    "price_precision": 5,
                    "profit_unit": "pips",
                    "profit_multiplier": 10000.0,
                    "formatted_symbol": "EUR/USD",
                    "base_currency": "EUR",
                    "quote_currency": "USD",
                },
                {
                    "symbol": "BTC-USD",
                    "asset_type": "crypto",
                    "price_precision": 8,
                    "profit_unit": "dollars",
                    "profit_multiplier": 1.0,
                    "formatted_symbol": "BTC/USD",
                    "base_currency": "BTC",
                    "quote_currency": "USD",
                },
                {
                    "symbol": "AAPL",
                    "asset_type": "stock",
                    "price_precision": 2,
                    "profit_unit": "points",
                    "profit_multiplier": 100.0,
                    "formatted_symbol": "AAPL",
                    "base_currency": None,
                    "quote_currency": "USD",
                },
            ]
        }
    }
