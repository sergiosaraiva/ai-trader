"""Tests for AssetMetadata Pydantic schema validation."""

import pytest
from pydantic import ValidationError
from src.api.schemas.asset import AssetMetadata


class TestAssetMetadata:
    """Test AssetMetadata schema validation."""

    # Valid Schema Tests

    def test_valid_forex_metadata(self):
        """Test creating valid forex metadata."""
        metadata = AssetMetadata(
            symbol="EURUSD",
            asset_type="forex",
            price_precision=5,
            profit_unit="pips",
            profit_multiplier=10000.0,
            formatted_symbol="EUR/USD",
            base_currency="EUR",
            quote_currency="USD",
        )

        assert metadata.symbol == "EURUSD"
        assert metadata.asset_type == "forex"
        assert metadata.price_precision == 5
        assert metadata.profit_unit == "pips"
        assert metadata.profit_multiplier == 10000.0
        assert metadata.formatted_symbol == "EUR/USD"
        assert metadata.base_currency == "EUR"
        assert metadata.quote_currency == "USD"

    def test_valid_crypto_metadata(self):
        """Test creating valid crypto metadata."""
        metadata = AssetMetadata(
            symbol="BTC-USD",
            asset_type="crypto",
            price_precision=8,
            profit_unit="dollars",
            profit_multiplier=1.0,
            formatted_symbol="BTC/USD",
            base_currency="BTC",
            quote_currency="USD",
        )

        assert metadata.asset_type == "crypto"
        assert metadata.price_precision == 8
        assert metadata.profit_unit == "dollars"

    def test_valid_stock_metadata(self):
        """Test creating valid stock metadata."""
        metadata = AssetMetadata(
            symbol="AAPL",
            asset_type="stock",
            price_precision=2,
            profit_unit="points",
            profit_multiplier=100.0,
            formatted_symbol="AAPL",
            base_currency=None,
            quote_currency="USD",
        )

        assert metadata.asset_type == "stock"
        assert metadata.price_precision == 2
        assert metadata.base_currency is None

    # Optional Fields Tests

    def test_optional_base_currency_none(self):
        """Test base_currency can be None (optional)."""
        metadata = AssetMetadata(
            symbol="AAPL",
            asset_type="stock",
            price_precision=2,
            profit_unit="points",
            profit_multiplier=100.0,
            formatted_symbol="AAPL",
            quote_currency="USD",
        )

        assert metadata.base_currency is None

    def test_optional_quote_currency_none(self):
        """Test quote_currency can be None (optional)."""
        metadata = AssetMetadata(
            symbol="GOLD",
            asset_type="commodity",
            price_precision=2,
            profit_unit="ticks",
            profit_multiplier=100.0,
            formatted_symbol="GOLD",
        )

        assert metadata.quote_currency is None

    def test_both_currencies_optional(self):
        """Test both base and quote currencies can be None."""
        metadata = AssetMetadata(
            symbol="INDEX",
            asset_type="index",
            price_precision=2,
            profit_unit="points",
            profit_multiplier=1.0,
            formatted_symbol="INDEX",
        )

        assert metadata.base_currency is None
        assert metadata.quote_currency is None

    # Required Fields Validation

    def test_missing_symbol_raises_error(self):
        """Test missing symbol raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            AssetMetadata(
                asset_type="forex",
                price_precision=5,
                profit_unit="pips",
                profit_multiplier=10000.0,
                formatted_symbol="EUR/USD",
            )

        errors = exc_info.value.errors()
        assert any(e["loc"] == ("symbol",) for e in errors)

    def test_missing_asset_type_raises_error(self):
        """Test missing asset_type raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            AssetMetadata(
                symbol="EURUSD",
                price_precision=5,
                profit_unit="pips",
                profit_multiplier=10000.0,
                formatted_symbol="EUR/USD",
            )

        errors = exc_info.value.errors()
        assert any(e["loc"] == ("asset_type",) for e in errors)

    def test_missing_price_precision_raises_error(self):
        """Test missing price_precision raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            AssetMetadata(
                symbol="EURUSD",
                asset_type="forex",
                profit_unit="pips",
                profit_multiplier=10000.0,
                formatted_symbol="EUR/USD",
            )

        errors = exc_info.value.errors()
        assert any(e["loc"] == ("price_precision",) for e in errors)

    def test_missing_profit_unit_raises_error(self):
        """Test missing profit_unit raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            AssetMetadata(
                symbol="EURUSD",
                asset_type="forex",
                price_precision=5,
                profit_multiplier=10000.0,
                formatted_symbol="EUR/USD",
            )

        errors = exc_info.value.errors()
        assert any(e["loc"] == ("profit_unit",) for e in errors)

    def test_missing_profit_multiplier_raises_error(self):
        """Test missing profit_multiplier raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            AssetMetadata(
                symbol="EURUSD",
                asset_type="forex",
                price_precision=5,
                profit_unit="pips",
                formatted_symbol="EUR/USD",
            )

        errors = exc_info.value.errors()
        assert any(e["loc"] == ("profit_multiplier",) for e in errors)

    def test_missing_formatted_symbol_raises_error(self):
        """Test missing formatted_symbol raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            AssetMetadata(
                symbol="EURUSD",
                asset_type="forex",
                price_precision=5,
                profit_unit="pips",
                profit_multiplier=10000.0,
            )

        errors = exc_info.value.errors()
        assert any(e["loc"] == ("formatted_symbol",) for e in errors)

    # Field Validation Tests

    def test_price_precision_negative_raises_error(self):
        """Test negative price_precision raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            AssetMetadata(
                symbol="EURUSD",
                asset_type="forex",
                price_precision=-1,
                profit_unit="pips",
                profit_multiplier=10000.0,
                formatted_symbol="EUR/USD",
            )

        errors = exc_info.value.errors()
        assert any(e["loc"] == ("price_precision",) for e in errors)

    def test_price_precision_too_high_raises_error(self):
        """Test price_precision > 8 raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            AssetMetadata(
                symbol="EURUSD",
                asset_type="forex",
                price_precision=9,
                profit_unit="pips",
                profit_multiplier=10000.0,
                formatted_symbol="EUR/USD",
            )

        errors = exc_info.value.errors()
        assert any(e["loc"] == ("price_precision",) for e in errors)

    def test_price_precision_boundary_values(self):
        """Test price_precision boundary values (0 and 8) are valid."""
        # Test 0
        metadata0 = AssetMetadata(
            symbol="TEST",
            asset_type="index",
            price_precision=0,
            profit_unit="points",
            profit_multiplier=1.0,
            formatted_symbol="TEST",
        )
        assert metadata0.price_precision == 0

        # Test 8
        metadata8 = AssetMetadata(
            symbol="TEST",
            asset_type="crypto",
            price_precision=8,
            profit_unit="dollars",
            profit_multiplier=1.0,
            formatted_symbol="TEST",
        )
        assert metadata8.price_precision == 8

    def test_profit_multiplier_zero_raises_error(self):
        """Test profit_multiplier = 0 raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            AssetMetadata(
                symbol="EURUSD",
                asset_type="forex",
                price_precision=5,
                profit_unit="pips",
                profit_multiplier=0.0,
                formatted_symbol="EUR/USD",
            )

        errors = exc_info.value.errors()
        assert any(e["loc"] == ("profit_multiplier",) for e in errors)

    def test_profit_multiplier_negative_raises_error(self):
        """Test negative profit_multiplier raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            AssetMetadata(
                symbol="EURUSD",
                asset_type="forex",
                price_precision=5,
                profit_unit="pips",
                profit_multiplier=-10000.0,
                formatted_symbol="EUR/USD",
            )

        errors = exc_info.value.errors()
        assert any(e["loc"] == ("profit_multiplier",) for e in errors)

    def test_profit_multiplier_positive_values(self):
        """Test various positive profit_multiplier values."""
        for multiplier in [0.1, 1.0, 100.0, 10000.0]:
            metadata = AssetMetadata(
                symbol="TEST",
                asset_type="forex",
                price_precision=5,
                profit_unit="pips",
                profit_multiplier=multiplier,
                formatted_symbol="TEST",
            )
            assert metadata.profit_multiplier == multiplier

    # Data Type Validation

    def test_string_fields_accept_strings(self):
        """Test string fields accept string values."""
        metadata = AssetMetadata(
            symbol="TEST",
            asset_type="forex",
            price_precision=5,
            profit_unit="pips",
            profit_multiplier=10000.0,
            formatted_symbol="TEST/USD",
            base_currency="TEST",
            quote_currency="USD",
        )

        assert isinstance(metadata.symbol, str)
        assert isinstance(metadata.asset_type, str)
        assert isinstance(metadata.profit_unit, str)
        assert isinstance(metadata.formatted_symbol, str)

    def test_numeric_fields_accept_numbers(self):
        """Test numeric fields accept numeric values."""
        metadata = AssetMetadata(
            symbol="TEST",
            asset_type="forex",
            price_precision=5,
            profit_unit="pips",
            profit_multiplier=10000.0,
            formatted_symbol="TEST",
        )

        assert isinstance(metadata.price_precision, int)
        assert isinstance(metadata.profit_multiplier, float)

    # JSON Serialization Tests

    def test_model_dump_json(self):
        """Test model can be serialized to JSON."""
        metadata = AssetMetadata(
            symbol="EURUSD",
            asset_type="forex",
            price_precision=5,
            profit_unit="pips",
            profit_multiplier=10000.0,
            formatted_symbol="EUR/USD",
            base_currency="EUR",
            quote_currency="USD",
        )

        json_str = metadata.model_dump_json()
        assert isinstance(json_str, str)
        assert "EURUSD" in json_str
        assert "forex" in json_str

    def test_model_dump_dict(self):
        """Test model can be converted to dictionary."""
        metadata = AssetMetadata(
            symbol="EURUSD",
            asset_type="forex",
            price_precision=5,
            profit_unit="pips",
            profit_multiplier=10000.0,
            formatted_symbol="EUR/USD",
            base_currency="EUR",
            quote_currency="USD",
        )

        data = metadata.model_dump()
        assert isinstance(data, dict)
        assert data["symbol"] == "EURUSD"
        assert data["asset_type"] == "forex"
        assert data["price_precision"] == 5

    def test_model_from_dict(self):
        """Test model can be created from dictionary."""
        data = {
            "symbol": "EURUSD",
            "asset_type": "forex",
            "price_precision": 5,
            "profit_unit": "pips",
            "profit_multiplier": 10000.0,
            "formatted_symbol": "EUR/USD",
            "base_currency": "EUR",
            "quote_currency": "USD",
        }

        metadata = AssetMetadata(**data)
        assert metadata.symbol == "EURUSD"
        assert metadata.asset_type == "forex"

    # Schema Examples Tests

    def test_schema_has_examples(self):
        """Test schema includes example data."""
        schema = AssetMetadata.model_json_schema()

        assert "examples" in schema
        examples = schema["examples"]
        assert len(examples) >= 3  # forex, crypto, stock

    def test_example_forex_is_valid(self):
        """Test forex example can create valid instance."""
        schema = AssetMetadata.model_json_schema()
        forex_example = schema["examples"][0]

        metadata = AssetMetadata(**forex_example)
        assert metadata.asset_type == "forex"

    def test_example_crypto_is_valid(self):
        """Test crypto example can create valid instance."""
        schema = AssetMetadata.model_json_schema()
        crypto_example = schema["examples"][1]

        metadata = AssetMetadata(**crypto_example)
        assert metadata.asset_type == "crypto"

    def test_example_stock_is_valid(self):
        """Test stock example can create valid instance."""
        schema = AssetMetadata.model_json_schema()
        stock_example = schema["examples"][2]

        metadata = AssetMetadata(**stock_example)
        assert metadata.asset_type == "stock"

    # Edge Cases

    def test_empty_string_symbol(self):
        """Test empty string symbol is allowed (service handles it)."""
        metadata = AssetMetadata(
            symbol="",
            asset_type="forex",
            price_precision=5,
            profit_unit="pips",
            profit_multiplier=10000.0,
            formatted_symbol="",
        )

        assert metadata.symbol == ""

    def test_unicode_in_symbol(self):
        """Test unicode characters in symbol."""
        metadata = AssetMetadata(
            symbol="TEST€",
            asset_type="forex",
            price_precision=5,
            profit_unit="pips",
            profit_multiplier=10000.0,
            formatted_symbol="TEST€/USD",
        )

        assert "€" in metadata.symbol

    def test_very_long_strings(self):
        """Test very long string values are accepted."""
        long_symbol = "A" * 100

        metadata = AssetMetadata(
            symbol=long_symbol,
            asset_type="forex",
            price_precision=5,
            profit_unit="pips",
            profit_multiplier=10000.0,
            formatted_symbol=long_symbol,
        )

        assert len(metadata.symbol) == 100
