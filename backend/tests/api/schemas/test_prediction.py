"""Tests for prediction schemas focusing on timestamp field validation."""

import pytest
from pydantic import ValidationError


class TestPredictionResponseSchema:
    """Test PredictionResponse schema with timestamp fields."""

    def test_prediction_response_with_data_timestamp(self):
        """Test PredictionResponse accepts data_timestamp field."""
        from src.api.schemas.prediction import PredictionResponse

        data = {
            "timestamp": "2024-01-15T14:05:32",
            "symbol": "EURUSD",
            "direction": "long",
            "confidence": 0.75,
            "prob_up": 0.75,
            "prob_down": 0.25,
            "should_trade": True,
            "agreement_count": 3,
            "agreement_score": 1.0,
            "all_agree": True,
            "market_regime": "trending",
            "component_directions": {"1H": 1, "4H": 1, "D": 1},
            "component_confidences": {"1H": 0.72, "4H": 0.75, "D": 0.68},
            "component_weights": {"1H": 0.6, "4H": 0.3, "D": 0.1},
            "data_timestamp": "2024-01-15T14:00:00",
        }

        response = PredictionResponse(**data)

        assert response.data_timestamp == "2024-01-15T14:00:00"
        assert response.timestamp == "2024-01-15T14:05:32"

    def test_prediction_response_with_next_prediction_at(self):
        """Test PredictionResponse accepts next_prediction_at field."""
        from src.api.schemas.prediction import PredictionResponse

        data = {
            "timestamp": "2024-01-15T14:05:32",
            "symbol": "EURUSD",
            "direction": "long",
            "confidence": 0.75,
            "prob_up": 0.75,
            "prob_down": 0.25,
            "should_trade": True,
            "agreement_count": 3,
            "agreement_score": 1.0,
            "all_agree": True,
            "market_regime": "trending",
            "component_directions": {"1H": 1, "4H": 1, "D": 1},
            "component_confidences": {"1H": 0.72, "4H": 0.75, "D": 0.68},
            "component_weights": {"1H": 0.6, "4H": 0.3, "D": 0.1},
            "next_prediction_at": "2024-01-15T15:01:00",
        }

        response = PredictionResponse(**data)

        assert response.next_prediction_at == "2024-01-15T15:01:00"

    def test_prediction_response_with_both_new_timestamps(self):
        """Test PredictionResponse accepts both data_timestamp and next_prediction_at."""
        from src.api.schemas.prediction import PredictionResponse

        data = {
            "timestamp": "2024-01-15T14:05:32",
            "symbol": "EURUSD",
            "direction": "long",
            "confidence": 0.75,
            "prob_up": 0.75,
            "prob_down": 0.25,
            "should_trade": True,
            "agreement_count": 3,
            "agreement_score": 1.0,
            "all_agree": True,
            "market_regime": "trending",
            "component_directions": {"1H": 1, "4H": 1, "D": 1},
            "component_confidences": {"1H": 0.72, "4H": 0.75, "D": 0.68},
            "component_weights": {"1H": 0.6, "4H": 0.3, "D": 0.1},
            "data_timestamp": "2024-01-15T14:00:00",
            "next_prediction_at": "2024-01-15T15:01:00",
        }

        response = PredictionResponse(**data)

        assert response.data_timestamp == "2024-01-15T14:00:00"
        assert response.next_prediction_at == "2024-01-15T15:01:00"
        assert response.timestamp == "2024-01-15T14:05:32"

    def test_prediction_response_data_timestamp_optional(self):
        """Test data_timestamp is optional (None is valid)."""
        from src.api.schemas.prediction import PredictionResponse

        data = {
            "timestamp": "2024-01-15T14:05:32",
            "symbol": "EURUSD",
            "direction": "long",
            "confidence": 0.75,
            "prob_up": 0.75,
            "prob_down": 0.25,
            "should_trade": True,
            "agreement_count": 3,
            "agreement_score": 1.0,
            "all_agree": True,
            "market_regime": "trending",
            "component_directions": {"1H": 1, "4H": 1, "D": 1},
            "component_confidences": {"1H": 0.72, "4H": 0.75, "D": 0.68},
            "component_weights": {"1H": 0.6, "4H": 0.3, "D": 0.1},
            # No data_timestamp field
        }

        response = PredictionResponse(**data)

        assert response.data_timestamp is None

    def test_prediction_response_next_prediction_at_optional(self):
        """Test next_prediction_at is optional (None is valid)."""
        from src.api.schemas.prediction import PredictionResponse

        data = {
            "timestamp": "2024-01-15T14:05:32",
            "symbol": "EURUSD",
            "direction": "long",
            "confidence": 0.75,
            "prob_up": 0.75,
            "prob_down": 0.25,
            "should_trade": True,
            "agreement_count": 3,
            "agreement_score": 1.0,
            "all_agree": True,
            "market_regime": "trending",
            "component_directions": {"1H": 1, "4H": 1, "D": 1},
            "component_confidences": {"1H": 0.72, "4H": 0.75, "D": 0.68},
            "component_weights": {"1H": 0.6, "4H": 0.3, "D": 0.1},
            # No next_prediction_at field
        }

        response = PredictionResponse(**data)

        assert response.next_prediction_at is None

    def test_prediction_response_data_timestamp_none_explicit(self):
        """Test data_timestamp can be explicitly None."""
        from src.api.schemas.prediction import PredictionResponse

        data = {
            "timestamp": "2024-01-15T14:05:32",
            "symbol": "EURUSD",
            "direction": "long",
            "confidence": 0.75,
            "prob_up": 0.75,
            "prob_down": 0.25,
            "should_trade": True,
            "agreement_count": 3,
            "agreement_score": 1.0,
            "all_agree": True,
            "market_regime": "trending",
            "component_directions": {"1H": 1, "4H": 1, "D": 1},
            "component_confidences": {"1H": 0.72, "4H": 0.75, "D": 0.68},
            "component_weights": {"1H": 0.6, "4H": 0.3, "D": 0.1},
            "data_timestamp": None,
        }

        response = PredictionResponse(**data)

        assert response.data_timestamp is None

    def test_prediction_response_required_fields_still_enforced(self):
        """Test that required fields are still enforced with new optional fields."""
        from src.api.schemas.prediction import PredictionResponse

        # Missing required field 'direction'
        data = {
            "timestamp": "2024-01-15T14:05:32",
            "symbol": "EURUSD",
            # "direction": "long",  # Missing
            "confidence": 0.75,
            "prob_up": 0.75,
            "prob_down": 0.25,
            "should_trade": True,
            "agreement_count": 3,
            "agreement_score": 1.0,
            "all_agree": True,
            "market_regime": "trending",
            "component_directions": {"1H": 1, "4H": 1, "D": 1},
            "component_confidences": {"1H": 0.72, "4H": 0.75, "D": 0.68},
            "component_weights": {"1H": 0.6, "4H": 0.3, "D": 0.1},
            "data_timestamp": "2024-01-15T14:00:00",
        }

        with pytest.raises(ValidationError) as exc_info:
            PredictionResponse(**data)

        # Verify the error is about missing 'direction'
        errors = exc_info.value.errors()
        assert any(error["loc"] == ("direction",) for error in errors)

    def test_prediction_response_backwards_compatible(self):
        """Test PredictionResponse is backwards compatible (works without new fields)."""
        from src.api.schemas.prediction import PredictionResponse

        # Old format without new timestamp fields
        data = {
            "timestamp": "2024-01-15T14:05:32",
            "symbol": "EURUSD",
            "direction": "long",
            "confidence": 0.75,
            "prob_up": 0.75,
            "prob_down": 0.25,
            "should_trade": True,
            "agreement_count": 3,
            "agreement_score": 1.0,
            "all_agree": True,
            "market_regime": "trending",
            "component_directions": {"1H": 1, "4H": 1, "D": 1},
            "component_confidences": {"1H": 0.72, "4H": 0.75, "D": 0.68},
            "component_weights": {"1H": 0.6, "4H": 0.3, "D": 0.1},
        }

        # Should work fine
        response = PredictionResponse(**data)

        assert response.timestamp == "2024-01-15T14:05:32"
        assert response.data_timestamp is None
        assert response.next_prediction_at is None

    def test_prediction_response_serialization_includes_all_fields(self):
        """Test serialized response includes all timestamp fields."""
        from src.api.schemas.prediction import PredictionResponse

        data = {
            "timestamp": "2024-01-15T14:05:32",
            "symbol": "EURUSD",
            "direction": "long",
            "confidence": 0.75,
            "prob_up": 0.75,
            "prob_down": 0.25,
            "should_trade": True,
            "agreement_count": 3,
            "agreement_score": 1.0,
            "all_agree": True,
            "market_regime": "trending",
            "component_directions": {"1H": 1, "4H": 1, "D": 1},
            "component_confidences": {"1H": 0.72, "4H": 0.75, "D": 0.68},
            "component_weights": {"1H": 0.6, "4H": 0.3, "D": 0.1},
            "data_timestamp": "2024-01-15T14:00:00",
            "next_prediction_at": "2024-01-15T15:01:00",
        }

        response = PredictionResponse(**data)
        serialized = response.model_dump()

        # All timestamp fields should be present
        assert "timestamp" in serialized
        assert "data_timestamp" in serialized
        assert "next_prediction_at" in serialized

        assert serialized["timestamp"] == "2024-01-15T14:05:32"
        assert serialized["data_timestamp"] == "2024-01-15T14:00:00"
        assert serialized["next_prediction_at"] == "2024-01-15T15:01:00"

    def test_prediction_response_json_serialization(self):
        """Test JSON serialization includes timestamp fields."""
        from src.api.schemas.prediction import PredictionResponse
        import json

        data = {
            "timestamp": "2024-01-15T14:05:32",
            "symbol": "EURUSD",
            "direction": "long",
            "confidence": 0.75,
            "prob_up": 0.75,
            "prob_down": 0.25,
            "should_trade": True,
            "agreement_count": 3,
            "agreement_score": 1.0,
            "all_agree": True,
            "market_regime": "trending",
            "component_directions": {"1H": 1, "4H": 1, "D": 1},
            "component_confidences": {"1H": 0.72, "4H": 0.75, "D": 0.68},
            "component_weights": {"1H": 0.6, "4H": 0.3, "D": 0.1},
            "data_timestamp": "2024-01-15T14:00:00",
            "next_prediction_at": "2024-01-15T15:01:00",
        }

        response = PredictionResponse(**data)
        json_str = response.model_dump_json()
        parsed = json.loads(json_str)

        # Verify JSON includes all fields
        assert "timestamp" in parsed
        assert "data_timestamp" in parsed
        assert "next_prediction_at" in parsed

    def test_prediction_response_with_asset_metadata(self):
        """Test PredictionResponse with optional asset_metadata and timestamp fields."""
        from src.api.schemas.prediction import PredictionResponse
        from src.api.schemas.asset import AssetMetadata

        asset_metadata = AssetMetadata(
            asset_type="forex",
            symbol="EURUSD",
            formatted_symbol="EUR/USD",
            price_precision=5,
            price_unit="pips",
            profit_unit="pips",
            profit_multiplier=10000.0,
        )

        data = {
            "timestamp": "2024-01-15T14:05:32",
            "symbol": "EURUSD",
            "direction": "long",
            "confidence": 0.75,
            "prob_up": 0.75,
            "prob_down": 0.25,
            "should_trade": True,
            "agreement_count": 3,
            "agreement_score": 1.0,
            "all_agree": True,
            "market_regime": "trending",
            "component_directions": {"1H": 1, "4H": 1, "D": 1},
            "component_confidences": {"1H": 0.72, "4H": 0.75, "D": 0.68},
            "component_weights": {"1H": 0.6, "4H": 0.3, "D": 0.1},
            "asset_metadata": asset_metadata,
            "data_timestamp": "2024-01-15T14:00:00",
            "next_prediction_at": "2024-01-15T15:01:00",
        }

        response = PredictionResponse(**data)

        assert response.asset_metadata is not None
        assert response.data_timestamp == "2024-01-15T14:00:00"
        assert response.next_prediction_at == "2024-01-15T15:01:00"


class TestPredictionResponseFieldTypes:
    """Test PredictionResponse field types and validation."""

    def test_data_timestamp_string_type(self):
        """Test data_timestamp accepts string type."""
        from src.api.schemas.prediction import PredictionResponse

        data = {
            "timestamp": "2024-01-15T14:05:32",
            "symbol": "EURUSD",
            "direction": "long",
            "confidence": 0.75,
            "prob_up": 0.75,
            "prob_down": 0.25,
            "should_trade": True,
            "agreement_count": 3,
            "agreement_score": 1.0,
            "all_agree": True,
            "market_regime": "trending",
            "component_directions": {"1H": 1},
            "component_confidences": {"1H": 0.72},
            "component_weights": {"1H": 1.0},
            "data_timestamp": "2024-01-15T14:00:00",  # String
        }

        response = PredictionResponse(**data)
        assert isinstance(response.data_timestamp, str)

    def test_next_prediction_at_string_type(self):
        """Test next_prediction_at accepts string type."""
        from src.api.schemas.prediction import PredictionResponse

        data = {
            "timestamp": "2024-01-15T14:05:32",
            "symbol": "EURUSD",
            "direction": "long",
            "confidence": 0.75,
            "prob_up": 0.75,
            "prob_down": 0.25,
            "should_trade": True,
            "agreement_count": 3,
            "agreement_score": 1.0,
            "all_agree": True,
            "market_regime": "trending",
            "component_directions": {"1H": 1},
            "component_confidences": {"1H": 0.72},
            "component_weights": {"1H": 1.0},
            "next_prediction_at": "2024-01-15T15:01:00",  # String
        }

        response = PredictionResponse(**data)
        assert isinstance(response.next_prediction_at, str)

    def test_timestamp_fields_iso_format(self):
        """Test timestamp fields accept ISO format strings."""
        from src.api.schemas.prediction import PredictionResponse

        data = {
            "timestamp": "2024-01-15T14:05:32.123456",  # With microseconds
            "symbol": "EURUSD",
            "direction": "long",
            "confidence": 0.75,
            "prob_up": 0.75,
            "prob_down": 0.25,
            "should_trade": True,
            "agreement_count": 3,
            "agreement_score": 1.0,
            "all_agree": True,
            "market_regime": "trending",
            "component_directions": {"1H": 1},
            "component_confidences": {"1H": 0.72},
            "component_weights": {"1H": 1.0},
            "data_timestamp": "2024-01-15T14:00:00.000000",
            "next_prediction_at": "2024-01-15T15:01:00+00:00",  # With timezone
        }

        response = PredictionResponse(**data)
        assert response.data_timestamp == "2024-01-15T14:00:00.000000"
        assert response.next_prediction_at == "2024-01-15T15:01:00+00:00"
