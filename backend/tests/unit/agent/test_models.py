"""Unit tests for agent data models.

Tests CycleResult, PredictionData, and SignalData dataclasses.
"""

import sys
from pathlib import Path
from datetime import datetime
import pytest
import importlib.util

# Add src to path
src_path = Path(__file__).parent.parent.parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# Load models module directly to avoid importing the full agent package
models_path = src_path / "agent" / "models.py"
spec = importlib.util.spec_from_file_location("agent.models", models_path)
models_module = importlib.util.module_from_spec(spec)
sys.modules["agent.models"] = models_module
spec.loader.exec_module(models_module)

CycleResult = models_module.CycleResult
PredictionData = models_module.PredictionData
SignalData = models_module.SignalData


class TestCycleResult:
    """Test CycleResult dataclass."""

    def test_create_with_required_fields(self):
        """Test creating CycleResult with required fields."""
        # Arrange
        cycle_number = 1
        timestamp = datetime.now()

        # Act
        result = CycleResult(cycle_number=cycle_number, timestamp=timestamp)

        # Assert
        assert result.cycle_number == cycle_number
        assert result.timestamp == timestamp
        assert result.prediction_made is False
        assert result.signal_generated is False
        assert result.action_taken == "none"
        assert result.prediction is None
        assert result.signal is None
        assert result.duration_ms == 0.0
        assert result.error is None
        assert result.reason == ""

    def test_create_with_optional_fields(self):
        """Test creating CycleResult with optional fields."""
        # Arrange
        cycle_number = 1
        timestamp = datetime.now()
        prediction = {"direction": "long", "confidence": 0.75}
        signal = {"action": "buy", "confidence": 0.75}

        # Act
        result = CycleResult(
            cycle_number=cycle_number,
            timestamp=timestamp,
            prediction_made=True,
            signal_generated=True,
            action_taken="signal_generated",
            prediction=prediction,
            signal=signal,
            duration_ms=150.5,
            error=None,
            reason="High confidence signal",
        )

        # Assert
        assert result.prediction_made is True
        assert result.signal_generated is True
        assert result.action_taken == "signal_generated"
        assert result.prediction == prediction
        assert result.signal == signal
        assert result.duration_ms == 150.5
        assert result.reason == "High confidence signal"

    def test_create_with_error(self):
        """Test creating CycleResult with error."""
        # Arrange
        cycle_number = 1
        timestamp = datetime.now()
        error_msg = "Model service not available"

        # Act
        result = CycleResult(
            cycle_number=cycle_number,
            timestamp=timestamp,
            error=error_msg,
            reason="Service unavailable",
        )

        # Assert
        assert result.error == error_msg
        assert result.reason == "Service unavailable"
        assert result.success is False

    def test_to_dict_serialization(self):
        """Test to_dict() serialization."""
        # Arrange
        cycle_number = 1
        timestamp = datetime.now()
        result = CycleResult(
            cycle_number=cycle_number,
            timestamp=timestamp,
            prediction_made=True,
            signal_generated=False,
            action_taken="hold",
            duration_ms=125.3,
            reason="Low confidence",
        )

        # Act
        result_dict = result.to_dict()

        # Assert
        assert result_dict["cycle_number"] == cycle_number
        assert result_dict["timestamp"] == timestamp.isoformat()
        assert result_dict["prediction_made"] is True
        assert result_dict["signal_generated"] is False
        assert result_dict["action_taken"] == "hold"
        assert result_dict["duration_ms"] == 125.3
        assert result_dict["error"] is None
        assert result_dict["reason"] == "Low confidence"

    def test_success_property_true_when_no_error(self):
        """Test success property returns True when no error."""
        # Arrange
        result = CycleResult(cycle_number=1, timestamp=datetime.now())

        # Act & Assert
        assert result.success is True

    def test_success_property_false_when_error(self):
        """Test success property returns False when error exists."""
        # Arrange
        result = CycleResult(
            cycle_number=1, timestamp=datetime.now(), error="Something went wrong"
        )

        # Act & Assert
        assert result.success is False

    def test_dataclass_defaults(self):
        """Test dataclass default values."""
        # Arrange & Act
        result = CycleResult(cycle_number=1, timestamp=datetime.now())

        # Assert - verify all defaults
        assert result.prediction_made is False
        assert result.signal_generated is False
        assert result.action_taken == "none"
        assert result.prediction is None
        assert result.signal is None
        assert result.duration_ms == 0.0
        assert result.error is None
        assert result.reason == ""


class TestPredictionData:
    """Test PredictionData dataclass."""

    def test_create_from_service_output(self):
        """Test creating PredictionData from model service output."""
        # Arrange
        service_output = {
            "direction": "long",
            "confidence": 0.72,
            "prob_up": 0.72,
            "prob_down": 0.28,
            "should_trade": True,
            "agreement_count": 3,
            "agreement_score": 1.0,
            "market_regime": "trending",
            "component_directions": {"1H": 1, "4H": 1, "D": 1},
            "component_confidences": {"1H": 0.75, "4H": 0.70, "D": 0.68},
            "timestamp": "2024-01-15T10:00:00",
            "symbol": "EURUSD",
        }

        # Act
        prediction = PredictionData.from_service_output(service_output)

        # Assert
        assert prediction.direction == "long"
        assert prediction.confidence == 0.72
        assert prediction.prob_up == 0.72
        assert prediction.prob_down == 0.28
        assert prediction.should_trade is True
        assert prediction.agreement_count == 3
        assert prediction.agreement_score == 1.0
        assert prediction.market_regime == "trending"
        assert prediction.component_directions == {"1H": 1, "4H": 1, "D": 1}
        assert prediction.component_confidences == {"1H": 0.75, "4H": 0.70, "D": 0.68}
        assert prediction.timestamp == datetime.fromisoformat("2024-01-15T10:00:00")
        assert prediction.symbol == "EURUSD"

    def test_from_service_output_handles_missing_fields(self):
        """Test from_service_output() handles missing fields with defaults."""
        # Arrange
        service_output = {
            "direction": "short",
            "confidence": 0.65,
        }

        # Act
        prediction = PredictionData.from_service_output(service_output)

        # Assert - verify defaults
        assert prediction.direction == "short"
        assert prediction.confidence == 0.65
        assert prediction.prob_up == 0.5
        assert prediction.prob_down == 0.5
        assert prediction.should_trade is False
        assert prediction.agreement_count == 0
        assert prediction.agreement_score == 0.0
        assert prediction.market_regime == "unknown"
        assert prediction.component_directions == {}
        assert prediction.component_confidences == {}
        assert prediction.symbol == "EURUSD"

    def test_from_service_output_raises_on_invalid_format(self):
        """Test from_service_output() raises ValueError on invalid data."""
        # Arrange - missing critical field that causes type error
        service_output = {"timestamp": "invalid-date-format"}

        # Act & Assert
        with pytest.raises(ValueError, match="Invalid prediction output format"):
            PredictionData.from_service_output(service_output)

    def test_to_dict_serialization(self):
        """Test to_dict() serialization."""
        # Arrange
        timestamp = datetime.now()
        prediction = PredictionData(
            direction="long",
            confidence=0.75,
            prob_up=0.75,
            prob_down=0.25,
            should_trade=True,
            agreement_count=2,
            agreement_score=0.67,
            market_regime="ranging",
            component_directions={"1H": 1, "4H": 1},
            component_confidences={"1H": 0.72, "4H": 0.71},
            timestamp=timestamp,
            symbol="EURUSD",
        )

        # Act
        prediction_dict = prediction.to_dict()

        # Assert
        assert prediction_dict["direction"] == "long"
        assert prediction_dict["confidence"] == 0.75
        assert prediction_dict["prob_up"] == 0.75
        assert prediction_dict["prob_down"] == 0.25
        assert prediction_dict["should_trade"] is True
        assert prediction_dict["agreement_count"] == 2
        assert prediction_dict["agreement_score"] == 0.67
        assert prediction_dict["market_regime"] == "ranging"
        assert prediction_dict["component_directions"] == {"1H": 1, "4H": 1}
        assert prediction_dict["component_confidences"] == {"1H": 0.72, "4H": 0.71}
        assert prediction_dict["timestamp"] == timestamp.isoformat()
        assert prediction_dict["symbol"] == "EURUSD"

    def test_all_fields_populated_correctly(self):
        """Test all fields are populated correctly from complete output."""
        # Arrange
        service_output = {
            "direction": "short",
            "confidence": 0.80,
            "prob_up": 0.20,
            "prob_down": 0.80,
            "should_trade": True,
            "agreement_count": 3,
            "agreement_score": 1.0,
            "market_regime": "trending",
            "component_directions": {"1H": -1, "4H": -1, "D": -1},
            "component_confidences": {"1H": 0.82, "4H": 0.78, "D": 0.75},
            "timestamp": "2024-01-15T14:30:00",
            "symbol": "GBPUSD",
        }

        # Act
        prediction = PredictionData.from_service_output(service_output)

        # Assert - verify every field
        assert prediction.direction == "short"
        assert prediction.confidence == 0.80
        assert prediction.prob_up == 0.20
        assert prediction.prob_down == 0.80
        assert prediction.should_trade is True
        assert prediction.agreement_count == 3
        assert prediction.agreement_score == 1.0
        assert prediction.market_regime == "trending"
        assert len(prediction.component_directions) == 3
        assert len(prediction.component_confidences) == 3
        assert isinstance(prediction.timestamp, datetime)
        assert prediction.symbol == "GBPUSD"


class TestSignalData:
    """Test SignalData dataclass."""

    def test_create_with_all_fields(self):
        """Test creating SignalData with all fields."""
        # Arrange
        timestamp = datetime.now()

        # Act
        signal = SignalData(
            action="buy",
            confidence=0.75,
            reason="High confidence long signal",
            position_size_pct=0.05,
            stop_loss_pct=0.02,
            take_profit_pct=0.04,
            timestamp=timestamp,
        )

        # Assert
        assert signal.action == "buy"
        assert signal.confidence == 0.75
        assert signal.reason == "High confidence long signal"
        assert signal.position_size_pct == 0.05
        assert signal.stop_loss_pct == 0.02
        assert signal.take_profit_pct == 0.04
        assert signal.timestamp == timestamp

    def test_to_dict_serialization(self):
        """Test to_dict() serialization."""
        # Arrange
        timestamp = datetime.now()
        signal = SignalData(
            action="sell",
            confidence=0.68,
            reason="Short signal from prediction",
            position_size_pct=0.04,
            stop_loss_pct=0.015,
            take_profit_pct=0.03,
            timestamp=timestamp,
        )

        # Act
        signal_dict = signal.to_dict()

        # Assert
        assert signal_dict["action"] == "sell"
        assert signal_dict["confidence"] == 0.68
        assert signal_dict["reason"] == "Short signal from prediction"
        assert signal_dict["position_size_pct"] == 0.04
        assert signal_dict["stop_loss_pct"] == 0.015
        assert signal_dict["take_profit_pct"] == 0.03
        assert signal_dict["timestamp"] == timestamp.isoformat()

    def test_action_enum_values(self):
        """Test signal action values."""
        # Arrange & Act
        buy_signal = SignalData(action="buy", confidence=0.75, reason="Test")
        sell_signal = SignalData(action="sell", confidence=0.75, reason="Test")
        hold_signal = SignalData(action="hold", confidence=0.50, reason="Test")

        # Assert
        assert buy_signal.action == "buy"
        assert sell_signal.action == "sell"
        assert hold_signal.action == "hold"

    def test_default_values_for_optional_fields(self):
        """Test default values for optional fields."""
        # Arrange & Act
        signal = SignalData(action="buy", confidence=0.75, reason="Test")

        # Assert - verify defaults
        assert signal.position_size_pct == 0.0
        assert signal.stop_loss_pct == 0.0
        assert signal.take_profit_pct == 0.0
        assert signal.timestamp is None

    def test_to_dict_handles_none_timestamp(self):
        """Test to_dict() handles None timestamp."""
        # Arrange
        signal = SignalData(
            action="hold", confidence=0.55, reason="Low confidence", timestamp=None
        )

        # Act
        signal_dict = signal.to_dict()

        # Assert
        assert signal_dict["timestamp"] is None
