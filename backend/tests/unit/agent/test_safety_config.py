"""Unit tests for SafetyConfig."""

import os
import sys
import pytest
from pathlib import Path
from unittest.mock import patch
import importlib.util

# Add src to path
src_path = Path(__file__).parent.parent.parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# Load safety_config module
agent_path = src_path / "agent"
safety_config_path = agent_path / "safety_config.py"
spec = importlib.util.spec_from_file_location("agent.safety_config", safety_config_path)
safety_config_module = importlib.util.module_from_spec(spec)
sys.modules["agent.safety_config"] = safety_config_module
spec.loader.exec_module(safety_config_module)

SafetyConfig = safety_config_module.SafetyConfig


class TestSafetyConfig:
    """Test SafetyConfig class."""

    def test_default_values(self):
        """Test config uses correct default values."""
        # Arrange & Act
        config = SafetyConfig()

        # Assert - Consecutive Loss Breaker
        assert config.max_consecutive_losses == 5
        assert config.consecutive_loss_action == "pause"

        # Assert - Drawdown Breaker
        assert config.max_drawdown_percent == 10.0
        assert config.drawdown_action == "stop"

        # Assert - Daily Loss Limit
        assert config.max_daily_loss_percent == 5.0
        assert config.max_daily_loss_amount == 5000.0
        assert config.daily_loss_action == "pause"

        # Assert - Model Degradation
        assert config.enable_model_degradation is False
        assert config.min_win_rate == 0.45
        assert config.degradation_window == 20

        # Assert - Kill Switch
        assert config.require_token_for_reset is True
        assert config.auto_reset_next_day is True
        assert config.max_disconnection_seconds == 60.0

        # Assert - Trade Limits
        assert config.max_daily_trades == 50
        assert config.max_trades_per_hour == 20

    @patch.dict(
        os.environ,
        {
            "AGENT_SAFETY_MAX_CONSECUTIVE_LOSSES": "3",
            "AGENT_SAFETY_CONSECUTIVE_LOSS_ACTION": "stop",
            "AGENT_SAFETY_MAX_DRAWDOWN_PERCENT": "15.0",
            "AGENT_SAFETY_DRAWDOWN_ACTION": "pause",
            "AGENT_SAFETY_MAX_DAILY_LOSS_PERCENT": "3.0",
            "AGENT_SAFETY_MAX_DAILY_LOSS_AMOUNT": "3000.0",
            "AGENT_SAFETY_DAILY_LOSS_ACTION": "stop",
            "AGENT_SAFETY_ENABLE_MODEL_DEGRADATION": "true",
            "AGENT_SAFETY_MIN_WIN_RATE": "0.50",
            "AGENT_SAFETY_DEGRADATION_WINDOW": "30",
            "AGENT_SAFETY_REQUIRE_TOKEN_FOR_RESET": "false",
            "AGENT_SAFETY_AUTO_RESET_NEXT_DAY": "false",
            "AGENT_SAFETY_MAX_DISCONNECTION_SECONDS": "120.0",
            "AGENT_SAFETY_MAX_DAILY_TRADES": "100",
            "AGENT_SAFETY_MAX_TRADES_PER_HOUR": "30",
        },
    )
    def test_from_env_loads_all_values(self):
        """Test loading all values from environment variables."""
        # Arrange & Act
        config = SafetyConfig.from_env()

        # Assert - all values loaded correctly
        assert config.max_consecutive_losses == 3
        assert config.consecutive_loss_action == "stop"
        assert config.max_drawdown_percent == 15.0
        assert config.drawdown_action == "pause"
        assert config.max_daily_loss_percent == 3.0
        assert config.max_daily_loss_amount == 3000.0
        assert config.daily_loss_action == "stop"
        assert config.enable_model_degradation is True
        assert config.min_win_rate == 0.50
        assert config.degradation_window == 30
        assert config.require_token_for_reset is False
        assert config.auto_reset_next_day is False
        assert config.max_disconnection_seconds == 120.0
        assert config.max_daily_trades == 100
        assert config.max_trades_per_hour == 30

    @patch.dict(os.environ, {}, clear=True)
    def test_from_env_uses_defaults_when_missing(self):
        """Test defaults are used when environment variables missing."""
        # Arrange & Act
        config = SafetyConfig.from_env()

        # Assert - defaults applied
        assert config.max_consecutive_losses == 5
        assert config.max_drawdown_percent == 10.0
        assert config.max_daily_loss_percent == 5.0

    @patch.dict(
        os.environ,
        {
            "AGENT_SAFETY_MAX_CONSECUTIVE_LOSSES": "7",
            "AGENT_SAFETY_MAX_DAILY_TRADES": "25",
        },
    )
    def test_from_env_partial_override(self):
        """Test partial environment override with defaults."""
        # Arrange & Act
        config = SafetyConfig.from_env()

        # Assert - overridden values
        assert config.max_consecutive_losses == 7
        assert config.max_daily_trades == 25

        # Assert - default values unchanged
        assert config.max_drawdown_percent == 10.0
        assert config.max_daily_loss_percent == 5.0

    def test_validate_passes_with_defaults(self):
        """Test validation passes with default config."""
        # Arrange
        config = SafetyConfig()

        # Act & Assert - should not raise
        config.validate()

    def test_validate_max_consecutive_losses_too_low(self):
        """Test validation fails when max_consecutive_losses < 1."""
        # Arrange
        config = SafetyConfig(max_consecutive_losses=0)

        # Act & Assert
        with pytest.raises(ValueError, match="max_consecutive_losses must be at least 1"):
            config.validate()

    def test_validate_invalid_consecutive_loss_action(self):
        """Test validation fails with invalid consecutive_loss_action."""
        # Arrange
        config = SafetyConfig(consecutive_loss_action="invalid")

        # Act & Assert
        with pytest.raises(ValueError, match="consecutive_loss_action must be 'pause' or 'stop'"):
            config.validate()

    def test_validate_max_drawdown_too_low(self):
        """Test validation fails when max_drawdown_percent <= 0."""
        # Arrange
        config = SafetyConfig(max_drawdown_percent=0.0)

        # Act & Assert
        with pytest.raises(ValueError, match="max_drawdown_percent must be between 0 and 100"):
            config.validate()

    def test_validate_max_drawdown_too_high(self):
        """Test validation fails when max_drawdown_percent > 100."""
        # Arrange
        config = SafetyConfig(max_drawdown_percent=150.0)

        # Act & Assert
        with pytest.raises(ValueError, match="max_drawdown_percent must be between 0 and 100"):
            config.validate()

    def test_validate_invalid_drawdown_action(self):
        """Test validation fails with invalid drawdown_action."""
        # Arrange
        config = SafetyConfig(drawdown_action="halt")

        # Act & Assert
        with pytest.raises(ValueError, match="drawdown_action must be 'pause' or 'stop'"):
            config.validate()

    def test_validate_max_daily_loss_percent_too_low(self):
        """Test validation fails when max_daily_loss_percent <= 0."""
        # Arrange
        config = SafetyConfig(max_daily_loss_percent=-1.0)

        # Act & Assert
        with pytest.raises(ValueError, match="max_daily_loss_percent must be between 0 and 100"):
            config.validate()

    def test_validate_max_daily_loss_percent_too_high(self):
        """Test validation fails when max_daily_loss_percent > 100."""
        # Arrange
        config = SafetyConfig(max_daily_loss_percent=200.0)

        # Act & Assert
        with pytest.raises(ValueError, match="max_daily_loss_percent must be between 0 and 100"):
            config.validate()

    def test_validate_max_daily_loss_amount_negative(self):
        """Test validation fails when max_daily_loss_amount <= 0."""
        # Arrange
        config = SafetyConfig(max_daily_loss_amount=-100.0)

        # Act & Assert
        with pytest.raises(ValueError, match="max_daily_loss_amount must be positive"):
            config.validate()

    def test_validate_min_win_rate_too_low(self):
        """Test validation fails when min_win_rate < 0."""
        # Arrange
        config = SafetyConfig(min_win_rate=-0.1)

        # Act & Assert
        with pytest.raises(ValueError, match="min_win_rate must be between 0.0 and 1.0"):
            config.validate()

    def test_validate_min_win_rate_too_high(self):
        """Test validation fails when min_win_rate > 1.0."""
        # Arrange
        config = SafetyConfig(min_win_rate=1.5)

        # Act & Assert
        with pytest.raises(ValueError, match="min_win_rate must be between 0.0 and 1.0"):
            config.validate()

    def test_validate_degradation_window_too_small(self):
        """Test validation fails when degradation_window < 5."""
        # Arrange
        config = SafetyConfig(degradation_window=3)

        # Act & Assert
        with pytest.raises(ValueError, match="degradation_window must be at least 5 trades"):
            config.validate()

    def test_validate_max_daily_trades_too_low(self):
        """Test validation fails when max_daily_trades < 1."""
        # Arrange
        config = SafetyConfig(max_daily_trades=0)

        # Act & Assert
        with pytest.raises(ValueError, match="max_daily_trades must be at least 1"):
            config.validate()

    def test_validate_max_trades_per_hour_too_low(self):
        """Test validation fails when max_trades_per_hour < 1."""
        # Arrange
        config = SafetyConfig(max_trades_per_hour=0)

        # Act & Assert
        with pytest.raises(ValueError, match="max_trades_per_hour must be at least 1"):
            config.validate()

    def test_validate_max_disconnection_seconds_too_low(self):
        """Test validation fails when max_disconnection_seconds < 1.0."""
        # Arrange
        config = SafetyConfig(max_disconnection_seconds=0.5)

        # Act & Assert
        with pytest.raises(ValueError, match="max_disconnection_seconds must be at least 1.0"):
            config.validate()

    def test_validate_boundary_values_pass(self):
        """Test validation passes with boundary values."""
        # Arrange
        config = SafetyConfig(
            max_consecutive_losses=1,
            max_drawdown_percent=0.1,
            max_daily_loss_percent=0.1,
            max_daily_loss_amount=0.01,
            min_win_rate=0.0,
            degradation_window=5,
            max_daily_trades=1,
            max_trades_per_hour=1,
            max_disconnection_seconds=1.0,
        )

        # Act & Assert - should not raise
        config.validate()

    def test_to_dict_returns_all_fields(self):
        """Test to_dict() returns all configuration fields."""
        # Arrange
        config = SafetyConfig(
            max_consecutive_losses=3,
            consecutive_loss_action="stop",
            max_drawdown_percent=12.0,
        )

        # Act
        result = config.to_dict()

        # Assert - all fields present
        assert result["max_consecutive_losses"] == 3
        assert result["consecutive_loss_action"] == "stop"
        assert result["max_drawdown_percent"] == 12.0
        assert result["drawdown_action"] == "stop"
        assert result["max_daily_loss_percent"] == 5.0
        assert result["max_daily_loss_amount"] == 5000.0
        assert result["daily_loss_action"] == "pause"
        assert result["enable_model_degradation"] is False
        assert result["min_win_rate"] == 0.45
        assert result["degradation_window"] == 20
        assert result["require_token_for_reset"] is True
        assert result["auto_reset_next_day"] is True
        assert result["max_disconnection_seconds"] == 60.0
        assert result["max_daily_trades"] == 50
        assert result["max_trades_per_hour"] == 20

    def test_to_dict_is_json_serializable(self):
        """Test to_dict() output can be serialized to JSON."""
        # Arrange
        import json

        config = SafetyConfig()

        # Act
        result = config.to_dict()
        json_str = json.dumps(result)

        # Assert - should not raise
        assert json_str is not None
        assert len(json_str) > 0

    def test_repr_contains_key_values(self):
        """Test __repr__() contains key configuration values."""
        # Arrange
        config = SafetyConfig(
            max_consecutive_losses=7,
            max_drawdown_percent=15.0,
            max_daily_loss_percent=8.0,
            enable_model_degradation=True,
        )

        # Act
        result = repr(config)

        # Assert - contains key values
        assert "SafetyConfig(" in result
        assert "consecutive_losses=7" in result
        assert "drawdown=15.0%" in result
        assert "daily_loss=8.0%" in result
        assert "model_degradation=True" in result

    def test_custom_prefix_from_env(self):
        """Test from_env() with custom prefix."""
        # Arrange
        with patch.dict(
            os.environ,
            {
                "CUSTOM_MAX_CONSECUTIVE_LOSSES": "8",
                "CUSTOM_MAX_DAILY_TRADES": "75",
            },
        ):
            # Act
            config = SafetyConfig.from_env(prefix="CUSTOM_")

            # Assert
            assert config.max_consecutive_losses == 8
            assert config.max_daily_trades == 75

    def test_boolean_env_parsing_true_variations(self):
        """Test boolean parsing from environment handles 'true' variations."""
        # Arrange
        with patch.dict(
            os.environ,
            {
                "AGENT_SAFETY_ENABLE_MODEL_DEGRADATION": "TRUE",
                "AGENT_SAFETY_REQUIRE_TOKEN_FOR_RESET": "True",
                "AGENT_SAFETY_AUTO_RESET_NEXT_DAY": "true",
            },
        ):
            # Act
            config = SafetyConfig.from_env()

            # Assert
            assert config.enable_model_degradation is True
            assert config.require_token_for_reset is True
            assert config.auto_reset_next_day is True

    def test_boolean_env_parsing_false_variations(self):
        """Test boolean parsing from environment handles 'false' variations."""
        # Arrange
        with patch.dict(
            os.environ,
            {
                "AGENT_SAFETY_ENABLE_MODEL_DEGRADATION": "FALSE",
                "AGENT_SAFETY_REQUIRE_TOKEN_FOR_RESET": "False",
                "AGENT_SAFETY_AUTO_RESET_NEXT_DAY": "anything_not_true",
            },
        ):
            # Act
            config = SafetyConfig.from_env()

            # Assert
            assert config.enable_model_degradation is False
            assert config.require_token_for_reset is False
            assert config.auto_reset_next_day is False
