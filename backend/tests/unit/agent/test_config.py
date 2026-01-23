"""Unit tests for AgentConfig."""

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

# Load config module directly to avoid importing the full API
agent_path = src_path / "agent"
config_path = agent_path / "config.py"
spec = importlib.util.spec_from_file_location("agent.config", config_path)
config_module = importlib.util.module_from_spec(spec)
sys.modules["agent.config"] = config_module
spec.loader.exec_module(config_module)

AgentConfig = config_module.AgentConfig


class TestAgentConfig:
    """Test AgentConfig class."""

    def test_default_values(self):
        """Test config default values."""
        # Arrange & Act
        config = AgentConfig()

        # Assert
        assert config.mode == "simulation"
        assert config.confidence_threshold == 0.70
        assert config.max_position_size == 0.1
        assert config.use_kelly_sizing is True
        assert config.cycle_interval_seconds == 60
        assert config.health_port == 8002
        assert config.initial_capital == 100000.0
        assert config.mt5_login is None
        assert config.mt5_password is None
        assert config.mt5_server is None

    @patch.dict(
        os.environ,
        {
            "AGENT_MODE": "paper",
            "AGENT_CONFIDENCE_THRESHOLD": "0.75",
            "AGENT_MAX_POSITION_SIZE": "0.2",
            "AGENT_USE_KELLY_SIZING": "false",
            "AGENT_CYCLE_INTERVAL": "30",
            "AGENT_HEALTH_PORT": "8003",
            "AGENT_INITIAL_CAPITAL": "50000.0",
            "DATABASE_URL": "sqlite:///test.db",
        },
    )
    def test_from_env_loads_all_values(self):
        """Test loading all values from environment variables."""
        # Arrange & Act
        config = AgentConfig.from_env()

        # Assert
        assert config.mode == "paper"
        assert config.confidence_threshold == 0.75
        assert config.max_position_size == 0.2
        assert config.use_kelly_sizing is False
        assert config.cycle_interval_seconds == 30
        assert config.health_port == 8003
        assert config.initial_capital == 50000.0
        assert config.database_url == "sqlite:///test.db"

    @patch.dict(
        os.environ,
        {
            "AGENT_MODE": "live",
            "AGENT_MT5_LOGIN": "12345678",
            "AGENT_MT5_PASSWORD": "secret_password",
            "AGENT_MT5_SERVER": "MetaQuotes-Demo",
        },
    )
    def test_from_env_loads_mt5_credentials(self):
        """Test loading MT5 credentials from environment."""
        # Arrange & Act
        config = AgentConfig.from_env()

        # Assert
        assert config.mode == "live"
        assert config.mt5_login == 12345678
        assert config.mt5_password == "secret_password"
        assert config.mt5_server == "MetaQuotes-Demo"

    @patch.dict(os.environ, {}, clear=True)
    def test_from_env_uses_defaults_when_env_vars_missing(self):
        """Test that defaults are used when environment variables are missing."""
        # Arrange & Act
        config = AgentConfig.from_env()

        # Assert
        assert config.mode == "simulation"
        assert config.confidence_threshold == 0.70
        assert config.cycle_interval_seconds == 60

    def test_validate_rejects_invalid_mode(self):
        """Test validation rejects invalid mode."""
        # Arrange
        config = AgentConfig(mode="invalid_mode")

        # Act & Assert
        with pytest.raises(ValueError, match="Invalid mode"):
            config.validate()

    def test_validate_accepts_valid_modes(self):
        """Test validation accepts all valid modes."""
        # Arrange
        valid_modes = ["simulation", "paper", "live"]

        for mode in valid_modes:
            # Act
            if mode == "live":
                # Live mode requires MT5 credentials
                config = AgentConfig(
                    mode=mode,
                    mt5_login=12345678,
                    mt5_password="password",
                    mt5_server="server",
                )
            else:
                config = AgentConfig(mode=mode)

            # Assert - should not raise
            config.validate()

    def test_validate_rejects_confidence_threshold_too_low(self):
        """Test validation rejects confidence threshold below 0."""
        # Arrange
        config = AgentConfig(confidence_threshold=-0.1)

        # Act & Assert
        with pytest.raises(ValueError, match="confidence_threshold must be between"):
            config.validate()

    def test_validate_rejects_confidence_threshold_too_high(self):
        """Test validation rejects confidence threshold above 1."""
        # Arrange
        config = AgentConfig(confidence_threshold=1.5)

        # Act & Assert
        with pytest.raises(ValueError, match="confidence_threshold must be between"):
            config.validate()

    def test_validate_accepts_boundary_confidence_values(self):
        """Test validation accepts boundary confidence values."""
        # Arrange & Act
        config1 = AgentConfig(confidence_threshold=0.0)
        config2 = AgentConfig(confidence_threshold=1.0)

        # Assert - should not raise
        config1.validate()
        config2.validate()

    def test_validate_rejects_negative_max_position_size(self):
        """Test validation rejects negative max position size."""
        # Arrange
        config = AgentConfig(max_position_size=-0.1)

        # Act & Assert
        with pytest.raises(ValueError, match="max_position_size must be positive"):
            config.validate()

    def test_validate_rejects_zero_max_position_size(self):
        """Test validation rejects zero max position size."""
        # Arrange
        config = AgentConfig(max_position_size=0.0)

        # Act & Assert
        with pytest.raises(ValueError, match="max_position_size must be positive"):
            config.validate()

    def test_validate_rejects_cycle_interval_less_than_one(self):
        """Test validation rejects cycle interval less than 1 second."""
        # Arrange
        config = AgentConfig(cycle_interval_seconds=0)

        # Act & Assert
        with pytest.raises(ValueError, match="cycle_interval_seconds must be at least 1"):
            config.validate()

    def test_validate_rejects_invalid_health_port_too_low(self):
        """Test validation rejects health port below 1024."""
        # Arrange
        config = AgentConfig(health_port=1023)

        # Act & Assert
        with pytest.raises(ValueError, match="health_port must be between"):
            config.validate()

    def test_validate_rejects_invalid_health_port_too_high(self):
        """Test validation rejects health port above 65535."""
        # Arrange
        config = AgentConfig(health_port=65536)

        # Act & Assert
        with pytest.raises(ValueError, match="health_port must be between"):
            config.validate()

    def test_validate_requires_mt5_credentials_for_live_mode(self):
        """Test validation requires MT5 credentials in live mode."""
        # Arrange
        config = AgentConfig(mode="live")

        # Act & Assert
        with pytest.raises(ValueError, match="MT5 credentials.*are required for live mode"):
            config.validate()

    def test_validate_accepts_live_mode_with_credentials(self):
        """Test validation accepts live mode with all credentials."""
        # Arrange
        config = AgentConfig(
            mode="live",
            mt5_login=12345678,
            mt5_password="password",
            mt5_server="server",
        )

        # Act & Assert - should not raise
        config.validate()

    def test_validate_rejects_negative_initial_capital(self):
        """Test validation rejects negative initial capital."""
        # Arrange
        config = AgentConfig(initial_capital=-1000.0)

        # Act & Assert
        with pytest.raises(ValueError, match="initial_capital must be positive"):
            config.validate()

    def test_validate_sets_default_database_url_if_missing(self):
        """Test validation sets default database URL if not provided."""
        # Arrange
        config = AgentConfig(database_url=None)

        # Act
        config.validate()

        # Assert
        assert config.database_url is not None
        assert config.database_url.startswith("sqlite:///")
        assert "trading.db" in config.database_url

    def test_to_dict_returns_all_fields(self):
        """Test to_dict returns all configuration fields."""
        # Arrange
        config = AgentConfig(
            mode="paper",
            confidence_threshold=0.75,
            max_position_size=0.2,
        )

        # Act
        config_dict = config.to_dict()

        # Assert
        assert config_dict["mode"] == "paper"
        assert config_dict["confidence_threshold"] == 0.75
        assert config_dict["max_position_size"] == 0.2
        assert "cycle_interval_seconds" in config_dict
        assert "health_port" in config_dict

    def test_to_dict_masks_mt5_password(self):
        """Test to_dict masks MT5 password."""
        # Arrange
        config = AgentConfig(
            mode="live",
            mt5_login=12345678,
            mt5_password="secret_password",
            mt5_server="server",
        )

        # Act
        config_dict = config.to_dict()

        # Assert
        assert config_dict["mt5_password"] == "***MASKED***"
        assert config_dict["mt5_login"] == 12345678
        assert config_dict["mt5_server"] == "server"

    def test_to_dict_does_not_mask_when_password_is_none(self):
        """Test to_dict handles None password without masking."""
        # Arrange
        config = AgentConfig(mt5_password=None)

        # Act
        config_dict = config.to_dict()

        # Assert
        assert config_dict["mt5_password"] is None

    def test_update_from_dict_updates_fields(self):
        """Test update_from_dict updates configuration fields."""
        # Arrange
        config = AgentConfig(mode="simulation", confidence_threshold=0.70)

        # Act
        config.update_from_dict({
            "mode": "paper",
            "confidence_threshold": 0.80,
            "max_position_size": 0.15,
        })

        # Assert
        assert config.mode == "paper"
        assert config.confidence_threshold == 0.80
        assert config.max_position_size == 0.15

    def test_update_from_dict_validates_after_update(self):
        """Test update_from_dict validates after updating."""
        # Arrange
        config = AgentConfig()

        # Act & Assert
        with pytest.raises(ValueError, match="Invalid mode"):
            config.update_from_dict({"mode": "invalid_mode"})

    def test_update_from_dict_rejects_unknown_fields(self):
        """Test update_from_dict rejects unknown fields."""
        # Arrange
        config = AgentConfig()

        # Act & Assert
        with pytest.raises(ValueError, match="Unknown configuration field"):
            config.update_from_dict({"unknown_field": "value"})

    def test_update_from_dict_partial_updates(self):
        """Test update_from_dict can update subset of fields."""
        # Arrange
        config = AgentConfig(mode="simulation", confidence_threshold=0.70)

        # Act
        config.update_from_dict({"confidence_threshold": 0.75})

        # Assert
        assert config.mode == "simulation"  # Unchanged
        assert config.confidence_threshold == 0.75  # Updated

    def test_repr_includes_masked_password(self):
        """Test __repr__ masks password in string representation."""
        # Arrange
        config = AgentConfig(
            mode="live",
            mt5_login=12345678,
            mt5_password="secret_password",
            mt5_server="server",
        )

        # Act
        repr_str = repr(config)

        # Assert
        assert "***MASKED***" in repr_str
        assert "secret_password" not in repr_str

    def test_repr_includes_config_name(self):
        """Test __repr__ includes class name."""
        # Arrange
        config = AgentConfig()

        # Act
        repr_str = repr(config)

        # Assert
        assert "AgentConfig" in repr_str
