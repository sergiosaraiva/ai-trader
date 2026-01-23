"""Agent configuration management.

Loads configuration from environment variables with validation and defaults.
"""

import os
import logging
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


@dataclass
class AgentConfig:
    """Configuration for the AI Trading Agent.

    All fields can be set via environment variables with AGENT_ prefix.
    Example: AGENT_MODE=paper, AGENT_CONFIDENCE_THRESHOLD=0.75
    """

    # Trading mode
    mode: str = "simulation"  # simulation, paper, live

    # Trading parameters
    symbol: str = "EURUSD"  # Trading symbol
    confidence_threshold: float = 0.70  # Minimum confidence to trade
    max_position_size: float = 0.1  # Maximum lot size
    use_kelly_sizing: bool = True  # Use Kelly Criterion for position sizing

    # Agent timing
    cycle_interval_seconds: int = 60  # How often to check for new predictions

    # MT5 credentials (required for live mode)
    mt5_login: Optional[int] = None
    mt5_password: Optional[str] = None
    mt5_server: Optional[str] = None

    # Health server
    health_port: int = 8002

    # Database
    database_url: Optional[str] = None

    # Initial capital (for simulation/paper)
    initial_capital: float = 100000.0

    # Safety settings
    max_consecutive_losses: int = 5
    max_drawdown_percent: float = 10.0
    max_daily_loss_percent: float = 5.0
    enable_model_degradation: bool = False

    # Timeout settings (configurable)
    db_timeout_seconds: float = 10.0  # Database operation timeout
    broker_timeout_seconds: float = 30.0  # Broker API timeout
    reconciliation_timeout_seconds: float = 30.0  # Orphaned trade reconciliation timeout

    # Reconnection settings
    max_reconnect_attempts: int = 5
    initial_reconnect_delay: float = 1.0  # Initial delay in seconds
    max_reconnect_delay: float = 60.0  # Maximum delay in seconds (exponential backoff cap)
    reconnect_backoff_multiplier: float = 2.0  # Exponential backoff multiplier

    # Graceful shutdown settings
    shutdown_timeout_seconds: float = 30.0  # Max time to wait for graceful shutdown
    close_positions_on_shutdown: bool = True  # Whether to close positions on stop

    @classmethod
    def from_env(cls) -> "AgentConfig":
        """Load configuration from environment variables.

        Returns:
            AgentConfig instance with values from environment

        Raises:
            ValueError: If configuration is invalid
        """
        config = cls(
            mode=os.getenv("AGENT_MODE", "simulation"),
            symbol=os.getenv("AGENT_SYMBOL", "EURUSD"),
            confidence_threshold=float(os.getenv("AGENT_CONFIDENCE_THRESHOLD", "0.70")),
            max_position_size=float(os.getenv("AGENT_MAX_POSITION_SIZE", "0.1")),
            use_kelly_sizing=os.getenv("AGENT_USE_KELLY_SIZING", "true").lower() == "true",
            cycle_interval_seconds=int(os.getenv("AGENT_CYCLE_INTERVAL", "60")),
            mt5_login=int(os.getenv("AGENT_MT5_LOGIN")) if os.getenv("AGENT_MT5_LOGIN") else None,
            mt5_password=os.getenv("AGENT_MT5_PASSWORD"),
            mt5_server=os.getenv("AGENT_MT5_SERVER"),
            health_port=int(os.getenv("AGENT_HEALTH_PORT", "8002")),
            database_url=os.getenv("DATABASE_URL"),
            initial_capital=float(os.getenv("AGENT_INITIAL_CAPITAL", "100000.0")),
            max_consecutive_losses=int(os.getenv("AGENT_MAX_CONSECUTIVE_LOSSES", "5")),
            max_drawdown_percent=float(os.getenv("AGENT_MAX_DRAWDOWN_PERCENT", "10.0")),
            max_daily_loss_percent=float(os.getenv("AGENT_MAX_DAILY_LOSS_PERCENT", "5.0")),
            enable_model_degradation=os.getenv("AGENT_ENABLE_MODEL_DEGRADATION", "false").lower() == "true",
            # Timeout settings
            db_timeout_seconds=float(os.getenv("AGENT_DB_TIMEOUT", "10.0")),
            broker_timeout_seconds=float(os.getenv("AGENT_BROKER_TIMEOUT", "30.0")),
            reconciliation_timeout_seconds=float(os.getenv("AGENT_RECONCILIATION_TIMEOUT", "30.0")),
            # Reconnection settings
            max_reconnect_attempts=int(os.getenv("AGENT_MAX_RECONNECT_ATTEMPTS", "5")),
            initial_reconnect_delay=float(os.getenv("AGENT_INITIAL_RECONNECT_DELAY", "1.0")),
            max_reconnect_delay=float(os.getenv("AGENT_MAX_RECONNECT_DELAY", "60.0")),
            reconnect_backoff_multiplier=float(os.getenv("AGENT_RECONNECT_BACKOFF", "2.0")),
            # Graceful shutdown
            shutdown_timeout_seconds=float(os.getenv("AGENT_SHUTDOWN_TIMEOUT", "30.0")),
            close_positions_on_shutdown=os.getenv("AGENT_CLOSE_ON_SHUTDOWN", "true").lower() == "true",
        )

        config.validate()
        return config

    def validate(self) -> None:
        """Validate configuration values.

        Raises:
            ValueError: If any configuration value is invalid
        """
        # Validate mode
        valid_modes = ["simulation", "paper", "live"]
        if self.mode not in valid_modes:
            raise ValueError(
                f"Invalid mode '{self.mode}'. Must be one of: {', '.join(valid_modes)}"
            )

        # Validate confidence threshold
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError(
                f"confidence_threshold must be between 0.0 and 1.0, got {self.confidence_threshold}"
            )

        # Validate max position size
        if self.max_position_size <= 0:
            raise ValueError(
                f"max_position_size must be positive, got {self.max_position_size}"
            )

        # Validate cycle interval
        if self.cycle_interval_seconds < 1:
            raise ValueError(
                f"cycle_interval_seconds must be at least 1, got {self.cycle_interval_seconds}"
            )

        # Validate health port
        if not 1024 <= self.health_port <= 65535:
            raise ValueError(
                f"health_port must be between 1024 and 65535, got {self.health_port}"
            )

        # Validate MT5 credentials for live mode
        if self.mode == "live":
            if not all([self.mt5_login, self.mt5_password, self.mt5_server]):
                raise ValueError(
                    "MT5 credentials (mt5_login, mt5_password, mt5_server) "
                    "are required for live mode"
                )

        # Validate initial capital
        if self.initial_capital <= 0:
            raise ValueError(
                f"initial_capital must be positive, got {self.initial_capital}"
            )

        # Validate database URL
        if not self.database_url:
            # Use default SQLite database if not provided
            db_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                "data", "db", "trading.db"
            )
            self.database_url = f"sqlite:///{db_path}"
            logger.warning(f"No DATABASE_URL provided, using default: {self.database_url}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary (for serialization).

        Returns:
            Dictionary representation of config
        """
        config_dict = asdict(self)
        # Mask sensitive values
        if config_dict.get("mt5_password"):
            config_dict["mt5_password"] = "***MASKED***"
        return config_dict

    def update_from_dict(self, updates: Dict[str, Any]) -> None:
        """Update configuration from dictionary.

        Args:
            updates: Dictionary of field names to new values

        Raises:
            ValueError: If updates contain invalid values
        """
        for key, value in updates.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Unknown configuration field: {key}")

        # Re-validate after updates
        self.validate()

    def __repr__(self) -> str:
        """String representation with masked sensitive values."""
        config_dict = self.to_dict()
        return f"AgentConfig({config_dict})"
