"""Safety configuration for the trading agent.

Configures circuit breakers, kill switch, and risk limits to protect capital
from catastrophic losses.
"""

import os
import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class SafetyConfig:
    """Safety configuration with sensible defaults.

    All safety limits are designed to prevent catastrophic losses while
    allowing the model to trade profitably. These defaults match the
    MODERATE risk profile from the backtesting results.
    """

    # Consecutive Loss Breaker
    max_consecutive_losses: int = 5  # Halt after 5 losses in a row
    consecutive_loss_action: str = "pause"  # "pause" or "stop"

    # Drawdown Breaker
    max_drawdown_percent: float = 10.0  # Halt at 10% drawdown from peak
    drawdown_action: str = "stop"  # "stop" immediately on max drawdown

    # Daily Loss Limit (via Kill Switch)
    max_daily_loss_percent: float = 5.0  # Kill switch at 5% daily loss
    max_daily_loss_amount: float = 5000.0  # Or $5000 absolute loss
    daily_loss_action: str = "pause"  # "pause" trading, don't stop agent

    # Model Degradation Breaker (optional)
    enable_model_degradation: bool = False  # Disabled by default
    min_win_rate: float = 0.45  # Trigger if win rate drops below 45%
    degradation_window: int = 20  # Rolling window of trades

    # Kill Switch
    require_token_for_reset: bool = True  # Require authorization code to reset
    auto_reset_next_day: bool = True  # Auto-reset kill switch at new trading day
    max_disconnection_seconds: float = 60.0  # Trigger on 60s MT5 disconnection

    # Daily Trade Limits
    max_daily_trades: int = 50  # Absolute maximum trades per day
    max_trades_per_hour: int = 20  # Maximum trades per hour

    @classmethod
    def from_env(cls, prefix: str = "AGENT_SAFETY_") -> "SafetyConfig":
        """Load safety configuration from environment variables.

        Args:
            prefix: Environment variable prefix (default: AGENT_SAFETY_)

        Returns:
            SafetyConfig instance with values from environment

        Example environment variables:
            AGENT_SAFETY_MAX_CONSECUTIVE_LOSSES=5
            AGENT_SAFETY_MAX_DRAWDOWN_PERCENT=10.0
            AGENT_SAFETY_MAX_DAILY_LOSS_PERCENT=5.0
        """
        config = cls(
            max_consecutive_losses=int(
                os.getenv(f"{prefix}MAX_CONSECUTIVE_LOSSES", "5")
            ),
            consecutive_loss_action=os.getenv(
                f"{prefix}CONSECUTIVE_LOSS_ACTION", "pause"
            ),
            max_drawdown_percent=float(
                os.getenv(f"{prefix}MAX_DRAWDOWN_PERCENT", "10.0")
            ),
            drawdown_action=os.getenv(f"{prefix}DRAWDOWN_ACTION", "stop"),
            max_daily_loss_percent=float(
                os.getenv(f"{prefix}MAX_DAILY_LOSS_PERCENT", "5.0")
            ),
            max_daily_loss_amount=float(
                os.getenv(f"{prefix}MAX_DAILY_LOSS_AMOUNT", "5000.0")
            ),
            daily_loss_action=os.getenv(f"{prefix}DAILY_LOSS_ACTION", "pause"),
            enable_model_degradation=os.getenv(
                f"{prefix}ENABLE_MODEL_DEGRADATION", "false"
            ).lower()
            == "true",
            min_win_rate=float(os.getenv(f"{prefix}MIN_WIN_RATE", "0.45")),
            degradation_window=int(os.getenv(f"{prefix}DEGRADATION_WINDOW", "20")),
            require_token_for_reset=os.getenv(
                f"{prefix}REQUIRE_TOKEN_FOR_RESET", "true"
            ).lower()
            == "true",
            auto_reset_next_day=os.getenv(
                f"{prefix}AUTO_RESET_NEXT_DAY", "true"
            ).lower()
            == "true",
            max_disconnection_seconds=float(
                os.getenv(f"{prefix}MAX_DISCONNECTION_SECONDS", "60.0")
            ),
            max_daily_trades=int(os.getenv(f"{prefix}MAX_DAILY_TRADES", "50")),
            max_trades_per_hour=int(os.getenv(f"{prefix}MAX_TRADES_PER_HOUR", "20")),
        )

        config.validate()
        return config

    def validate(self) -> None:
        """Validate safety configuration values.

        Raises:
            ValueError: If any configuration value is invalid
        """
        # Validate consecutive losses
        if self.max_consecutive_losses < 1:
            raise ValueError(
                f"max_consecutive_losses must be at least 1, got {self.max_consecutive_losses}"
            )

        if self.consecutive_loss_action not in ("pause", "stop"):
            raise ValueError(
                f"consecutive_loss_action must be 'pause' or 'stop', got {self.consecutive_loss_action}"
            )

        # Validate drawdown
        if not 0.0 < self.max_drawdown_percent <= 100.0:
            raise ValueError(
                f"max_drawdown_percent must be between 0 and 100, got {self.max_drawdown_percent}"
            )

        if self.drawdown_action not in ("pause", "stop"):
            raise ValueError(
                f"drawdown_action must be 'pause' or 'stop', got {self.drawdown_action}"
            )

        # Validate daily loss limits
        if not 0.0 < self.max_daily_loss_percent <= 100.0:
            raise ValueError(
                f"max_daily_loss_percent must be between 0 and 100, got {self.max_daily_loss_percent}"
            )

        if self.max_daily_loss_amount <= 0:
            raise ValueError(
                f"max_daily_loss_amount must be positive, got {self.max_daily_loss_amount}"
            )

        # Validate model degradation
        if not 0.0 <= self.min_win_rate <= 1.0:
            raise ValueError(
                f"min_win_rate must be between 0.0 and 1.0, got {self.min_win_rate}"
            )

        if self.degradation_window < 5:
            raise ValueError(
                f"degradation_window must be at least 5 trades, got {self.degradation_window}"
            )

        # Validate trade limits
        if self.max_daily_trades < 1:
            raise ValueError(
                f"max_daily_trades must be at least 1, got {self.max_daily_trades}"
            )

        if self.max_trades_per_hour < 1:
            raise ValueError(
                f"max_trades_per_hour must be at least 1, got {self.max_trades_per_hour}"
            )

        # Validate disconnection timeout
        if self.max_disconnection_seconds < 1.0:
            raise ValueError(
                f"max_disconnection_seconds must be at least 1.0, got {self.max_disconnection_seconds}"
            )

        logger.info(f"Safety config validated: {self}")

    def to_dict(self) -> dict:
        """Convert safety config to dictionary for serialization."""
        return {
            "max_consecutive_losses": self.max_consecutive_losses,
            "consecutive_loss_action": self.consecutive_loss_action,
            "max_drawdown_percent": self.max_drawdown_percent,
            "drawdown_action": self.drawdown_action,
            "max_daily_loss_percent": self.max_daily_loss_percent,
            "max_daily_loss_amount": self.max_daily_loss_amount,
            "daily_loss_action": self.daily_loss_action,
            "enable_model_degradation": self.enable_model_degradation,
            "min_win_rate": self.min_win_rate,
            "degradation_window": self.degradation_window,
            "require_token_for_reset": self.require_token_for_reset,
            "auto_reset_next_day": self.auto_reset_next_day,
            "max_disconnection_seconds": self.max_disconnection_seconds,
            "max_daily_trades": self.max_daily_trades,
            "max_trades_per_hour": self.max_trades_per_hour,
        }

    def __repr__(self) -> str:
        """String representation for logging."""
        return (
            f"SafetyConfig("
            f"consecutive_losses={self.max_consecutive_losses}, "
            f"drawdown={self.max_drawdown_percent}%, "
            f"daily_loss={self.max_daily_loss_percent}%, "
            f"model_degradation={self.enable_model_degradation})"
        )
