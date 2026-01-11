"""Application settings and configuration."""

from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Global application settings."""

    # Application
    app_name: str = "AI Assets Trader"
    app_version: str = "0.1.0"
    debug: bool = False
    log_level: str = "INFO"

    # Paths
    base_dir: Path = Path(__file__).parent.parent.parent
    data_dir: Path = Field(default_factory=lambda: Path("data"))
    models_dir: Path = Field(default_factory=lambda: Path("models"))
    mlruns_dir: Path = Field(default_factory=lambda: Path("mlruns"))

    # Database
    database_url: str = "postgresql://localhost:5432/ai_trader"
    redis_url: str = "redis://localhost:6379/0"

    # Data Sources
    mt5_login: Optional[int] = None
    mt5_password: Optional[str] = None
    mt5_server: Optional[str] = None
    alpaca_api_key: Optional[str] = None
    alpaca_secret_key: Optional[str] = None
    alpaca_base_url: str = "https://paper-api.alpaca.markets"

    # MLflow
    mlflow_tracking_uri: str = "mlruns"
    mlflow_experiment_name: str = "ai-trader"

    # Trading
    default_symbol: str = "EURUSD"
    default_timeframe: str = "1H"
    max_position_size: float = 0.02  # 2% of account
    max_daily_loss: float = 0.05  # 5% max daily loss

    # Model defaults
    batch_size: int = 64
    learning_rate: float = 1e-4
    epochs: int = 100
    early_stopping_patience: int = 15

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
