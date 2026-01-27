"""Centralized trading configuration system.

This module provides a singleton configuration manager that:
- Loads configuration from defaults, environment variables, and database
- Supports hot reload without service restart
- Provides thread-safe access to configuration
- Maintains audit trail of all changes
- Validates configuration changes before applying
"""

import logging
import os
import platform
import threading
from dataclasses import dataclass, field, asdict
from threading import Lock, RLock
from typing import Dict, Any, Optional, Callable, List
from datetime import datetime

from .indicator_config import IndicatorParameters
from .model_config import ModelHyperparameters
from .feature_config import FeatureParameters as FeatureEngineeringParameters
from .training_config import TrainingParameters
from .labeling_config import LabelingParameters

logger = logging.getLogger(__name__)


@dataclass
class TradingParameters:
    """Trading execution parameters."""

    confidence_threshold: float = 0.60  # Minimum confidence to trade (Config C optimized)
    default_lot_size: float = 0.1  # Standard lot (10K units)
    pip_value: float = 10.0  # $10 per pip for 0.1 lot EURUSD

    # Risk parameters (1H timeframe)
    default_tp_pips: float = 25.0
    default_sl_pips: float = 15.0
    max_holding_hours: int = 12

    # Account settings
    initial_balance: float = 100000.0  # $100K starting balance

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class ModelParameters:
    """Model ensemble parameters."""

    # Timeframe weights
    weight_1h: float = 0.6
    weight_4h: float = 0.3
    weight_daily: float = 0.1

    # Agreement bonus
    agreement_bonus: float = 0.05  # +5% confidence when all models agree

    # Regime adjustment
    use_regime_adjustment: bool = True

    def get_weights(self) -> Dict[str, float]:
        """Get weights as dict."""
        return {
            "1H": self.weight_1h,
            "4H": self.weight_4h,
            "D": self.weight_daily,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class RiskParameters:
    """Risk management parameters."""

    # Loss limits
    max_consecutive_losses: int = 5
    max_drawdown_percent: float = 15.0  # Circuit breaker threshold (CRITICAL: 15% not 10%)
    max_daily_loss_percent: float = 5.0
    max_daily_loss_amount: float = 5000.0  # Absolute daily loss limit ($)

    # Model degradation monitoring
    enable_model_degradation: bool = False
    min_win_rate: float = 0.45  # Trigger if win rate drops below 45%
    degradation_window: int = 20  # Rolling window of trades

    # Trade limits
    max_trades_per_day: int = 50  # Absolute maximum trades per day
    max_trades_per_hour: int = 20  # Maximum trades per hour

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class SystemParameters:
    """System operational parameters."""

    cache_ttl_seconds: int = 60  # Prediction cache TTL
    scheduler_enabled: bool = True  # Enable background scheduler
    db_timeout_seconds: float = 10.0
    broker_timeout_seconds: float = 30.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class TimeframeParameters:
    """Per-timeframe trading parameters."""

    timeframe: str  # "1H", "4H", or "D"
    tp_pips: float  # Take profit in pips
    sl_pips: float  # Stop loss in pips
    max_holding_bars: int  # Maximum holding period in bars
    weight: float  # Ensemble weight (0-1)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class AgentParameters:
    """Agent operational settings."""

    mode: str = "simulation"  # "simulation" or "live"
    symbol: str = "EURUSD"
    max_position_size: float = 0.1  # Maximum lot size
    use_kelly_sizing: bool = True  # Use Kelly criterion for position sizing
    cycle_interval_seconds: int = 60  # Agent cycle interval
    health_port: int = 8002  # Health check endpoint port
    max_reconnect_attempts: int = 5  # Max broker reconnection attempts
    max_reconnect_delay: float = 60.0  # Max broker disconnection tolerance (seconds)
    shutdown_timeout_seconds: float = 30.0  # Graceful shutdown timeout
    close_positions_on_shutdown: bool = True  # Close positions on shutdown

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class CacheParameters:
    """Cache configuration for various data types."""

    prediction_cache_ttl_seconds: int = 60  # Prediction cache TTL
    prediction_cache_max_size: int = 100  # Max cached predictions
    price_cache_max_size: int = 50  # Max cached price data points
    ohlcv_cache_max_size: int = 20  # Max cached OHLCV datasets
    asset_cache_max_size: int = 100  # Max cached asset records

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class SchedulerParameters:
    """Scheduler job intervals and cron expressions."""

    pipeline_cron_minute: int = 55  # Pipeline job minute (hourly at :55)
    prediction_cron_minute: int = 1  # Prediction job minute (hourly at :01)
    market_data_interval_minutes: int = 5  # Market data update interval
    position_check_interval_minutes: int = 5  # Position monitoring interval
    misfire_grace_time_seconds: int = 300  # Grace time for misfired jobs (5 min)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class FeatureParameters:
    """Feature engineering settings."""

    use_regime_detection: bool = True  # Enable market regime detection
    regime_lookback_periods: int = 50  # Lookback for regime detection
    use_sentiment_1h: bool = False  # Sentiment for 1H timeframe
    use_sentiment_4h: bool = False  # Sentiment for 4H timeframe
    use_sentiment_daily: bool = True  # Sentiment for Daily timeframe
    sentiment_cache_ttl_seconds: int = 3600  # Sentiment cache TTL (1 hour)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class ThresholdParameters:
    """Dynamic confidence threshold parameters."""

    # Enable/disable dynamic threshold system
    use_dynamic_threshold: bool = True  # Use dynamic threshold vs static

    # Lookback windows (in days) - optimized for stability + responsiveness
    short_term_window_days: int = 14  # Short-term predictions window (quick response)
    medium_term_window_days: int = 21  # Medium-term predictions window (PRIMARY)
    long_term_window_days: int = 45  # Long-term predictions window (stability anchor)

    # Blending weights (must sum to 1.0)
    short_term_weight: float = 0.20  # Weight for short-term component
    medium_term_weight: float = 0.60  # Weight for medium-term component (PRIMARY)
    long_term_weight: float = 0.20  # Weight for long-term component

    # Quantile parameters
    quantile: float = 0.60  # Top 40% of predictions (60th percentile)

    # Performance adjustment
    performance_lookback_trades: int = 30  # Recent trades for win rate (better sample)
    target_win_rate: float = 0.54  # Target win rate (from backtests)
    adjustment_factor: float = 0.10  # Adjustment magnitude per 10% win rate delta

    # Hard bounds
    min_threshold: float = 0.55  # Absolute minimum threshold
    max_threshold: float = 0.75  # Absolute maximum threshold

    # Divergence check
    max_divergence_from_long_term: float = 0.08  # Max deviation from long-term

    # Minimum data requirements
    min_predictions_required: int = 50  # Min predictions before using dynamic
    min_trades_for_adjustment: int = 10  # Min trades for performance adjustment

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class ConservativeHybridParameters:
    """Conservative Hybrid position sizing parameters.

    Combines confidence scaling with fixed risk management and comprehensive
    circuit breakers for safe position sizing.
    """

    # Base risk parameters
    base_risk_percent: float = 1.5  # Base risk as % of balance
    confidence_scaling_factor: float = 0.5  # How much confidence affects risk
    min_risk_percent: float = 0.8  # Minimum risk cap
    max_risk_percent: float = 2.5  # Maximum risk cap

    # Circuit breakers
    daily_loss_limit_percent: float = -3.0  # Daily loss limit (negative)
    consecutive_loss_limit: int = 5  # Max consecutive losses before stopping

    # Progressive Risk Reduction
    enable_progressive_reduction: bool = True  # Use progressive reduction instead of hard stop
    risk_reduction_per_loss: float = 0.20  # 20% reduction per consecutive loss
    min_risk_factor: float = 0.20  # Minimum 20% of normal risk (never zero)
    recovery_per_win: float = 0.20  # 20% recovery per winning trade

    # Trading parameters
    confidence_threshold: float = 0.60  # Minimum confidence to trade (Config C)
    pip_value: float = 10.0  # $ per pip for 0.1 lot EUR/USD
    lot_size: float = 100000.0  # Standard lot size

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class TradingConfig:
    """Singleton configuration manager.

    Provides centralized configuration with:
    - Hierarchical loading (defaults → env → database)
    - Hot reload capability
    - Thread-safe access
    - Change callbacks for dependent services
    - Audit trail
    """

    _instance: Optional["TradingConfig"] = None
    _lock = Lock()

    def __new__(cls):
        """Ensure singleton pattern."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize configuration manager."""
        # Only initialize once
        if hasattr(self, '_initialized'):
            return

        self._initialized = False
        self._update_lock = RLock()  # Reentrant lock for nested updates

        # Configuration sections
        self.trading = TradingParameters()
        self.model = ModelParameters()
        self.risk = RiskParameters()
        self.system = SystemParameters()

        # Extended configuration sections
        self.timeframes = {
            "1H": TimeframeParameters(
                timeframe="1H",
                tp_pips=25.0,
                sl_pips=15.0,
                max_holding_bars=12,
                weight=0.6,
            ),
            "4H": TimeframeParameters(
                timeframe="4H",
                tp_pips=50.0,
                sl_pips=25.0,
                max_holding_bars=18,
                weight=0.3,
            ),
            "D": TimeframeParameters(
                timeframe="D",
                tp_pips=150.0,
                sl_pips=75.0,
                max_holding_bars=15,
                weight=0.1,
            ),
        }
        self.agent = AgentParameters()
        self.cache = CacheParameters()
        self.scheduler = SchedulerParameters()
        self.features = FeatureParameters()
        self.threshold = ThresholdParameters()
        self.conservative_hybrid = ConservativeHybridParameters()

        # New configuration sections (Week 1 - Configuration Centralization)
        self.indicators = IndicatorParameters()
        self.hyperparameters = ModelHyperparameters()
        self.feature_engineering = FeatureEngineeringParameters()  # Feature engineering parameters (lag, session, cyclical)
        self.training = TrainingParameters()
        self.labeling = LabelingParameters()

        # Change callbacks: services register callbacks to react to config changes
        self._callbacks: Dict[str, List[Callable[[Any], None]]] = {
            "trading": [],
            "model": [],
            "risk": [],
            "system": [],
            "timeframes": [],
            "agent": [],
            "cache": [],
            "scheduler": [],
            "features": [],
            "threshold": [],
            "conservative_hybrid": [],
            # New sections (Week 1)
            "indicators": [],
            "hyperparameters": [],
            "feature_engineering": [],
            "training": [],
            "labeling": [],
        }

        # Track loaded state
        self._db_loaded = False
        self._last_reload = None

        # Configuration version for cache invalidation
        self._config_version = 0

        logger.info("TradingConfig instance created (singleton)")

    def initialize(self, db_session=None) -> None:
        """Initialize configuration from all sources.

        Loading order:
        1. Defaults (already set in dataclasses)
        2. Environment variables
        3. Database (if session provided)

        Args:
            db_session: Optional database session for loading DB config
        """
        if self._initialized:
            logger.debug("TradingConfig already initialized")
            return

        logger.info("Initializing TradingConfig...")

        # Load from environment variables
        self._load_from_env()

        # Load from database
        if db_session is not None:
            self._load_from_db(db_session)

        self._initialized = True
        self._last_reload = datetime.utcnow()
        logger.info("TradingConfig initialized successfully")

    def _load_from_env(self) -> None:
        """Load configuration from environment variables."""
        logger.debug("Loading configuration from environment variables")

        # Trading parameters
        if val := os.getenv("CONFIDENCE_THRESHOLD"):
            self.trading.confidence_threshold = float(val)
        if val := os.getenv("DEFAULT_LOT_SIZE"):
            self.trading.default_lot_size = float(val)
        if val := os.getenv("DEFAULT_TP_PIPS"):
            self.trading.default_tp_pips = float(val)
        if val := os.getenv("DEFAULT_SL_PIPS"):
            self.trading.default_sl_pips = float(val)
        if val := os.getenv("INITIAL_BALANCE"):
            self.trading.initial_balance = float(val)

        # Model parameters
        if val := os.getenv("MODEL_WEIGHT_1H"):
            self.model.weight_1h = float(val)
        if val := os.getenv("MODEL_WEIGHT_4H"):
            self.model.weight_4h = float(val)
        if val := os.getenv("MODEL_WEIGHT_DAILY"):
            self.model.weight_daily = float(val)
        if val := os.getenv("MODEL_AGREEMENT_BONUS"):
            self.model.agreement_bonus = float(val)

        # Risk parameters
        if val := os.getenv("MAX_CONSECUTIVE_LOSSES"):
            self.risk.max_consecutive_losses = int(val)
        if val := os.getenv("MAX_DRAWDOWN_PERCENT"):
            self.risk.max_drawdown_percent = float(val)
        if val := os.getenv("MAX_DAILY_LOSS_PERCENT"):
            self.risk.max_daily_loss_percent = float(val)
        if val := os.getenv("MAX_DAILY_LOSS_AMOUNT"):
            self.risk.max_daily_loss_amount = float(val)
        if val := os.getenv("ENABLE_MODEL_DEGRADATION"):
            self.risk.enable_model_degradation = val.lower() in ("true", "1", "yes")
        if val := os.getenv("MIN_WIN_RATE"):
            self.risk.min_win_rate = float(val)
        if val := os.getenv("DEGRADATION_WINDOW"):
            self.risk.degradation_window = int(val)
        if val := os.getenv("MAX_TRADES_PER_DAY"):
            self.risk.max_trades_per_day = int(val)
        if val := os.getenv("MAX_TRADES_PER_HOUR"):
            self.risk.max_trades_per_hour = int(val)

        # System parameters
        if val := os.getenv("SCHEDULER_ENABLED"):
            self.system.scheduler_enabled = val.lower() in ("true", "1", "yes")
        if val := os.getenv("CACHE_TTL_SECONDS"):
            self.system.cache_ttl_seconds = int(val)

        # Agent parameters
        if val := os.getenv("AGENT_MODE"):
            self.agent.mode = val
        if val := os.getenv("AGENT_SYMBOL"):
            self.agent.symbol = val
        if val := os.getenv("AGENT_MAX_POSITION_SIZE"):
            self.agent.max_position_size = float(val)
        if val := os.getenv("AGENT_USE_KELLY_SIZING"):
            self.agent.use_kelly_sizing = val.lower() in ("true", "1", "yes")
        if val := os.getenv("AGENT_CYCLE_INTERVAL_SECONDS"):
            self.agent.cycle_interval_seconds = int(val)
        if val := os.getenv("AGENT_HEALTH_PORT"):
            self.agent.health_port = int(val)
        if val := os.getenv("AGENT_MAX_RECONNECT_ATTEMPTS"):
            self.agent.max_reconnect_attempts = int(val)
        if val := os.getenv("AGENT_MAX_RECONNECT_DELAY"):
            self.agent.max_reconnect_delay = float(val)
        if val := os.getenv("AGENT_SHUTDOWN_TIMEOUT_SECONDS"):
            self.agent.shutdown_timeout_seconds = float(val)
        if val := os.getenv("AGENT_CLOSE_POSITIONS_ON_SHUTDOWN"):
            self.agent.close_positions_on_shutdown = val.lower() in ("true", "1", "yes")

        # Cache parameters
        if val := os.getenv("CACHE_PREDICTION_TTL_SECONDS"):
            self.cache.prediction_cache_ttl_seconds = int(val)
        if val := os.getenv("CACHE_PREDICTION_MAX_SIZE"):
            self.cache.prediction_cache_max_size = int(val)
        if val := os.getenv("CACHE_PRICE_MAX_SIZE"):
            self.cache.price_cache_max_size = int(val)
        if val := os.getenv("CACHE_OHLCV_MAX_SIZE"):
            self.cache.ohlcv_cache_max_size = int(val)
        if val := os.getenv("CACHE_ASSET_MAX_SIZE"):
            self.cache.asset_cache_max_size = int(val)

        # Scheduler parameters
        if val := os.getenv("SCHEDULER_PIPELINE_CRON_MINUTE"):
            self.scheduler.pipeline_cron_minute = int(val)
        if val := os.getenv("SCHEDULER_PREDICTION_CRON_MINUTE"):
            self.scheduler.prediction_cron_minute = int(val)
        if val := os.getenv("SCHEDULER_MARKET_DATA_INTERVAL_MINUTES"):
            self.scheduler.market_data_interval_minutes = int(val)
        if val := os.getenv("SCHEDULER_POSITION_CHECK_INTERVAL_MINUTES"):
            self.scheduler.position_check_interval_minutes = int(val)
        if val := os.getenv("SCHEDULER_MISFIRE_GRACE_TIME_SECONDS"):
            self.scheduler.misfire_grace_time_seconds = int(val)

        # Feature parameters
        if val := os.getenv("FEATURES_USE_REGIME_DETECTION"):
            self.features.use_regime_detection = val.lower() in ("true", "1", "yes")
        if val := os.getenv("FEATURES_REGIME_LOOKBACK_PERIODS"):
            self.features.regime_lookback_periods = int(val)
        if val := os.getenv("FEATURES_USE_SENTIMENT_1H"):
            self.features.use_sentiment_1h = val.lower() in ("true", "1", "yes")
        if val := os.getenv("FEATURES_USE_SENTIMENT_4H"):
            self.features.use_sentiment_4h = val.lower() in ("true", "1", "yes")
        if val := os.getenv("FEATURES_USE_SENTIMENT_DAILY"):
            self.features.use_sentiment_daily = val.lower() in ("true", "1", "yes")
        if val := os.getenv("FEATURES_SENTIMENT_CACHE_TTL_SECONDS"):
            self.features.sentiment_cache_ttl_seconds = int(val)

        # Timeframe parameters (1H)
        if val := os.getenv("TIMEFRAME_1H_TP_PIPS"):
            self.timeframes["1H"].tp_pips = float(val)
        if val := os.getenv("TIMEFRAME_1H_SL_PIPS"):
            self.timeframes["1H"].sl_pips = float(val)
        if val := os.getenv("TIMEFRAME_1H_MAX_HOLDING_BARS"):
            self.timeframes["1H"].max_holding_bars = int(val)
        if val := os.getenv("TIMEFRAME_1H_WEIGHT"):
            self.timeframes["1H"].weight = float(val)

        # Timeframe parameters (4H)
        if val := os.getenv("TIMEFRAME_4H_TP_PIPS"):
            self.timeframes["4H"].tp_pips = float(val)
        if val := os.getenv("TIMEFRAME_4H_SL_PIPS"):
            self.timeframes["4H"].sl_pips = float(val)
        if val := os.getenv("TIMEFRAME_4H_MAX_HOLDING_BARS"):
            self.timeframes["4H"].max_holding_bars = int(val)
        if val := os.getenv("TIMEFRAME_4H_WEIGHT"):
            self.timeframes["4H"].weight = float(val)

        # Timeframe parameters (Daily)
        if val := os.getenv("TIMEFRAME_D_TP_PIPS"):
            self.timeframes["D"].tp_pips = float(val)
        if val := os.getenv("TIMEFRAME_D_SL_PIPS"):
            self.timeframes["D"].sl_pips = float(val)
        if val := os.getenv("TIMEFRAME_D_MAX_HOLDING_BARS"):
            self.timeframes["D"].max_holding_bars = int(val)
        if val := os.getenv("TIMEFRAME_D_WEIGHT"):
            self.timeframes["D"].weight = float(val)

        # Conservative Hybrid parameters
        if val := os.getenv("CONSERVATIVE_HYBRID_BASE_RISK"):
            self.conservative_hybrid.base_risk_percent = float(val)
        if val := os.getenv("CONSERVATIVE_HYBRID_CONFIDENCE_SCALING"):
            self.conservative_hybrid.confidence_scaling_factor = float(val)
        if val := os.getenv("CONSERVATIVE_HYBRID_MIN_RISK"):
            self.conservative_hybrid.min_risk_percent = float(val)
        if val := os.getenv("CONSERVATIVE_HYBRID_MAX_RISK"):
            self.conservative_hybrid.max_risk_percent = float(val)
        if val := os.getenv("CONSERVATIVE_HYBRID_DAILY_LOSS_LIMIT"):
            self.conservative_hybrid.daily_loss_limit_percent = float(val)
        if val := os.getenv("CONSERVATIVE_HYBRID_CONSECUTIVE_LOSS_LIMIT"):
            self.conservative_hybrid.consecutive_loss_limit = int(val)
        if val := os.getenv("CONSERVATIVE_HYBRID_ENABLE_PROGRESSIVE_REDUCTION"):
            self.conservative_hybrid.enable_progressive_reduction = val.lower() in ("true", "1", "yes")
        if val := os.getenv("CONSERVATIVE_HYBRID_RISK_REDUCTION_PER_LOSS"):
            self.conservative_hybrid.risk_reduction_per_loss = float(val)
        if val := os.getenv("CONSERVATIVE_HYBRID_MIN_RISK_FACTOR"):
            self.conservative_hybrid.min_risk_factor = float(val)
        if val := os.getenv("CONSERVATIVE_HYBRID_RECOVERY_PER_WIN"):
            self.conservative_hybrid.recovery_per_win = float(val)
        if val := os.getenv("CONSERVATIVE_HYBRID_CONFIDENCE_THRESHOLD"):
            self.conservative_hybrid.confidence_threshold = float(val)
        if val := os.getenv("CONSERVATIVE_HYBRID_PIP_VALUE"):
            self.conservative_hybrid.pip_value = float(val)
        if val := os.getenv("CONSERVATIVE_HYBRID_LOT_SIZE"):
            self.conservative_hybrid.lot_size = float(val)

        logger.info("Configuration loaded from environment")

    def _load_from_db(self, db_session) -> None:
        """Load configuration from database.

        Database values override environment and defaults.

        Args:
            db_session: SQLAlchemy database session
        """
        try:
            from ..api.database.models import ConfigurationSetting

            logger.debug("Loading configuration from database")

            # Query all configuration settings
            settings = db_session.query(ConfigurationSetting).all()

            if not settings:
                logger.info("No configuration settings in database, using defaults")
                return

            # Apply each setting
            for setting in settings:
                self._apply_db_setting(setting)

            self._db_loaded = True
            logger.info(f"Loaded {len(settings)} configuration settings from database")

        except Exception as e:
            logger.error(f"Failed to load configuration from database: {e}")
            logger.warning("Continuing with default/environment configuration")

    def _apply_db_setting(self, setting) -> None:
        """Apply a single database setting to configuration.

        Args:
            setting: ConfigurationSetting ORM object
        """
        category = setting.category
        key = setting.key
        value = setting.value

        # Map to appropriate parameter object
        if category == "trading":
            if hasattr(self.trading, key):
                setattr(self.trading, key, value)
        elif category == "model":
            if hasattr(self.model, key):
                setattr(self.model, key, value)
        elif category == "risk":
            if hasattr(self.risk, key):
                setattr(self.risk, key, value)
        elif category == "system":
            if hasattr(self.system, key):
                setattr(self.system, key, value)
        elif category == "agent":
            if hasattr(self.agent, key):
                setattr(self.agent, key, value)
        elif category == "cache":
            if hasattr(self.cache, key):
                setattr(self.cache, key, value)
        elif category == "scheduler":
            if hasattr(self.scheduler, key):
                setattr(self.scheduler, key, value)
        elif category == "features":
            if hasattr(self.features, key):
                setattr(self.features, key, value)
        elif category == "threshold":
            if hasattr(self.threshold, key):
                setattr(self.threshold, key, value)
        elif category == "conservative_hybrid":
            if hasattr(self.conservative_hybrid, key):
                setattr(self.conservative_hybrid, key, value)
        elif category == "indicators":
            if hasattr(self.indicators, key):
                setattr(self.indicators, key, value)
        elif category == "hyperparameters":
            if hasattr(self.hyperparameters, key):
                setattr(self.hyperparameters, key, value)
        elif category == "feature_engineering":
            if hasattr(self.feature_engineering, key):
                setattr(self.feature_engineering, key, value)
        elif category == "training":
            if hasattr(self.training, key):
                setattr(self.training, key, value)
        elif category == "labeling":
            if hasattr(self.labeling, key):
                setattr(self.labeling, key, value)
        elif category == "timeframes":
            # Handle timeframe-specific settings (e.g., "1H", "4H", "D")
            # Key format: "{timeframe}.{parameter}" e.g., "1H.tp_pips"
            if "." in key:
                timeframe, param = key.split(".", 1)
                if timeframe in self.timeframes and hasattr(self.timeframes[timeframe], param):
                    setattr(self.timeframes[timeframe], param, value)

    def validate(self) -> List[str]:
        """Validate configuration parameters.

        Returns:
            List of validation errors (empty if valid)
        """
        errors = []

        # Validate trading parameters
        if not 0.0 <= self.trading.confidence_threshold <= 1.0:
            errors.append(
                f"confidence_threshold must be between 0.0 and 1.0, got {self.trading.confidence_threshold}"
            )

        if self.trading.default_lot_size <= 0:
            errors.append(f"default_lot_size must be positive, got {self.trading.default_lot_size}")

        if self.trading.pip_value <= 0:
            errors.append(f"pip_value must be positive, got {self.trading.pip_value}")

        if self.trading.default_tp_pips <= 0:
            errors.append(f"default_tp_pips must be positive, got {self.trading.default_tp_pips}")

        if self.trading.default_sl_pips <= 0:
            errors.append(f"default_sl_pips must be positive, got {self.trading.default_sl_pips}")

        if self.trading.max_holding_hours <= 0:
            errors.append(f"max_holding_hours must be positive, got {self.trading.max_holding_hours}")

        if self.trading.initial_balance <= 0:
            errors.append(f"initial_balance must be positive, got {self.trading.initial_balance}")

        # Validate model weights sum to ~1.0
        weights_sum = self.model.weight_1h + self.model.weight_4h + self.model.weight_daily
        if not (0.99 <= weights_sum <= 1.01):
            errors.append(f"Model weights must sum to 1.0, got {weights_sum:.3f}")

        # Validate risk parameters
        if self.risk.max_consecutive_losses <= 0:
            errors.append(f"max_consecutive_losses must be positive, got {self.risk.max_consecutive_losses}")

        if self.risk.max_drawdown_percent <= 0:
            errors.append(f"max_drawdown_percent must be positive, got {self.risk.max_drawdown_percent}")

        if self.risk.max_daily_loss_percent <= 0:
            errors.append(f"max_daily_loss_percent must be positive, got {self.risk.max_daily_loss_percent}")

        if self.risk.max_daily_loss_amount <= 0:
            errors.append(f"max_daily_loss_amount must be positive, got {self.risk.max_daily_loss_amount}")

        if not 0.0 <= self.risk.min_win_rate <= 1.0:
            errors.append(f"min_win_rate must be between 0.0 and 1.0, got {self.risk.min_win_rate}")

        if self.risk.degradation_window < 5:
            errors.append(f"degradation_window must be at least 5, got {self.risk.degradation_window}")

        if self.risk.max_trades_per_day <= 0:
            errors.append(f"max_trades_per_day must be positive, got {self.risk.max_trades_per_day}")

        if self.risk.max_trades_per_hour <= 0:
            errors.append(f"max_trades_per_hour must be positive, got {self.risk.max_trades_per_hour}")

        # Validate system parameters
        if self.system.cache_ttl_seconds <= 0:
            errors.append(f"cache_ttl_seconds must be positive, got {self.system.cache_ttl_seconds}")

        if self.system.db_timeout_seconds <= 0:
            errors.append(f"db_timeout_seconds must be positive, got {self.system.db_timeout_seconds}")

        if self.system.broker_timeout_seconds <= 0:
            errors.append(f"broker_timeout_seconds must be positive, got {self.system.broker_timeout_seconds}")

        # Validate agent parameters
        if self.agent.mode not in ("simulation", "live"):
            errors.append(f"agent.mode must be 'simulation' or 'live', got {self.agent.mode}")

        if self.agent.max_position_size <= 0:
            errors.append(f"agent.max_position_size must be positive, got {self.agent.max_position_size}")

        if self.agent.cycle_interval_seconds <= 0:
            errors.append(f"agent.cycle_interval_seconds must be positive, got {self.agent.cycle_interval_seconds}")

        if self.agent.health_port <= 0 or self.agent.health_port > 65535:
            errors.append(f"agent.health_port must be 1-65535, got {self.agent.health_port}")

        if self.agent.max_reconnect_attempts < 0:
            errors.append(f"agent.max_reconnect_attempts must be non-negative, got {self.agent.max_reconnect_attempts}")

        if self.agent.max_reconnect_delay <= 0:
            errors.append(f"agent.max_reconnect_delay must be positive, got {self.agent.max_reconnect_delay}")

        if self.agent.shutdown_timeout_seconds <= 0:
            errors.append(f"agent.shutdown_timeout_seconds must be positive, got {self.agent.shutdown_timeout_seconds}")

        # Validate cache parameters
        if self.cache.prediction_cache_ttl_seconds <= 0:
            errors.append(f"cache.prediction_cache_ttl_seconds must be positive, got {self.cache.prediction_cache_ttl_seconds}")

        if self.cache.prediction_cache_max_size <= 0:
            errors.append(f"cache.prediction_cache_max_size must be positive, got {self.cache.prediction_cache_max_size}")

        if self.cache.price_cache_max_size <= 0:
            errors.append(f"cache.price_cache_max_size must be positive, got {self.cache.price_cache_max_size}")

        if self.cache.ohlcv_cache_max_size <= 0:
            errors.append(f"cache.ohlcv_cache_max_size must be positive, got {self.cache.ohlcv_cache_max_size}")

        if self.cache.asset_cache_max_size <= 0:
            errors.append(f"cache.asset_cache_max_size must be positive, got {self.cache.asset_cache_max_size}")

        # Validate scheduler parameters
        if not (0 <= self.scheduler.pipeline_cron_minute <= 59):
            errors.append(f"scheduler.pipeline_cron_minute must be 0-59, got {self.scheduler.pipeline_cron_minute}")

        if not (0 <= self.scheduler.prediction_cron_minute <= 59):
            errors.append(f"scheduler.prediction_cron_minute must be 0-59, got {self.scheduler.prediction_cron_minute}")

        if self.scheduler.market_data_interval_minutes <= 0:
            errors.append(f"scheduler.market_data_interval_minutes must be positive, got {self.scheduler.market_data_interval_minutes}")

        if self.scheduler.position_check_interval_minutes <= 0:
            errors.append(f"scheduler.position_check_interval_minutes must be positive, got {self.scheduler.position_check_interval_minutes}")

        if self.scheduler.misfire_grace_time_seconds <= 0:
            errors.append(f"scheduler.misfire_grace_time_seconds must be positive, got {self.scheduler.misfire_grace_time_seconds}")

        # Validate feature parameters
        if self.features.regime_lookback_periods <= 0:
            errors.append(f"features.regime_lookback_periods must be positive, got {self.features.regime_lookback_periods}")

        if self.features.sentiment_cache_ttl_seconds <= 0:
            errors.append(f"features.sentiment_cache_ttl_seconds must be positive, got {self.features.sentiment_cache_ttl_seconds}")

        # Validate threshold parameters
        if self.threshold.short_term_window_days <= 0:
            errors.append(f"threshold.short_term_window_days must be positive, got {self.threshold.short_term_window_days}")

        if self.threshold.medium_term_window_days <= 0:
            errors.append(f"threshold.medium_term_window_days must be positive, got {self.threshold.medium_term_window_days}")

        if self.threshold.long_term_window_days <= 0:
            errors.append(f"threshold.long_term_window_days must be positive, got {self.threshold.long_term_window_days}")

        # Validate threshold blending weights sum to 1.0
        threshold_weights_sum = (
            self.threshold.short_term_weight +
            self.threshold.medium_term_weight +
            self.threshold.long_term_weight
        )
        if not (0.99 <= threshold_weights_sum <= 1.01):
            errors.append(f"Threshold weights must sum to 1.0, got {threshold_weights_sum:.3f}")

        if not (0.0 <= self.threshold.quantile <= 1.0):
            errors.append(f"threshold.quantile must be between 0.0 and 1.0, got {self.threshold.quantile}")

        if self.threshold.performance_lookback_trades <= 0:
            errors.append(f"threshold.performance_lookback_trades must be positive, got {self.threshold.performance_lookback_trades}")

        if not (0.0 <= self.threshold.target_win_rate <= 1.0):
            errors.append(f"threshold.target_win_rate must be between 0.0 and 1.0, got {self.threshold.target_win_rate}")

        if not (0.0 <= self.threshold.min_threshold <= 1.0):
            errors.append(f"threshold.min_threshold must be between 0.0 and 1.0, got {self.threshold.min_threshold}")

        if not (0.0 <= self.threshold.max_threshold <= 1.0):
            errors.append(f"threshold.max_threshold must be between 0.0 and 1.0, got {self.threshold.max_threshold}")

        if self.threshold.min_threshold >= self.threshold.max_threshold:
            errors.append(f"threshold.min_threshold must be < max_threshold, got {self.threshold.min_threshold} >= {self.threshold.max_threshold}")

        if self.threshold.min_predictions_required < 0:
            errors.append(f"threshold.min_predictions_required must be non-negative, got {self.threshold.min_predictions_required}")

        if self.threshold.min_trades_for_adjustment < 0:
            errors.append(f"threshold.min_trades_for_adjustment must be non-negative, got {self.threshold.min_trades_for_adjustment}")

        # Validate timeframe parameters
        total_weight = 0.0
        for tf, params in self.timeframes.items():
            if params.tp_pips <= 0:
                errors.append(f"timeframes.{tf}.tp_pips must be positive, got {params.tp_pips}")

            if params.sl_pips <= 0:
                errors.append(f"timeframes.{tf}.sl_pips must be positive, got {params.sl_pips}")

            if params.max_holding_bars <= 0:
                errors.append(f"timeframes.{tf}.max_holding_bars must be positive, got {params.max_holding_bars}")

            if not (0.0 <= params.weight <= 1.0):
                errors.append(f"timeframes.{tf}.weight must be 0-1, got {params.weight}")

            total_weight += params.weight

        # Validate total timeframe weights sum to ~1.0
        if not (0.99 <= total_weight <= 1.01):
            errors.append(f"Timeframe weights must sum to 1.0, got {total_weight:.3f}")

        # Validate conservative_hybrid parameters
        if not (0.0 < self.conservative_hybrid.base_risk_percent <= 100.0):
            errors.append(f"conservative_hybrid.base_risk_percent must be 0-100, got {self.conservative_hybrid.base_risk_percent}")

        if not (0.0 <= self.conservative_hybrid.confidence_scaling_factor <= 5.0):
            errors.append(f"conservative_hybrid.confidence_scaling_factor must be 0-5, got {self.conservative_hybrid.confidence_scaling_factor}")

        if not (0.0 < self.conservative_hybrid.min_risk_percent <= 100.0):
            errors.append(f"conservative_hybrid.min_risk_percent must be 0-100, got {self.conservative_hybrid.min_risk_percent}")

        if not (0.0 < self.conservative_hybrid.max_risk_percent <= 100.0):
            errors.append(f"conservative_hybrid.max_risk_percent must be 0-100, got {self.conservative_hybrid.max_risk_percent}")

        if self.conservative_hybrid.min_risk_percent >= self.conservative_hybrid.max_risk_percent:
            errors.append(f"conservative_hybrid.min_risk_percent must be < max_risk_percent")

        if not (-100.0 <= self.conservative_hybrid.daily_loss_limit_percent < 0.0):
            errors.append(f"conservative_hybrid.daily_loss_limit_percent must be negative and > -100, got {self.conservative_hybrid.daily_loss_limit_percent}")

        if self.conservative_hybrid.consecutive_loss_limit <= 0:
            errors.append(f"conservative_hybrid.consecutive_loss_limit must be positive, got {self.conservative_hybrid.consecutive_loss_limit}")

        if not (0.0 < self.conservative_hybrid.risk_reduction_per_loss <= 1.0):
            errors.append(f"conservative_hybrid.risk_reduction_per_loss must be 0-1, got {self.conservative_hybrid.risk_reduction_per_loss}")

        if not (0.0 < self.conservative_hybrid.min_risk_factor <= 1.0):
            errors.append(f"conservative_hybrid.min_risk_factor must be 0-1, got {self.conservative_hybrid.min_risk_factor}")

        if not (0.0 < self.conservative_hybrid.recovery_per_win <= 1.0):
            errors.append(f"conservative_hybrid.recovery_per_win must be 0-1, got {self.conservative_hybrid.recovery_per_win}")

        if not (0.0 <= self.conservative_hybrid.confidence_threshold <= 1.0):
            errors.append(f"conservative_hybrid.confidence_threshold must be 0-1, got {self.conservative_hybrid.confidence_threshold}")

        if self.conservative_hybrid.pip_value <= 0:
            errors.append(f"conservative_hybrid.pip_value must be positive, got {self.conservative_hybrid.pip_value}")

        if self.conservative_hybrid.lot_size <= 0:
            errors.append(f"conservative_hybrid.lot_size must be positive, got {self.conservative_hybrid.lot_size}")

        return errors

    def update(
        self,
        category: str,
        updates: Dict[str, Any],
        updated_by: Optional[str] = None,
        reason: Optional[str] = None,
        db_session=None,
    ) -> Dict[str, Any]:
        """Update configuration parameters.

        Args:
            category: Configuration category (trading, model, risk, system)
            updates: Dictionary of parameter updates
            updated_by: User/service making the change
            reason: Optional reason for change
            db_session: Database session for persisting changes

        Returns:
            Dictionary with update result including callback status

        Raises:
            ValueError: If validation fails
        """
        with self._update_lock:
            # Get target parameter object
            if category == "trading":
                target = self.trading
            elif category == "model":
                target = self.model
            elif category == "risk":
                target = self.risk
            elif category == "system":
                target = self.system
            elif category == "agent":
                target = self.agent
            elif category == "cache":
                target = self.cache
            elif category == "scheduler":
                target = self.scheduler
            elif category == "features":
                target = self.features
            elif category == "threshold":
                target = self.threshold
            elif category == "conservative_hybrid":
                target = self.conservative_hybrid
            elif category == "indicators":
                target = self.indicators
            elif category == "hyperparameters":
                target = self.hyperparameters
            elif category == "feature_engineering":
                target = self.feature_engineering
            elif category == "training":
                target = self.training
            elif category == "labeling":
                target = self.labeling
            elif category == "timeframes":
                # Special handling for timeframes (dict of TimeframeParameters)
                raise ValueError("Use update_timeframe() method for timeframe updates")
            else:
                raise ValueError(f"Invalid category: {category}")

            # Store old values for rollback
            old_values = {}
            for key in updates:
                if hasattr(target, key):
                    old_values[key] = getattr(target, key)
                else:
                    raise ValueError(f"Unknown parameter: {category}.{key}")

            # Apply updates for validation
            try:
                for key, value in updates.items():
                    setattr(target, key, value)

                # Validate BEFORE persisting
                errors = self.validate()
                if errors:
                    # Rollback
                    for key, old_val in old_values.items():
                        setattr(target, key, old_val)
                    raise ValueError(f"Validation failed: {'; '.join(errors)}")

                # Persist to database BEFORE updating memory (for consistency)
                # If DB persist fails, we rollback memory
                if db_session is not None:
                    self._persist_updates(
                        category, updates, old_values, updated_by, reason, db_session
                    )

                # Increment config version for cache invalidation
                self._config_version += 1

                # Trigger callbacks (track failures)
                callback_failures = self._trigger_callbacks(category, target)

                logger.info(
                    f"Configuration updated: {category} - {list(updates.keys())} by {updated_by or 'system'} (version: {self._config_version})"
                )

                result = {
                    "status": "success",
                    "category": category,
                    "updated": list(updates.keys()),
                    "config_version": self._config_version,
                    "timestamp": datetime.utcnow().isoformat(),
                }

                # Include callback failure info if any
                if callback_failures:
                    result["callback_failures"] = callback_failures
                    logger.warning(f"Some callbacks failed: {callback_failures}")

                return result

            except Exception as e:
                # Rollback memory on any error (DB already rolled back in _persist_updates)
                for key, old_val in old_values.items():
                    setattr(target, key, old_val)
                logger.error(f"Configuration update failed, rolled back: {e}")
                raise

    def _persist_updates(
        self,
        category: str,
        updates: Dict[str, Any],
        old_values: Dict[str, Any],
        updated_by: Optional[str],
        reason: Optional[str],
        db_session,
    ) -> None:
        """Persist configuration updates to database with timeout and transaction management.

        Args:
            category: Configuration category
            updates: New values
            old_values: Previous values
            updated_by: User/service making change
            reason: Optional reason
            db_session: Database session

        Raises:
            Exception: If database operation fails or times out
        """
        from ..api.database.models import ConfigurationSetting, ConfigurationHistory

        timeout_seconds = int(self.system.db_timeout_seconds)
        timeout_occurred = [False]  # Use list to allow modification in nested function

        # Platform-specific timeout implementation
        is_windows = platform.system() == 'Windows'

        def timeout_handler():
            """Handler called when timeout expires (threading.Timer version)."""
            timeout_occurred[0] = True
            logger.error(f"Database operation timed out after {timeout_seconds}s")

        # Set up timeout mechanism based on platform
        if is_windows:
            # Use threading.Timer for Windows (SIGALRM not available)
            timeout_timer = threading.Timer(timeout_seconds, timeout_handler)
            timeout_timer.daemon = True
            timeout_timer.start()
        else:
            # Use signal-based timeout for Unix/Linux/macOS
            import signal

            def signal_timeout_handler(signum, frame):
                raise TimeoutError("Database operation timed out")

            old_handler = signal.signal(signal.SIGALRM, signal_timeout_handler)
            signal.alarm(timeout_seconds)

        try:
            # Begin explicit transaction (create savepoint for nested transactions)
            savepoint = db_session.begin_nested() if db_session.in_transaction() else None

            try:
                for key, new_value in updates.items():
                    # Check for timeout (Windows path)
                    if is_windows and timeout_occurred[0]:
                        raise TimeoutError("Database operation timed out")

                    # Find or create setting
                    setting = db_session.query(ConfigurationSetting).filter_by(
                        category=category, key=key
                    ).first()

                    if setting is None:
                        # Create new setting
                        setting = ConfigurationSetting(
                            category=category,
                            key=key,
                            value=new_value,
                            value_type=type(new_value).__name__,
                            version=1,
                            updated_by=updated_by,
                        )
                        db_session.add(setting)
                        db_session.flush()  # Get ID

                        # Record history
                        history = ConfigurationHistory(
                            setting_id=setting.id,
                            category=category,
                            key=key,
                            old_value=None,
                            new_value=new_value,
                            version=1,
                            changed_by=updated_by,
                            reason=reason,
                        )
                        db_session.add(history)
                    else:
                        # Update existing setting
                        old_value = old_values.get(key)
                        setting.value = new_value
                        setting.value_type = type(new_value).__name__
                        setting.version += 1
                        setting.updated_by = updated_by
                        setting.updated_at = datetime.utcnow()

                        # Record history
                        history = ConfigurationHistory(
                            setting_id=setting.id,
                            category=category,
                            key=key,
                            old_value=old_value,
                            new_value=new_value,
                            version=setting.version,
                            changed_by=updated_by,
                            reason=reason,
                        )
                        db_session.add(history)

                # Final timeout check (Windows path)
                if is_windows and timeout_occurred[0]:
                    raise TimeoutError("Database operation timed out")

                # Commit the changes
                db_session.commit()
                logger.debug(f"Persisted {len(updates)} configuration changes to database")

            except Exception as e:
                # Rollback to savepoint if exists, otherwise full rollback
                if savepoint:
                    savepoint.rollback()
                else:
                    db_session.rollback()
                logger.error(f"Failed to persist configuration to database: {e}")
                raise

        except TimeoutError as e:
            db_session.rollback()
            logger.error(f"Database operation timed out after {timeout_seconds}s")
            raise
        finally:
            # Clean up timeout mechanism based on platform
            if is_windows:
                # Cancel the timer
                timeout_timer.cancel()
            else:
                # Cancel alarm and restore signal handler
                import signal
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)

    def reload(self, db_session=None) -> Dict[str, Any]:
        """Hot reload configuration from database.

        Args:
            db_session: Database session

        Returns:
            Dictionary with reload result
        """
        with self._update_lock:
            logger.info("Reloading configuration from database...")

            if db_session is None:
                return {
                    "status": "error",
                    "message": "Database session required for reload",
                }

            try:
                # Reload from database
                self._load_from_db(db_session)

                # Validate
                errors = self.validate()
                if errors:
                    raise ValueError(f"Validation failed after reload: {'; '.join(errors)}")

                # Increment config version for cache invalidation
                self._config_version += 1

                # Trigger all callbacks
                self._trigger_callbacks("trading", self.trading)
                self._trigger_callbacks("model", self.model)
                self._trigger_callbacks("risk", self.risk)
                self._trigger_callbacks("system", self.system)
                self._trigger_callbacks("agent", self.agent)
                self._trigger_callbacks("cache", self.cache)
                self._trigger_callbacks("scheduler", self.scheduler)
                self._trigger_callbacks("features", self.features)
                self._trigger_callbacks("threshold", self.threshold)
                self._trigger_callbacks("conservative_hybrid", self.conservative_hybrid)
                self._trigger_callbacks("indicators", self.indicators)
                self._trigger_callbacks("hyperparameters", self.hyperparameters)
                self._trigger_callbacks("feature_engineering", self.feature_engineering)
                self._trigger_callbacks("training", self.training)
                self._trigger_callbacks("labeling", self.labeling)
                self._trigger_callbacks("timeframes", self.timeframes)

                self._last_reload = datetime.utcnow()

                logger.info(f"Configuration reloaded successfully (version: {self._config_version})")

                return {
                    "status": "success",
                    "config_version": self._config_version,
                    "timestamp": self._last_reload.isoformat(),
                    "message": "Configuration reloaded from database",
                }

            except Exception as e:
                logger.error(f"Configuration reload failed: {e}")
                return {
                    "status": "error",
                    "message": str(e),
                }

    def get_config_version(self) -> int:
        """Get the current configuration version.

        This version number is incremented on every configuration update,
        allowing services to invalidate caches when config changes.

        Returns:
            Current configuration version number
        """
        return self._config_version

    def update_timeframe(
        self,
        timeframe: str,
        updates: Dict[str, Any],
        updated_by: Optional[str] = None,
        reason: Optional[str] = None,
        db_session=None,
    ) -> Dict[str, Any]:
        """Update timeframe-specific parameters.

        Args:
            timeframe: Timeframe identifier ("1H", "4H", "D")
            updates: Dictionary of parameter updates
            updated_by: User/service making the change
            reason: Optional reason for change
            db_session: Database session for persisting changes

        Returns:
            Dictionary with update result

        Raises:
            ValueError: If validation fails
        """
        with self._update_lock:
            if timeframe not in self.timeframes:
                raise ValueError(f"Invalid timeframe: {timeframe}")

            target = self.timeframes[timeframe]

            # Store old values for rollback
            old_values = {}
            for key in updates:
                if hasattr(target, key):
                    old_values[key] = getattr(target, key)
                else:
                    raise ValueError(f"Unknown parameter: timeframes.{timeframe}.{key}")

            # Apply updates for validation
            try:
                for key, value in updates.items():
                    setattr(target, key, value)

                # Validate
                errors = self.validate()
                if errors:
                    # Rollback
                    for key, old_val in old_values.items():
                        setattr(target, key, old_val)
                    raise ValueError(f"Validation failed: {'; '.join(errors)}")

                # Persist to database (using category "timeframes" and key "{timeframe}.{param}")
                if db_session is not None:
                    db_updates = {f"{timeframe}.{k}": v for k, v in updates.items()}
                    db_old_values = {f"{timeframe}.{k}": v for k, v in old_values.items()}
                    self._persist_updates(
                        "timeframes", db_updates, db_old_values, updated_by, reason, db_session
                    )

                # Increment config version
                self._config_version += 1

                # Trigger callbacks
                callback_failures = self._trigger_callbacks("timeframes", self.timeframes)

                logger.info(
                    f"Timeframe configuration updated: {timeframe} - {list(updates.keys())} by {updated_by or 'system'}"
                )

                result = {
                    "status": "success",
                    "category": "timeframes",
                    "timeframe": timeframe,
                    "updated": list(updates.keys()),
                    "config_version": self._config_version,
                    "timestamp": datetime.utcnow().isoformat(),
                }

                if callback_failures:
                    result["callback_failures"] = callback_failures
                    logger.warning(f"Some callbacks failed: {callback_failures}")

                return result

            except Exception as e:
                # Rollback
                for key, old_val in old_values.items():
                    setattr(target, key, old_val)
                logger.error(f"Timeframe configuration update failed, rolled back: {e}")
                raise

    def register_callback(self, category: str, callback: Callable[[Any], None]) -> None:
        """Register a callback for configuration changes.

        Args:
            category: Configuration category to watch
            callback: Function to call when category changes (receives parameter object)
        """
        if category not in self._callbacks:
            raise ValueError(f"Invalid category: {category}")

        self._callbacks[category].append(callback)
        logger.debug(f"Registered callback for {category} configuration changes")

    def _trigger_callbacks(self, category: str, params: Any) -> List[str]:
        """Trigger callbacks for a category.

        Args:
            category: Configuration category
            params: Parameter object to pass to callbacks

        Returns:
            List of callback failure descriptions (empty if all succeeded)
        """
        failures = []
        for i, callback in enumerate(self._callbacks.get(category, [])):
            try:
                callback(params)
            except Exception as e:
                error_msg = f"Callback {i} for {category}: {str(e)}"
                logger.error(f"Configuration callback failed - {error_msg}")
                failures.append(error_msg)
        return failures

    def get_all(self) -> Dict[str, Dict[str, Any]]:
        """Get all configuration as nested dictionary.

        Returns:
            Dictionary with all configuration parameters
        """
        return {
            "trading": self.trading.to_dict(),
            "model": self.model.to_dict(),
            "risk": self.risk.to_dict(),
            "system": self.system.to_dict(),
            "agent": self.agent.to_dict(),
            "cache": self.cache.to_dict(),
            "scheduler": self.scheduler.to_dict(),
            "features": self.features.to_dict(),
            "threshold": self.threshold.to_dict(),
            "conservative_hybrid": self.conservative_hybrid.to_dict(),
            "indicators": self.indicators.to_dict(),
            "hyperparameters": self.hyperparameters.to_dict(),
            "feature_engineering": self.feature_engineering.to_dict(),
            "training": self.training.to_dict(),
            "labeling": self.labeling.to_dict(),
            "timeframes": {
                tf: params.to_dict() for tf, params in self.timeframes.items()
            },
            "metadata": {
                "initialized": self._initialized,
                "db_loaded": self._db_loaded,
                "last_reload": self._last_reload.isoformat() if self._last_reload else None,
                "config_version": self._config_version,
            },
        }

    def get_category(self, category: str) -> Dict[str, Any]:
        """Get configuration for a specific category.

        Args:
            category: Configuration category

        Returns:
            Dictionary with category parameters
        """
        if category == "trading":
            return self.trading.to_dict()
        elif category == "model":
            return self.model.to_dict()
        elif category == "risk":
            return self.risk.to_dict()
        elif category == "system":
            return self.system.to_dict()
        elif category == "agent":
            return self.agent.to_dict()
        elif category == "cache":
            return self.cache.to_dict()
        elif category == "scheduler":
            return self.scheduler.to_dict()
        elif category == "features":
            return self.features.to_dict()
        elif category == "threshold":
            return self.threshold.to_dict()
        elif category == "conservative_hybrid":
            return self.conservative_hybrid.to_dict()
        elif category == "indicators":
            return self.indicators.to_dict()
        elif category == "hyperparameters":
            return self.hyperparameters.to_dict()
        elif category == "feature_engineering":
            return self.feature_engineering.to_dict()
        elif category == "training":
            return self.training.to_dict()
        elif category == "labeling":
            return self.labeling.to_dict()
        elif category == "timeframes":
            return {tf: params.to_dict() for tf, params in self.timeframes.items()}
        else:
            raise ValueError(f"Invalid category: {category}")

    def reset_to_defaults(
        self,
        category: Optional[str] = None,
        key: Optional[str] = None,
        db_session=None,
    ) -> Dict[str, Any]:
        """Reset configuration to defaults.

        Args:
            category: Specific category to reset (None = all)
            key: Specific key to reset (requires category)
            db_session: Database session for persisting reset

        Returns:
            Dictionary with reset result
        """
        with self._update_lock:
            defaults = {
                "trading": TradingParameters(),
                "model": ModelParameters(),
                "risk": RiskParameters(),
                "system": SystemParameters(),
                "agent": AgentParameters(),
                "cache": CacheParameters(),
                "scheduler": SchedulerParameters(),
                "features": FeatureParameters(),
                "threshold": ThresholdParameters(),
                "conservative_hybrid": ConservativeHybridParameters(),
                "indicators": IndicatorParameters(),
                "hyperparameters": ModelHyperparameters(),
                "feature_engineering": FeatureEngineeringParameters(),
                "training": TrainingParameters(),
                "labeling": LabelingParameters(),
            }

            if key is not None and category is None:
                raise ValueError("category required when resetting specific key")

            if category and key:
                # Reset specific key (this already persists via update method)
                if category == "timeframes":
                    # Special handling for timeframes (key format: "{timeframe}.{param}")
                    if "." in key:
                        timeframe, param = key.split(".", 1)
                        default_timeframes = {
                            "1H": TimeframeParameters(
                                timeframe="1H", tp_pips=25.0, sl_pips=15.0,
                                max_holding_bars=12, weight=0.6
                            ),
                            "4H": TimeframeParameters(
                                timeframe="4H", tp_pips=50.0, sl_pips=25.0,
                                max_holding_bars=18, weight=0.3
                            ),
                            "D": TimeframeParameters(
                                timeframe="D", tp_pips=150.0, sl_pips=75.0,
                                max_holding_bars=15, weight=0.1
                            ),
                        }
                        if timeframe in default_timeframes:
                            default_value = getattr(default_timeframes[timeframe], param)
                            self.update_timeframe(
                                timeframe, {param: default_value}, "system", "Reset to default", db_session
                            )
                            return {
                                "status": "success",
                                "message": f"Reset timeframes.{key} to default",
                                "value": default_value,
                            }
                    raise ValueError(f"Unknown parameter: timeframes.{key}")
                else:
                    default_obj = defaults[category]
                    if not hasattr(default_obj, key):
                        raise ValueError(f"Unknown parameter: {category}.{key}")

                    default_value = getattr(default_obj, key)
                    self.update(category, {key: default_value}, "system", "Reset to default", db_session)

                    return {
                        "status": "success",
                        "message": f"Reset {category}.{key} to default",
                        "value": default_value,
                    }

            elif category:
                # Reset entire category - convert to updates dict for persistence
                if category == "timeframes":
                    # Special handling for timeframes
                    default_timeframes = {
                        "1H": TimeframeParameters(
                            timeframe="1H", tp_pips=25.0, sl_pips=15.0,
                            max_holding_bars=12, weight=0.6
                        ),
                        "4H": TimeframeParameters(
                            timeframe="4H", tp_pips=50.0, sl_pips=25.0,
                            max_holding_bars=18, weight=0.3
                        ),
                        "D": TimeframeParameters(
                            timeframe="D", tp_pips=150.0, sl_pips=75.0,
                            max_holding_bars=15, weight=0.1
                        ),
                    }
                    for tf, default_params in default_timeframes.items():
                        updates = {
                            key: getattr(default_params, key)
                            for key in default_params.__dataclass_fields__.keys()
                            if key != "timeframe"  # Don't update timeframe identifier
                        }
                        if db_session:
                            self.update_timeframe(tf, updates, "system", f"Reset {tf} to defaults", db_session)
                        else:
                            self.timeframes[tf] = default_params
                            self._trigger_callbacks("timeframes", self.timeframes)
                    return {
                        "status": "success",
                        "message": "Reset all timeframes to defaults",
                    }
                else:
                    default_obj = defaults[category]
                    updates = {
                        key: getattr(default_obj, key)
                        for key in default_obj.__dataclass_fields__.keys()
                    }

                    # Use update method to ensure DB persistence
                    if db_session:
                        self.update(category, updates, "system", f"Reset {category} to defaults", db_session)
                    else:
                        # No DB session - update memory only (for backwards compatibility)
                        setattr(self, category, default_obj)
                        self._trigger_callbacks(category, getattr(self, category))

                    return {
                        "status": "success",
                        "message": f"Reset {category} to defaults",
                    }

            else:
                # Reset all categories
                for cat, default_obj in defaults.items():
                    updates = {
                        key: getattr(default_obj, key)
                        for key in default_obj.__dataclass_fields__.keys()
                    }

                    # Use update method to ensure DB persistence
                    if db_session:
                        self.update(cat, updates, "system", "Reset all to defaults", db_session)
                    else:
                        # No DB session - update memory only (for backwards compatibility)
                        setattr(self, cat, default_obj)
                        self._trigger_callbacks(cat, getattr(self, cat))

                # Also reset timeframes
                default_timeframes = {
                    "1H": TimeframeParameters(
                        timeframe="1H", tp_pips=25.0, sl_pips=15.0,
                        max_holding_bars=12, weight=0.6
                    ),
                    "4H": TimeframeParameters(
                        timeframe="4H", tp_pips=50.0, sl_pips=25.0,
                        max_holding_bars=18, weight=0.3
                    ),
                    "D": TimeframeParameters(
                        timeframe="D", tp_pips=150.0, sl_pips=75.0,
                        max_holding_bars=15, weight=0.1
                    ),
                }
                for tf, default_params in default_timeframes.items():
                    updates = {
                        key: getattr(default_params, key)
                        for key in default_params.__dataclass_fields__.keys()
                        if key != "timeframe"
                    }
                    if db_session:
                        self.update_timeframe(tf, updates, "system", "Reset all to defaults", db_session)
                    else:
                        self.timeframes[tf] = default_params
                        self._trigger_callbacks("timeframes", self.timeframes)

                return {
                    "status": "success",
                    "message": "Reset all configuration to defaults",
                }


# Global singleton instance
trading_config = TradingConfig()
