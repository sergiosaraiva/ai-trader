"""SQLAlchemy database models for AI-Trader."""

from datetime import datetime
from typing import Optional

from sqlalchemy import (
    Column,
    Integer,
    Float,
    String,
    DateTime,
    Boolean,
    JSON,
    Text,
    ForeignKey,
    Enum as SQLEnum,
    Index,
)
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class Prediction(Base):
    """Store model predictions."""

    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    symbol = Column(String(20), nullable=False, default="EURUSD")

    # Prediction details
    direction = Column(String(10), nullable=False)  # "long" or "short"
    confidence = Column(Float, nullable=False)
    prob_up = Column(Float, nullable=True)
    prob_down = Column(Float, nullable=True)

    # Component model predictions
    pred_1h = Column(Integer, nullable=True)  # 0=down, 1=up
    conf_1h = Column(Float, nullable=True)
    pred_4h = Column(Integer, nullable=True)
    conf_4h = Column(Float, nullable=True)
    pred_d = Column(Integer, nullable=True)
    conf_d = Column(Float, nullable=True)

    # Agreement info
    agreement_count = Column(Integer, nullable=True)
    agreement_score = Column(Float, nullable=True)

    # Market context
    market_regime = Column(String(20), nullable=True)
    market_price = Column(Float, nullable=True)
    vix_value = Column(Float, nullable=True)

    # Whether a trade was executed based on this prediction
    trade_executed = Column(Boolean, default=False)

    # Whether confidence meets 70% threshold for trading
    should_trade = Column(Boolean, nullable=False, default=True)

    # Agent tracking fields
    used_by_agent = Column(Boolean, default=False)
    agent_cycle_number = Column(Integer, nullable=True)

    # Dynamic threshold tracking
    dynamic_threshold_used = Column(Float, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("idx_predictions_timestamp_symbol", "timestamp", "symbol"),
        Index("idx_predictions_agent_cycle", "agent_cycle_number"),
    )


class Trade(Base):
    """Store paper trade history."""

    __tablename__ = "trades"

    id = Column(Integer, primary_key=True, autoincrement=True)
    prediction_id = Column(Integer, ForeignKey("predictions.id", ondelete="SET NULL"), nullable=True)  # Link to prediction
    symbol = Column(String(20), nullable=False, default="EURUSD")

    # Trade details
    direction = Column(String(10), nullable=False)  # "long" or "short"
    entry_price = Column(Float, nullable=False)
    entry_time = Column(DateTime, nullable=False, index=True)

    # Exit details (null if still open)
    exit_price = Column(Float, nullable=True)
    exit_time = Column(DateTime, nullable=True)
    exit_reason = Column(String(20), nullable=True)  # "tp", "sl", "timeout", "manual"

    # Position sizing
    lot_size = Column(Float, nullable=False, default=0.1)
    position_value = Column(Float, nullable=True)  # USD value

    # Risk parameters
    take_profit = Column(Float, nullable=True)
    stop_loss = Column(Float, nullable=True)
    max_holding_bars = Column(Integer, nullable=True)

    # Result (null if still open)
    pips = Column(Float, nullable=True)
    pnl_usd = Column(Float, nullable=True)
    is_winner = Column(Boolean, nullable=True)

    # Trade metadata
    confidence = Column(Float, nullable=True)
    risk_percentage_used = Column(Float, nullable=True)  # Risk % used for position sizing
    status = Column(String(20), nullable=False, default="open")  # "open", "closed"

    # Agent execution fields
    execution_mode = Column(String(20), nullable=False, default="simulation")  # "simulation", "paper", "live"
    broker = Column(String(50), nullable=True)  # "mt5", "alpaca", etc.
    mt5_ticket = Column(Integer, nullable=True)  # MT5 order ticket number
    explanation_id = Column(Integer, nullable=True)  # Note: TradeExplanation has FK to trades, not vice versa

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        Index("idx_trades_status", "status"),
        Index("idx_trades_entry_time", "entry_time"),
        Index("idx_trades_execution_mode", "execution_mode"),
    )


class PerformanceSnapshot(Base):
    """Store periodic performance snapshots."""

    __tablename__ = "performance_snapshots"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, nullable=False, index=True)

    # Account metrics
    balance = Column(Float, nullable=False)
    equity = Column(Float, nullable=False)
    unrealized_pnl = Column(Float, nullable=False, default=0.0)

    # Trading metrics
    total_trades = Column(Integer, nullable=False, default=0)
    winning_trades = Column(Integer, nullable=False, default=0)
    losing_trades = Column(Integer, nullable=False, default=0)
    win_rate = Column(Float, nullable=True)

    # P&L metrics
    total_pips = Column(Float, nullable=False, default=0.0)
    total_pnl_usd = Column(Float, nullable=False, default=0.0)
    profit_factor = Column(Float, nullable=True)
    avg_pips_per_trade = Column(Float, nullable=True)

    # Risk metrics
    max_drawdown = Column(Float, nullable=True)
    sharpe_ratio = Column(Float, nullable=True)

    # Prediction accuracy
    predictions_made = Column(Integer, nullable=False, default=0)
    predictions_correct = Column(Integer, nullable=False, default=0)
    prediction_accuracy = Column(Float, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (Index("idx_performance_timestamp", "timestamp"),)


class MarketData(Base):
    """Cache for market data (optional, for quick lookup)."""

    __tablename__ = "market_data"

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)

    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Float, nullable=True)

    timeframe = Column(String(10), nullable=False, default="5min")

    __table_args__ = (
        Index("idx_market_data_symbol_timestamp", "symbol", "timestamp"),
    )


class AgentCommand(Base):
    """Commands from backend to agent (start, stop, pause, resume, kill, update_config)."""

    __tablename__ = "agent_commands"

    id = Column(Integer, primary_key=True, autoincrement=True)
    command = Column(String(50), nullable=False)  # start, stop, pause, resume, kill, update_config
    payload = Column(JSON, nullable=True)  # Command parameters
    status = Column(String(20), nullable=False, default="pending")  # pending, processing, completed, failed
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    processed_at = Column(DateTime, nullable=True)
    result = Column(JSON, nullable=True)  # Execution result
    error_message = Column(Text, nullable=True)

    __table_args__ = (
        Index("idx_agent_commands_status", "status"),
        Index("idx_agent_commands_created", "created_at"),
    )


class AgentState(Base):
    """Current agent status for crash recovery and status reporting."""

    __tablename__ = "agent_state"

    id = Column(Integer, primary_key=True, autoincrement=True)
    status = Column(String(20), nullable=False)  # stopped, starting, running, paused, stopping, error
    mode = Column(String(20), nullable=False)  # simulation, paper, live
    cycle_count = Column(Integer, nullable=False, default=0)
    last_cycle_at = Column(DateTime, nullable=True)
    last_prediction = Column(JSON, nullable=True)
    last_signal = Column(JSON, nullable=True)
    account_equity = Column(Float, nullable=True)
    open_positions = Column(Integer, nullable=False, default=0)
    circuit_breaker_state = Column(String(50), nullable=True)
    kill_switch_active = Column(Boolean, nullable=False, default=False)
    error_message = Column(Text, nullable=True)
    config = Column(JSON, nullable=False)
    started_at = Column(DateTime, nullable=True)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow, index=True)

    __table_args__ = (
        Index("idx_agent_state_status", "status"),
        Index("idx_agent_state_updated", "updated_at"),
    )


class TradeExplanation(Base):
    """LLM explanations linked to trades."""

    __tablename__ = "trade_explanations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    trade_id = Column(Integer, ForeignKey("trades.id"), nullable=False, index=True)
    prediction_id = Column(Integer, ForeignKey("predictions.id"), nullable=True, index=True)
    explanation = Column(Text, nullable=False)
    confidence_factors = Column(JSON, nullable=True)
    risk_factors = Column(JSON, nullable=True)
    llm_model = Column(String(50), nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    __table_args__ = (
        Index("idx_trade_explanations_trade", "trade_id"),
        Index("idx_trade_explanations_prediction", "prediction_id"),
    )


class CircuitBreakerEvent(Base):
    """Audit trail for safety system triggers.

    Persists circuit breaker state across service restarts to enforce daily loss limits.
    """

    __tablename__ = "circuit_breaker_events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    breaker_type = Column(String(50), nullable=False, index=True)  # daily_loss_limit, consecutive_losses, etc.
    action = Column(String(20), nullable=False)  # "triggered" or "recovered"
    triggered_at = Column(DateTime(timezone=True), nullable=False, index=True)
    recovered_at = Column(DateTime(timezone=True), nullable=True)
    value = Column(Float, nullable=False)  # The value that caused trigger
    event_metadata = Column(JSON, nullable=True)  # Additional context (renamed from 'metadata' to avoid SQLAlchemy reserved word)

    created_at = Column(DateTime(timezone=True), default=lambda: datetime.utcnow())

    __table_args__ = (
        Index("idx_circuit_breaker_type_action", "breaker_type", "action"),
        Index("idx_circuit_breaker_triggered", "triggered_at"),
    )


class ConfigurationSetting(Base):
    """Centralized configuration settings with versioning."""

    __tablename__ = "configuration_settings"

    id = Column(Integer, primary_key=True, autoincrement=True)
    category = Column(String(50), nullable=False, index=True)  # trading, model, risk, system
    key = Column(String(100), nullable=False, index=True)
    value = Column(JSON, nullable=False)  # Stores any JSON-serializable value
    value_type = Column(String(20), nullable=False)  # int, float, str, bool, dict, list
    description = Column(Text, nullable=True)

    # Versioning
    version = Column(Integer, nullable=False, default=1)

    # Audit trail
    updated_by = Column(String(100), nullable=True)  # User/service that made the change
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow, index=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    # Validation constraints (optional JSON schema for validation)
    constraints = Column(JSON, nullable=True)  # {"min": 0, "max": 1, "choices": [...]}

    __table_args__ = (
        Index("idx_config_category_key", "category", "key", unique=True),
        Index("idx_config_updated", "updated_at"),
    )


class ConfigurationHistory(Base):
    """Change history for configuration settings."""

    __tablename__ = "configuration_history"

    id = Column(Integer, primary_key=True, autoincrement=True)
    setting_id = Column(Integer, ForeignKey("configuration_settings.id", ondelete="CASCADE"), nullable=False, index=True)

    # Snapshot of change
    category = Column(String(50), nullable=False)
    key = Column(String(100), nullable=False)
    old_value = Column(JSON, nullable=True)
    new_value = Column(JSON, nullable=False)
    version = Column(Integer, nullable=False)

    # Audit info
    changed_by = Column(String(100), nullable=True)
    changed_at = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    reason = Column(Text, nullable=True)  # Optional reason for change

    __table_args__ = (
        Index("idx_config_history_setting", "setting_id"),
        Index("idx_config_history_changed", "changed_at"),
        Index("idx_config_history_category_key", "category", "key"),
    )


class ThresholdHistory(Base):
    """Historical record of dynamic confidence threshold calculations."""

    __tablename__ = "threshold_history"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, nullable=False, index=True)

    # Calculated threshold value
    threshold_value = Column(Float, nullable=False)

    # Threshold components
    short_term_component = Column(Float, nullable=True)
    medium_term_component = Column(Float, nullable=True)
    long_term_component = Column(Float, nullable=True)
    blended_value = Column(Float, nullable=True)
    performance_adjustment = Column(Float, nullable=True)

    # Data counts used
    prediction_count_7d = Column(Integer, nullable=False, default=0)
    prediction_count_14d = Column(Integer, nullable=False, default=0)
    prediction_count_30d = Column(Integer, nullable=False, default=0)

    # Performance metrics
    trade_win_rate_25 = Column(Float, nullable=True)
    trade_count_25 = Column(Integer, nullable=True)

    # Metadata
    reason = Column(String(200), nullable=True)  # "dynamic", "fallback_insufficient_data", etc.
    config_version = Column(Integer, nullable=True)  # Track which config was used

    __table_args__ = (
        Index("idx_threshold_history_timestamp", "timestamp"),
        Index("idx_threshold_history_reason", "reason"),
    )


class RiskReductionState(Base):
    """Track consecutive losses and progressive risk reduction state.

    Single-row singleton table that persists the current risk reduction state
    to survive service restarts.
    """

    __tablename__ = "risk_reduction_state"

    id = Column(Integer, primary_key=True, autoincrement=True)
    consecutive_losses = Column(Integer, nullable=False, default=0)
    risk_reduction_factor = Column(Float, nullable=False, default=1.0)
    last_trade_id = Column(Integer, ForeignKey("trades.id", ondelete="SET NULL"), nullable=True, index=True)
    updated_at = Column(DateTime, nullable=False, default=lambda: datetime.utcnow(), onupdate=lambda: datetime.utcnow())

    __table_args__ = (
        Index("idx_risk_reduction_updated", "updated_at"),
    )
