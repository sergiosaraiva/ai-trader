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

    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("idx_predictions_timestamp_symbol", "timestamp", "symbol"),
    )


class Trade(Base):
    """Store paper trade history."""

    __tablename__ = "trades"

    id = Column(Integer, primary_key=True, autoincrement=True)
    prediction_id = Column(Integer, nullable=True)  # Link to prediction
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
    status = Column(String(20), nullable=False, default="open")  # "open", "closed"

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        Index("idx_trades_status", "status"),
        Index("idx_trades_entry_time", "entry_time"),
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
