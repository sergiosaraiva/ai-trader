"""Data models for agent trading cycles.

Defines result types and intermediate data structures for the agent's
trading cycle execution.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any


@dataclass
class CycleResult:
    """Result of a single trading cycle.

    Captures what happened during one iteration of the trading loop:
    - Whether a prediction was made
    - Whether a signal was generated
    - What action was taken
    - Any errors encountered
    """

    cycle_number: int
    timestamp: datetime

    # What happened in this cycle
    prediction_made: bool = False
    signal_generated: bool = False
    action_taken: str = "none"  # none, signal_generated, hold

    # Details (if available)
    prediction: Optional[Dict[str, Any]] = None
    signal: Optional[Dict[str, Any]] = None

    # Performance
    duration_ms: float = 0.0

    # Error tracking
    error: Optional[str] = None

    # Reason for action (or no action)
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/storage."""
        return {
            "cycle_number": self.cycle_number,
            "timestamp": self.timestamp.isoformat(),
            "prediction_made": self.prediction_made,
            "signal_generated": self.signal_generated,
            "action_taken": self.action_taken,
            "duration_ms": self.duration_ms,
            "error": self.error,
            "reason": self.reason,
        }

    @property
    def success(self) -> bool:
        """Whether cycle completed without errors."""
        return self.error is None


@dataclass
class PredictionData:
    """Prediction details from model service.

    Simplified view of model_service prediction output for state storage.
    """

    direction: str  # "long" or "short"
    confidence: float
    prob_up: float
    prob_down: float
    should_trade: bool
    agreement_count: int
    agreement_score: float
    market_regime: str
    component_directions: Dict[str, int]
    component_confidences: Dict[str, float]
    timestamp: datetime
    symbol: str

    @classmethod
    def from_service_output(cls, output: Dict[str, Any]) -> "PredictionData":
        """Create from model_service.predict() output.

        Safely extracts fields with defaults for missing keys.
        """
        try:
            return cls(
                direction=output.get("direction", "short"),
                confidence=output.get("confidence", 0.0),
                prob_up=output.get("prob_up", 0.5),
                prob_down=output.get("prob_down", 0.5),
                should_trade=output.get("should_trade", False),
                agreement_count=output.get("agreement_count", 0),
                agreement_score=output.get("agreement_score", 0.0),
                market_regime=output.get("market_regime", "unknown"),
                component_directions=output.get("component_directions", {}),
                component_confidences=output.get("component_confidences", {}),
                timestamp=datetime.fromisoformat(output.get("timestamp", datetime.now().isoformat())),
                symbol=output.get("symbol", "EURUSD"),
            )
        except (KeyError, ValueError, TypeError) as e:
            # Log the error and re-raise with more context
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Failed to parse prediction output: {e}, output keys: {list(output.keys())}")
            raise ValueError(f"Invalid prediction output format: {e}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "direction": self.direction,
            "confidence": self.confidence,
            "prob_up": self.prob_up,
            "prob_down": self.prob_down,
            "should_trade": self.should_trade,
            "agreement_count": self.agreement_count,
            "agreement_score": self.agreement_score,
            "market_regime": self.market_regime,
            "component_directions": self.component_directions,
            "component_confidences": self.component_confidences,
            "timestamp": self.timestamp.isoformat(),
            "symbol": self.symbol,
        }


@dataclass
class SignalData:
    """Trading signal details.

    Simplified view of signal for state storage.
    """

    action: str  # "buy", "sell", "hold"
    confidence: float
    reason: str
    position_size_pct: float = 0.0
    stop_loss_pct: float = 0.0
    take_profit_pct: float = 0.0
    timestamp: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "action": self.action,
            "confidence": self.confidence,
            "reason": self.reason,
            "position_size_pct": self.position_size_pct,
            "stop_loss_pct": self.stop_loss_pct,
            "take_profit_pct": self.take_profit_pct,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
        }


@dataclass
class TradeResult:
    """Result of trade execution."""
    success: bool
    trade_id: Optional[int] = None
    mt5_ticket: Optional[int] = None
    entry_price: Optional[float] = None
    error: Optional[str] = None


@dataclass
class PositionStatus:
    """Status of an open position."""
    trade_id: int
    mt5_ticket: int
    current_price: float
    unrealized_pnl: float
    should_close: bool
    close_reason: Optional[str] = None  # "tp", "sl", "timeout"


@dataclass
class ExitSignal:
    """Signal to exit a position."""
    trade_id: int
    reason: str  # "take_profit", "stop_loss", "timeout"
    exit_price: float
