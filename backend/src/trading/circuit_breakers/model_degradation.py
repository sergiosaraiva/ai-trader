"""
Model Degradation Circuit Breaker.

Detects when ML model performance has degraded due to
market regime changes or model miscalibration.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional
from collections import deque

from .base import (
    CircuitBreaker,
    CircuitBreakerAction,
    TradeResult,
    TradingState,
    Severity,
    RecoveryRequirement,
)


@dataclass
class ModelDegradationBreaker(CircuitBreaker):
    """
    Detects when model performance has degraded.

    DISTINGUISHES BETWEEN:
    1. Normal variance: Model is still valid, just unlucky
    2. Model degradation: Market has changed, model needs recalibration
    3. Market unpredictability: No model would work in current conditions

    DETECTION SIGNALS:
    - Rolling accuracy drops below threshold
    - Ensemble disagreement increases
    - Confidence is high but accuracy is low (miscalibration)
    - Feature distribution shift

    KEY INSIGHT:
    High confidence + low accuracy = BROKEN MODEL
    Low confidence + low accuracy = TOUGH MARKET (less severe)
    """

    # Thresholds
    min_rolling_accuracy: float = 0.45  # Below this = degraded
    rolling_window: int = 20  # Number of trades to consider
    max_ensemble_disagreement: float = 0.30  # Std dev of predictions
    miscalibration_threshold: float = 0.20  # Confidence - accuracy gap

    # Tracking
    recent_trades: List[TradeResult] = field(default_factory=list)

    @property
    def name(self) -> str:
        return "ModelDegradationBreaker"

    def record_trade(
        self,
        trade: TradeResult,
    ) -> CircuitBreakerAction:
        """
        Record trade and analyze model performance.

        Args:
            trade: Completed trade with confidence and agreement info

        Returns:
            CircuitBreakerAction based on analysis
        """
        self.recent_trades.append(trade)

        # Keep only rolling window
        if len(self.recent_trades) > self.rolling_window:
            self.recent_trades.pop(0)

        return self.check(
            confidence=trade.confidence_at_entry,
            ensemble_agreement=trade.ensemble_agreement,
        )

    def check(
        self,
        confidence: float = None,
        ensemble_agreement: float = None,
        **kwargs
    ) -> CircuitBreakerAction:
        """
        Analyze model performance and return action.

        Args:
            confidence: Most recent prediction confidence
            ensemble_agreement: Most recent ensemble agreement

        Returns:
            CircuitBreakerAction based on analysis
        """
        # Need minimum data
        if len(self.recent_trades) < self.rolling_window // 2:
            return CircuitBreakerAction(action=TradingState.ACTIVE)

        # Calculate rolling metrics
        wins = sum(1 for t in self.recent_trades if t.is_win)
        rolling_accuracy = wins / len(self.recent_trades)

        # Calculate average confidence and agreement
        avg_confidence = sum(t.confidence_at_entry for t in self.recent_trades) / len(self.recent_trades)
        avg_agreement = sum(t.ensemble_agreement for t in self.recent_trades) / len(self.recent_trades)

        # Check for miscalibration (MOST SEVERE)
        # High confidence + low accuracy = broken model
        calibration_gap = avg_confidence - rolling_accuracy

        if calibration_gap > self.miscalibration_threshold and rolling_accuracy < 0.45:
            return CircuitBreakerAction(
                action=TradingState.HALTED,
                reason=f"Model miscalibrated: {avg_confidence:.1%} confidence, {rolling_accuracy:.1%} accuracy",
                severity=Severity.CRITICAL,
                recovery_requirement=RecoveryRequirement(
                    cooldown_hours=168,  # 1 week - needs investigation
                    requires_recalibration=True,
                    reduced_size_on_resume=0.25,
                    wins_to_restore=5,
                )
            )

        # High ensemble disagreement = uncertain market
        if ensemble_agreement is not None and ensemble_agreement < (1 - self.max_ensemble_disagreement):
            return CircuitBreakerAction(
                action=TradingState.REDUCED,
                reason=f"High model disagreement: agreement={ensemble_agreement:.1%}",
                severity=Severity.MEDIUM,
                size_multiplier=0.5,
                min_confidence_override=0.80,
            )

        # Low accuracy but well-calibrated = tough market (less severe)
        if rolling_accuracy < self.min_rolling_accuracy:
            # Check if model "knows" it's struggling (low confidence)
            if avg_confidence < 0.60:
                # Model is appropriately uncertain - just reduce size
                return CircuitBreakerAction(
                    action=TradingState.REDUCED,
                    reason=f"Rolling accuracy low ({rolling_accuracy:.1%}), but model is appropriately uncertain",
                    severity=Severity.LOW,
                    size_multiplier=0.50,
                )
            else:
                # Model is confident but wrong - more serious
                return CircuitBreakerAction(
                    action=TradingState.REDUCED,
                    reason=f"Rolling accuracy low ({rolling_accuracy:.1%}) with confidence {avg_confidence:.1%}",
                    severity=Severity.MEDIUM,
                    size_multiplier=0.25,
                    min_confidence_override=0.85,
                )

        return CircuitBreakerAction(action=TradingState.ACTIVE)

    def reset(self) -> None:
        """Reset breaker state."""
        self.recent_trades.clear()

    def get_stats(self) -> dict:
        """Get current breaker statistics."""
        if not self.recent_trades:
            return {
                'trades_tracked': 0,
                'rolling_window': self.rolling_window,
            }

        wins = sum(1 for t in self.recent_trades if t.is_win)
        rolling_accuracy = wins / len(self.recent_trades)
        avg_confidence = sum(t.confidence_at_entry for t in self.recent_trades) / len(self.recent_trades)
        avg_agreement = sum(t.ensemble_agreement for t in self.recent_trades) / len(self.recent_trades)

        return {
            'trades_tracked': len(self.recent_trades),
            'rolling_window': self.rolling_window,
            'rolling_accuracy': rolling_accuracy,
            'avg_confidence': avg_confidence,
            'avg_agreement': avg_agreement,
            'calibration_gap': avg_confidence - rolling_accuracy,
            'min_accuracy_threshold': self.min_rolling_accuracy,
        }
