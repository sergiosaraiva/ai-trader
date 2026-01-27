"""Conservative Hybrid position sizing implementation.

This module provides the ConservativeHybridSizer class that combines:
- Confidence-based scaling (higher confidence = higher risk)
- Fixed base risk with min/max caps
- No-leverage constraint (position notional <= account balance)
"""

import logging
from typing import Dict, Any, Tuple

logger = logging.getLogger(__name__)

# Minimum position size for most brokers (0.01 lots = 1 micro lot)
MIN_POSITION_SIZE = 0.01


class ConservativeHybridSizer:
    """Conservative Hybrid position sizing strategy.

    Combines confidence scaling with fixed risk management:
    1. Base risk adjusted by confidence multiplier
    2. Apply min/max caps
    3. Calculate position from risk
    4. Apply no-leverage constraint

    Formula:
        confidence_multiplier = 1.0 + (confidence - threshold) * scaling_factor
        adjusted_risk_pct = base_risk_pct * confidence_multiplier
        risk_pct_used = clamp(adjusted_risk_pct, min_risk, max_risk)
        position_lots = (balance * risk_pct_used / 100) / (sl_pips * pip_value)
        position_lots = min(position_lots, balance / lot_size)  # No leverage
    """

    def __init__(self):
        """Initialize position sizer."""
        pass

    def calculate_position_size(
        self,
        balance: float,
        confidence: float,
        sl_pips: float,
        config,
        pip_value: float = 10.0,
        lot_size: float = 100000.0,
        risk_reduction_factor: float = 1.0
    ) -> Tuple[float, float, Dict[str, Any]]:
        """Calculate position size using Conservative Hybrid strategy.

        Args:
            balance: Current account balance (USD)
            confidence: Model confidence (0.0-1.0)
            sl_pips: Stop loss in pips
            config: ConservativeHybridParameters configuration object
            pip_value: $ per pip per lot (default: 10.0 for 0.1 lot EUR/USD)
            lot_size: Notional value per lot (default: 100,000)
            risk_reduction_factor: Progressive risk reduction (0.2-1.0, default 1.0)

        Returns:
            Tuple of:
                - position_lots: Position size in lots
                - risk_pct_used: Actual risk percentage used
                - metadata: Dict with calculation details
        """
        # Step 1: Check confidence threshold
        if confidence < config.confidence_threshold:
            logger.debug(
                f"Confidence {confidence:.1%} below threshold {config.confidence_threshold:.1%}, "
                f"no position taken"
            )
            return 0.0, 0.0, {
                "reason": "confidence_below_threshold",
                "confidence": confidence,
                "threshold": config.confidence_threshold
            }

        # Step 2: Calculate confidence multiplier
        # Higher confidence = higher risk (but capped)
        confidence_multiplier = 1.0 + (confidence - config.confidence_threshold) * config.confidence_scaling_factor

        # Step 3: Adjust risk percentage
        adjusted_risk_pct = config.base_risk_percent * confidence_multiplier

        # Step 4: Apply min/max caps
        risk_pct_used = max(config.min_risk_percent, min(adjusted_risk_pct, config.max_risk_percent))

        # Step 4.5: Apply progressive risk reduction factor
        # This scales down the risk based on consecutive losses (never zero)
        if risk_reduction_factor < 1.0:
            logger.info(
                f"Applying progressive risk reduction: {risk_pct_used:.2f}% -> "
                f"{risk_pct_used * risk_reduction_factor:.2f}% (factor: {risk_reduction_factor:.2f})"
            )
        risk_pct_used = risk_pct_used * risk_reduction_factor

        # Step 5: Calculate position from risk
        # Risk = Position Size × SL Pips × Pip Value
        # Position Size (lots) = (Balance × Risk%) / (SL Pips × Pip Value)
        risk_amount = balance * (risk_pct_used / 100.0)
        if sl_pips <= 0:
            logger.warning(f"Invalid sl_pips: {sl_pips}, cannot calculate position")
            return 0.0, 0.0, {
                "reason": "invalid_sl_pips",
                "sl_pips": sl_pips
            }

        # M1: Division by zero protection for pip_value
        if pip_value <= 0:
            logger.warning(f"Invalid pip_value: {pip_value}, cannot calculate position")
            return 0.0, 0.0, {
                "reason": "invalid_pip_value",
                "pip_value": pip_value
            }

        desired_position_lots = risk_amount / (sl_pips * pip_value)

        # Step 6: Apply no-leverage constraint
        # Position notional value cannot exceed available balance
        # M1: Division by zero protection for lot_size
        if lot_size <= 0:
            logger.warning(f"Invalid lot_size: {lot_size}, cannot calculate position")
            return 0.0, 0.0, {
                "reason": "invalid_lot_size",
                "lot_size": lot_size
            }

        max_position_no_leverage = balance / lot_size
        final_position_lots = min(desired_position_lots, max_position_no_leverage)

        # M4: Check minimum position size
        if 0 < final_position_lots < MIN_POSITION_SIZE:
            logger.info(
                f"Position size {final_position_lots:.4f} below minimum {MIN_POSITION_SIZE}"
            )
            return 0.0, 0.0, {
                "reason": "below_minimum_position_size",
                "calculated_position": final_position_lots,
                "minimum_required": MIN_POSITION_SIZE
            }

        # Check if position was limited by cash constraint
        limited_by_cash = final_position_lots < desired_position_lots

        metadata = {
            "confidence": confidence,
            "confidence_multiplier": confidence_multiplier,
            "base_risk_pct": config.base_risk_percent,
            "adjusted_risk_pct": adjusted_risk_pct,
            "risk_pct_used": risk_pct_used,
            "risk_reduction_factor": risk_reduction_factor,
            "risk_amount_usd": risk_amount,
            "sl_pips": sl_pips,
            "desired_position_lots": desired_position_lots,
            "max_position_no_leverage": max_position_no_leverage,
            "final_position_lots": final_position_lots,
            "limited_by_cash": limited_by_cash,
        }

        if limited_by_cash:
            logger.info(
                f"Position limited by no-leverage constraint: {final_position_lots:.4f} lots "
                f"(desired: {desired_position_lots:.4f}, balance: ${balance:.2f})"
            )

        logger.debug(
            f"Position calculated: {final_position_lots:.4f} lots at {risk_pct_used:.2f}% risk "
            f"(confidence: {confidence:.1%}, balance: ${balance:.2f})"
        )

        return final_position_lots, risk_pct_used, metadata
