"""Model hyperparameter configuration.

This module defines XGBoost hyperparameters for each timeframe model:
- 1H Model: Short-term (highest weight 60%)
- 4H Model: Medium-term (30% weight)
- Daily Model: Long-term (10% weight + sentiment)
"""

from dataclasses import dataclass, field
from typing import Dict, Any


@dataclass
class XGBoostHyperparameters:
    """XGBoost hyperparameters for a single model."""

    n_estimators: int
    max_depth: int
    learning_rate: float
    min_child_weight: int = 3
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_alpha: float = 0.1
    reg_lambda: float = 1.0
    gamma: float = 0.1
    eval_metric: str = "logloss"
    random_state: int = 42

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "min_child_weight": self.min_child_weight,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "reg_alpha": self.reg_alpha,
            "reg_lambda": self.reg_lambda,
            "gamma": self.gamma,
            "eval_metric": self.eval_metric,
            "random_state": self.random_state,
        }


@dataclass
class ModelHyperparameters:
    """Hyperparameters for all timeframe models."""

    # 1H Model: Short-term (highest weight 60%)
    model_1h: XGBoostHyperparameters = field(
        default_factory=lambda: XGBoostHyperparameters(
            n_estimators=150, max_depth=5, learning_rate=0.03
        )
    )

    # 4H Model: Medium-term (30% weight)
    model_4h: XGBoostHyperparameters = field(
        default_factory=lambda: XGBoostHyperparameters(
            n_estimators=120, max_depth=4, learning_rate=0.03
        )
    )

    # Daily Model: Long-term (10% weight + sentiment)
    model_daily: XGBoostHyperparameters = field(
        default_factory=lambda: XGBoostHyperparameters(
            n_estimators=80, max_depth=3, learning_rate=0.03
        )
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model_1h": self.model_1h.to_dict(),
            "model_4h": self.model_4h.to_dict(),
            "model_daily": self.model_daily.to_dict(),
        }
