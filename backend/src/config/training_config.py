"""Training configuration.

This module defines parameters for model training:
- Data split parameters (train/val/test ratios)
- Stacking parameters (cross-validation settings)
- Early stopping parameters
"""

from dataclasses import dataclass, field
from typing import Dict, Any


@dataclass
class DataSplitParameters:
    """Train/validation/test split configuration."""

    train_ratio: float = 0.6  # 60%
    validation_ratio: float = 0.2  # 20%
    test_ratio: float = 0.2  # 20% (implicit)

    # Ensure chronological order for time series
    enforce_chronological: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "train_ratio": self.train_ratio,
            "validation_ratio": self.validation_ratio,
            "test_ratio": self.test_ratio,
            "enforce_chronological": self.enforce_chronological,
        }


@dataclass
class StackingParameters:
    """Stacking meta-learner configuration."""

    n_folds: int = 5
    min_train_size: int = 500
    shuffle: bool = False  # Time series - no shuffle
    stratified: bool = True

    # Meta-learner hyperparameters (can be same as base or different)
    use_base_hyperparams: bool = True
    custom_hyperparams: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "n_folds": self.n_folds,
            "min_train_size": self.min_train_size,
            "shuffle": self.shuffle,
            "stratified": self.stratified,
            "use_base_hyperparams": self.use_base_hyperparams,
            "custom_hyperparams": self.custom_hyperparams,
        }


@dataclass
class EarlyStoppingParameters:
    """Early stopping configuration for XGBoost."""

    enabled: bool = True
    stopping_rounds: int = 10
    eval_metric: str = "logloss"
    verbose: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "enabled": self.enabled,
            "stopping_rounds": self.stopping_rounds,
            "eval_metric": self.eval_metric,
            "verbose": self.verbose,
        }


@dataclass
class TrainingParameters:
    """Complete training configuration."""

    splits: DataSplitParameters = field(default_factory=DataSplitParameters)
    stacking: StackingParameters = field(default_factory=StackingParameters)
    early_stopping: EarlyStoppingParameters = field(
        default_factory=EarlyStoppingParameters
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "splits": self.splits.to_dict(),
            "stacking": self.stacking.to_dict(),
            "early_stopping": self.early_stopping.to_dict(),
        }
