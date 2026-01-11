"""Training configuration classes.

Provides comprehensive configuration for training sessions including:
- Optimizer settings
- Learning rate schedulers
- Early stopping criteria
- Model checkpointing
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


class SchedulerType(Enum):
    """Learning rate scheduler types."""

    NONE = "none"
    STEP = "step"
    COSINE = "cosine"
    COSINE_WARM_RESTARTS = "cosine_warm_restarts"
    ONE_CYCLE = "one_cycle"
    REDUCE_ON_PLATEAU = "reduce_on_plateau"
    EXPONENTIAL = "exponential"


class OptimizerType(Enum):
    """Optimizer types."""

    ADAM = "adam"
    ADAMW = "adamw"
    SGD = "sgd"
    RMSPROP = "rmsprop"
    RADAM = "radam"


@dataclass
class OptimizerConfig:
    """Optimizer configuration.

    Attributes:
        optimizer_type: Type of optimizer to use.
        learning_rate: Initial learning rate.
        weight_decay: L2 regularization coefficient.
        betas: Adam betas for momentum.
        momentum: SGD momentum (if using SGD).
        eps: Epsilon for numerical stability.
    """

    optimizer_type: OptimizerType = OptimizerType.ADAMW
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    betas: tuple = (0.9, 0.999)
    momentum: float = 0.9
    eps: float = 1e-8

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for optimizer creation."""
        return {
            "optimizer_type": self.optimizer_type.value,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "betas": list(self.betas),  # Convert tuple to list for YAML serialization
            "momentum": self.momentum,
            "eps": self.eps,
        }


@dataclass
class SchedulerConfig:
    """Learning rate scheduler configuration.

    Attributes:
        scheduler_type: Type of scheduler.
        step_size: Step size for StepLR.
        gamma: Decay factor.
        T_max: Max iterations for CosineAnnealing.
        T_0: Initial period for CosineWarmRestarts.
        T_mult: Period multiplier for CosineWarmRestarts.
        eta_min: Minimum learning rate.
        patience: Patience for ReduceOnPlateau.
        factor: Reduction factor for ReduceOnPlateau.
        pct_start: Percentage of cycle for warmup (OneCycle).
        div_factor: Initial LR divisor (OneCycle).
        final_div_factor: Final LR divisor (OneCycle).
    """

    scheduler_type: SchedulerType = SchedulerType.COSINE
    step_size: int = 10
    gamma: float = 0.1
    T_max: int = 100
    T_0: int = 10
    T_mult: int = 2
    eta_min: float = 1e-7
    patience: int = 10
    factor: float = 0.5
    pct_start: float = 0.3
    div_factor: float = 25.0
    final_div_factor: float = 10000.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "scheduler_type": self.scheduler_type.value,
            "step_size": self.step_size,
            "gamma": self.gamma,
            "T_max": self.T_max,
            "T_0": self.T_0,
            "T_mult": self.T_mult,
            "eta_min": self.eta_min,
            "patience": self.patience,
            "factor": self.factor,
            "pct_start": self.pct_start,
            "div_factor": self.div_factor,
            "final_div_factor": self.final_div_factor,
        }


@dataclass
class EarlyStoppingConfig:
    """Early stopping configuration.

    Attributes:
        enabled: Whether early stopping is enabled.
        patience: Number of epochs to wait for improvement.
        min_delta: Minimum change to qualify as improvement.
        monitor: Metric to monitor ('val_loss', 'val_accuracy', etc.).
        mode: 'min' for loss, 'max' for accuracy.
        restore_best_weights: Whether to restore best weights on stop.
        baseline: Minimum required value before stopping starts.
        start_from_epoch: Epoch to start monitoring from.
    """

    enabled: bool = True
    patience: int = 15
    min_delta: float = 1e-5
    monitor: str = "val_loss"
    mode: str = "min"
    restore_best_weights: bool = True
    baseline: Optional[float] = None
    start_from_epoch: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "enabled": self.enabled,
            "patience": self.patience,
            "min_delta": self.min_delta,
            "monitor": self.monitor,
            "mode": self.mode,
            "restore_best_weights": self.restore_best_weights,
            "baseline": self.baseline,
            "start_from_epoch": self.start_from_epoch,
        }


@dataclass
class CheckpointConfig:
    """Model checkpoint configuration.

    Attributes:
        enabled: Whether checkpointing is enabled.
        save_dir: Directory to save checkpoints.
        save_best_only: Only save when metric improves.
        save_freq: Save frequency ('epoch' or number of epochs).
        monitor: Metric to monitor for best model.
        mode: 'min' for loss, 'max' for accuracy.
        max_to_keep: Maximum checkpoints to keep.
        save_weights_only: Only save weights, not full model.
    """

    enabled: bool = True
    save_dir: Union[str, Path] = "checkpoints"
    save_best_only: bool = True
    save_freq: Union[str, int] = "epoch"
    monitor: str = "val_loss"
    mode: str = "min"
    max_to_keep: int = 3
    save_weights_only: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "enabled": self.enabled,
            "save_dir": str(self.save_dir),
            "save_best_only": self.save_best_only,
            "save_freq": self.save_freq,
            "monitor": self.monitor,
            "mode": self.mode,
            "max_to_keep": self.max_to_keep,
            "save_weights_only": self.save_weights_only,
        }


@dataclass
class TrainingConfig:
    """Complete training configuration.

    This is the main configuration class that combines all training settings.

    Attributes:
        name: Name for the training run.
        batch_size: Training batch size.
        epochs: Maximum number of training epochs.
        gradient_clip: Maximum gradient norm for clipping.
        accumulation_steps: Gradient accumulation steps.
        mixed_precision: Whether to use mixed precision training.
        seed: Random seed for reproducibility.
        num_workers: DataLoader workers.
        pin_memory: Pin memory for faster GPU transfer.
        optimizer: Optimizer configuration.
        scheduler: Learning rate scheduler configuration.
        early_stopping: Early stopping configuration.
        checkpoint: Checkpoint configuration.
        device: Device to train on ('auto', 'cpu', 'cuda', 'mps').
        verbose: Verbosity level (0=silent, 1=progress, 2=detailed).
        log_every_n_steps: Log metrics every N steps.
        val_check_interval: Validation check interval.
        architecture_config: Architecture-specific configuration.
    """

    name: str = "training_run"
    batch_size: int = 64
    epochs: int = 100
    gradient_clip: float = 1.0
    accumulation_steps: int = 1
    mixed_precision: bool = False
    seed: int = 42
    num_workers: int = 0
    pin_memory: bool = True
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    early_stopping: EarlyStoppingConfig = field(default_factory=EarlyStoppingConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    device: str = "auto"
    verbose: int = 1
    log_every_n_steps: int = 10
    val_check_interval: Union[float, int] = 1.0
    architecture_config: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Post-initialization processing."""
        if isinstance(self.optimizer, dict):
            self.optimizer = OptimizerConfig(**self.optimizer)
        if isinstance(self.scheduler, dict):
            self.scheduler = SchedulerConfig(**self.scheduler)
        if isinstance(self.early_stopping, dict):
            self.early_stopping = EarlyStoppingConfig(**self.early_stopping)
        if isinstance(self.checkpoint, dict):
            self.checkpoint = CheckpointConfig(**self.checkpoint)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "gradient_clip": self.gradient_clip,
            "accumulation_steps": self.accumulation_steps,
            "mixed_precision": self.mixed_precision,
            "seed": self.seed,
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
            "optimizer": self.optimizer.to_dict(),
            "scheduler": self.scheduler.to_dict(),
            "early_stopping": self.early_stopping.to_dict(),
            "checkpoint": self.checkpoint.to_dict(),
            "device": self.device,
            "verbose": self.verbose,
            "log_every_n_steps": self.log_every_n_steps,
            "val_check_interval": self.val_check_interval,
            "architecture_config": self.architecture_config,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrainingConfig":
        """Create from dictionary."""
        return cls(**data)

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "TrainingConfig":
        """Load configuration from YAML file."""
        import yaml

        with open(path, "r") as f:
            data = yaml.safe_load(f)

        return cls.from_dict(data)

    def save_yaml(self, path: Union[str, Path]) -> None:
        """Save configuration to YAML file."""
        import yaml

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)

    def get_device(self) -> str:
        """Resolve the actual device to use."""
        if self.device != "auto":
            return self.device

        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass

        return "cpu"
