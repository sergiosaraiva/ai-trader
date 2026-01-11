"""Training session management.

Provides training state tracking, metrics history, and session management
for reproducible and monitorable training runs.
"""

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)


class SessionStatus(Enum):
    """Training session status."""

    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    STOPPED_EARLY = "stopped_early"
    FAILED = "failed"


@dataclass
class EpochMetrics:
    """Metrics for a single training epoch.

    Attributes:
        epoch: Epoch number (1-indexed).
        train_loss: Average training loss.
        val_loss: Average validation loss.
        train_metrics: Additional training metrics.
        val_metrics: Additional validation metrics.
        learning_rate: Learning rate at this epoch.
        duration_seconds: Time taken for this epoch.
        timestamp: When this epoch completed.
    """

    epoch: int
    train_loss: float
    val_loss: Optional[float] = None
    train_metrics: Dict[str, float] = field(default_factory=dict)
    val_metrics: Dict[str, float] = field(default_factory=dict)
    learning_rate: float = 0.0
    duration_seconds: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "epoch": self.epoch,
            "train_loss": self.train_loss,
            "val_loss": self.val_loss,
            "train_metrics": self.train_metrics,
            "val_metrics": self.val_metrics,
            "learning_rate": self.learning_rate,
            "duration_seconds": self.duration_seconds,
            "timestamp": self.timestamp.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EpochMetrics":
        """Create from dictionary."""
        data = data.copy()
        if isinstance(data.get("timestamp"), str):
            data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        return cls(**data)


@dataclass
class TrainingState:
    """Complete training state for checkpointing and resumption.

    Attributes:
        current_epoch: Current epoch (0-indexed for internal use).
        total_epochs: Total epochs to train.
        best_epoch: Epoch with best validation metric.
        best_val_loss: Best validation loss achieved.
        best_val_metric: Best validation metric value.
        monitor_metric: Metric being monitored for best model.
        epochs_without_improvement: Consecutive epochs without improvement.
        global_step: Total training steps completed.
        history: List of epoch metrics.
        status: Current session status.
        stop_reason: Reason for stopping if stopped early.
        started_at: When training started.
        ended_at: When training ended.
        total_duration_seconds: Total training time.
    """

    current_epoch: int = 0
    total_epochs: int = 100
    best_epoch: int = 0
    best_val_loss: float = float("inf")
    best_val_metric: float = float("inf")
    monitor_metric: str = "val_loss"
    epochs_without_improvement: int = 0
    global_step: int = 0
    history: List[EpochMetrics] = field(default_factory=list)
    status: SessionStatus = SessionStatus.PENDING
    stop_reason: Optional[str] = None
    started_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None
    total_duration_seconds: float = 0.0

    def update_best(
        self,
        epoch: int,
        val_loss: float,
        val_metric: Optional[float] = None,
        mode: str = "min",
    ) -> bool:
        """Update best model tracking.

        Args:
            epoch: Current epoch.
            val_loss: Current validation loss.
            val_metric: Current validation metric (uses val_loss if None).
            mode: 'min' if lower is better, 'max' if higher is better.

        Returns:
            True if this is a new best model.
        """
        metric_value = val_metric if val_metric is not None else val_loss

        is_improvement = False
        if mode == "min":
            is_improvement = metric_value < self.best_val_metric
        else:
            is_improvement = metric_value > self.best_val_metric

        if is_improvement:
            self.best_epoch = epoch
            self.best_val_loss = val_loss
            self.best_val_metric = metric_value
            self.epochs_without_improvement = 0
            return True
        else:
            self.epochs_without_improvement += 1
            return False

    def should_stop_early(self, patience: int, min_delta: float = 0.0) -> bool:
        """Check if training should stop early.

        Args:
            patience: Number of epochs to wait for improvement.
            min_delta: Minimum improvement to reset patience.

        Returns:
            True if training should stop.
        """
        return self.epochs_without_improvement >= patience

    def get_best_metrics(self) -> Dict[str, Any]:
        """Get metrics from best epoch."""
        if not self.history:
            return {}

        for metrics in self.history:
            if metrics.epoch == self.best_epoch:
                return metrics.to_dict()

        return {}

    def get_metric_history(self, metric: str) -> List[float]:
        """Get history of a specific metric.

        Args:
            metric: Metric name (e.g., 'train_loss', 'val_loss').

        Returns:
            List of metric values across epochs.
        """
        values = []
        for epoch_metrics in self.history:
            if metric == "train_loss":
                values.append(epoch_metrics.train_loss)
            elif metric == "val_loss":
                values.append(epoch_metrics.val_loss or float("nan"))
            elif metric == "learning_rate":
                values.append(epoch_metrics.learning_rate)
            elif metric in epoch_metrics.train_metrics:
                values.append(epoch_metrics.train_metrics[metric])
            elif metric in epoch_metrics.val_metrics:
                values.append(epoch_metrics.val_metrics[metric])
            else:
                values.append(float("nan"))
        return values

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "current_epoch": self.current_epoch,
            "total_epochs": self.total_epochs,
            "best_epoch": self.best_epoch,
            "best_val_loss": self.best_val_loss,
            "best_val_metric": self.best_val_metric,
            "monitor_metric": self.monitor_metric,
            "epochs_without_improvement": self.epochs_without_improvement,
            "global_step": self.global_step,
            "history": [m.to_dict() for m in self.history],
            "status": self.status.value,
            "stop_reason": self.stop_reason,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "ended_at": self.ended_at.isoformat() if self.ended_at else None,
            "total_duration_seconds": self.total_duration_seconds,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrainingState":
        """Create from dictionary."""
        data = data.copy()
        data["history"] = [EpochMetrics.from_dict(m) for m in data.get("history", [])]
        data["status"] = SessionStatus(data.get("status", "pending"))
        if data.get("started_at"):
            data["started_at"] = datetime.fromisoformat(data["started_at"])
        if data.get("ended_at"):
            data["ended_at"] = datetime.fromisoformat(data["ended_at"])
        return cls(**data)


class TrainingSession:
    """Manages a complete training session.

    Handles state tracking, checkpointing, and session lifecycle.

    Example:
        ```python
        session = TrainingSession(
            name="my_training",
            total_epochs=100,
            checkpoint_dir="checkpoints",
        )

        session.start()

        for epoch in range(session.state.total_epochs):
            train_loss = train_one_epoch()
            val_loss = validate()

            session.record_epoch(
                epoch=epoch + 1,
                train_loss=train_loss,
                val_loss=val_loss,
            )

            if session.should_stop():
                break

        session.end()
        ```
    """

    def __init__(
        self,
        name: str = "training_session",
        total_epochs: int = 100,
        checkpoint_dir: Optional[Union[str, Path]] = None,
        monitor: str = "val_loss",
        mode: str = "min",
        resume_from: Optional[Union[str, Path]] = None,
    ):
        """Initialize training session.

        Args:
            name: Session name.
            total_epochs: Total epochs to train.
            checkpoint_dir: Directory for checkpoints.
            monitor: Metric to monitor for best model.
            mode: 'min' or 'max' for monitored metric.
            resume_from: Path to resume from existing session.
        """
        self.name = name
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self.monitor = monitor
        self.mode = mode

        if resume_from:
            self.state = self._load_state(resume_from)
            logger.info(f"Resumed session from epoch {self.state.current_epoch}")
        else:
            self.state = TrainingState(
                total_epochs=total_epochs,
                monitor_metric=monitor,
            )
            # Set appropriate initial best value based on mode
            if mode == "max":
                self.state.best_val_metric = -float("inf")

        self._epoch_start_time: Optional[float] = None

    def start(self) -> None:
        """Start the training session."""
        self.state.status = SessionStatus.RUNNING
        self.state.started_at = datetime.now()
        logger.info(f"Training session '{self.name}' started")

    def end(self, reason: Optional[str] = None) -> None:
        """End the training session.

        Args:
            reason: Reason for ending (if early stopping).
        """
        self.state.ended_at = datetime.now()
        if self.state.started_at:
            self.state.total_duration_seconds = (
                self.state.ended_at - self.state.started_at
            ).total_seconds()

        if reason:
            self.state.stop_reason = reason
            self.state.status = SessionStatus.STOPPED_EARLY
        else:
            self.state.status = SessionStatus.COMPLETED

        logger.info(
            f"Training session '{self.name}' ended. "
            f"Status: {self.state.status.value}, "
            f"Best epoch: {self.state.best_epoch}, "
            f"Best val_loss: {self.state.best_val_loss:.6f}"
        )

        if self.checkpoint_dir:
            self.save_state()

    def start_epoch(self) -> None:
        """Mark the start of an epoch."""
        self._epoch_start_time = time.time()

    def record_epoch(
        self,
        epoch: int,
        train_loss: float,
        val_loss: Optional[float] = None,
        train_metrics: Optional[Dict[str, float]] = None,
        val_metrics: Optional[Dict[str, float]] = None,
        learning_rate: float = 0.0,
    ) -> bool:
        """Record metrics for a completed epoch.

        Args:
            epoch: Epoch number (1-indexed).
            train_loss: Training loss.
            val_loss: Validation loss.
            train_metrics: Additional training metrics.
            val_metrics: Additional validation metrics.
            learning_rate: Current learning rate.

        Returns:
            True if this is a new best model.
        """
        duration = (
            time.time() - self._epoch_start_time
            if self._epoch_start_time
            else 0.0
        )

        metrics = EpochMetrics(
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            train_metrics=train_metrics or {},
            val_metrics=val_metrics or {},
            learning_rate=learning_rate,
            duration_seconds=duration,
        )

        self.state.history.append(metrics)
        self.state.current_epoch = epoch

        # Update best model tracking
        is_best = False
        if val_loss is not None:
            # Get monitored metric
            if self.monitor == "val_loss":
                monitor_value = val_loss
            elif self.monitor in (val_metrics or {}):
                monitor_value = val_metrics[self.monitor]
            else:
                monitor_value = val_loss

            is_best = self.state.update_best(
                epoch=epoch,
                val_loss=val_loss,
                val_metric=monitor_value,
                mode=self.mode,
            )

        return is_best

    def should_stop(
        self,
        patience: int = 15,
        min_delta: float = 1e-5,
    ) -> bool:
        """Check if training should stop.

        Args:
            patience: Epochs to wait for improvement.
            min_delta: Minimum improvement threshold.

        Returns:
            True if training should stop.
        """
        # Check if we've reached max epochs
        if self.state.current_epoch >= self.state.total_epochs:
            return True

        # Check early stopping
        return self.state.should_stop_early(patience, min_delta)

    def get_stop_reason(self, patience: int = 15) -> str:
        """Get the reason for stopping.

        Args:
            patience: Patience used for early stopping.

        Returns:
            Human-readable stop reason.
        """
        if self.state.current_epoch >= self.state.total_epochs:
            return f"Completed all {self.state.total_epochs} epochs"
        elif self.state.epochs_without_improvement >= patience:
            return (
                f"Early stopped after {patience} epochs without improvement. "
                f"Best epoch: {self.state.best_epoch}"
            )
        else:
            return "Unknown"

    def save_state(self, path: Optional[Union[str, Path]] = None) -> Path:
        """Save session state to disk.

        Args:
            path: Path to save to (uses checkpoint_dir if None).

        Returns:
            Path where state was saved.
        """
        if path is None:
            if self.checkpoint_dir is None:
                raise ValueError("No checkpoint directory configured")
            path = self.checkpoint_dir / f"{self.name}_state.json"

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            json.dump(self.state.to_dict(), f, indent=2)

        logger.debug(f"Saved session state to {path}")
        return path

    def _load_state(self, path: Union[str, Path]) -> TrainingState:
        """Load session state from disk.

        Args:
            path: Path to load from.

        Returns:
            Loaded TrainingState.
        """
        path = Path(path)

        with open(path, "r") as f:
            data = json.load(f)

        return TrainingState.from_dict(data)

    def get_summary(self) -> Dict[str, Any]:
        """Get session summary.

        Returns:
            Dictionary with session summary.
        """
        return {
            "name": self.name,
            "status": self.state.status.value,
            "current_epoch": self.state.current_epoch,
            "total_epochs": self.state.total_epochs,
            "best_epoch": self.state.best_epoch,
            "best_val_loss": self.state.best_val_loss,
            "epochs_without_improvement": self.state.epochs_without_improvement,
            "total_duration_seconds": self.state.total_duration_seconds,
            "final_train_loss": (
                self.state.history[-1].train_loss if self.state.history else None
            ),
            "final_val_loss": (
                self.state.history[-1].val_loss if self.state.history else None
            ),
        }

    def plot_history(
        self,
        metrics: Optional[List[str]] = None,
        save_path: Optional[Union[str, Path]] = None,
    ) -> None:
        """Plot training history.

        Args:
            metrics: Metrics to plot (defaults to losses).
            save_path: Path to save plot (displays if None).
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not installed, cannot plot history")
            return

        metrics = metrics or ["train_loss", "val_loss"]

        fig, ax = plt.subplots(figsize=(10, 6))

        for metric in metrics:
            values = self.state.get_metric_history(metric)
            epochs = list(range(1, len(values) + 1))
            ax.plot(epochs, values, label=metric)

        # Mark best epoch
        if self.state.best_epoch > 0:
            ax.axvline(
                x=self.state.best_epoch,
                color="green",
                linestyle="--",
                label=f"Best epoch ({self.state.best_epoch})",
            )

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Value")
        ax.set_title(f"Training History - {self.name}")
        ax.legend()
        ax.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close()
        else:
            plt.show()
