"""Training callbacks for monitoring and control.

Provides extensible callback system for:
- Early stopping
- Model checkpointing
- Learning rate scheduling
- Metrics logging
- Progress display
"""

import json
import logging
import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TrainingLogs:
    """Container for training logs passed to callbacks."""

    epoch: int = 0
    batch: int = 0
    train_loss: float = 0.0
    val_loss: Optional[float] = None
    train_metrics: Dict[str, float] = None
    val_metrics: Dict[str, float] = None
    learning_rate: float = 0.0
    model: Any = None
    optimizer: Any = None
    scheduler: Any = None

    def __post_init__(self):
        if self.train_metrics is None:
            self.train_metrics = {}
        if self.val_metrics is None:
            self.val_metrics = {}

    def get(self, key: str, default: Any = None) -> Any:
        """Get a log value by key."""
        if hasattr(self, key):
            return getattr(self, key)
        if key in self.train_metrics:
            return self.train_metrics[key]
        if key in self.val_metrics:
            return self.val_metrics[key]
        return default


class Callback(ABC):
    """Base class for training callbacks.

    Callbacks can be used to perform actions at various points during training:
    - on_train_begin: Called once at the start of training
    - on_train_end: Called once at the end of training
    - on_epoch_begin: Called at the start of each epoch
    - on_epoch_end: Called at the end of each epoch
    - on_batch_begin: Called at the start of each batch
    - on_batch_end: Called at the end of each batch
    - on_validation_begin: Called at the start of validation
    - on_validation_end: Called at the end of validation
    """

    def set_model(self, model: Any) -> None:
        """Set the model reference."""
        self.model = model

    def set_trainer(self, trainer: Any) -> None:
        """Set the trainer reference."""
        self.trainer = trainer

    def on_train_begin(self, logs: Optional[TrainingLogs] = None) -> None:
        """Called at the start of training."""
        pass

    def on_train_end(self, logs: Optional[TrainingLogs] = None) -> None:
        """Called at the end of training."""
        pass

    def on_epoch_begin(self, epoch: int, logs: Optional[TrainingLogs] = None) -> None:
        """Called at the start of each epoch."""
        pass

    def on_epoch_end(self, epoch: int, logs: Optional[TrainingLogs] = None) -> bool:
        """Called at the end of each epoch.

        Returns:
            True to stop training, False to continue.
        """
        return False

    def on_batch_begin(self, batch: int, logs: Optional[TrainingLogs] = None) -> None:
        """Called at the start of each batch."""
        pass

    def on_batch_end(self, batch: int, logs: Optional[TrainingLogs] = None) -> None:
        """Called at the end of each batch."""
        pass

    def on_validation_begin(self, logs: Optional[TrainingLogs] = None) -> None:
        """Called at the start of validation."""
        pass

    def on_validation_end(self, logs: Optional[TrainingLogs] = None) -> None:
        """Called at the end of validation."""
        pass


class CallbackList:
    """Container for managing multiple callbacks."""

    def __init__(self, callbacks: Optional[List[Callback]] = None):
        """Initialize callback list.

        Args:
            callbacks: List of callbacks.
        """
        self.callbacks = callbacks or []

    def append(self, callback: Callback) -> None:
        """Add a callback."""
        self.callbacks.append(callback)

    def set_model(self, model: Any) -> None:
        """Set model on all callbacks."""
        for callback in self.callbacks:
            callback.set_model(model)

    def set_trainer(self, trainer: Any) -> None:
        """Set trainer on all callbacks."""
        for callback in self.callbacks:
            callback.set_trainer(trainer)

    def on_train_begin(self, logs: Optional[TrainingLogs] = None) -> None:
        """Call on_train_begin on all callbacks."""
        for callback in self.callbacks:
            callback.on_train_begin(logs)

    def on_train_end(self, logs: Optional[TrainingLogs] = None) -> None:
        """Call on_train_end on all callbacks."""
        for callback in self.callbacks:
            callback.on_train_end(logs)

    def on_epoch_begin(self, epoch: int, logs: Optional[TrainingLogs] = None) -> None:
        """Call on_epoch_begin on all callbacks."""
        for callback in self.callbacks:
            callback.on_epoch_begin(epoch, logs)

    def on_epoch_end(self, epoch: int, logs: Optional[TrainingLogs] = None) -> bool:
        """Call on_epoch_end on all callbacks.

        Returns:
            True if any callback requests to stop training.
        """
        stop = False
        for callback in self.callbacks:
            if callback.on_epoch_end(epoch, logs):
                stop = True
        return stop

    def on_batch_begin(self, batch: int, logs: Optional[TrainingLogs] = None) -> None:
        """Call on_batch_begin on all callbacks."""
        for callback in self.callbacks:
            callback.on_batch_begin(batch, logs)

    def on_batch_end(self, batch: int, logs: Optional[TrainingLogs] = None) -> None:
        """Call on_batch_end on all callbacks."""
        for callback in self.callbacks:
            callback.on_batch_end(batch, logs)

    def on_validation_begin(self, logs: Optional[TrainingLogs] = None) -> None:
        """Call on_validation_begin on all callbacks."""
        for callback in self.callbacks:
            callback.on_validation_begin(logs)

    def on_validation_end(self, logs: Optional[TrainingLogs] = None) -> None:
        """Call on_validation_end on all callbacks."""
        for callback in self.callbacks:
            callback.on_validation_end(logs)


class EarlyStoppingCallback(Callback):
    """Early stopping callback.

    Stops training when a monitored metric stops improving.

    Example:
        ```python
        early_stopping = EarlyStoppingCallback(
            monitor="val_loss",
            patience=10,
            min_delta=1e-4,
            mode="min",
            restore_best_weights=True,
        )
        ```
    """

    def __init__(
        self,
        monitor: str = "val_loss",
        patience: int = 15,
        min_delta: float = 1e-5,
        mode: str = "min",
        restore_best_weights: bool = True,
        baseline: Optional[float] = None,
        start_from_epoch: int = 0,
        verbose: bool = True,
    ):
        """Initialize early stopping callback.

        Args:
            monitor: Metric to monitor.
            patience: Epochs to wait for improvement.
            min_delta: Minimum improvement threshold.
            mode: 'min' or 'max'.
            restore_best_weights: Restore best weights on stop.
            baseline: Baseline value for the metric.
            start_from_epoch: Epoch to start monitoring from.
            verbose: Whether to print messages.
        """
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        self.baseline = baseline
        self.start_from_epoch = start_from_epoch
        self.verbose = verbose

        self.wait = 0
        self.stopped_epoch = 0
        self.best = None
        self.best_weights = None
        self.best_epoch = 0

        if mode == "min":
            self.monitor_op = np.less
            self.min_delta *= -1
        else:
            self.monitor_op = np.greater

    def on_train_begin(self, logs: Optional[TrainingLogs] = None) -> None:
        """Reset state at training start."""
        self.wait = 0
        self.stopped_epoch = 0
        self.best = self.baseline if self.baseline is not None else (
            np.inf if self.mode == "min" else -np.inf
        )
        self.best_weights = None
        self.best_epoch = 0

    def on_epoch_end(self, epoch: int, logs: Optional[TrainingLogs] = None) -> bool:
        """Check for improvement at epoch end."""
        if epoch < self.start_from_epoch:
            return False

        current = self._get_monitor_value(logs)
        if current is None:
            return False

        if self.monitor_op(current - self.min_delta, self.best):
            self.best = current
            self.best_epoch = epoch
            self.wait = 0
            if self.restore_best_weights:
                self.best_weights = self._get_model_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                if self.verbose:
                    logger.info(
                        f"Early stopping triggered at epoch {epoch}. "
                        f"Best epoch: {self.best_epoch}, Best {self.monitor}: {self.best:.6f}"
                    )
                return True

        return False

    def on_train_end(self, logs: Optional[TrainingLogs] = None) -> None:
        """Restore best weights if configured."""
        if self.restore_best_weights and self.best_weights is not None:
            if self.verbose:
                logger.info(f"Restoring model weights from epoch {self.best_epoch}")
            self._set_model_weights(self.best_weights)

    def _get_monitor_value(self, logs: Optional[TrainingLogs]) -> Optional[float]:
        """Get the monitored metric value."""
        if logs is None:
            return None
        return logs.get(self.monitor)

    def _get_model_weights(self) -> Optional[Dict]:
        """Get current model weights."""
        if hasattr(self, "model") and self.model is not None:
            try:
                import torch
                return {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            except Exception:
                pass
        return None

    def _set_model_weights(self, weights: Dict) -> None:
        """Set model weights."""
        if hasattr(self, "model") and self.model is not None:
            try:
                self.model.load_state_dict(weights)
            except Exception:
                pass


class ModelCheckpointCallback(Callback):
    """Model checkpoint callback.

    Saves model weights during training.

    Example:
        ```python
        checkpoint = ModelCheckpointCallback(
            filepath="checkpoints/model_{epoch:03d}_{val_loss:.4f}.pt",
            save_best_only=True,
            monitor="val_loss",
            mode="min",
        )
        ```
    """

    def __init__(
        self,
        filepath: Union[str, Path] = "checkpoints/model.pt",
        save_best_only: bool = True,
        monitor: str = "val_loss",
        mode: str = "min",
        max_to_keep: int = 5,
        save_weights_only: bool = True,
        verbose: bool = True,
    ):
        """Initialize checkpoint callback.

        Args:
            filepath: Path template for saving (supports {epoch}, {val_loss}, etc.).
            save_best_only: Only save when metric improves.
            monitor: Metric to monitor.
            mode: 'min' or 'max'.
            max_to_keep: Maximum checkpoints to retain.
            save_weights_only: Only save weights, not full model.
            verbose: Whether to print messages.
        """
        self.filepath = Path(filepath)
        self.save_best_only = save_best_only
        self.monitor = monitor
        self.mode = mode
        self.max_to_keep = max_to_keep
        self.save_weights_only = save_weights_only
        self.verbose = verbose

        self.best = np.inf if mode == "min" else -np.inf
        self.saved_checkpoints: List[Path] = []

        if mode == "min":
            self.monitor_op = np.less
        else:
            self.monitor_op = np.greater

    def on_epoch_end(self, epoch: int, logs: Optional[TrainingLogs] = None) -> bool:
        """Save checkpoint at epoch end."""
        current = logs.get(self.monitor) if logs else None

        if self.save_best_only:
            if current is None:
                return False

            if self.monitor_op(current, self.best):
                self.best = current
                self._save_checkpoint(epoch, logs)
        else:
            self._save_checkpoint(epoch, logs)

        return False

    def _save_checkpoint(self, epoch: int, logs: Optional[TrainingLogs]) -> None:
        """Save a checkpoint."""
        # Format filepath
        filepath_str = str(self.filepath)
        if "{epoch" in filepath_str:
            filepath_str = filepath_str.format(
                epoch=epoch,
                val_loss=logs.val_loss if logs else 0,
                train_loss=logs.train_loss if logs else 0,
            )
        filepath = Path(filepath_str)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Save model
        if hasattr(self, "model") and self.model is not None:
            try:
                import torch

                if self.save_weights_only:
                    torch.save(self.model.state_dict(), filepath)
                else:
                    torch.save({
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": (
                            logs.optimizer.state_dict() if logs and logs.optimizer else None
                        ),
                        "val_loss": logs.val_loss if logs else None,
                    }, filepath)

                if self.verbose:
                    logger.info(f"Saved checkpoint to {filepath}")

                self.saved_checkpoints.append(filepath)
                self._cleanup_old_checkpoints()

            except Exception as e:
                logger.warning(f"Failed to save checkpoint: {e}")

    def _cleanup_old_checkpoints(self) -> None:
        """Remove old checkpoints beyond max_to_keep."""
        while len(self.saved_checkpoints) > self.max_to_keep:
            old_checkpoint = self.saved_checkpoints.pop(0)
            try:
                old_checkpoint.unlink()
            except Exception:
                pass


class LRSchedulerCallback(Callback):
    """Learning rate scheduler callback.

    Wraps PyTorch LR schedulers and updates them at appropriate times.
    """

    def __init__(
        self,
        scheduler: Any = None,
        monitor: str = "val_loss",
        mode: str = "min",
        verbose: bool = True,
    ):
        """Initialize LR scheduler callback.

        Args:
            scheduler: PyTorch LR scheduler instance.
            monitor: Metric to monitor (for ReduceLROnPlateau).
            mode: 'min' or 'max' (for ReduceLROnPlateau).
            verbose: Whether to print LR changes.
        """
        self.scheduler = scheduler
        self.monitor = monitor
        self.mode = mode
        self.verbose = verbose
        self._last_lr = None

    def on_epoch_end(self, epoch: int, logs: Optional[TrainingLogs] = None) -> bool:
        """Update scheduler at epoch end."""
        if self.scheduler is None:
            return False

        try:
            import torch.optim.lr_scheduler as lr_scheduler

            # Handle ReduceLROnPlateau specially
            if isinstance(self.scheduler, lr_scheduler.ReduceLROnPlateau):
                metric = logs.get(self.monitor) if logs else None
                if metric is not None:
                    self.scheduler.step(metric)
            else:
                self.scheduler.step()

            # Log LR change
            if self.verbose:
                current_lr = self._get_lr()
                if self._last_lr is not None and current_lr != self._last_lr:
                    logger.info(f"Learning rate changed: {self._last_lr:.2e} -> {current_lr:.2e}")
                self._last_lr = current_lr

        except Exception as e:
            logger.warning(f"Failed to step scheduler: {e}")

        return False

    def _get_lr(self) -> float:
        """Get current learning rate."""
        if self.scheduler is None:
            return 0.0

        try:
            if hasattr(self.scheduler, "get_last_lr"):
                return self.scheduler.get_last_lr()[0]
            elif hasattr(self.scheduler, "_last_lr"):
                return self.scheduler._last_lr[0]
        except Exception:
            pass

        return 0.0


class MetricsLoggerCallback(Callback):
    """Metrics logging callback.

    Logs metrics to file and optionally to MLflow.
    """

    def __init__(
        self,
        log_dir: Optional[Union[str, Path]] = None,
        log_to_mlflow: bool = False,
        log_frequency: int = 1,
    ):
        """Initialize metrics logger.

        Args:
            log_dir: Directory to save logs.
            log_to_mlflow: Whether to log to MLflow.
            log_frequency: Log every N epochs.
        """
        self.log_dir = Path(log_dir) if log_dir else None
        self.log_to_mlflow = log_to_mlflow
        self.log_frequency = log_frequency
        self.history: List[Dict[str, Any]] = []

    def on_train_begin(self, logs: Optional[TrainingLogs] = None) -> None:
        """Initialize logging at training start."""
        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)

        if self.log_to_mlflow:
            try:
                import mlflow
                if mlflow.active_run() is None:
                    mlflow.start_run()
            except ImportError:
                logger.warning("MLflow not installed, disabling MLflow logging")
                self.log_to_mlflow = False

    def on_epoch_end(self, epoch: int, logs: Optional[TrainingLogs] = None) -> bool:
        """Log metrics at epoch end."""
        if epoch % self.log_frequency != 0:
            return False

        metrics = {
            "epoch": epoch,
            "train_loss": logs.train_loss if logs else 0,
            "val_loss": logs.val_loss if logs else None,
            "learning_rate": logs.learning_rate if logs else 0,
            "timestamp": datetime.now().isoformat(),
        }

        if logs:
            metrics.update({f"train_{k}": v for k, v in logs.train_metrics.items()})
            metrics.update({f"val_{k}": v for k, v in logs.val_metrics.items()})

        self.history.append(metrics)

        # Log to MLflow
        if self.log_to_mlflow:
            try:
                import mlflow
                for key, value in metrics.items():
                    if isinstance(value, (int, float)) and value is not None:
                        mlflow.log_metric(key, value, step=epoch)
            except Exception as e:
                logger.warning(f"Failed to log to MLflow: {e}")

        return False

    def on_train_end(self, logs: Optional[TrainingLogs] = None) -> None:
        """Save logs at training end."""
        if self.log_dir:
            log_file = self.log_dir / "training_history.json"
            with open(log_file, "w") as f:
                json.dump(self.history, f, indent=2)

        if self.log_to_mlflow:
            try:
                import mlflow
                mlflow.end_run()
            except Exception:
                pass


class ProgressCallback(Callback):
    """Progress display callback.

    Shows training progress with optional tqdm support.
    """

    def __init__(
        self,
        use_tqdm: bool = True,
        show_metrics: List[str] = None,
    ):
        """Initialize progress callback.

        Args:
            use_tqdm: Use tqdm for progress bars.
            show_metrics: Metrics to display in progress bar.
        """
        self.use_tqdm = use_tqdm
        self.show_metrics = show_metrics or ["train_loss", "val_loss"]
        self._pbar = None
        self._epoch_pbar = None

    def on_train_begin(self, logs: Optional[TrainingLogs] = None) -> None:
        """Initialize progress bar at training start."""
        if not self.use_tqdm:
            return

        try:
            from tqdm import tqdm
            total_epochs = getattr(self, "total_epochs", 100)
            self._pbar = tqdm(total=total_epochs, desc="Training", unit="epoch")
        except ImportError:
            self.use_tqdm = False

    def on_epoch_end(self, epoch: int, logs: Optional[TrainingLogs] = None) -> bool:
        """Update progress at epoch end."""
        if self._pbar is not None:
            # Update progress bar
            self._pbar.update(1)

            # Update description with metrics
            if logs:
                desc_parts = [f"Epoch {epoch}"]
                for metric in self.show_metrics:
                    value = logs.get(metric)
                    if value is not None:
                        desc_parts.append(f"{metric}={value:.4f}")
                self._pbar.set_description(" | ".join(desc_parts))

        elif logs:
            # Fallback to print
            metrics_str = " | ".join(
                f"{m}={logs.get(m):.4f}"
                for m in self.show_metrics
                if logs.get(m) is not None
            )
            print(f"Epoch {epoch}: {metrics_str}")

        return False

    def on_train_end(self, logs: Optional[TrainingLogs] = None) -> None:
        """Close progress bar at training end."""
        if self._pbar is not None:
            self._pbar.close()
