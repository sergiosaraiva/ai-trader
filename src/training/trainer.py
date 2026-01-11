"""Training orchestrator.

Provides a complete training pipeline that combines:
- Architecture creation from registry
- Optimizer and scheduler setup
- Training loop with validation
- Callbacks for monitoring and early stopping
- Model checkpointing and artifact saving
"""

import json
import logging
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .architectures import ArchitectureRegistry, BaseArchitecture
from .callbacks import (
    Callback,
    CallbackList,
    EarlyStoppingCallback,
    LRSchedulerCallback,
    MetricsLoggerCallback,
    ModelCheckpointCallback,
    ProgressCallback,
    TrainingLogs,
)
from .config import (
    CheckpointConfig,
    EarlyStoppingConfig,
    OptimizerType,
    SchedulerConfig,
    SchedulerType,
    TrainingConfig,
)
from .session import TrainingSession, TrainingState

logger = logging.getLogger(__name__)


class MultiTaskLoss(nn.Module):
    """Multi-task loss for combined price, direction, and confidence predictions."""

    def __init__(
        self,
        price_weight: float = 1.0,
        direction_weight: float = 1.0,
        confidence_weight: float = 0.5,
        use_uncertainty_weighting: bool = True,
    ):
        super().__init__()
        self.price_weight = price_weight
        self.direction_weight = direction_weight
        self.confidence_weight = confidence_weight
        self.use_uncertainty_weighting = use_uncertainty_weighting

        # Learnable uncertainty weights (if enabled)
        if use_uncertainty_weighting:
            self.log_vars = nn.Parameter(torch.zeros(3))

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute multi-task loss.

        Args:
            predictions: Model predictions dict.
            targets: Target values dict.

        Returns:
            Tuple of (total_loss, loss_components dict).
        """
        losses = {}

        # Price loss (Huber for robustness)
        if "price" in predictions and "price" in targets:
            price_loss = nn.functional.huber_loss(
                predictions["price"],
                targets["price"],
                delta=1.0,
            )
            losses["price"] = price_loss

        # Direction loss (Cross entropy with class weights for imbalance)
        if "direction_logits" in predictions and "direction" in targets:
            direction_logits = predictions["direction_logits"]
            direction_targets = targets["direction"].long()

            # Handle multi-horizon case
            if direction_logits.dim() == 3:
                batch, horizons, classes = direction_logits.shape
                direction_logits = direction_logits.view(-1, classes)
                direction_targets = direction_targets.view(-1)

            # NOTE: _prepare_batch already maps -1,0,1 to 0,1,2, so no need to remap here
            direction_targets_mapped = direction_targets.clamp(0, 2)  # Just ensure valid range

            # Use simple cross-entropy without class weights or label smoothing
            # Dynamic class weights per batch can cause instability
            direction_loss = nn.functional.cross_entropy(
                direction_logits,
                direction_targets_mapped,
            )
            losses["direction"] = direction_loss

        # Confidence loss (Beta NLL)
        if "alpha" in predictions and "beta" in predictions:
            alpha = predictions["alpha"]
            beta = predictions["beta"]

            # Direction probability from targets
            if "direction" in targets:
                # Convert direction to probability
                direction = targets["direction"].float()
                if direction.dim() < alpha.dim():
                    direction = direction.unsqueeze(-1).expand_as(alpha)

                # Map direction labels to probability:
                # -1 (down) -> 0.0, 0 (neutral) -> 0.5, 1 (up) -> 1.0
                target_prob = (direction + 1) / 2
                target_prob = target_prob.clamp(0, 1)  # Safety clamp

                # Beta NLL loss
                confidence_loss = self._beta_nll_loss(alpha, beta, target_prob)
                losses["confidence"] = confidence_loss

        # Combine losses
        total_loss = torch.tensor(0.0, device=next(iter(predictions.values())).device)

        if self.use_uncertainty_weighting:
            # Uncertainty-weighted multi-task loss
            for i, (name, loss) in enumerate(losses.items()):
                if i < len(self.log_vars):
                    precision = torch.exp(-self.log_vars[i])
                    total_loss = total_loss + precision * loss + self.log_vars[i]
                else:
                    total_loss = total_loss + loss
        else:
            # Fixed-weight combination
            weights = {
                "price": self.price_weight,
                "direction": self.direction_weight,
                "confidence": self.confidence_weight,
            }
            for name, loss in losses.items():
                total_loss = total_loss + weights.get(name, 1.0) * loss

        loss_values = {k: v.item() for k, v in losses.items()}
        loss_values["total"] = total_loss.item()

        return total_loss, loss_values

    def _beta_nll_loss(
        self,
        alpha: torch.Tensor,
        beta: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Beta distribution negative log-likelihood loss."""
        # Clamp for numerical stability
        target = torch.clamp(target, 1e-6, 1 - 1e-6)
        alpha = torch.clamp(alpha, 1.0, 100.0)
        beta = torch.clamp(beta, 1.0, 100.0)

        # Beta NLL
        nll = (
            torch.lgamma(alpha + beta)
            - torch.lgamma(alpha)
            - torch.lgamma(beta)
            + (alpha - 1) * torch.log(target)
            + (beta - 1) * torch.log(1 - target)
        )

        return -nll.mean()


class Trainer:
    """Training orchestrator.

    Manages the complete training pipeline including:
    - Model creation from architecture registry
    - Optimizer and scheduler setup
    - Training loop with validation
    - Early stopping and checkpointing
    - Metrics logging and experiment tracking

    Example:
        ```python
        config = TrainingConfig(
            name="my_experiment",
            batch_size=64,
            epochs=100,
            early_stopping=EarlyStoppingConfig(patience=15),
        )

        trainer = Trainer(
            architecture="cnn_lstm_attention",
            config=config,
            input_dim=50,
            sequence_length=168,
            prediction_horizons=[1, 4, 12, 24],
        )

        results = trainer.fit(train_loader, val_loader)
        trainer.save("models/my_model")
        ```
    """

    def __init__(
        self,
        architecture: Union[str, BaseArchitecture],
        config: Optional[TrainingConfig] = None,
        loss_fn: Optional[nn.Module] = None,
        callbacks: Optional[List[Callback]] = None,
        **architecture_kwargs,
    ):
        """Initialize trainer.

        Args:
            architecture: Architecture name or instance.
            config: Training configuration.
            loss_fn: Custom loss function (uses MultiTaskLoss if None).
            callbacks: Additional callbacks.
            **architecture_kwargs: Arguments passed to architecture.
        """
        self.config = config or TrainingConfig()

        # Set random seed
        self._set_seed(self.config.seed)

        # Resolve device
        self.device = torch.device(self.config.get_device())
        logger.info(f"Using device: {self.device}")

        # Create or use provided architecture
        if isinstance(architecture, str):
            # Merge config and kwargs
            arch_config = {**self.config.architecture_config, **architecture_kwargs}
            self.model = ArchitectureRegistry.create(architecture, **arch_config)
        else:
            self.model = architecture

        self.model = self.model.to(self.device)
        logger.info(f"Model: {self.model.summary()}")

        # Setup loss function
        self.loss_fn = loss_fn or MultiTaskLoss()
        if isinstance(self.loss_fn, nn.Module):
            self.loss_fn = self.loss_fn.to(self.device)

        # Setup optimizer and scheduler
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()

        # Setup callbacks
        self.callbacks = self._setup_callbacks(callbacks)

        # Training state
        self.session: Optional[TrainingSession] = None
        self.best_model_state: Optional[Dict] = None

    def _set_seed(self, seed: int) -> None:
        """Set random seeds for reproducibility."""
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer from config."""
        opt_config = self.config.optimizer

        params = self.model.parameters()

        if opt_config.optimizer_type == OptimizerType.ADAM:
            return torch.optim.Adam(
                params,
                lr=opt_config.learning_rate,
                betas=opt_config.betas,
                eps=opt_config.eps,
                weight_decay=opt_config.weight_decay,
            )
        elif opt_config.optimizer_type == OptimizerType.ADAMW:
            return torch.optim.AdamW(
                params,
                lr=opt_config.learning_rate,
                betas=opt_config.betas,
                eps=opt_config.eps,
                weight_decay=opt_config.weight_decay,
            )
        elif opt_config.optimizer_type == OptimizerType.SGD:
            return torch.optim.SGD(
                params,
                lr=opt_config.learning_rate,
                momentum=opt_config.momentum,
                weight_decay=opt_config.weight_decay,
            )
        elif opt_config.optimizer_type == OptimizerType.RMSPROP:
            return torch.optim.RMSprop(
                params,
                lr=opt_config.learning_rate,
                eps=opt_config.eps,
                weight_decay=opt_config.weight_decay,
            )
        else:
            return torch.optim.AdamW(
                params,
                lr=opt_config.learning_rate,
                weight_decay=opt_config.weight_decay,
            )

    def _create_scheduler(self) -> Optional[torch.optim.lr_scheduler.LRScheduler]:
        """Create learning rate scheduler from config."""
        sched_config = self.config.scheduler

        if sched_config.scheduler_type == SchedulerType.NONE:
            return None
        elif sched_config.scheduler_type == SchedulerType.STEP:
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=sched_config.step_size,
                gamma=sched_config.gamma,
            )
        elif sched_config.scheduler_type == SchedulerType.COSINE:
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=sched_config.T_max,
                eta_min=sched_config.eta_min,
            )
        elif sched_config.scheduler_type == SchedulerType.COSINE_WARM_RESTARTS:
            return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=sched_config.T_0,
                T_mult=sched_config.T_mult,
                eta_min=sched_config.eta_min,
            )
        elif sched_config.scheduler_type == SchedulerType.REDUCE_ON_PLATEAU:
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=sched_config.factor,
                patience=sched_config.patience,
            )
        elif sched_config.scheduler_type == SchedulerType.EXPONENTIAL:
            return torch.optim.lr_scheduler.ExponentialLR(
                self.optimizer,
                gamma=sched_config.gamma,
            )
        else:
            return None

    def _setup_callbacks(
        self,
        custom_callbacks: Optional[List[Callback]],
    ) -> CallbackList:
        """Setup default and custom callbacks."""
        callbacks = []

        # Early stopping callback
        if self.config.early_stopping.enabled:
            es_config = self.config.early_stopping
            callbacks.append(
                EarlyStoppingCallback(
                    monitor=es_config.monitor,
                    patience=es_config.patience,
                    min_delta=es_config.min_delta,
                    mode=es_config.mode,
                    restore_best_weights=es_config.restore_best_weights,
                    baseline=es_config.baseline,
                    start_from_epoch=es_config.start_from_epoch,
                    verbose=self.config.verbose > 0,
                )
            )

        # Checkpoint callback
        if self.config.checkpoint.enabled:
            ckpt_config = self.config.checkpoint
            callbacks.append(
                ModelCheckpointCallback(
                    filepath=Path(ckpt_config.save_dir) / f"{self.config.name}_best.pt",
                    save_best_only=ckpt_config.save_best_only,
                    monitor=ckpt_config.monitor,
                    mode=ckpt_config.mode,
                    max_to_keep=ckpt_config.max_to_keep,
                    save_weights_only=ckpt_config.save_weights_only,
                    verbose=self.config.verbose > 0,
                )
            )

        # LR scheduler callback
        if self.scheduler is not None:
            callbacks.append(
                LRSchedulerCallback(
                    scheduler=self.scheduler,
                    monitor="val_loss",
                    verbose=self.config.verbose > 0,
                )
            )

        # Progress callback
        if self.config.verbose > 0:
            callbacks.append(
                ProgressCallback(
                    use_tqdm=True,
                    show_metrics=["train_loss", "val_loss"],
                )
            )

        # Add custom callbacks
        if custom_callbacks:
            callbacks.extend(custom_callbacks)

        callback_list = CallbackList(callbacks)
        callback_list.set_model(self.model)
        callback_list.set_trainer(self)

        return callback_list

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        resume_from: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Any]:
        """Train the model.

        Args:
            train_loader: Training data loader.
            val_loader: Validation data loader.
            resume_from: Path to resume from checkpoint.

        Returns:
            Dictionary with training results.
        """
        # Initialize session
        self.session = TrainingSession(
            name=self.config.name,
            total_epochs=self.config.epochs,
            checkpoint_dir=self.config.checkpoint.save_dir if self.config.checkpoint.enabled else None,
            monitor=self.config.early_stopping.monitor,
            mode=self.config.early_stopping.mode,
            resume_from=resume_from,
        )

        self.session.start()
        self.callbacks.on_train_begin()

        start_epoch = self.session.state.current_epoch
        best_val_loss = float("inf")

        try:
            for epoch in range(start_epoch, self.config.epochs):
                self.session.start_epoch()
                self.callbacks.on_epoch_begin(epoch + 1)

                # Training phase
                train_loss, train_metrics = self._train_epoch(train_loader, epoch)

                # Validation phase
                val_loss, val_metrics = None, {}
                if val_loader is not None:
                    self.callbacks.on_validation_begin()
                    val_loss, val_metrics = self._validate_epoch(val_loader)
                    self.callbacks.on_validation_end()

                # Get current learning rate
                current_lr = self.optimizer.param_groups[0]["lr"]

                # Record epoch
                is_best = self.session.record_epoch(
                    epoch=epoch + 1,
                    train_loss=train_loss,
                    val_loss=val_loss,
                    train_metrics=train_metrics,
                    val_metrics=val_metrics,
                    learning_rate=current_lr,
                )

                # Save best model state
                if is_best and val_loss is not None:
                    best_val_loss = val_loss
                    self.best_model_state = {
                        k: v.cpu().clone() for k, v in self.model.state_dict().items()
                    }

                # Create logs for callbacks
                logs = TrainingLogs(
                    epoch=epoch + 1,
                    train_loss=train_loss,
                    val_loss=val_loss,
                    train_metrics=train_metrics,
                    val_metrics=val_metrics,
                    learning_rate=current_lr,
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                )

                # Check for early stopping via callbacks
                should_stop = self.callbacks.on_epoch_end(epoch + 1, logs)

                # Also check session's built-in early stopping
                if self.config.early_stopping.enabled:
                    if self.session.should_stop(
                        patience=self.config.early_stopping.patience,
                        min_delta=self.config.early_stopping.min_delta,
                    ):
                        should_stop = True

                if should_stop:
                    reason = self.session.get_stop_reason(self.config.early_stopping.patience)
                    self.session.end(reason)
                    break

            else:
                self.session.end()

        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
            self.session.end("User interrupted")

        except Exception as e:
            logger.error(f"Training failed: {e}")
            self.session.state.status = "failed"
            raise

        finally:
            self.callbacks.on_train_end()

        # Restore best weights if available
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)

        return self.session.get_summary()

    def _train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int,
    ) -> Tuple[float, Dict[str, float]]:
        """Train for one epoch.

        Args:
            train_loader: Training data loader.
            epoch: Current epoch number.

        Returns:
            Tuple of (average_loss, metrics_dict).
        """
        self.model.train()
        total_loss = 0.0
        total_samples = 0
        all_losses = {}

        for batch_idx, batch in enumerate(train_loader):
            self.callbacks.on_batch_begin(batch_idx)

            # Move batch to device
            inputs, targets = self._prepare_batch(batch)

            # Forward pass
            self.optimizer.zero_grad()
            predictions = self.model(inputs)

            # Compute loss
            loss, loss_components = self.loss_fn(predictions, targets)

            # Backward pass
            loss.backward()

            # Gradient clipping
            if self.config.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.gradient_clip,
                )

            self.optimizer.step()

            # Accumulate losses
            batch_size = inputs.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            for name, value in loss_components.items():
                if name not in all_losses:
                    all_losses[name] = 0.0
                all_losses[name] += value * batch_size

            self.callbacks.on_batch_end(batch_idx)

        # Average losses
        avg_loss = total_loss / total_samples
        metrics = {k: v / total_samples for k, v in all_losses.items()}

        return avg_loss, metrics

    def _validate_epoch(
        self,
        val_loader: DataLoader,
    ) -> Tuple[float, Dict[str, float]]:
        """Validate for one epoch.

        Args:
            val_loader: Validation data loader.

        Returns:
            Tuple of (average_loss, metrics_dict).
        """
        self.model.eval()
        total_loss = 0.0
        total_samples = 0
        all_losses = {}

        with torch.no_grad():
            for batch in val_loader:
                inputs, targets = self._prepare_batch(batch)

                predictions = self.model(inputs)
                loss, loss_components = self.loss_fn(predictions, targets)

                batch_size = inputs.size(0)
                total_loss += loss.item() * batch_size
                total_samples += batch_size

                for name, value in loss_components.items():
                    if name not in all_losses:
                        all_losses[name] = 0.0
                    all_losses[name] += value * batch_size

        avg_loss = total_loss / total_samples
        metrics = {k: v / total_samples for k, v in all_losses.items()}

        return avg_loss, metrics

    def _prepare_batch(
        self,
        batch: Union[Tuple, Dict],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Prepare batch for training.

        Args:
            batch: Batch from data loader.

        Returns:
            Tuple of (inputs, targets_dict).
        """
        if isinstance(batch, (tuple, list)):
            inputs, labels = batch[0], batch[1]
        elif isinstance(batch, dict):
            inputs = batch["features"]
            labels = batch.get("labels", batch.get("targets"))
        else:
            raise ValueError(f"Unsupported batch type: {type(batch)}")

        inputs = inputs.to(self.device)

        # Convert labels to targets dict
        if isinstance(labels, torch.Tensor):
            labels = labels.to(self.device)
            # Handle different label formats:
            # - Binary labels (0, 1): Use directly for 2-class classification
            # - Ternary labels (-1, 0, 1): Map to (0, 1, 2) for 3-class
            # - Continuous returns: Use sign for direction
            if labels.is_floating_point():
                # Check if binary (0.0 or 1.0) vs continuous
                unique_vals = torch.unique(labels)
                is_binary = len(unique_vals) <= 2 and all(v in [0.0, 1.0] for v in unique_vals.tolist())
                if is_binary:
                    # Binary labels: 0=down, 1=up
                    direction = labels.long()
                else:
                    # Continuous returns: use sign and map -1,0,1 -> 0,1,2
                    direction = torch.sign(labels).long() + 1
            else:
                # Integer labels: check if already 0/1 or needs mapping
                if labels.min() >= 0:
                    direction = labels.long()
                else:
                    # -1,0,1 labels: map to 0,1,2
                    direction = labels.long() + 1
            targets = {
                "price": labels,
                "direction": direction,
            }
        elif isinstance(labels, dict):
            targets = {k: v.to(self.device) for k, v in labels.items()}
        else:
            targets = {"price": torch.tensor(labels, device=self.device)}

        return inputs, targets

    def predict(
        self,
        inputs: Union[np.ndarray, torch.Tensor],
    ) -> Dict[str, np.ndarray]:
        """Make predictions.

        Args:
            inputs: Input data of shape (batch, seq, features).

        Returns:
            Dictionary of predictions.
        """
        self.model.eval()

        if isinstance(inputs, np.ndarray):
            inputs = torch.FloatTensor(inputs)

        inputs = inputs.to(self.device)

        with torch.no_grad():
            predictions = self.model(inputs)

        return {k: v.cpu().numpy() for k, v in predictions.items()}

    def save(
        self,
        path: Union[str, Path],
        save_optimizer: bool = True,
    ) -> None:
        """Save trainer state and model.

        Args:
            path: Directory to save to.
            save_optimizer: Whether to save optimizer state.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save model weights
        torch.save(self.model.state_dict(), path / "model.pt")

        # Save optimizer
        if save_optimizer:
            torch.save(self.optimizer.state_dict(), path / "optimizer.pt")

        # Save config
        with open(path / "config.json", "w") as f:
            json.dump(self.config.to_dict(), f, indent=2)

        # Save architecture config
        arch_config = self.model.config.to_dict()
        arch_config["architecture_name"] = self.model.name
        with open(path / "architecture.json", "w") as f:
            json.dump(arch_config, f, indent=2)

        # Save session state if available
        if self.session is not None:
            self.session.save_state(path / "session.json")

        logger.info(f"Saved trainer to {path}")

    @classmethod
    def load(
        cls,
        path: Union[str, Path],
        device: Optional[str] = None,
    ) -> "Trainer":
        """Load trainer from saved state.

        Args:
            path: Directory to load from.
            device: Device to load to.

        Returns:
            Loaded Trainer instance.
        """
        path = Path(path)

        # Load configs
        with open(path / "config.json", "r") as f:
            config_dict = json.load(f)
        config = TrainingConfig.from_dict(config_dict)

        with open(path / "architecture.json", "r") as f:
            arch_config = json.load(f)

        arch_name = arch_config.pop("architecture_name", "cnn_lstm_attention")

        # Override device if specified
        if device:
            config.device = device

        # Create trainer
        trainer = cls(
            architecture=arch_name,
            config=config,
            **arch_config,
        )

        # Load model weights
        trainer.model.load_state_dict(
            torch.load(path / "model.pt", map_location=trainer.device)
        )

        # Load optimizer if available
        optimizer_path = path / "optimizer.pt"
        if optimizer_path.exists():
            trainer.optimizer.load_state_dict(
                torch.load(optimizer_path, map_location=trainer.device)
            )

        logger.info(f"Loaded trainer from {path}")
        return trainer
