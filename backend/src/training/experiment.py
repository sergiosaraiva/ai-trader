"""Experiment management with MLflow integration.

Provides end-to-end experiment tracking, hyperparameter search,
and reproducible training runs.
"""

import hashlib
import json
import logging
import os
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .config import TrainingConfig
from .trainer import Trainer

logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """Configuration for an experiment run.

    Attributes:
        name: Experiment name.
        description: Description of the experiment.
        tags: Tags for categorization.
        tracking_uri: MLflow tracking URI.
        artifact_location: Where to store artifacts.
        log_models: Whether to log models to MLflow.
        log_artifacts: Whether to log additional artifacts.
        seed: Random seed for reproducibility.
    """

    name: str = "default_experiment"
    description: str = ""
    tags: Dict[str, str] = field(default_factory=dict)
    tracking_uri: str = "mlruns"
    artifact_location: Optional[str] = None
    log_models: bool = True
    log_artifacts: bool = True
    seed: int = 42


@dataclass
class ExperimentResult:
    """Result of an experiment run.

    Attributes:
        run_id: MLflow run ID.
        experiment_id: MLflow experiment ID.
        status: Run status (completed, failed, etc.).
        metrics: Final metrics from training.
        best_epoch: Best epoch number.
        best_val_loss: Best validation loss.
        model_path: Path to saved model.
        duration_seconds: Total training time.
        config: Training configuration used.
    """

    run_id: str
    experiment_id: str
    status: str
    metrics: Dict[str, float]
    best_epoch: int
    best_val_loss: float
    model_path: Optional[str] = None
    duration_seconds: float = 0.0
    config: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "run_id": self.run_id,
            "experiment_id": self.experiment_id,
            "status": self.status,
            "metrics": self.metrics,
            "best_epoch": self.best_epoch,
            "best_val_loss": self.best_val_loss,
            "model_path": self.model_path,
            "duration_seconds": self.duration_seconds,
            "config": self.config,
        }


class ExperimentManager:
    """Manages ML experiments with MLflow tracking.

    Provides:
    - Experiment tracking with MLflow
    - Hyperparameter logging
    - Metric logging during training
    - Model artifact storage
    - Reproducible experiments

    Example:
        ```python
        manager = ExperimentManager(
            config=ExperimentConfig(name="forex_models")
        )

        result = manager.run_experiment(
            trainer=trainer,
            train_loader=train_loader,
            val_loader=val_loader,
            training_config=training_config,
        )
        ```
    """

    def __init__(
        self,
        config: Optional[ExperimentConfig] = None,
        use_mlflow: bool = True,
    ):
        """Initialize experiment manager.

        Args:
            config: Experiment configuration.
            use_mlflow: Whether to use MLflow for tracking.
        """
        self.config = config or ExperimentConfig()
        self.use_mlflow = use_mlflow
        self._mlflow = None
        self._experiment_id = None

        if use_mlflow:
            self._setup_mlflow()

    def _setup_mlflow(self) -> None:
        """Set up MLflow tracking."""
        try:
            import mlflow

            self._mlflow = mlflow

            # Set tracking URI
            mlflow.set_tracking_uri(self.config.tracking_uri)

            # Get or create experiment
            experiment = mlflow.get_experiment_by_name(self.config.name)
            if experiment is None:
                self._experiment_id = mlflow.create_experiment(
                    name=self.config.name,
                    artifact_location=self.config.artifact_location,
                    tags=self.config.tags,
                )
            else:
                self._experiment_id = experiment.experiment_id

            logger.info(
                f"MLflow experiment '{self.config.name}' "
                f"(ID: {self._experiment_id})"
            )

        except ImportError:
            logger.warning("MLflow not installed. Experiment tracking disabled.")
            self.use_mlflow = False
            self._mlflow = None

    def _set_seeds(self, seed: int) -> None:
        """Set random seeds for reproducibility."""
        import random

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def _generate_run_name(self, training_config: TrainingConfig) -> str:
        """Generate a unique run name."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        config_hash = hashlib.md5(
            json.dumps(training_config.to_dict(), sort_keys=True).encode()
        ).hexdigest()[:8]
        return f"{training_config.name}_{timestamp}_{config_hash}"

    def _log_params(self, config: TrainingConfig) -> None:
        """Log training parameters to MLflow."""
        if not self.use_mlflow:
            return

        params = {
            "batch_size": config.batch_size,
            "epochs": config.epochs,
            "learning_rate": config.optimizer.learning_rate,
            "weight_decay": config.optimizer.weight_decay,
            "optimizer_type": config.optimizer.optimizer_type.value,
            "scheduler_type": config.scheduler.scheduler_type.value,
            "gradient_clip": config.gradient_clip,
            "early_stopping_patience": config.early_stopping.patience,
            "early_stopping_min_delta": config.early_stopping.min_delta,
            "seed": config.seed,
            "device": config.device,
        }

        # Add architecture config
        for key, value in config.architecture_config.items():
            if isinstance(value, (int, float, str, bool)):
                params[f"arch_{key}"] = value

        self._mlflow.log_params(params)

    def _log_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None,
    ) -> None:
        """Log metrics to MLflow."""
        if not self.use_mlflow:
            return

        self._mlflow.log_metrics(metrics, step=step)

    def _log_model(
        self,
        trainer: Trainer,
        model_name: str = "model",
    ) -> str:
        """Log model to MLflow and return artifact path."""
        if not self.use_mlflow or not self.config.log_models:
            return ""

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / model_name
            trainer.save(model_path)

            # Log as artifact
            self._mlflow.log_artifacts(tmpdir, artifact_path="model")

        return f"model/{model_name}"

    def run_experiment(
        self,
        trainer: Trainer,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        training_config: Optional[TrainingConfig] = None,
        run_name: Optional[str] = None,
        nested: bool = False,
    ) -> ExperimentResult:
        """Run a single training experiment.

        Args:
            trainer: Configured trainer instance.
            train_loader: Training data loader.
            val_loader: Validation data loader.
            training_config: Training configuration (uses trainer's if None).
            run_name: Custom run name.
            nested: Whether this is a nested run (for hyperparameter search).

        Returns:
            ExperimentResult with training outcomes.
        """
        training_config = training_config or trainer.config
        run_name = run_name or self._generate_run_name(training_config)

        # Set seeds
        self._set_seeds(training_config.seed)

        start_time = datetime.now()

        if self.use_mlflow:
            with self._mlflow.start_run(
                experiment_id=self._experiment_id,
                run_name=run_name,
                nested=nested,
            ) as run:
                result = self._execute_training(
                    trainer=trainer,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    training_config=training_config,
                    run_id=run.info.run_id,
                    start_time=start_time,
                )
        else:
            result = self._execute_training(
                trainer=trainer,
                train_loader=train_loader,
                val_loader=val_loader,
                training_config=training_config,
                run_id="local_run",
                start_time=start_time,
            )

        return result

    def _execute_training(
        self,
        trainer: Trainer,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        training_config: TrainingConfig,
        run_id: str,
        start_time: datetime,
    ) -> ExperimentResult:
        """Execute the training loop with logging."""
        # Log parameters
        self._log_params(training_config)

        # Log tags
        if self.use_mlflow:
            self._mlflow.set_tags({
                "model_type": trainer.model.__class__.__name__,
                "description": self.config.description,
                **self.config.tags,
            })

        try:
            # Run training
            train_results = trainer.fit(train_loader, val_loader)

            # Log final metrics
            final_metrics = {
                "final_train_loss": train_results.get("final_train_loss", 0),
                "best_val_loss": train_results.get("best_val_loss", float("inf")),
                "best_epoch": train_results.get("best_epoch", 0),
                "total_epochs": train_results.get("current_epoch", 0),
            }
            self._log_metrics(final_metrics)

            # Log model
            model_path = ""
            if self.config.log_models:
                model_path = self._log_model(trainer)

            duration = (datetime.now() - start_time).total_seconds()

            return ExperimentResult(
                run_id=run_id,
                experiment_id=self._experiment_id or "local",
                status="completed",
                metrics=final_metrics,
                best_epoch=train_results.get("best_epoch", 0),
                best_val_loss=train_results.get("best_val_loss", float("inf")),
                model_path=model_path,
                duration_seconds=duration,
                config=training_config.to_dict(),
            )

        except Exception as e:
            logger.error(f"Training failed: {e}")
            if self.use_mlflow:
                self._mlflow.set_tag("error", str(e))

            return ExperimentResult(
                run_id=run_id,
                experiment_id=self._experiment_id or "local",
                status="failed",
                metrics={},
                best_epoch=0,
                best_val_loss=float("inf"),
                config=training_config.to_dict(),
            )

    def hyperparameter_search(
        self,
        create_trainer_fn: Callable[[Dict[str, Any]], Trainer],
        train_loader: DataLoader,
        val_loader: DataLoader,
        search_space: Dict[str, Any],
        n_trials: int = 20,
        direction: str = "minimize",
        metric: str = "best_val_loss",
        pruning: bool = True,
        timeout: Optional[int] = None,
    ) -> Tuple[Dict[str, Any], ExperimentResult]:
        """Run hyperparameter search using Optuna.

        Args:
            create_trainer_fn: Function that takes params dict and returns Trainer.
            train_loader: Training data loader.
            val_loader: Validation data loader.
            search_space: Dictionary defining search space.
            n_trials: Number of trials to run.
            direction: 'minimize' or 'maximize'.
            metric: Metric to optimize.
            pruning: Whether to enable trial pruning.
            timeout: Optional timeout in seconds.

        Returns:
            Tuple of (best_params, best_result).

        Example:
            ```python
            search_space = {
                "learning_rate": ("log_uniform", 1e-5, 1e-2),
                "hidden_dim": ("categorical", [64, 128, 256]),
                "dropout": ("uniform", 0.1, 0.5),
                "batch_size": ("categorical", [32, 64, 128]),
            }

            best_params, best_result = manager.hyperparameter_search(
                create_trainer_fn=create_trainer,
                train_loader=train_loader,
                val_loader=val_loader,
                search_space=search_space,
                n_trials=50,
            )
            ```
        """
        try:
            import optuna
            from optuna.integration import PyTorchLightningPruningCallback
        except ImportError:
            raise ImportError("Optuna required for hyperparameter search: pip install optuna")

        def objective(trial: optuna.Trial) -> float:
            # Sample hyperparameters
            params = self._sample_params(trial, search_space)

            # Create trainer with sampled params
            trainer = create_trainer_fn(params)

            # Run experiment
            result = self.run_experiment(
                trainer=trainer,
                train_loader=train_loader,
                val_loader=val_loader,
                run_name=f"trial_{trial.number}",
                nested=True,
            )

            # Return metric
            return result.metrics.get(metric, float("inf"))

        # Create study
        study = optuna.create_study(
            direction=direction,
            study_name=f"{self.config.name}_hpo",
            pruner=optuna.pruners.MedianPruner() if pruning else None,
        )

        # Run optimization
        if self.use_mlflow:
            with self._mlflow.start_run(
                experiment_id=self._experiment_id,
                run_name="hyperparameter_search",
            ):
                study.optimize(
                    objective,
                    n_trials=n_trials,
                    timeout=timeout,
                    show_progress_bar=True,
                )

                # Log best params
                self._mlflow.log_params({
                    f"best_{k}": v for k, v in study.best_params.items()
                })
                self._mlflow.log_metric("best_value", study.best_value)
        else:
            study.optimize(
                objective,
                n_trials=n_trials,
                timeout=timeout,
                show_progress_bar=True,
            )

        # Get best result
        best_trainer = create_trainer_fn(study.best_params)
        best_result = self.run_experiment(
            trainer=best_trainer,
            train_loader=train_loader,
            val_loader=val_loader,
            run_name="best_model",
        )

        return study.best_params, best_result

    def _sample_params(
        self,
        trial: "optuna.Trial",
        search_space: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Sample parameters from search space."""
        params = {}

        for name, spec in search_space.items():
            if isinstance(spec, tuple):
                dist_type = spec[0]

                if dist_type == "uniform":
                    params[name] = trial.suggest_float(name, spec[1], spec[2])
                elif dist_type == "log_uniform":
                    params[name] = trial.suggest_float(name, spec[1], spec[2], log=True)
                elif dist_type == "int":
                    params[name] = trial.suggest_int(name, spec[1], spec[2])
                elif dist_type == "categorical":
                    params[name] = trial.suggest_categorical(name, spec[1])
                else:
                    raise ValueError(f"Unknown distribution type: {dist_type}")
            else:
                # Fixed value
                params[name] = spec

        return params

    def compare_runs(
        self,
        run_ids: List[str],
        metrics: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Compare multiple experiment runs.

        Args:
            run_ids: List of MLflow run IDs to compare.
            metrics: Metrics to include (all if None).

        Returns:
            DataFrame with run comparison.
        """
        if not self.use_mlflow:
            raise RuntimeError("MLflow required for run comparison")

        runs_data = []

        for run_id in run_ids:
            run = self._mlflow.get_run(run_id)
            run_data = {
                "run_id": run_id,
                "run_name": run.info.run_name,
                "status": run.info.status,
                "start_time": run.info.start_time,
                "end_time": run.info.end_time,
            }

            # Add params
            for k, v in run.data.params.items():
                run_data[f"param_{k}"] = v

            # Add metrics
            for k, v in run.data.metrics.items():
                if metrics is None or k in metrics:
                    run_data[f"metric_{k}"] = v

            runs_data.append(run_data)

        return pd.DataFrame(runs_data)

    def get_best_run(
        self,
        metric: str = "best_val_loss",
        mode: str = "min",
    ) -> Optional[str]:
        """Get the best run ID based on a metric.

        Args:
            metric: Metric to use for comparison.
            mode: 'min' or 'max'.

        Returns:
            Run ID of the best run.
        """
        if not self.use_mlflow:
            return None

        runs = self._mlflow.search_runs(
            experiment_ids=[self._experiment_id],
            order_by=[f"metrics.{metric} {'ASC' if mode == 'min' else 'DESC'}"],
            max_results=1,
        )

        if len(runs) > 0:
            return runs.iloc[0]["run_id"]

        return None

    def load_model_from_run(
        self,
        run_id: str,
        model_name: str = "model",
    ) -> Trainer:
        """Load a model from an MLflow run.

        Args:
            run_id: MLflow run ID.
            model_name: Name of the model artifact.

        Returns:
            Loaded Trainer instance.
        """
        if not self.use_mlflow:
            raise RuntimeError("MLflow required to load model from run")

        # Download artifact
        artifact_path = self._mlflow.artifacts.download_artifacts(
            run_id=run_id,
            artifact_path=f"model/{model_name}",
        )

        # Load trainer
        return Trainer.load(artifact_path)


def create_experiment_from_config(
    config_path: Union[str, Path],
    experiment_name: Optional[str] = None,
) -> ExperimentManager:
    """Create experiment manager from configuration file.

    Args:
        config_path: Path to YAML configuration file.
        experiment_name: Override experiment name.

    Returns:
        Configured ExperimentManager.
    """
    import yaml

    with open(config_path, "r") as f:
        config_data = yaml.safe_load(f)

    experiment_config = ExperimentConfig(
        name=experiment_name or config_data.get("experiment_name", "default"),
        description=config_data.get("description", ""),
        tags=config_data.get("tags", {}),
        tracking_uri=config_data.get("tracking_uri", "mlruns"),
        seed=config_data.get("seed", 42),
    )

    return ExperimentManager(config=experiment_config)
