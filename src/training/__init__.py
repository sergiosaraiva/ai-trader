"""Training module for model training orchestration.

This module provides:
- Plugin-based model architecture system
- Training configuration with hyperparameters
- Training sessions with early stopping
- Callbacks for monitoring and checkpointing
- Trainer orchestrator for end-to-end training
- Experiment management with MLflow
- Model evaluation with trading metrics
"""

from .config import (
    TrainingConfig,
    EarlyStoppingConfig,
    SchedulerConfig,
    SchedulerType,
    CheckpointConfig,
    OptimizerConfig,
    OptimizerType,
)
from .session import (
    TrainingSession,
    TrainingState,
    EpochMetrics,
    SessionStatus,
)
from .callbacks import (
    Callback,
    CallbackList,
    EarlyStoppingCallback,
    ModelCheckpointCallback,
    LRSchedulerCallback,
    MetricsLoggerCallback,
    ProgressCallback,
)
from .trainer import Trainer
from .architectures import (
    ArchitectureRegistry,
    BaseArchitecture,
    ArchitectureConfig,
    CNNLSTMAttention,
    TemporalFusionTransformer,
    NBEATSTransformer,
)
from .experiment import (
    ExperimentConfig,
    ExperimentResult,
    ExperimentManager,
)
from .evaluation import (
    ModelEvaluator,
    EvaluationResult,
    DirectionMetrics,
    CalibrationMetrics,
    TradingMetrics,
)

__all__ = [
    # Config
    "TrainingConfig",
    "EarlyStoppingConfig",
    "SchedulerConfig",
    "SchedulerType",
    "CheckpointConfig",
    "OptimizerConfig",
    "OptimizerType",
    # Session
    "TrainingSession",
    "TrainingState",
    "EpochMetrics",
    "SessionStatus",
    # Callbacks
    "Callback",
    "CallbackList",
    "EarlyStoppingCallback",
    "ModelCheckpointCallback",
    "LRSchedulerCallback",
    "MetricsLoggerCallback",
    "ProgressCallback",
    # Trainer
    "Trainer",
    # Architectures
    "ArchitectureRegistry",
    "BaseArchitecture",
    "ArchitectureConfig",
    "CNNLSTMAttention",
    "TemporalFusionTransformer",
    "NBEATSTransformer",
    # Experiment
    "ExperimentConfig",
    "ExperimentResult",
    "ExperimentManager",
    # Evaluation
    "ModelEvaluator",
    "EvaluationResult",
    "DirectionMetrics",
    "CalibrationMetrics",
    "TradingMetrics",
]
