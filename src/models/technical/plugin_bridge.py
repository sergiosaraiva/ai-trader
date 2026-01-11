"""Bridge between existing technical models and new plugin architecture.

This module provides backward compatibility by wrapping the new architecture
system in the existing BaseModel interface.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch

from ..base import BaseModel, Prediction, ModelRegistry
from src.training.architectures import (
    ArchitectureRegistry,
    BaseArchitecture,
    CNNLSTMAttention,
    TemporalFusionTransformer,
    NBEATSTransformer,
)
from src.training.config import TrainingConfig
from src.training.trainer import Trainer

logger = logging.getLogger(__name__)


class PluginModel(BaseModel):
    """Wrapper that bridges new architectures to BaseModel interface.

    This allows any architecture from the registry to be used through
    the existing BaseModel API.

    Example:
        ```python
        # Create model from architecture name
        model = PluginModel.from_architecture(
            "cnn_lstm_attention",
            input_dim=50,
            sequence_length=168,
        )

        # Train using BaseModel interface
        history = model.train(X_train, y_train, X_val, y_val)

        # Predict
        prediction = model.predict(X_test)
        ```
    """

    DEFAULT_CONFIG = {
        "architecture": "cnn_lstm_attention",
        "input_dim": 50,
        "sequence_length": 100,
        "hidden_dim": 256,
        "num_layers": 2,
        "dropout": 0.3,
        "prediction_horizons": [1],
        "num_classes": 3,
        "batch_size": 64,
        "learning_rate": 1e-4,
        "epochs": 100,
        "early_stopping_patience": 15,
        "device": "auto",
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize plugin model.

        Args:
            config: Configuration dictionary with architecture and training params.
        """
        super().__init__(config)

        # Merge with defaults
        full_config = {**self.DEFAULT_CONFIG, **self.config}
        self.config = full_config

        self.architecture: Optional[BaseArchitecture] = None
        self.trainer: Optional[Trainer] = None

    @classmethod
    def from_architecture(
        cls,
        architecture: str,
        **kwargs,
    ) -> "PluginModel":
        """Create model from architecture name.

        Args:
            architecture: Architecture name from registry.
            **kwargs: Architecture and training configuration.

        Returns:
            Configured PluginModel instance.
        """
        config = {"architecture": architecture, **kwargs}
        model = cls(config)
        model.build()
        return model

    def build(self) -> None:
        """Build model architecture."""
        # Extract architecture config
        arch_config = {
            "input_dim": self.config.get("input_dim", 50),
            "sequence_length": self.config.get("sequence_length", 100),
            "hidden_dim": self.config.get("hidden_dim", 256),
            "num_layers": self.config.get("num_layers", 2),
            "dropout": self.config.get("dropout", 0.3),
            "prediction_horizons": self.config.get("prediction_horizons", [1]),
            "num_classes": self.config.get("num_classes", 3),
        }

        # Create architecture from registry
        arch_name = self.config.get("architecture", "cnn_lstm_attention")
        self.architecture = ArchitectureRegistry.create(arch_name, **arch_config)

        # Create trainer
        training_config = TrainingConfig(
            name=self.name,
            batch_size=self.config.get("batch_size", 64),
            epochs=self.config.get("epochs", 100),
            device=self.config.get("device", "auto"),
        )
        training_config.optimizer.learning_rate = self.config.get("learning_rate", 1e-4)
        training_config.early_stopping.patience = self.config.get("early_stopping_patience", 15)

        self.trainer = Trainer(
            architecture=self.architecture,
            config=training_config,
        )

        self.model = self.architecture
        logger.info(f"Built {arch_name} architecture with {self.architecture.get_num_parameters():,} parameters")

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Train the model.

        Args:
            X_train: Training features (batch, sequence, features).
            y_train: Training targets.
            X_val: Validation features.
            y_val: Validation targets.

        Returns:
            Training history/metrics.
        """
        if self.trainer is None:
            self.build()

        # Create data loaders
        from torch.utils.data import DataLoader, TensorDataset

        train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.FloatTensor(y_train),
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.get("batch_size", 64),
            shuffle=True,
        )

        val_loader = None
        if X_val is not None and y_val is not None:
            val_dataset = TensorDataset(
                torch.FloatTensor(X_val),
                torch.FloatTensor(y_val),
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config.get("batch_size", 64),
                shuffle=False,
            )

        # Train
        results = self.trainer.fit(train_loader, val_loader)
        self.is_trained = True

        return results

    def predict(self, X: np.ndarray) -> Prediction:
        """Make single prediction.

        Args:
            X: Input features (sequence, features) or (batch, sequence, features).

        Returns:
            Prediction object.
        """
        if self.trainer is None or self.architecture is None:
            raise ValueError("Model not built or trained")

        # Ensure 3D input
        if X.ndim == 2:
            X = X[np.newaxis, ...]

        outputs = self.trainer.predict(X)

        # Extract predictions
        price_pred = outputs.get("price", np.array([0.0]))[0]
        if price_pred.ndim > 0:
            price_pred = price_pred[0]  # First horizon

        # Direction from logits
        direction_logits = outputs.get("direction_logits", np.array([[0, 1, 0]]))[0]
        if direction_logits.ndim > 1:
            direction_logits = direction_logits[0]  # First horizon
        direction_probs = np.exp(direction_logits) / np.exp(direction_logits).sum()
        direction_idx = np.argmax(direction_probs)
        direction = ["bearish", "neutral", "bullish"][direction_idx]
        direction_prob = direction_probs[direction_idx]

        # Confidence from Beta distribution
        alpha = outputs.get("alpha", np.array([2.0]))[0]
        beta = outputs.get("beta", np.array([2.0]))[0]
        if alpha.ndim > 0:
            alpha, beta = alpha[0], beta[0]
        confidence = alpha / (alpha + beta)  # Mean of Beta distribution

        from datetime import datetime

        return Prediction(
            timestamp=datetime.now(),
            symbol=self.config.get("symbol", "UNKNOWN"),
            price_prediction=float(price_pred),
            direction=direction,
            direction_probability=float(direction_prob),
            confidence=float(confidence),
            model_name=self.name,
            model_version=self.version,
            prediction_horizon=self.config.get("prediction_horizons", [1])[0],
        )

    def predict_batch(self, X: np.ndarray) -> List[Prediction]:
        """Make batch predictions.

        Args:
            X: Batch of inputs (batch, sequence, features).

        Returns:
            List of Prediction objects.
        """
        if self.trainer is None or self.architecture is None:
            raise ValueError("Model not built or trained")

        outputs = self.trainer.predict(X)
        predictions = []

        for i in range(X.shape[0]):
            price_pred = outputs.get("price", np.zeros(len(X)))[i]
            if price_pred.ndim > 0:
                price_pred = price_pred[0]

            direction_logits = outputs.get("direction_logits", np.zeros((len(X), 3)))[i]
            if direction_logits.ndim > 1:
                direction_logits = direction_logits[0]
            direction_probs = np.exp(direction_logits) / np.exp(direction_logits).sum()
            direction_idx = np.argmax(direction_probs)
            direction = ["bearish", "neutral", "bullish"][direction_idx]
            direction_prob = direction_probs[direction_idx]

            alpha = outputs.get("alpha", np.full(len(X), 2.0))[i]
            beta = outputs.get("beta", np.full(len(X), 2.0))[i]
            if isinstance(alpha, np.ndarray) and alpha.ndim > 0:
                alpha, beta = alpha[0], beta[0]
            confidence = alpha / (alpha + beta)

            from datetime import datetime

            predictions.append(
                Prediction(
                    timestamp=datetime.now(),
                    symbol=self.config.get("symbol", "UNKNOWN"),
                    price_prediction=float(price_pred),
                    direction=direction,
                    direction_probability=float(direction_prob),
                    confidence=float(confidence),
                    model_name=self.name,
                    model_version=self.version,
                    prediction_horizon=self.config.get("prediction_horizons", [1])[0],
                )
            )

        return predictions

    def _save_weights(self, path: Path) -> None:
        """Save model weights."""
        if self.trainer is not None:
            self.trainer.save(path)

    def _load_weights(self, path: Path) -> None:
        """Load model weights."""
        if path.exists():
            self.trainer = Trainer.load(path)
            self.architecture = self.trainer.model
            self.model = self.architecture


# Register the plugin model
ModelRegistry.register("plugin", PluginModel)


def create_model(
    architecture: str,
    timeframe: str = "medium_term",
    **kwargs,
) -> PluginModel:
    """Factory function to create models for any timeframe.

    This replaces the old paradigm of having separate ShortTermModel,
    MediumTermModel, LongTermModel classes. Any architecture can now
    be used at any timeframe.

    Args:
        architecture: Architecture name ('cnn_lstm_attention', 'tft', 'nbeats').
        timeframe: Timeframe preset for default configs.
        **kwargs: Override any configuration parameter.

    Returns:
        Configured PluginModel.

    Example:
        ```python
        # Use CNN-LSTM at daily timeframe (was medium_term specific)
        model = create_model("cnn_lstm_attention", timeframe="daily")

        # Use TFT at hourly timeframe (was short_term specific)
        model = create_model("tft", timeframe="hourly")

        # Custom configuration
        model = create_model(
            "nbeats_transformer",
            timeframe="weekly",
            hidden_dim=512,
            epochs=200,
        )
        ```
    """
    # Timeframe presets
    TIMEFRAME_PRESETS = {
        "short_term": {
            "sequence_length": 168,
            "prediction_horizons": [1, 4, 12, 24],
        },
        "hourly": {
            "sequence_length": 168,
            "prediction_horizons": [1, 4, 12, 24],
        },
        "medium_term": {
            "sequence_length": 90,
            "prediction_horizons": [1, 3, 5, 7],
        },
        "daily": {
            "sequence_length": 90,
            "prediction_horizons": [1, 3, 5, 7],
        },
        "long_term": {
            "sequence_length": 52,
            "prediction_horizons": [1, 2, 4],
        },
        "weekly": {
            "sequence_length": 52,
            "prediction_horizons": [1, 2, 4],
        },
    }

    # Start with timeframe preset
    config = TIMEFRAME_PRESETS.get(timeframe, TIMEFRAME_PRESETS["medium_term"]).copy()
    config["architecture"] = architecture

    # Override with kwargs
    config.update(kwargs)

    return PluginModel.from_architecture(**config)
