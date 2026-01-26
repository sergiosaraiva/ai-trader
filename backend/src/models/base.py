"""Base model class and model registry."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Type, Tuple
import json

import numpy as np
import pandas as pd


@dataclass
class Prediction:
    """Model prediction output."""

    timestamp: datetime
    symbol: str

    # Price predictions
    price_prediction: float
    price_predictions_multi: Dict[str, float] = field(default_factory=dict)

    # Direction
    direction: str = "neutral"  # bullish, bearish, neutral
    direction_probability: float = 0.5

    # Confidence
    confidence: float = 0.5
    prediction_lower: float = 0.0
    prediction_upper: float = 0.0

    # Model info
    model_name: str = ""
    model_version: str = ""
    prediction_horizon: int = 1

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "symbol": self.symbol,
            "price_prediction": self.price_prediction,
            "price_predictions_multi": self.price_predictions_multi,
            "direction": self.direction,
            "direction_probability": self.direction_probability,
            "confidence": self.confidence,
            "prediction_lower": self.prediction_lower,
            "prediction_upper": self.prediction_upper,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "prediction_horizon": self.prediction_horizon,
        }


class BaseModel(ABC):
    """Abstract base class for all prediction models."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize model.

        Args:
            config: Model configuration dictionary
        """
        self.config = config or {}
        self.name = self.config.get("name", self.__class__.__name__)
        self.version = self.config.get("version", "1.0.0")
        self.model = None
        self.scalers: Dict[str, Any] = {}
        self.feature_names: List[str] = []
        self.is_trained = False

    @abstractmethod
    def build(self) -> None:
        """Build model architecture."""
        pass

    @abstractmethod
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Train the model.

        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets

        Returns:
            Training history/metrics
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> Prediction:
        """
        Make prediction.

        Args:
            X: Input features

        Returns:
            Prediction object
        """
        pass

    @abstractmethod
    def predict_batch(self, X: np.ndarray) -> List[Prediction]:
        """
        Make batch predictions.

        Args:
            X: Batch of input features

        Returns:
            List of Prediction objects
        """
        pass

    def save(self, path: Path) -> None:
        """
        Save model to disk.

        Args:
            path: Directory to save model
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save metadata
        metadata = {
            "name": self.name,
            "version": self.version,
            "config": self.config,
            "feature_names": self.feature_names,
            "is_trained": self.is_trained,
            "saved_at": datetime.now().isoformat(),
        }

        with open(path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        # Save scalers
        if self.scalers:
            import pickle
            with open(path / "scalers.pkl", "wb") as f:
                pickle.dump(self.scalers, f)

        # Save model weights (implemented by subclasses)
        self._save_weights(path)

    def load(self, path: Path) -> None:
        """
        Load model from disk.

        Args:
            path: Directory containing saved model
        """
        path = Path(path)

        # Load metadata
        with open(path / "metadata.json", "r") as f:
            metadata = json.load(f)

        self.name = metadata["name"]
        self.version = metadata["version"]
        self.config = metadata["config"]
        self.feature_names = metadata["feature_names"]
        self.is_trained = metadata["is_trained"]

        # Load scalers
        scalers_path = path / "scalers.pkl"
        if scalers_path.exists():
            import pickle
            with open(scalers_path, "rb") as f:
                self.scalers = pickle.load(f)

        # Build and load weights
        self.build()
        self._load_weights(path)

    def _save_weights(self, path: Path) -> None:
        """Save model weights. Override in subclasses."""
        pass

    def _load_weights(self, path: Path) -> None:
        """Load model weights. Override in subclasses."""
        pass

    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> Dict[str, float]:
        """
        Evaluate model performance.

        Args:
            X_test: Test features
            y_test: Test targets

        Returns:
            Dictionary of metrics
        """
        predictions = self.predict_batch(X_test)
        pred_values = np.array([p.price_prediction for p in predictions])

        # Regression metrics
        mae = np.mean(np.abs(y_test - pred_values))
        mse = np.mean((y_test - pred_values) ** 2)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((y_test - pred_values) / (y_test + 1e-10))) * 100

        # Direction accuracy
        actual_direction = np.sign(np.diff(np.concatenate([[y_test[0]], y_test])))
        pred_direction = np.sign(np.diff(np.concatenate([[pred_values[0]], pred_values])))
        direction_accuracy = np.mean(actual_direction == pred_direction)

        return {
            "mae": float(mae),
            "mse": float(mse),
            "rmse": float(rmse),
            "mape": float(mape),
            "direction_accuracy": float(direction_accuracy),
        }


class ModelRegistry:
    """Registry for managing model classes."""

    _models: Dict[str, Type[BaseModel]] = {}

    @classmethod
    def register(cls, name: str, model_class: Type[BaseModel]) -> None:
        """Register a model class."""
        cls._models[name.lower()] = model_class

    @classmethod
    def create(cls, name: str, config: Optional[Dict[str, Any]] = None) -> BaseModel:
        """Create a model instance by name."""
        model_class = cls._models.get(name.lower())
        if model_class is None:
            available = ", ".join(cls._models.keys())
            raise ValueError(f"Unknown model: {name}. Available: {available}")
        return model_class(config)

    @classmethod
    def available_models(cls) -> List[str]:
        """Get list of available model names."""
        return list(cls._models.keys())
