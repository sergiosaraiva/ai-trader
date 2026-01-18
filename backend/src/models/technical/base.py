"""Base class for technical analysis models."""

from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np

from ..base import BaseModel, Prediction, ModelRegistry


class TechnicalBaseModel(BaseModel):
    """Base class for technical analysis models."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize technical model."""
        super().__init__(config)
        self.device = self._get_device()

    def _get_device(self) -> str:
        """Get compute device."""
        device = self.config.get("device", "auto")

        if device == "auto":
            try:
                import torch
                if torch.cuda.is_available():
                    return "cuda"
                elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    return "mps"
            except ImportError:
                pass
            return "cpu"

        return device

    def _save_weights(self, path: Path) -> None:
        """Save PyTorch model weights."""
        if self.model is not None:
            try:
                import torch
                torch.save(self.model.state_dict(), path / "model.pt")
            except ImportError:
                pass

    def _load_weights(self, path: Path) -> None:
        """Load PyTorch model weights."""
        weights_path = path / "model.pt"
        if weights_path.exists() and self.model is not None:
            try:
                import torch
                self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
                self.model.eval()
            except ImportError:
                pass

    def predict_with_uncertainty(
        self,
        X: np.ndarray,
        n_samples: int = 100,
    ) -> Dict[str, Any]:
        """
        Make prediction with uncertainty estimation using MC Dropout.

        Args:
            X: Input features
            n_samples: Number of MC samples

        Returns:
            Dictionary with mean, std, and percentiles
        """
        if self.model is None:
            raise ValueError("Model not built")

        try:
            import torch

            self.model.train()  # Enable dropout
            X_tensor = torch.FloatTensor(X).to(self.device)

            predictions = []
            for _ in range(n_samples):
                with torch.no_grad():
                    pred = self.model(X_tensor.unsqueeze(0))
                    predictions.append(pred.cpu().numpy())

            predictions = np.array(predictions).squeeze()
            self.model.eval()

            return {
                "mean": np.mean(predictions),
                "std": np.std(predictions),
                "p10": np.percentile(predictions, 10),
                "p25": np.percentile(predictions, 25),
                "p50": np.percentile(predictions, 50),
                "p75": np.percentile(predictions, 75),
                "p90": np.percentile(predictions, 90),
            }

        except ImportError:
            return {"mean": 0.0, "std": 0.0}
