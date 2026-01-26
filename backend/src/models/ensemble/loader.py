"""Utilities for loading ensemble models.

Provides convenient functions to discover and load trained models
and create ensemble predictors.
"""

import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch

from .predictor import EnsembleConfig, EnsemblePredictor

logger = logging.getLogger(__name__)


def discover_trained_models(
    models_dir: Union[str, Path] = "models/trained",
    symbol: Optional[str] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """Discover trained models in a directory.

    Args:
        models_dir: Directory containing trained models.
        symbol: Filter by symbol (e.g., "EURUSD").

    Returns:
        Dictionary mapping model type to list of model info dicts.
        Each info dict contains: path, symbol, timestamp, type.

    Example:
        ```python
        models = discover_trained_models("models/trained", symbol="EURUSD")
        # {
        #     "short_term": [
        #         {"path": "models/trained/short_term_EURUSD_20260108_211711", ...},
        #     ],
        #     "medium_term": [...],
        #     "long_term": [...],
        # }
        ```
    """
    models_dir = Path(models_dir)
    if not models_dir.exists():
        logger.warning(f"Models directory not found: {models_dir}")
        return {}

    # Pattern: {type}_{symbol}_{timestamp}
    pattern = re.compile(r"^(short_term|medium_term|long_term)_([A-Z]+)_(\d{8}_\d{6})$")

    discovered = {
        "short_term": [],
        "medium_term": [],
        "long_term": [],
    }

    for item in models_dir.iterdir():
        if not item.is_dir():
            continue

        match = pattern.match(item.name)
        if not match:
            continue

        model_type = match.group(1)
        model_symbol = match.group(2)
        timestamp_str = match.group(3)

        # Filter by symbol if specified
        if symbol and model_symbol != symbol:
            continue

        # Check if model.pt exists
        if not (item / "model.pt").exists():
            logger.debug(f"Skipping {item.name}: no model.pt found")
            continue

        # Parse timestamp
        try:
            timestamp = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
        except ValueError:
            timestamp = None

        info = {
            "path": str(item),
            "name": item.name,
            "type": model_type,
            "symbol": model_symbol,
            "timestamp": timestamp,
            "timestamp_str": timestamp_str,
        }

        discovered[model_type].append(info)

    # Sort by timestamp (newest first)
    for model_type in discovered:
        discovered[model_type].sort(
            key=lambda x: x["timestamp"] or datetime.min,
            reverse=True,
        )

    # Log summary
    total = sum(len(v) for v in discovered.values())
    logger.info(
        f"Discovered {total} trained models in {models_dir}: "
        f"short_term={len(discovered['short_term'])}, "
        f"medium_term={len(discovered['medium_term'])}, "
        f"long_term={len(discovered['long_term'])}"
    )

    return discovered


def get_latest_models(
    models_dir: Union[str, Path] = "models/trained",
    symbol: str = "EURUSD",
) -> Dict[str, str]:
    """Get paths to the latest trained models for each type.

    Args:
        models_dir: Directory containing trained models.
        symbol: Symbol to filter by.

    Returns:
        Dictionary mapping model type to path.

    Raises:
        FileNotFoundError: If no models found for any type.
    """
    discovered = discover_trained_models(models_dir, symbol=symbol)

    latest = {}
    missing = []

    for model_type in ["short_term", "medium_term", "long_term"]:
        models = discovered.get(model_type, [])
        if models:
            latest[model_type] = models[0]["path"]
        else:
            missing.append(model_type)

    if missing:
        raise FileNotFoundError(
            f"No trained models found for: {missing}. "
            f"Run training first with: python scripts/train_model.py --model all --symbol {symbol}"
        )

    return latest


def load_ensemble(
    models_dir: Union[str, Path] = "models/trained",
    symbol: str = "EURUSD",
    device: str = "auto",
    config: Optional[EnsembleConfig] = None,
) -> EnsemblePredictor:
    """Load all latest models and create an EnsemblePredictor.

    This is the main convenience function for loading a complete ensemble.

    Args:
        models_dir: Directory containing trained models.
        symbol: Symbol to load models for.
        device: Device for inference ("cpu", "cuda", "mps", "auto").
        config: Optional ensemble configuration.

    Returns:
        Configured EnsemblePredictor ready for predictions.

    Example:
        ```python
        ensemble = load_ensemble(
            models_dir="models/trained",
            symbol="EURUSD",
            device="cuda",
        )

        prediction = ensemble.predict(features, symbol="EURUSD")
        ```
    """
    # Resolve device
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    logger.info(f"Loading ensemble for {symbol} on device={device}")

    # Get latest model paths
    model_paths = get_latest_models(models_dir, symbol)

    # Create ensemble predictor
    predictor = EnsemblePredictor.from_trained_models(
        model_paths=model_paths,
        config=config,
        device=device,
    )

    logger.info(f"Loaded ensemble with models: {list(model_paths.keys())}")
    return predictor


def load_single_model(
    model_path: Union[str, Path],
    device: str = "cpu",
) -> torch.nn.Module:
    """Load a single trained model.

    Args:
        model_path: Path to saved model directory.
        device: Device to load model to.

    Returns:
        Loaded PyTorch model.
    """
    from src.training.trainer import Trainer

    trainer = Trainer.load(model_path, device=device)
    model = trainer.model
    model.eval()

    return model


def get_model_info(model_path: Union[str, Path]) -> Dict[str, Any]:
    """Get information about a saved model.

    Args:
        model_path: Path to saved model directory.

    Returns:
        Dictionary with model information.
    """
    import json

    model_path = Path(model_path)

    info = {
        "path": str(model_path),
        "name": model_path.name,
        "exists": model_path.exists(),
    }

    if not model_path.exists():
        return info

    # Check for model files
    info["has_model"] = (model_path / "model.pt").exists()
    info["has_config"] = (model_path / "config.json").exists()
    info["has_architecture"] = (model_path / "architecture.json").exists()

    # Load config if available
    if info["has_config"]:
        try:
            with open(model_path / "config.json") as f:
                config = json.load(f)
            info["config"] = config
        except Exception as e:
            info["config_error"] = str(e)

    # Load architecture config if available
    if info["has_architecture"]:
        try:
            with open(model_path / "architecture.json") as f:
                arch = json.load(f)
            info["architecture"] = arch
        except Exception as e:
            info["architecture_error"] = str(e)

    # Check for scalers
    scaler_path = model_path.parent / f"{model_path.name}_scalers.pkl"
    info["has_scalers"] = scaler_path.exists()
    if info["has_scalers"]:
        info["scalers_path"] = str(scaler_path)

    return info


def validate_ensemble_models(
    model_paths: Dict[str, Union[str, Path]],
) -> Tuple[bool, List[str]]:
    """Validate that all ensemble models can be loaded.

    Args:
        model_paths: Dictionary mapping model names to paths.

    Returns:
        Tuple of (all_valid, list of error messages).
    """
    errors = []

    for name, path in model_paths.items():
        path = Path(path)

        if not path.exists():
            errors.append(f"{name}: Path does not exist: {path}")
            continue

        if not (path / "model.pt").exists():
            errors.append(f"{name}: model.pt not found in {path}")
            continue

        if not (path / "config.json").exists():
            errors.append(f"{name}: config.json not found in {path}")
            continue

        if not (path / "architecture.json").exists():
            errors.append(f"{name}: architecture.json not found in {path}")
            continue

        # Try to load the model
        try:
            from src.training.trainer import Trainer
            trainer = Trainer.load(path, device="cpu")
            del trainer  # Free memory
        except Exception as e:
            errors.append(f"{name}: Failed to load model: {e}")

    return len(errors) == 0, errors
