#!/usr/bin/env python3
"""Training script for AI-Trader models.

This script provides end-to-end model training including:
- Data loading from CSV files
- Feature engineering with technical indicators
- Model training with early stopping
- Evaluation and model saving
- MLflow experiment tracking

Usage:
    # Train a single model
    python scripts/train_model.py --model short_term --symbol EURUSD

    # Train all models
    python scripts/train_model.py --model all --symbol EURUSD

    # Train with custom configuration
    python scripts/train_model.py --model short_term --symbol EURUSD --epochs 50 --batch-size 32

    # Use specific config file
    python scripts/train_model.py --config configs/training/experiment_1.yaml
"""

import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import torch
import yaml

from src.data.loaders.training_loader import (
    DataLoaderConfig,
    LabelMethod,
    TrainingDataLoader,
)
from src.features.technical.calculator import TechnicalIndicatorCalculator
from src.training import (
    ArchitectureRegistry,
    EarlyStoppingConfig,
    SchedulerConfig,
    SchedulerType,
    Trainer,
    TrainingConfig,
)
from src.training.evaluation import EvaluationResult, ModelEvaluator
from src.training.experiment import ExperimentConfig, ExperimentManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# Model configurations for different types
MODEL_CONFIGS = {
    "short_term": {
        "architecture": "cnn_lstm_attention",
        "sequence_length": 50,  # Reduced for sample data
        "hidden_dim": 64,
        "num_layers": 2,
        "prediction_horizons": [1],
        "description": "Short-term CNN-LSTM-Attention model",
    },
    "medium_term": {
        "architecture": "tft",
        "sequence_length": 60,
        "hidden_dim": 64,
        "num_layers": 2,
        "prediction_horizons": [1],
        "description": "Medium-term Temporal Fusion Transformer model",
    },
    "long_term": {
        "architecture": "nbeats",
        "sequence_length": 90,
        "hidden_dim": 64,
        "num_layers": 2,
        "prediction_horizons": [1],
        "description": "Long-term N-BEATS model",
    },
}


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train AI-Trader models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Model selection
    parser.add_argument(
        "--model",
        type=str,
        choices=["short_term", "medium_term", "long_term", "all"],
        default="short_term",
        help="Model type to train",
    )

    # Data options
    parser.add_argument(
        "--symbol",
        type=str,
        default="EURUSD",
        help="Symbol to train on (e.g., EURUSD, GBPUSD)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/sample",
        help="Directory containing sample data",
    )

    # Training options
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Maximum training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Training batch size",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Learning rate",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Early stopping patience",
    )

    # Output options
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/trained",
        help="Directory to save trained models",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="MLflow experiment name",
    )

    # Configuration file
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML configuration file",
    )

    # Other options
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use (cpu, cuda, mps, auto)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        help="Verbosity level (0=silent, 1=progress, 2=detailed)",
    )
    parser.add_argument(
        "--no-mlflow",
        action="store_true",
        help="Disable MLflow tracking",
    )

    return parser.parse_args()


def load_data(symbol: str, data_dir: str) -> pd.DataFrame:
    """Load price data from CSV file.

    Args:
        symbol: Trading symbol (e.g., EURUSD).
        data_dir: Directory containing data files.

    Returns:
        DataFrame with OHLCV data.
    """
    data_path = Path(data_dir)

    # Try different filename patterns
    patterns = [
        f"{symbol}_daily.csv",
        f"{symbol.lower()}_daily.csv",
        f"{symbol}_Daily.csv",
        f"{symbol}.csv",
    ]

    for pattern in patterns:
        filepath = data_path / pattern
        if filepath.exists():
            logger.info(f"Loading data from {filepath}")
            df = pd.read_csv(filepath, parse_dates=["Date"], index_col="Date")

            # Normalize column names
            df.columns = [c.lower() for c in df.columns]

            logger.info(f"Loaded {len(df)} rows from {symbol}")
            return df

    raise FileNotFoundError(
        f"No data file found for {symbol} in {data_dir}. "
        f"Tried patterns: {patterns}"
    )


def prepare_features(df: pd.DataFrame, model_type: str) -> pd.DataFrame:
    """Calculate technical indicators for the DataFrame.

    Args:
        df: OHLCV DataFrame.
        model_type: Model type for indicator selection.

    Returns:
        DataFrame with features.
    """
    logger.info(f"Calculating features for {model_type} model...")

    # Use the calculator
    calculator = TechnicalIndicatorCalculator(model_type=model_type)
    df_features = calculator.calculate(df)

    logger.info(f"Generated {len(calculator.get_feature_names())} features")
    return df_features


def create_data_loaders(
    df: pd.DataFrame,
    model_type: str,
    batch_size: int,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader, TrainingDataLoader]:
    """Create training, validation, and test data loaders.

    Args:
        df: Feature DataFrame.
        model_type: Model type.
        batch_size: Batch size.

    Returns:
        Tuple of (train_loader, val_loader, test_loader, loader_instance).
    """
    model_config = MODEL_CONFIGS[model_type]
    sequence_length = model_config["sequence_length"]

    # Configure data loader
    config = DataLoaderConfig(
        sequence_length=sequence_length,
        prediction_horizon=1,
        batch_size=batch_size,
        train_ratio=0.7,
        val_ratio=0.15,
        label_method=LabelMethod.DIRECTION,
        label_column="close",
        normalization="zscore",
        drop_ohlcv=False,
    )

    loader = TrainingDataLoader(config)
    train_loader, val_loader, test_loader = loader.create_dataloaders(df)

    logger.info(
        f"Created loaders - Train: {loader.train_size}, "
        f"Val: {loader.val_size}, Test: {loader.test_size}"
    )

    return train_loader, val_loader, test_loader, loader


def create_trainer(
    model_type: str,
    input_dim: int,
    training_config: TrainingConfig,
) -> Trainer:
    """Create a trainer for the specified model type.

    Args:
        model_type: Model type.
        input_dim: Number of input features.
        training_config: Training configuration.

    Returns:
        Configured Trainer instance.
    """
    model_config = MODEL_CONFIGS[model_type]

    trainer = Trainer(
        architecture=model_config["architecture"],
        config=training_config,
        input_dim=input_dim,
        sequence_length=model_config["sequence_length"],
        hidden_dim=model_config["hidden_dim"],
        num_layers=model_config["num_layers"],
        prediction_horizons=model_config["prediction_horizons"],
    )

    logger.info(f"Created {model_type} model with {trainer.model.get_num_parameters():,} parameters")
    return trainer


def train_model(
    model_type: str,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    input_dim: int,
    args: argparse.Namespace,
    experiment_manager: Optional[ExperimentManager] = None,
) -> Tuple[Trainer, Dict[str, Any]]:
    """Train a single model.

    Args:
        model_type: Model type.
        train_loader: Training data loader.
        val_loader: Validation data loader.
        input_dim: Number of input features.
        args: Command line arguments.
        experiment_manager: Optional MLflow experiment manager.

    Returns:
        Tuple of (trained_trainer, training_results).
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Training {model_type.upper()} model")
    logger.info(f"{'='*60}")

    # Create training configuration
    training_config = TrainingConfig(
        name=f"{model_type}_{args.symbol}",
        batch_size=args.batch_size,
        epochs=args.epochs,
        verbose=args.verbose,
        device=args.device,
        seed=args.seed,
        early_stopping=EarlyStoppingConfig(
            enabled=True,
            patience=args.patience,
            min_delta=1e-4,
        ),
        scheduler=SchedulerConfig(
            scheduler_type=SchedulerType.COSINE,
            T_max=args.epochs,
        ),
    )
    training_config.optimizer.learning_rate = args.learning_rate

    # Create trainer
    trainer = create_trainer(model_type, input_dim, training_config)

    # Train with or without MLflow
    if experiment_manager:
        result = experiment_manager.run_experiment(
            trainer=trainer,
            train_loader=train_loader,
            val_loader=val_loader,
            training_config=training_config,
            run_name=f"{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        )
        results = result.metrics
    else:
        results = trainer.fit(train_loader, val_loader)

    logger.info(f"Training completed - Best val_loss: {results.get('best_val_loss', 'N/A'):.6f}")
    return trainer, results


def evaluate_model(
    trainer: Trainer,
    test_loader: torch.utils.data.DataLoader,
    prices: pd.Series,
    model_name: str,
) -> EvaluationResult:
    """Evaluate a trained model.

    Args:
        trainer: Trained trainer instance.
        test_loader: Test data loader.
        prices: Price series for trading simulation.
        model_name: Name of the model.

    Returns:
        Evaluation result.
    """
    logger.info(f"Evaluating {model_name}...")

    evaluator = ModelEvaluator(device=str(trainer.device))
    result = evaluator.evaluate(
        model=trainer.model,
        test_loader=test_loader,
        prices=prices,
    )

    # Print report
    report = evaluator.generate_report(result, model_name)
    print(report)

    return result


def save_model(
    trainer: Trainer,
    output_dir: str,
    model_type: str,
    symbol: str,
    loader: TrainingDataLoader,
) -> Path:
    """Save trained model.

    Args:
        trainer: Trained trainer instance.
        output_dir: Output directory.
        model_type: Model type.
        symbol: Trading symbol.
        loader: Data loader (for saving scalers).

    Returns:
        Path to saved model.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"{model_type}_{symbol}_{timestamp}"

    # Save model
    model_path = output_path / model_name
    trainer.save(model_path)

    # Save scalers
    scaler_path = output_path / f"{model_name}_scalers.pkl"
    loader.save_scalers(scaler_path)

    logger.info(f"Saved model to {model_path}")
    return model_path


def main():
    """Main training function."""
    args = parse_args()

    # Load config from file if specified
    if args.config:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
        # Override args with config values
        for key, value in config.items():
            if hasattr(args, key.replace("-", "_")):
                setattr(args, key.replace("-", "_"), value)

    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Determine which models to train
    if args.model == "all":
        model_types = ["short_term", "medium_term", "long_term"]
    else:
        model_types = [args.model]

    # Setup experiment tracking
    experiment_manager = None
    if not args.no_mlflow:
        try:
            exp_name = args.experiment_name or f"ai_trader_{args.symbol}"
            experiment_manager = ExperimentManager(
                config=ExperimentConfig(
                    name=exp_name,
                    description=f"Training run for {args.symbol}",
                    seed=args.seed,
                ),
            )
        except Exception as e:
            logger.warning(f"Failed to setup MLflow: {e}. Continuing without tracking.")

    # Load data
    try:
        df = load_data(args.symbol, args.data_dir)
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)

    # Store results
    trained_models = {}
    evaluation_results = {}

    # Train each model type
    for model_type in model_types:
        try:
            # Prepare features
            df_features = prepare_features(df.copy(), model_type)

            # Create data loaders
            train_loader, val_loader, test_loader, loader = create_data_loaders(
                df_features,
                model_type,
                args.batch_size,
            )

            # Get input dimension
            input_dim = len(loader.feature_names)

            # Train model
            trainer, train_results = train_model(
                model_type=model_type,
                train_loader=train_loader,
                val_loader=val_loader,
                input_dim=input_dim,
                args=args,
                experiment_manager=experiment_manager,
            )

            # Evaluate model
            eval_result = evaluate_model(
                trainer=trainer,
                test_loader=test_loader,
                prices=df["close"],
                model_name=f"{model_type}_{args.symbol}",
            )

            # Save model
            model_path = save_model(
                trainer=trainer,
                output_dir=args.output_dir,
                model_type=model_type,
                symbol=args.symbol,
                loader=loader,
            )

            # Store results
            trained_models[model_type] = {
                "trainer": trainer,
                "path": str(model_path),
                "train_results": train_results,
            }
            evaluation_results[model_type] = eval_result

        except Exception as e:
            logger.error(f"Failed to train {model_type}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Print summary
    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)

    for model_type, results in trained_models.items():
        eval_result = evaluation_results.get(model_type)
        print(f"\n{model_type.upper()}:")
        print(f"  Model saved to: {results['path']}")
        if eval_result:
            print(f"  Direction Accuracy: {eval_result.direction_metrics.accuracy:.2%}")
            print(f"  Sharpe Ratio: {eval_result.trading_metrics.sharpe_ratio:.2f}")
            print(f"  Max Drawdown: {eval_result.trading_metrics.max_drawdown:.2%}")

    print("\n" + "=" * 60)
    print(f"Training complete! Models saved to {args.output_dir}")
    print("=" * 60)

    return trained_models, evaluation_results


if __name__ == "__main__":
    main()
