#!/usr/bin/env python3
"""Train each time-resolution model individually on its proper timeframe data.

Short-term: 1H data (135K samples)
Medium-term: 4H data (135K samples)
Long-term: 1D data (135K samples)
"""

import argparse
import logging
import sys
import json
from datetime import datetime
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_timeframe_data(timeframe: str, subsample: float = 1.0) -> pd.DataFrame:
    """Load data for a specific timeframe.

    Args:
        timeframe: The timeframe to load (1H, 4H, 1D)
        subsample: Fraction of data to use (0.0-1.0)
    """
    # Use properly aggregated data (not sliding window)
    data_dir = project_root / "data" / "forex" / "derived_proper" / timeframe
    filepath = data_dir / f"EURUSD_{timeframe}.parquet"

    logger.info(f"Loading {timeframe} data from {filepath}")

    df = pd.read_parquet(filepath)
    df.columns = [c.lower() for c in df.columns]

    if "timestamp" in df.columns:
        df = df.set_index("timestamp")
        df.index = pd.to_datetime(df.index)

    # Drop metadata columns
    for col in ["slice_id", "base_minutes", "target_minutes", "candles_aggregated"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    df = df.sort_index()

    # Subsample if requested (use last portion to keep most recent data)
    if 0 < subsample < 1.0:
        n_samples = int(len(df) * subsample)
        df = df.iloc[-n_samples:]
        logger.info(f"Subsampled to {len(df)} rows ({subsample:.0%} of data)")

    logger.info(f"Loaded {len(df)} rows: {df.index.min()} to {df.index.max()}")

    return df


def calculate_features(df: pd.DataFrame, model_type: str) -> pd.DataFrame:
    """Calculate technical indicators."""
    from src.features.technical.calculator import TechnicalIndicatorCalculator

    calculator = TechnicalIndicatorCalculator(model_type=model_type)
    df_features = calculator.calculate(df)

    n_features = len(calculator.get_feature_names())
    logger.info(f"Calculated {n_features} features, {len(df_features)} samples after NaN handling")

    return df_features


def train_model(
    model_type: str,
    timeframe: str,
    df_features: pd.DataFrame,
    config: dict,
) -> dict:
    """Train a model and return results."""
    from src.data.loaders.training_loader import (
        DataLoaderConfig,
        LabelMethod,
        TrainingDataLoader,
    )
    from src.training import (
        EarlyStoppingConfig,
        SchedulerConfig,
        SchedulerType,
        Trainer,
        TrainingConfig,
    )
    from src.training.evaluation import ModelEvaluator

    logger.info(f"\n{'='*60}")
    logger.info(f"TRAINING {model_type.upper()} MODEL ON {timeframe} DATA")
    logger.info(f"{'='*60}")

    # Data loader - Use 3-class labels with timeframe-specific threshold
    # Thresholds are tuned per timeframe to get balanced label distribution
    loader_config = DataLoaderConfig(
        sequence_length=config["sequence_length"],
        prediction_horizon=config.get("prediction_horizon", 1),
        batch_size=config["batch_size"],
        train_ratio=0.7,
        val_ratio=0.15,
        label_method=LabelMethod.DIRECTION_THREE,  # 3 classes: down(-1), neutral(0), up(1)
        label_column="close",
        normalization="zscore",
        target_threshold=config.get("target_threshold", 0.0005),  # Timeframe-specific threshold
    )

    loader = TrainingDataLoader(loader_config)
    train_loader, val_loader, test_loader = loader.create_dataloaders(df_features)

    input_dim = len(loader.feature_names)

    logger.info(f"Data splits:")
    logger.info(f"  Train: {loader.train_size} samples")
    logger.info(f"  Val: {loader.val_size} samples")
    logger.info(f"  Test: {loader.test_size} samples")
    logger.info(f"  Input dim: {input_dim}")
    logger.info(f"  Sequence length: {config['sequence_length']}")

    # Training config
    training_config = TrainingConfig(
        name=f"{model_type}_{timeframe}",
        batch_size=config["batch_size"],
        epochs=config["epochs"],
        verbose=1,
        device="cpu",
        seed=42,
        early_stopping=EarlyStoppingConfig(
            enabled=True,
            patience=config.get("patience", 10),
            min_delta=1e-4,
        ),
        scheduler=SchedulerConfig(
            scheduler_type=SchedulerType.COSINE,
            T_max=config["epochs"],
        ),
    )
    training_config.optimizer.learning_rate = config.get("learning_rate", 1e-3)

    # Create trainer with direction-focused loss weights
    # Set price_weight=0 and confidence_weight=0 to focus entirely on direction prediction
    from src.training.trainer import MultiTaskLoss

    loss_fn = MultiTaskLoss(
        price_weight=0.0,       # Disable price regression loss
        direction_weight=1.0,   # Focus on direction classification
        confidence_weight=0.0,  # Disable confidence/uncertainty loss
        use_uncertainty_weighting=False,  # Use fixed weights, not learnable
    )

    trainer = Trainer(
        architecture=config["architecture"],
        config=training_config,
        input_dim=input_dim,
        sequence_length=config["sequence_length"],
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"],
        prediction_horizons=[1],
        loss_fn=loss_fn,
    )

    logger.info(f"Model: {config['architecture']}")
    logger.info(f"Parameters: {trainer.model.get_num_parameters():,}")

    # Train
    logger.info(f"\nStarting training for up to {config['epochs']} epochs...")
    start_time = datetime.now()

    train_results = trainer.fit(train_loader, val_loader)

    train_time = (datetime.now() - start_time).total_seconds()
    logger.info(f"Training completed in {train_time:.1f} seconds")
    logger.info(f"Best epoch: {train_results.get('best_epoch')}")
    logger.info(f"Best val_loss: {train_results.get('best_val_loss', 'N/A'):.6f}")

    # Evaluate on test set
    logger.info(f"\nEvaluating on test set...")
    evaluator = ModelEvaluator(device="cpu")
    eval_result = evaluator.evaluate(
        model=trainer.model,
        test_loader=test_loader,
        prices=df_features["close"],
    )

    # Print detailed results
    logger.info(f"\n{'='*40}")
    logger.info(f"TEST SET RESULTS - {model_type.upper()}")
    logger.info(f"{'='*40}")
    logger.info(f"Direction Accuracy: {eval_result.direction_metrics.accuracy:.2%}")
    logger.info(f"Precision (up): {eval_result.direction_metrics.precision.get('up', 0):.2%}")
    logger.info(f"Recall (up): {eval_result.direction_metrics.recall.get('up', 0):.2%}")
    logger.info(f"Precision (down): {eval_result.direction_metrics.precision.get('down', 0):.2%}")
    logger.info(f"Recall (down): {eval_result.direction_metrics.recall.get('down', 0):.2%}")
    logger.info(f"Sharpe Ratio: {eval_result.trading_metrics.sharpe_ratio:.2f}")
    logger.info(f"Max Drawdown: {eval_result.trading_metrics.max_drawdown:.2%}")
    logger.info(f"Total Return: {eval_result.trading_metrics.total_return:.2%}")
    logger.info(f"Win Rate: {eval_result.trading_metrics.win_rate:.2%}")

    # Save model
    output_dir = project_root / "models" / "individual_models" / f"{model_type}_{timeframe}"
    trainer.save(output_dir)
    logger.info(f"Model saved to {output_dir}")

    return {
        "model_type": model_type,
        "timeframe": timeframe,
        "train_samples": loader.train_size,
        "val_samples": loader.val_size,
        "test_samples": loader.test_size,
        "parameters": trainer.model.get_num_parameters(),
        "best_epoch": train_results.get("best_epoch"),
        "best_val_loss": train_results.get("best_val_loss"),
        "train_time_seconds": train_time,
        "accuracy": eval_result.direction_metrics.accuracy,
        "precision_up": eval_result.direction_metrics.precision.get("up", 0),
        "recall_up": eval_result.direction_metrics.recall.get("up", 0),
        "precision_down": eval_result.direction_metrics.precision.get("down", 0),
        "recall_down": eval_result.direction_metrics.recall.get("down", 0),
        "sharpe_ratio": eval_result.trading_metrics.sharpe_ratio,
        "max_drawdown": eval_result.trading_metrics.max_drawdown,
        "total_return": eval_result.trading_metrics.total_return,
        "win_rate": eval_result.trading_metrics.win_rate,
    }


def main():
    parser = argparse.ArgumentParser(description="Train individual time-resolution models")
    parser.add_argument("--model", type=str, choices=["short_term", "medium_term", "long_term", "all"],
                        default="all", help="Which model to train")
    parser.add_argument("--epochs", type=int, default=30, help="Max training epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size (larger = faster on CPU)")
    parser.add_argument("--patience", type=int, default=7, help="Early stopping patience")
    parser.add_argument("--subsample", type=float, default=1.0,
                        help="Fraction of data to use (0.0-1.0). Use 0.3 for faster testing.")
    args = parser.parse_args()

    torch.manual_seed(42)
    np.random.seed(42)

    print("\n" + "=" * 70)
    print("INDIVIDUAL MODEL TRAINING - CPU OPTIMIZED")
    print("=" * 70)

    # Model configurations - CPU-OPTIMIZED for reasonable training time
    # Key changes: smaller hidden_dim, shorter sequences, fewer layers
    # This reduces params from ~4M to ~200K while maintaining meaningful learning
    MODEL_CONFIGS = {
        "short_term": {
            "timeframe": "1H",
            "architecture": "cnn_lstm_attention",
            "sequence_length": 24,  # 1 day of hourly data (CPU-friendly)
            "hidden_dim": 128,      # Fixed: was 24, causing severe bottleneck (512->24->12)
            "num_layers": 2,        # Restored to 2 for better learning
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "patience": args.patience,
            "learning_rate": 5e-4,  # Lower LR for larger model
            # Threshold tuned for balanced labels: ~43% down, 15% neutral, 43% up
            "target_threshold": 0.0001,
        },
        "medium_term": {
            "timeframe": "4H",
            "architecture": "tft",
            "sequence_length": 42,  # 1 week of 4H data (CPU-friendly)
            "hidden_dim": 128,      # Fixed: was 24, causing bottleneck
            "num_layers": 2,        # Restored to 2
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "patience": args.patience,
            "learning_rate": 5e-4,
            # Threshold tuned for balanced labels: ~32% down, 35% neutral, 33% up
            "target_threshold": 0.0005,
        },
        "long_term": {
            "timeframe": "1D",
            "architecture": "nbeats",
            "sequence_length": 30,  # 1 month of daily data (CPU-friendly)
            "hidden_dim": 128,      # Fixed: was 24, causing bottleneck
            "num_layers": 2,        # Restored to 2
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "patience": args.patience,
            "learning_rate": 5e-4,
            # Threshold tuned for balanced labels: ~36% down, 30% neutral, 35% up
            "target_threshold": 0.001,
        },
    }

    models_to_train = [args.model] if args.model != "all" else ["short_term", "medium_term", "long_term"]

    print(f"Configuration:")
    print(f"  Epochs: {args.epochs}, Batch size: {args.batch_size}")
    print(f"  Early stopping patience: {args.patience}")
    print(f"  Data subsample: {args.subsample:.0%}")
    print(f"  Models to train: {', '.join(models_to_train)}")
    print("=" * 70)

    all_results = {}

    for model_type in models_to_train:
        config = MODEL_CONFIGS[model_type]
        timeframe = config["timeframe"]

        # Load data
        df = load_timeframe_data(timeframe, subsample=args.subsample)

        # Calculate features
        df_features = calculate_features(df, model_type)

        # Train and evaluate
        results = train_model(model_type, timeframe, df_features, config)
        all_results[model_type] = results

    # Summary
    print("\n" + "=" * 70)
    print("TRAINING SUMMARY")
    print("=" * 70)

    for model_type, results in all_results.items():
        print(f"\n{model_type.upper()} ({results['timeframe']}):")
        print(f"  Samples: {results['train_samples']} train, {results['test_samples']} test")
        print(f"  Parameters: {results['parameters']:,}")
        print(f"  Training time: {results['train_time_seconds']:.1f}s")
        print(f"  Accuracy: {results['accuracy']:.2%}")
        print(f"  Precision (up/down): {results['precision_up']:.2%} / {results['precision_down']:.2%}")
        print(f"  Recall (up/down): {results['recall_up']:.2%} / {results['recall_down']:.2%}")
        print(f"  Sharpe: {results['sharpe_ratio']:.2f}")
        print(f"  Return: {results['total_return']:.2%}")

    # Save results
    output_dir = project_root / "models" / "individual_models"
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "training_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\nResults saved to {output_dir / 'training_results.json'}")
    print("=" * 70)

    return all_results


if __name__ == "__main__":
    main()
