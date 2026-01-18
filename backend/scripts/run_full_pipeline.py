#!/usr/bin/env python3
"""Full pipeline execution script.

Executes a complete training pipeline:
1. Load raw forex data
2. Calculate technical indicators
3. Train model
4. Evaluate on test set
5. Generate conclusions
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import torch

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_forex_data(data_path: Path) -> pd.DataFrame:
    """Load forex data from parquet or CSV.

    Args:
        data_path: Path to data file.

    Returns:
        DataFrame with OHLCV data.
    """
    logger.info(f"Loading data from {data_path}")

    if data_path.suffix == ".parquet":
        df = pd.read_parquet(data_path)
    else:
        df = pd.read_csv(data_path, parse_dates=True, index_col=0)

    # Ensure lowercase columns
    df.columns = [c.lower() for c in df.columns]

    # If timestamp column exists, set it as index
    if "timestamp" in df.columns:
        df = df.set_index("timestamp")
        df.index = pd.to_datetime(df.index)

    # Ensure datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    # Drop non-OHLCV columns that are metadata
    cols_to_drop = ["slice_id", "base_minutes", "target_minutes", "candles_aggregated"]
    for col in cols_to_drop:
        if col in df.columns:
            df = df.drop(columns=[col])

    # Sort by date
    df = df.sort_index()

    logger.info(f"Loaded {len(df)} rows from {df.index.min()} to {df.index.max()}")
    logger.info(f"Columns: {df.columns.tolist()}")

    return df


def calculate_features(df: pd.DataFrame, model_type: str = "short_term") -> pd.DataFrame:
    """Calculate technical indicators.

    Args:
        df: OHLCV DataFrame.
        model_type: Model type for indicator selection.

    Returns:
        DataFrame with features.
    """
    from src.features.technical.calculator import TechnicalIndicatorCalculator

    logger.info(f"Calculating features for {model_type} model...")

    calculator = TechnicalIndicatorCalculator(model_type=model_type)
    df_features = calculator.calculate(df)

    feature_names = calculator.get_feature_names()
    logger.info(f"Generated {len(feature_names)} features")
    logger.info(f"Sample features: {feature_names[:10]}")

    return df_features


def train_model(
    df_features: pd.DataFrame,
    model_type: str = "short_term",
    epochs: int = 30,
    batch_size: int = 32,
    sequence_length: int = 50,
) -> dict:
    """Train a model on the data.

    Args:
        df_features: Feature DataFrame.
        model_type: Type of model to train.
        epochs: Maximum training epochs.
        batch_size: Batch size.
        sequence_length: Input sequence length.

    Returns:
        Dictionary with training results.
    """
    from src.data.loaders.training_loader import (
        DataLoaderConfig,
        LabelMethod,
        TrainingDataLoader,
    )
    from src.training import (
        ArchitectureRegistry,
        EarlyStoppingConfig,
        SchedulerConfig,
        SchedulerType,
        Trainer,
        TrainingConfig,
    )
    from src.training.evaluation import ModelEvaluator

    # Architecture configs
    MODEL_CONFIGS = {
        "short_term": {
            "architecture": "cnn_lstm_attention",
            "hidden_dim": 64,
            "num_layers": 2,
            "prediction_horizons": [1],
        },
        "medium_term": {
            "architecture": "tft",
            "hidden_dim": 64,
            "num_layers": 2,
            "prediction_horizons": [1],
        },
        "long_term": {
            "architecture": "nbeats",
            "hidden_dim": 64,
            "num_layers": 2,
            "prediction_horizons": [1],
        },
    }

    model_config = MODEL_CONFIGS[model_type]

    # Configure data loader
    loader_config = DataLoaderConfig(
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

    loader = TrainingDataLoader(loader_config)
    train_loader, val_loader, test_loader = loader.create_dataloaders(df_features)

    input_dim = len(loader.feature_names)

    logger.info(f"Data splits - Train: {loader.train_size}, Val: {loader.val_size}, Test: {loader.test_size}")
    logger.info(f"Input dimension: {input_dim}")
    logger.info(f"Split dates: {loader.split_dates}")

    # Configure training
    training_config = TrainingConfig(
        name=f"{model_type}_pipeline",
        batch_size=batch_size,
        epochs=epochs,
        verbose=1,
        device="auto",
        seed=42,
        early_stopping=EarlyStoppingConfig(
            enabled=True,
            patience=10,
            min_delta=1e-4,
        ),
        scheduler=SchedulerConfig(
            scheduler_type=SchedulerType.COSINE,
            T_max=epochs,
        ),
    )
    training_config.optimizer.learning_rate = 1e-3

    # Create trainer
    trainer = Trainer(
        architecture=model_config["architecture"],
        config=training_config,
        input_dim=input_dim,
        sequence_length=sequence_length,
        hidden_dim=model_config["hidden_dim"],
        num_layers=model_config["num_layers"],
        prediction_horizons=model_config["prediction_horizons"],
    )

    logger.info(f"Model parameters: {trainer.model.get_num_parameters():,}")

    # Train
    logger.info("Starting training...")
    train_results = trainer.fit(train_loader, val_loader)

    logger.info(f"Training completed - Best val_loss: {train_results.get('best_val_loss', 'N/A'):.6f}")

    # Evaluate on test set
    logger.info("Evaluating on test set...")
    evaluator = ModelEvaluator(device=str(trainer.device))
    eval_result = evaluator.evaluate(
        model=trainer.model,
        test_loader=test_loader,
        prices=df_features["close"],
    )

    # Print report
    report = evaluator.generate_report(eval_result, f"{model_type}_EURUSD")
    print(report)

    return {
        "trainer": trainer,
        "train_results": train_results,
        "eval_result": eval_result,
        "loader": loader,
        "input_dim": input_dim,
        "report": report,
    }


def analyze_results(results: dict) -> str:
    """Analyze results and generate conclusions.

    Args:
        results: Training results dictionary.

    Returns:
        Conclusions string.
    """
    eval_result = results["eval_result"]
    train_results = results["train_results"]

    conclusions = []
    conclusions.append("\n" + "=" * 70)
    conclusions.append("PIPELINE EXECUTION CONCLUSIONS")
    conclusions.append("=" * 70)

    # Training analysis
    conclusions.append("\n## Training Analysis")
    conclusions.append(f"- Final training epochs: {train_results.get('current_epoch', 'N/A')}")
    conclusions.append(f"- Best validation loss: {train_results.get('best_val_loss', 'N/A'):.6f}")
    conclusions.append(f"- Best epoch: {train_results.get('best_epoch', 'N/A')}")
    conclusions.append(f"- Status: {train_results.get('status', 'N/A')}")

    # Direction prediction analysis
    conclusions.append("\n## Direction Prediction Quality")
    accuracy = eval_result.direction_metrics.accuracy
    conclusions.append(f"- Directional Accuracy: {accuracy:.2%}")

    if accuracy > 0.55:
        conclusions.append("  [PASS] Meets target accuracy (>55%)")
    elif accuracy > 0.52:
        conclusions.append("  [MARGINAL] Slightly above random (>52%)")
    else:
        conclusions.append("  [FAIL] Not significantly better than random")

    conclusions.append(f"- Precision (up): {eval_result.direction_metrics.precision.get('up', 0):.2%}")
    conclusions.append(f"- Recall (up): {eval_result.direction_metrics.recall.get('up', 0):.2%}")
    conclusions.append(f"- F1 Score (up): {eval_result.direction_metrics.f1_score.get('up', 0):.2%}")

    # Trading metrics analysis
    conclusions.append("\n## Simulated Trading Performance")
    sharpe = eval_result.trading_metrics.sharpe_ratio
    max_dd = eval_result.trading_metrics.max_drawdown
    total_ret = eval_result.trading_metrics.total_return

    conclusions.append(f"- Sharpe Ratio: {sharpe:.2f}")
    if sharpe > 1.5:
        conclusions.append("  [PASS] Excellent risk-adjusted returns (>1.5)")
    elif sharpe > 1.0:
        conclusions.append("  [GOOD] Acceptable risk-adjusted returns (>1.0)")
    elif sharpe > 0.0:
        conclusions.append("  [MARGINAL] Positive but weak risk-adjusted returns")
    else:
        conclusions.append("  [FAIL] Negative risk-adjusted returns")

    conclusions.append(f"- Maximum Drawdown: {max_dd:.2%}")
    if max_dd < 0.15:
        conclusions.append("  [PASS] Within acceptable drawdown (<15%)")
    elif max_dd < 0.25:
        conclusions.append("  [WARNING] Moderate drawdown (15-25%)")
    else:
        conclusions.append("  [FAIL] Excessive drawdown (>25%)")

    conclusions.append(f"- Total Return: {total_ret:.2%}")
    conclusions.append(f"- Win Rate: {eval_result.trading_metrics.win_rate:.2%}")
    conclusions.append(f"- Profit Factor: {eval_result.trading_metrics.profit_factor:.2f}")

    # Calibration analysis
    conclusions.append("\n## Confidence Calibration")
    ece = eval_result.calibration_metrics.ece
    conclusions.append(f"- Expected Calibration Error (ECE): {ece:.4f}")
    if ece < 0.05:
        conclusions.append("  [PASS] Well-calibrated predictions")
    elif ece < 0.10:
        conclusions.append("  [MARGINAL] Moderately calibrated")
    else:
        conclusions.append("  [FAIL] Poorly calibrated - consider calibration training")

    # Recommendations
    conclusions.append("\n## Recommendations")

    recommendations = []

    if accuracy < 0.55:
        recommendations.append("1. ACCURACY: Consider adding more features, longer sequences, or different architectures")

    if sharpe < 1.0:
        recommendations.append("2. SHARPE: Implement confidence-based position sizing to improve risk-adjusted returns")

    if max_dd > 0.15:
        recommendations.append("3. DRAWDOWN: Add stop-loss logic and position size limits")

    if ece > 0.05:
        recommendations.append("4. CALIBRATION: Apply temperature scaling or Platt scaling post-training")

    if not recommendations:
        recommendations.append("Model meets all performance targets!")

    conclusions.extend(recommendations)

    conclusions.append("\n" + "=" * 70)

    return "\n".join(conclusions)


def main():
    """Main pipeline execution."""
    parser = argparse.ArgumentParser(description="Run full AI-Trader training pipeline")
    parser.add_argument(
        "--data",
        type=str,
        default="data/forex/derived/1H/EURUSD_1H.parquet",
        help="Path to data file",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["short_term", "medium_term", "long_term"],
        default="short_term",
        help="Model type to train",
    )
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
        help="Batch size",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=50,
        help="Input sequence length",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/pipeline_run",
        help="Output directory",
    )

    args = parser.parse_args()

    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)

    print("\n" + "=" * 70)
    print("AI-TRADER FULL PIPELINE EXECUTION")
    print("=" * 70)
    print(f"Data: {args.data}")
    print(f"Model: {args.model}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Sequence Length: {args.sequence_length}")
    print("=" * 70 + "\n")

    # Step 1: Load data
    data_path = project_root / args.data
    df = load_forex_data(data_path)

    print(f"\nData Summary:")
    print(f"  Shape: {df.shape}")
    print(f"  Date Range: {df.index.min()} to {df.index.max()}")
    print(f"  Columns: {df.columns.tolist()}")
    print(f"\nData Statistics:")
    print(df.describe())

    # Step 2: Calculate features
    df_features = calculate_features(df, args.model)

    print(f"\nFeature Summary:")
    print(f"  Shape: {df_features.shape}")
    print(f"  NaN count after features: {df_features.isna().sum().sum()}")

    # Step 3: Train model
    results = train_model(
        df_features,
        model_type=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
    )

    # Step 4: Analyze and save results
    conclusions = analyze_results(results)
    print(conclusions)

    # Save results
    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save model
    model_path = output_dir / f"{args.model}_{timestamp}"
    results["trainer"].save(model_path)
    logger.info(f"Model saved to {model_path}")

    # Save conclusions
    conclusions_path = output_dir / f"conclusions_{timestamp}.txt"
    with open(conclusions_path, "w") as f:
        f.write(results["report"])
        f.write("\n\n")
        f.write(conclusions)
    logger.info(f"Conclusions saved to {conclusions_path}")

    return results


if __name__ == "__main__":
    main()
