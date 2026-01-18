#!/usr/bin/env python3
"""End-to-end test pipeline for AI-Trader.

This script executes a complete trading system test:
1. Load multi-timeframe data (1H, 4H, 1D)
2. Train all three time-resolution models
3. Create ensemble model with calibration
4. Run simulated trading session on out-of-sample data

Usage:
    python scripts/run_full_e2e_test.py --train-year 2020 --test-months 6
"""

import argparse
import logging
import sys
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import torch

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# Model configurations - optimized for CPU training
# Using smaller models that can still learn meaningful patterns
MODEL_CONFIGS = {
    "short_term": {
        "architecture": "cnn_lstm_attention",
        "timeframe": "1H",
        "hidden_dim": 32,
        "num_layers": 1,
        "sequence_length": 24,  # 1 day of hourly data
        "prediction_horizons": [1],
    },
    "medium_term": {
        "architecture": "tft",
        "timeframe": "4H",
        "hidden_dim": 32,
        "num_layers": 1,
        "sequence_length": 30,  # ~5 days of 4H data
        "prediction_horizons": [1],
    },
    "long_term": {
        "architecture": "nbeats",
        "timeframe": "1D",
        "hidden_dim": 32,
        "num_layers": 1,
        "sequence_length": 20,  # ~1 month of daily data
        "prediction_horizons": [1],
    },
}


def load_timeframe_data(timeframe: str, data_dir: Path) -> pd.DataFrame:
    """Load data for a specific timeframe."""
    filepath = data_dir / timeframe / f"EURUSD_{timeframe}.parquet"
    logger.info(f"Loading {timeframe} data from {filepath}")

    df = pd.read_parquet(filepath)
    df.columns = [c.lower() for c in df.columns]

    # Set timestamp as index
    if "timestamp" in df.columns:
        df = df.set_index("timestamp")
        df.index = pd.to_datetime(df.index)

    # Drop metadata columns
    cols_to_drop = ["slice_id", "base_minutes", "target_minutes", "candles_aggregated"]
    for col in cols_to_drop:
        if col in df.columns:
            df = df.drop(columns=[col])

    df = df.sort_index()
    logger.info(f"  Loaded {len(df)} rows from {df.index.min()} to {df.index.max()}")

    return df


def calculate_features(df: pd.DataFrame, model_type: str) -> pd.DataFrame:
    """Calculate technical indicators."""
    from src.features.technical.calculator import TechnicalIndicatorCalculator

    calculator = TechnicalIndicatorCalculator(model_type=model_type)
    df_features = calculator.calculate(df)

    logger.info(f"  Calculated {len(calculator.get_feature_names())} features")
    return df_features


def split_data_by_date(
    df: pd.DataFrame,
    train_end: str,
    val_end: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split data chronologically by date."""
    train = df[df.index < train_end]
    val = df[(df.index >= train_end) & (df.index < val_end)]
    test = df[df.index >= val_end]

    logger.info(f"  Train: {len(train)} ({train.index.min()} to {train.index.max()})")
    logger.info(f"  Val: {len(val)} ({val.index.min()} to {val.index.max()})")
    logger.info(f"  Test: {len(test)} ({test.index.min()} to {test.index.max()})")

    return train, val, test


def train_single_model(
    model_type: str,
    df_features: pd.DataFrame,
    epochs: int = 5,
    batch_size: int = 16,
) -> Tuple[Any, Dict]:
    """Train a single model."""
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

    config = MODEL_CONFIGS[model_type]

    # Data loader config
    loader_config = DataLoaderConfig(
        sequence_length=config["sequence_length"],
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

    # Training config - optimized for CPU
    training_config = TrainingConfig(
        name=f"{model_type}_e2e",
        batch_size=batch_size,
        epochs=epochs,
        verbose=1,
        device="cpu",
        seed=42,
        early_stopping=EarlyStoppingConfig(
            enabled=True,
            patience=3,
            min_delta=1e-4,
        ),
        scheduler=SchedulerConfig(
            scheduler_type=SchedulerType.COSINE,
            T_max=epochs,
        ),
    )
    training_config.optimizer.learning_rate = 1e-3

    # Create and train
    trainer = Trainer(
        architecture=config["architecture"],
        config=training_config,
        input_dim=input_dim,
        sequence_length=config["sequence_length"],
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"],
        prediction_horizons=config["prediction_horizons"],
    )

    logger.info(f"  Model parameters: {trainer.model.get_num_parameters():,}")

    results = trainer.fit(train_loader, val_loader)

    return trainer, results, test_loader, loader


def evaluate_model(trainer, test_loader, prices: pd.Series) -> Dict:
    """Evaluate a trained model."""
    from src.training.evaluation import ModelEvaluator

    evaluator = ModelEvaluator(device="cpu")
    result = evaluator.evaluate(
        model=trainer.model,
        test_loader=test_loader,
        prices=prices,
    )

    return {
        "accuracy": result.direction_metrics.accuracy,
        "sharpe": result.trading_metrics.sharpe_ratio,
        "max_drawdown": result.trading_metrics.max_drawdown,
        "total_return": result.trading_metrics.total_return,
        "win_rate": result.trading_metrics.win_rate,
    }


class SimplifiedBacktester:
    """Simplified backtester for ensemble predictions."""

    def __init__(
        self,
        initial_balance: float = 10000.0,
        position_size_pct: float = 0.02,
        commission: float = 0.0001,
    ):
        self.initial_balance = initial_balance
        self.position_size_pct = position_size_pct
        self.commission = commission
        self.reset()

    def reset(self):
        self.balance = self.initial_balance
        self.equity_history = []
        self.trade_history = []
        self.position = 0  # 1 = long, -1 = short, 0 = flat
        self.entry_price = 0

    def run(
        self,
        predictions: List[Dict],
        prices: pd.Series,
        min_confidence: float = 0.6,
    ) -> Dict:
        """Run backtest with predictions."""
        self.reset()

        for i, pred in enumerate(predictions):
            if i >= len(prices) - 1:
                break

            current_price = prices.iloc[i]
            next_price = prices.iloc[i + 1]

            # Record equity
            unrealized_pnl = 0
            if self.position != 0:
                price_change = (current_price - self.entry_price) / self.entry_price
                unrealized_pnl = self.position * price_change * self.balance * self.position_size_pct

            self.equity_history.append({
                "timestamp": prices.index[i],
                "equity": self.balance + unrealized_pnl,
            })

            # Get prediction
            direction = pred.get("direction", 0)
            confidence = pred.get("confidence", 0)

            # Trading logic
            if confidence >= min_confidence:
                target_position = 1 if direction > 0 else (-1 if direction < 0 else 0)
            else:
                target_position = 0

            # Execute trades
            if target_position != self.position:
                # Close existing position
                if self.position != 0:
                    price_change = (current_price - self.entry_price) / self.entry_price
                    pnl = self.position * price_change * self.balance * self.position_size_pct
                    pnl -= abs(pnl) * self.commission
                    self.balance += pnl

                    self.trade_history.append({
                        "exit_time": prices.index[i],
                        "exit_price": current_price,
                        "pnl": pnl,
                        "direction": "long" if self.position > 0 else "short",
                    })

                # Open new position
                if target_position != 0:
                    self.position = target_position
                    self.entry_price = current_price
                else:
                    self.position = 0
                    self.entry_price = 0

        # Close final position
        if self.position != 0 and len(prices) > 0:
            final_price = prices.iloc[-1]
            price_change = (final_price - self.entry_price) / self.entry_price
            pnl = self.position * price_change * self.balance * self.position_size_pct
            self.balance += pnl

        return self._calculate_metrics()

    def _calculate_metrics(self) -> Dict:
        """Calculate performance metrics."""
        if not self.equity_history:
            return {"error": "No equity history"}

        equity = pd.Series([e["equity"] for e in self.equity_history])
        returns = equity.pct_change().dropna()

        # Sharpe ratio
        if len(returns) > 0 and returns.std() > 0:
            sharpe = returns.mean() / returns.std() * np.sqrt(252)
        else:
            sharpe = 0.0

        # Max drawdown
        cummax = equity.cummax()
        drawdown = (equity - cummax) / cummax
        max_dd = abs(drawdown.min())

        # Trade statistics
        trades = self.trade_history
        winning = [t for t in trades if t["pnl"] > 0]
        losing = [t for t in trades if t["pnl"] <= 0]

        win_rate = len(winning) / len(trades) if trades else 0

        avg_win = np.mean([t["pnl"] for t in winning]) if winning else 0
        avg_loss = np.mean([t["pnl"] for t in losing]) if losing else 0

        profit_factor = abs(sum(t["pnl"] for t in winning) / sum(t["pnl"] for t in losing)) if losing and sum(t["pnl"] for t in losing) != 0 else float("inf")

        return {
            "initial_balance": self.initial_balance,
            "final_balance": self.balance,
            "total_return": (self.balance - self.initial_balance) / self.initial_balance,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_dd,
            "total_trades": len(trades),
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
        }


def create_ensemble_predictions(
    models: Dict[str, Any],
    test_data: Dict[str, pd.DataFrame],
    weights: Dict[str, float] = None,
) -> List[Dict]:
    """Generate ensemble predictions from trained models."""

    if weights is None:
        weights = {"short_term": 0.4, "medium_term": 0.35, "long_term": 0.25}

    # Normalize weights
    total = sum(weights.values())
    weights = {k: v/total for k, v in weights.items()}

    # Get predictions from each model
    all_preds = {}

    for model_type, (trainer, _, test_loader, _) in models.items():
        trainer.model.eval()
        preds = []

        with torch.no_grad():
            for batch in test_loader:
                inputs = batch[0].to(trainer.device)
                outputs = trainer.model(inputs)

                # Extract direction logits
                if "direction_logits" in outputs:
                    logits = outputs["direction_logits"]
                    # Shape: (batch, horizons, classes) or (batch, classes)
                    if logits.dim() == 3:
                        logits = logits[:, 0, :]  # First horizon
                    probs = torch.softmax(logits, dim=-1)
                    # Classes: 0=down, 1=neutral, 2=up
                    direction = probs[:, 2] - probs[:, 0]  # Positive = bullish

                    # Confidence from Beta if available
                    if "alpha" in outputs and "beta" in outputs:
                        alpha = outputs["alpha"][:, 0] if outputs["alpha"].dim() > 1 else outputs["alpha"]
                        beta = outputs["beta"][:, 0] if outputs["beta"].dim() > 1 else outputs["beta"]
                        confidence = (alpha / (alpha + beta)).cpu().numpy()
                    else:
                        confidence = probs.max(dim=-1).values.cpu().numpy()

                    for d, c in zip(direction.cpu().numpy(), confidence):
                        preds.append({"direction": float(d), "confidence": float(c)})

        all_preds[model_type] = preds

    # Combine predictions
    min_len = min(len(p) for p in all_preds.values())
    ensemble_preds = []

    for i in range(min_len):
        combined_direction = sum(
            weights[m] * all_preds[m][i]["direction"]
            for m in weights if m in all_preds
        )
        combined_confidence = sum(
            weights[m] * all_preds[m][i]["confidence"]
            for m in weights if m in all_preds
        )

        # Agreement score
        directions = [np.sign(all_preds[m][i]["direction"]) for m in weights if m in all_preds]
        majority = np.sign(combined_direction)
        agreement = sum(1 for d in directions if d == majority) / len(directions)

        # Scale confidence by agreement
        final_confidence = combined_confidence * (0.5 + 0.5 * agreement)

        ensemble_preds.append({
            "direction": combined_direction,
            "confidence": final_confidence,
            "agreement": agreement,
        })

    return ensemble_preds


def main():
    parser = argparse.ArgumentParser(description="End-to-end AI-Trader test")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs per model")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--output-dir", type=str, default="models/e2e_test", help="Output directory")
    parser.add_argument("--train-start", type=str, default="2020-07-01", help="Training start date")
    parser.add_argument("--train-end", type=str, default="2021-04-01", help="Training end date")
    parser.add_argument("--val-end", type=str, default="2021-07-01", help="Validation end date")
    args = parser.parse_args()

    # Set seeds
    torch.manual_seed(42)
    np.random.seed(42)

    print("\n" + "=" * 70)
    print("AI-TRADER END-TO-END PIPELINE TEST")
    print("=" * 70)

    data_dir = project_root / "data" / "forex" / "derived"
    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Date splits
    train_start = args.train_start  # Start of training period
    train_end = args.train_end      # End of training (9 months)
    val_end = args.val_end          # End of validation (3 months)

    print(f"\nData splits:")
    print(f"  Training: {train_start} - {train_end}")
    print(f"  Validation: {train_end} - {val_end}")
    print(f"  Testing: {val_end} - End")

    # Store models and results
    trained_models = {}
    model_results = {}

    # Train each model
    for model_type in ["short_term", "medium_term", "long_term"]:
        print(f"\n{'='*60}")
        print(f"TRAINING {model_type.upper()} MODEL")
        print(f"{'='*60}")

        config = MODEL_CONFIGS[model_type]
        timeframe = config["timeframe"]

        # Load data
        df = load_timeframe_data(timeframe, data_dir)

        # Filter to training period for features (train_start to val_end)
        df_period = df[(df.index >= train_start) & (df.index < val_end)]

        # Calculate features
        print(f"\nCalculating features...")
        df_features = calculate_features(df_period, model_type)

        # Train model
        print(f"\nTraining {model_type} model...")
        trainer, results, test_loader, loader = train_single_model(
            model_type,
            df_features,
            epochs=args.epochs,
            batch_size=args.batch_size,
        )

        trained_models[model_type] = (trainer, results, test_loader, loader)

        # Evaluate
        print(f"\nEvaluating {model_type}...")
        metrics = evaluate_model(trainer, test_loader, df_features["close"])
        model_results[model_type] = metrics

        print(f"  Accuracy: {metrics['accuracy']:.2%}")
        print(f"  Sharpe: {metrics['sharpe']:.2f}")
        print(f"  Max DD: {metrics['max_drawdown']:.2%}")

        # Save model
        model_path = output_dir / model_type
        trainer.save(model_path)
        print(f"  Saved to {model_path}")

    # Create ensemble predictions
    print(f"\n{'='*60}")
    print("CREATING ENSEMBLE")
    print(f"{'='*60}")

    # Load test data for backtesting
    df_1h = load_timeframe_data("1H", data_dir)
    df_test = df_1h[df_1h.index >= val_end]
    print(f"\nTest period: {df_test.index.min()} to {df_test.index.max()}")
    print(f"Test samples: {len(df_test)}")

    # Generate ensemble predictions
    print("\nGenerating ensemble predictions...")
    ensemble_preds = create_ensemble_predictions(
        trained_models,
        {"1H": df_test},
        weights={"short_term": 0.4, "medium_term": 0.35, "long_term": 0.25},
    )
    print(f"Generated {len(ensemble_preds)} predictions")

    # Run backtest
    print(f"\n{'='*60}")
    print("RUNNING BACKTEST SIMULATION")
    print(f"{'='*60}")

    backtester = SimplifiedBacktester(
        initial_balance=10000.0,
        position_size_pct=0.02,
        commission=0.0001,
    )

    # Align predictions with test prices
    test_prices = df_test["close"]
    if len(ensemble_preds) < len(test_prices):
        test_prices = test_prices.iloc[:len(ensemble_preds)]

    backtest_results = backtester.run(
        predictions=ensemble_preds,
        prices=test_prices,
        min_confidence=0.55,
    )

    # Print results
    print(f"\n{'='*60}")
    print("FINAL RESULTS")
    print(f"{'='*60}")

    print("\n## Individual Model Performance:")
    for model_type, metrics in model_results.items():
        print(f"\n  {model_type.upper()}:")
        print(f"    Accuracy: {metrics['accuracy']:.2%}")
        print(f"    Sharpe: {metrics['sharpe']:.2f}")
        print(f"    Max Drawdown: {metrics['max_drawdown']:.2%}")

    print("\n## Ensemble Backtest Results:")
    print(f"  Initial Balance: ${backtest_results['initial_balance']:,.2f}")
    print(f"  Final Balance: ${backtest_results['final_balance']:,.2f}")
    print(f"  Total Return: {backtest_results['total_return']:.2%}")
    print(f"  Sharpe Ratio: {backtest_results['sharpe_ratio']:.2f}")
    print(f"  Max Drawdown: {backtest_results['max_drawdown']:.2%}")
    print(f"  Total Trades: {backtest_results['total_trades']}")
    print(f"  Win Rate: {backtest_results['win_rate']:.2%}")
    print(f"  Profit Factor: {backtest_results['profit_factor']:.2f}")

    # Target assessment
    print("\n## Target Assessment:")
    targets = {
        "Accuracy > 55%": any(m["accuracy"] > 0.55 for m in model_results.values()),
        "Sharpe > 1.5": backtest_results["sharpe_ratio"] > 1.5,
        "Max DD < 15%": backtest_results["max_drawdown"] < 0.15,
        "Positive Return": backtest_results["total_return"] > 0,
    }

    for target, met in targets.items():
        status = "PASS" if met else "FAIL"
        print(f"  {target}: {status}")

    # Save results
    results_file = output_dir / "e2e_results.json"
    with open(results_file, "w") as f:
        json.dump({
            "model_results": model_results,
            "backtest_results": backtest_results,
            "targets": targets,
            "timestamp": datetime.now().isoformat(),
        }, f, indent=2, default=str)
    print(f"\nResults saved to {results_file}")

    print("\n" + "=" * 70)
    print("END-TO-END TEST COMPLETE")
    print("=" * 70)

    return model_results, backtest_results


if __name__ == "__main__":
    main()
