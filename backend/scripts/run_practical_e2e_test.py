#!/usr/bin/env python3
"""Practical end-to-end test using daily sample data.

This script executes a complete trading system test with data that
can be trained on CPU in reasonable time:
1. Uses daily EURUSD sample data (~1300 rows)
2. Trains 3 models with different lookback windows
3. Creates ensemble
4. Runs backtesting simulation

This demonstrates the full pipeline while being CPU-feasible.
"""

import argparse
import logging
import sys
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

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


# Small models for daily data (quick training)
MODEL_CONFIGS = {
    "short_term": {
        "architecture": "cnn_lstm_attention",
        "sequence_length": 10,  # 2 weeks lookback
        "hidden_dim": 24,
        "num_layers": 1,
    },
    "medium_term": {
        "architecture": "tft",
        "sequence_length": 20,  # 1 month lookback
        "hidden_dim": 24,
        "num_layers": 1,
    },
    "long_term": {
        "architecture": "nbeats",
        "sequence_length": 40,  # 2 months lookback
        "hidden_dim": 24,
        "num_layers": 1,
    },
}


def load_sample_data() -> pd.DataFrame:
    """Load EURUSD daily sample data."""
    filepath = project_root / "data" / "sample" / "EURUSD_daily.csv"
    logger.info(f"Loading data from {filepath}")

    df = pd.read_csv(filepath, parse_dates=["Date"], index_col="Date")
    df.columns = [c.lower() for c in df.columns]
    df = df.sort_index()

    logger.info(f"Loaded {len(df)} rows from {df.index.min()} to {df.index.max()}")
    return df


def calculate_features(df: pd.DataFrame, model_type: str = "short_term") -> pd.DataFrame:
    """Calculate technical indicators."""
    from src.features.technical.calculator import TechnicalIndicatorCalculator

    calculator = TechnicalIndicatorCalculator(model_type=model_type)
    df_features = calculator.calculate(df)
    logger.info(f"Calculated {len(calculator.get_feature_names())} features")
    return df_features


def train_model(
    model_type: str,
    df_features: pd.DataFrame,
    epochs: int,
    batch_size: int,
) -> Tuple[Any, Dict, Any, Any]:
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

    loader_config = DataLoaderConfig(
        sequence_length=config["sequence_length"],
        prediction_horizon=1,
        batch_size=batch_size,
        train_ratio=0.7,
        val_ratio=0.15,
        label_method=LabelMethod.DIRECTION,
        label_column="close",
        normalization="zscore",
    )

    loader = TrainingDataLoader(loader_config)
    train_loader, val_loader, test_loader = loader.create_dataloaders(df_features)

    input_dim = len(loader.feature_names)
    logger.info(f"  Train: {loader.train_size}, Val: {loader.val_size}, Test: {loader.test_size}")

    training_config = TrainingConfig(
        name=f"{model_type}_practical",
        batch_size=batch_size,
        epochs=epochs,
        verbose=1,
        device="cpu",
        seed=42,
        early_stopping=EarlyStoppingConfig(enabled=True, patience=5, min_delta=1e-4),
        scheduler=SchedulerConfig(scheduler_type=SchedulerType.COSINE, T_max=epochs),
    )
    training_config.optimizer.learning_rate = 1e-3

    trainer = Trainer(
        architecture=config["architecture"],
        config=training_config,
        input_dim=input_dim,
        sequence_length=config["sequence_length"],
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"],
        prediction_horizons=[1],
    )

    logger.info(f"  Parameters: {trainer.model.get_num_parameters():,}")

    results = trainer.fit(train_loader, val_loader)

    return trainer, results, test_loader, loader


def evaluate_model(trainer, test_loader, prices: pd.Series) -> Dict:
    """Evaluate model on test set."""
    from src.training.evaluation import ModelEvaluator

    evaluator = ModelEvaluator(device="cpu")
    result = evaluator.evaluate(
        model=trainer.model,
        test_loader=test_loader,
        prices=prices,
    )

    return {
        "accuracy": result.direction_metrics.accuracy,
        "precision_up": result.direction_metrics.precision.get("up", 0),
        "recall_up": result.direction_metrics.recall.get("up", 0),
        "sharpe": result.trading_metrics.sharpe_ratio,
        "max_drawdown": result.trading_metrics.max_drawdown,
        "total_return": result.trading_metrics.total_return,
        "win_rate": result.trading_metrics.win_rate,
    }


def generate_ensemble_predictions(
    models: Dict[str, Tuple],
    weights: Dict[str, float],
) -> List[Dict]:
    """Generate ensemble predictions."""
    # Normalize weights
    total = sum(weights.values())
    weights = {k: v/total for k, v in weights.items()}

    # Get predictions from each model
    all_preds = {}
    min_len = float('inf')

    for model_type, (trainer, _, test_loader, _) in models.items():
        trainer.model.eval()
        preds = []

        with torch.no_grad():
            for batch in test_loader:
                inputs = batch[0].to(trainer.device)
                outputs = trainer.model(inputs)

                if "direction_logits" in outputs:
                    logits = outputs["direction_logits"]
                    if logits.dim() == 3:
                        logits = logits[:, 0, :]
                    probs = torch.softmax(logits, dim=-1)
                    direction = probs[:, 2] - probs[:, 0]  # up - down

                    if "alpha" in outputs and "beta" in outputs:
                        alpha = outputs["alpha"][:, 0] if outputs["alpha"].dim() > 1 else outputs["alpha"]
                        beta = outputs["beta"][:, 0] if outputs["beta"].dim() > 1 else outputs["beta"]
                        confidence = (alpha / (alpha + beta)).cpu().numpy()
                    else:
                        confidence = probs.max(dim=-1).values.cpu().numpy()

                    for d, c in zip(direction.cpu().numpy(), confidence):
                        preds.append({"direction": float(d), "confidence": float(c)})

        all_preds[model_type] = preds
        min_len = min(min_len, len(preds))

    # Combine predictions
    ensemble_preds = []
    for i in range(int(min_len)):
        combined_direction = sum(
            weights[m] * all_preds[m][i]["direction"]
            for m in weights if m in all_preds
        )
        combined_confidence = sum(
            weights[m] * all_preds[m][i]["confidence"]
            for m in weights if m in all_preds
        )

        # Agreement
        directions = [np.sign(all_preds[m][i]["direction"]) for m in weights if m in all_preds]
        majority = np.sign(combined_direction) if combined_direction != 0 else 0
        agreement = sum(1 for d in directions if d == majority) / len(directions) if directions else 0

        ensemble_preds.append({
            "direction": combined_direction,
            "confidence": combined_confidence * (0.5 + 0.5 * agreement),
            "agreement": agreement,
        })

    return ensemble_preds


class SimpleBacktester:
    """Simple backtester for ensemble predictions."""

    def __init__(self, initial_balance: float = 10000.0, position_size: float = 0.02):
        self.initial_balance = initial_balance
        self.position_size = position_size
        self.reset()

    def reset(self):
        self.balance = self.initial_balance
        self.position = 0
        self.entry_price = 0
        self.equity_history = []
        self.trades = []

    def run(self, predictions: List[Dict], prices: pd.Series, min_confidence: float = 0.55) -> Dict:
        self.reset()

        for i, pred in enumerate(predictions):
            if i >= len(prices) - 1:
                break

            price = prices.iloc[i]

            # Update equity
            unrealized = 0
            if self.position != 0:
                pct_change = (price - self.entry_price) / self.entry_price
                unrealized = self.position * pct_change * self.balance * self.position_size

            self.equity_history.append(self.balance + unrealized)

            # Trading logic
            target = 0
            if pred["confidence"] >= min_confidence:
                if pred["direction"] > 0.1:
                    target = 1
                elif pred["direction"] < -0.1:
                    target = -1

            # Execute
            if target != self.position:
                if self.position != 0:
                    pct_change = (price - self.entry_price) / self.entry_price
                    pnl = self.position * pct_change * self.balance * self.position_size
                    self.balance += pnl
                    self.trades.append(pnl)

                if target != 0:
                    self.position = target
                    self.entry_price = price
                else:
                    self.position = 0

        # Close final position
        if self.position != 0:
            pct_change = (prices.iloc[-1] - self.entry_price) / self.entry_price
            pnl = self.position * pct_change * self.balance * self.position_size
            self.balance += pnl
            self.trades.append(pnl)

        return self._calc_metrics()

    def _calc_metrics(self) -> Dict:
        equity = pd.Series(self.equity_history)
        returns = equity.pct_change().dropna()

        sharpe = returns.mean() / returns.std() * np.sqrt(252) if len(returns) > 0 and returns.std() > 0 else 0

        cummax = equity.cummax()
        dd = (equity - cummax) / cummax
        max_dd = abs(dd.min()) if len(dd) > 0 else 0

        wins = [t for t in self.trades if t > 0]
        losses = [t for t in self.trades if t <= 0]
        win_rate = len(wins) / len(self.trades) if self.trades else 0

        return {
            "initial_balance": self.initial_balance,
            "final_balance": self.balance,
            "total_return": (self.balance - self.initial_balance) / self.initial_balance,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_dd,
            "total_trades": len(self.trades),
            "win_rate": win_rate,
            "avg_win": np.mean(wins) if wins else 0,
            "avg_loss": np.mean(losses) if losses else 0,
        }


def main():
    parser = argparse.ArgumentParser(description="Practical E2E test")
    parser.add_argument("--epochs", type=int, default=15, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--output-dir", type=str, default="models/practical_e2e", help="Output")
    args = parser.parse_args()

    torch.manual_seed(42)
    np.random.seed(42)

    print("\n" + "=" * 70)
    print("AI-TRADER PRACTICAL END-TO-END TEST")
    print("=" * 70)
    print(f"Using EURUSD daily sample data")
    print(f"Training 3 models with different lookback windows")
    print("=" * 70)

    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    df = load_sample_data()

    # Store trained models
    trained_models = {}
    model_metrics = {}

    # Train each model
    for model_type in ["short_term", "medium_term", "long_term"]:
        print(f"\n{'='*50}")
        print(f"TRAINING {model_type.upper()}")
        print(f"{'='*50}")

        # Calculate features
        df_features = calculate_features(df, model_type)
        print(f"Features: {df_features.shape[1]}, Samples: {len(df_features)}")

        # Train
        trainer, results, test_loader, loader = train_model(
            model_type, df_features, args.epochs, args.batch_size
        )

        trained_models[model_type] = (trainer, results, test_loader, loader)

        # Evaluate
        metrics = evaluate_model(trainer, test_loader, df_features["close"])
        model_metrics[model_type] = metrics

        print(f"\nResults:")
        print(f"  Accuracy: {metrics['accuracy']:.2%}")
        print(f"  Precision (up): {metrics['precision_up']:.2%}")
        print(f"  Sharpe: {metrics['sharpe']:.2f}")
        print(f"  Max DD: {metrics['max_drawdown']:.2%}")

        # Save
        trainer.save(output_dir / model_type)

    # Generate ensemble predictions
    print(f"\n{'='*50}")
    print("ENSEMBLE & BACKTEST")
    print(f"{'='*50}")

    weights = {"short_term": 0.4, "medium_term": 0.35, "long_term": 0.25}
    ensemble_preds = generate_ensemble_predictions(trained_models, weights)
    print(f"Generated {len(ensemble_preds)} ensemble predictions")

    # Run backtest
    backtester = SimpleBacktester(initial_balance=10000.0, position_size=0.02)

    # Get test prices from the shortest model's test set
    test_prices = df["close"].iloc[-len(ensemble_preds):]

    backtest_results = backtester.run(ensemble_preds, test_prices, min_confidence=0.55)

    # Final report
    print(f"\n{'='*70}")
    print("FINAL RESULTS")
    print(f"{'='*70}")

    print("\n## Individual Models:")
    for model_type, metrics in model_metrics.items():
        print(f"\n  {model_type.upper()}:")
        print(f"    Accuracy: {metrics['accuracy']:.2%}")
        print(f"    Sharpe: {metrics['sharpe']:.2f}")
        print(f"    Return: {metrics['total_return']:.2%}")

    print("\n## Ensemble Backtest:")
    print(f"  Initial: ${backtest_results['initial_balance']:,.2f}")
    print(f"  Final: ${backtest_results['final_balance']:,.2f}")
    print(f"  Return: {backtest_results['total_return']:.2%}")
    print(f"  Sharpe: {backtest_results['sharpe_ratio']:.2f}")
    print(f"  Max DD: {backtest_results['max_drawdown']:.2%}")
    print(f"  Trades: {backtest_results['total_trades']}")
    print(f"  Win Rate: {backtest_results['win_rate']:.2%}")

    # Target assessment
    print("\n## Target Assessment:")
    targets = {
        "Accuracy > 55%": any(m["accuracy"] > 0.55 for m in model_metrics.values()),
        "Sharpe > 1.5": backtest_results["sharpe_ratio"] > 1.5,
        "Max DD < 15%": backtest_results["max_drawdown"] < 0.15,
        "Positive Return": backtest_results["total_return"] > 0,
    }

    for target, passed in targets.items():
        print(f"  {target}: {'PASS' if passed else 'FAIL'}")

    # Save results
    results = {
        "model_metrics": model_metrics,
        "backtest_results": backtest_results,
        "targets": targets,
        "timestamp": datetime.now().isoformat(),
    }

    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to {output_dir}")
    print("=" * 70)

    return results


if __name__ == "__main__":
    main()
