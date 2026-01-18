#!/usr/bin/env python3
"""
Training script for sentiment-enhanced technical analysis models.

This script trains models with or without sentiment features and provides
comparison metrics to evaluate the impact of sentiment integration.

Usage:
    # Train with sentiment
    python scripts/train_with_sentiment.py --pair EURUSD --model short_term --sentiment

    # Train without sentiment (baseline)
    python scripts/train_with_sentiment.py --pair EURUSD --model short_term --no-sentiment

    # Train both and compare
    python scripts/train_with_sentiment.py --pair EURUSD --model short_term --compare

    # Full training with all options
    python scripts/train_with_sentiment.py \
        --pair EURUSD \
        --model short_term \
        --sentiment \
        --epochs 100 \
        --batch-size 64 \
        --lr 1e-4 \
        --save-dir models/trained
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_price_data(pair: str, data_dir: str = "data") -> pd.DataFrame:
    """
    Load price data for a trading pair.

    Args:
        pair: Trading pair (e.g., 'EURUSD', 'BTCUSDT')
        data_dir: Base data directory

    Returns:
        DataFrame with OHLCV data and DatetimeIndex
    """
    # Forex data files
    forex_files = {
        'EURUSD': 'forex/EURUSD_20200101_20251231_5min_combined.csv',
        'GBPUSD': 'forex/GBPUSD_20200101_20251231_5min_combined.csv',
        'USDJPY': 'forex/USDJPY_20200101_20251231_5min_combined.csv',
        'AUDUSD': 'forex/AUDUSD_20200101_20251231_5min_combined.csv',
        'EURGBP': 'forex/EURGBP_20200101_20251231_5min_combined.csv',
    }

    # Crypto data files
    crypto_files = {
        'BTCUSDT': 'crypto/BTCUSDT_20200101_20251231_5m.csv',
        'ETHUSDT': 'crypto/ETHUSDT_20200101_20251231_5m.csv',
        'BNBUSDT': 'crypto/BNBUSDT_20200101_20251231_5m.csv',
        'SOLUSDT': 'crypto/SOLUSDT_20200101_20251231_5m.csv',
        'XRPUSDT': 'crypto/XRPUSDT_20200101_20251231_5m.csv',
    }

    all_files = {**forex_files, **crypto_files}
    pair_upper = pair.upper()

    if pair_upper not in all_files:
        raise ValueError(f"Unknown pair: {pair}. Available: {list(all_files.keys())}")

    file_path = Path(data_dir) / all_files[pair_upper]

    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")

    # Load data - handle different date column names
    df = pd.read_csv(file_path)

    # Find the date column (could be 'Date', 'timestamp', 'datetime', etc.)
    date_col = None
    for col in df.columns:
        if col.lower() in ['date', 'timestamp', 'datetime', 'time']:
            date_col = col
            break

    if date_col:
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.set_index(date_col)
    else:
        # Try to parse the first column as datetime
        df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0])
        df = df.set_index(df.columns[0])

    # Standardize column names
    df.columns = [c.lower() for c in df.columns]

    # Remove spread column if present (not needed)
    if 'spread' in df.columns:
        df = df.drop('spread', axis=1)

    return df


def prepare_training_data(
    pair: str,
    include_sentiment: bool = True,
    sequence_length: int = 168,
    prediction_horizon: int = 1,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    max_rows: Optional[int] = None,
    resample_timeframe: Optional[str] = None,
) -> Dict:
    """
    Prepare complete training dataset with optional sentiment.

    Args:
        pair: Trading pair
        include_sentiment: Whether to include sentiment features
        sequence_length: Input sequence length
        prediction_horizon: Steps ahead to predict
        train_ratio: Training data ratio
        val_ratio: Validation data ratio
        max_rows: Maximum rows to use (for faster testing)
        resample_timeframe: Resample to this timeframe (e.g., '1h', '4h', '1d')

    Returns:
        Dictionary with training data and metadata
    """
    from src.data.processors import FeatureProcessor

    # Load price data
    print(f"Loading price data for {pair}...")
    price_df = load_price_data(pair)

    if max_rows:
        price_df = price_df.iloc[-max_rows:]

    print(f"  Loaded {len(price_df):,} candles")

    # Resample if requested
    if resample_timeframe:
        print(f"  Resampling to {resample_timeframe}...")
        ohlc_dict = {
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum',
        }
        price_df = price_df.resample(resample_timeframe).agg(ohlc_dict).dropna()
        print(f"  After resampling: {len(price_df):,} candles")

    # Initialize feature processor
    processor = FeatureProcessor(include_sentiment=include_sentiment)

    # Prepare all features
    print("Calculating features...")
    df = processor.prepare_all_features(
        price_df=price_df,
        pair=pair,
        include_technical=True,
        include_temporal=True,
        include_session=True,
    )

    # Log feature counts
    sentiment_cols = [c for c in df.columns if 'sentiment' in c.lower() or 'sent_' in c.lower()]
    print(f"  Total features: {len(df.columns)}")
    print(f"  Sentiment features: {len(sentiment_cols)}")

    if include_sentiment and sentiment_cols:
        print(f"  Sentiment columns: {sentiment_cols[:5]}..." if len(sentiment_cols) > 5 else f"  Sentiment columns: {sentiment_cols}")

    # Handle remaining NaN
    initial_len = len(df)
    df = df.dropna()
    if initial_len - len(df) > 0:
        print(f"  Dropped {initial_len - len(df)} rows with NaN values")

    # Prepare for training
    print("Creating sequences...")
    data = processor.prepare_for_training(
        df=df,
        target_column='close',
        sequence_length=sequence_length,
        prediction_horizon=prediction_horizon,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
    )

    # Add metadata
    data['pair'] = pair
    data['include_sentiment'] = include_sentiment
    data['n_features'] = len(df.columns)
    data['n_sentiment_features'] = len(sentiment_cols)
    data['sentiment_feature_names'] = sentiment_cols

    print(f"  X shape: {data['X_train'].shape}")
    print(f"  Train: {len(data['X_train']):,} | Val: {len(data['X_val']):,} | Test: {len(data['X_test']):,}")

    return data


def train_model(
    model_type: str,
    data: Dict,
    epochs: int = 100,
    batch_size: int = 64,
    learning_rate: float = 1e-4,
    device: str = 'auto',
) -> Tuple:
    """
    Train a model with prepared data.

    Args:
        model_type: Model type ('short_term', 'medium_term', 'long_term')
        data: Training data dictionary
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        device: Device to use ('auto', 'cuda', 'cpu')

    Returns:
        Tuple of (trained_model, training_history, evaluation_metrics)
    """
    from src.models.technical import ShortTermModel, MediumTermModel, LongTermModel

    # Select model class
    model_classes = {
        'short_term': ShortTermModel,
        'medium_term': MediumTermModel,
        'long_term': LongTermModel,
    }

    if model_type not in model_classes:
        raise ValueError(f"Unknown model: {model_type}. Available: {list(model_classes.keys())}")

    # Create model config
    config = {
        'name': f"{model_type}_sentiment" if data['include_sentiment'] else model_type,
        'version': '2.0.0' if data['include_sentiment'] else '1.0.0',
        'sequence_length': data['sequence_length'],
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'epochs': epochs,
    }

    # Initialize model
    print(f"\nInitializing {model_type} model...")
    model = model_classes[model_type](config)
    model.build()

    # Train
    print("Training model...")
    history = model.train(
        X_train=data['X_train'].astype(np.float32),
        y_train=data['y_train'].astype(np.float32),
        X_val=data['X_val'].astype(np.float32),
        y_val=data['y_val'].astype(np.float32),
    )

    # Evaluate
    print("\nEvaluating on test set...")
    metrics = evaluate_model(
        model=model,
        X_test=data['X_test'].astype(np.float32),
        y_test=data['y_test'].astype(np.float32),
    )

    print(f"\nTest Metrics:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.6f}")

    return model, history, metrics


def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
    """
    Evaluate model on test data.

    Args:
        model: Trained model
        X_test: Test features
        y_test: Test targets

    Returns:
        Dictionary with evaluation metrics
    """
    predictions = model.predict_batch(X_test)

    # Extract price predictions
    pred_prices = np.array([p.price_prediction for p in predictions])

    # Calculate metrics
    mae = np.mean(np.abs(pred_prices - y_test))
    mse = np.mean((pred_prices - y_test) ** 2)
    rmse = np.sqrt(mse)

    # Direction accuracy
    actual_direction = np.sign(np.diff(np.concatenate([[y_test[0]], y_test])))
    pred_direction = np.sign(pred_prices - np.concatenate([[y_test[0]], y_test[:-1]]))
    direction_accuracy = np.mean(actual_direction[1:] == pred_direction[1:])

    # MAPE (Mean Absolute Percentage Error)
    mape = np.mean(np.abs((y_test - pred_prices) / y_test)) * 100

    return {
        'mae': float(mae),
        'mse': float(mse),
        'rmse': float(rmse),
        'mape': float(mape),
        'direction_accuracy': float(direction_accuracy),
    }


def save_model(model, save_path: Path, data: Dict, metrics: Dict) -> None:
    """Save model and metadata."""
    save_path.mkdir(parents=True, exist_ok=True)

    # Save model
    model.save(save_path / 'model')

    # Save metadata
    metadata = {
        'pair': data['pair'],
        'include_sentiment': data['include_sentiment'],
        'n_features': data['n_features'],
        'n_sentiment_features': data['n_sentiment_features'],
        'sequence_length': data['sequence_length'],
        'prediction_horizon': data['prediction_horizon'],
        'feature_names': data['feature_names'],
        'sentiment_feature_names': data.get('sentiment_feature_names', []),
        'metrics': metrics,
        'trained_at': datetime.now().isoformat(),
    }

    with open(save_path / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"Model saved to {save_path}")


def run_comparison(
    pair: str,
    model_type: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    save_dir: Path,
    sequence_length: int = 168,
    max_rows: Optional[int] = None,
    resample_timeframe: Optional[str] = None,
) -> Dict:
    """
    Run A/B comparison training with and without sentiment.

    Args:
        pair: Trading pair
        model_type: Model type
        epochs: Training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        save_dir: Directory to save models
        sequence_length: Input sequence length
        max_rows: Maximum rows to use
        resample_timeframe: Resample timeframe

    Returns:
        Dictionary with comparison results
    """
    results = {}

    # Train WITHOUT sentiment (baseline)
    print("\n" + "=" * 60)
    print("TRAINING BASELINE (WITHOUT SENTIMENT)")
    print("=" * 60)

    data_baseline = prepare_training_data(
        pair=pair,
        include_sentiment=False,
        sequence_length=sequence_length,
        max_rows=max_rows,
        resample_timeframe=resample_timeframe,
    )

    model_baseline, history_baseline, metrics_baseline = train_model(
        model_type=model_type,
        data=data_baseline,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
    )

    save_model(
        model=model_baseline,
        save_path=save_dir / f"{pair}_{model_type}_baseline",
        data=data_baseline,
        metrics=metrics_baseline,
    )

    results['baseline'] = {
        'metrics': metrics_baseline,
        'n_features': data_baseline['n_features'],
        'final_train_loss': history_baseline['train_loss'][-1] if history_baseline['train_loss'] else None,
        'final_val_loss': history_baseline['val_loss'][-1] if history_baseline.get('val_loss') else None,
    }

    # Train WITH sentiment
    print("\n" + "=" * 60)
    print("TRAINING WITH SENTIMENT")
    print("=" * 60)

    data_sentiment = prepare_training_data(
        pair=pair,
        include_sentiment=True,
        sequence_length=sequence_length,
        max_rows=max_rows,
        resample_timeframe=resample_timeframe,
    )

    model_sentiment, history_sentiment, metrics_sentiment = train_model(
        model_type=model_type,
        data=data_sentiment,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
    )

    save_model(
        model=model_sentiment,
        save_path=save_dir / f"{pair}_{model_type}_sentiment",
        data=data_sentiment,
        metrics=metrics_sentiment,
    )

    results['sentiment'] = {
        'metrics': metrics_sentiment,
        'n_features': data_sentiment['n_features'],
        'n_sentiment_features': data_sentiment['n_sentiment_features'],
        'final_train_loss': history_sentiment['train_loss'][-1] if history_sentiment['train_loss'] else None,
        'final_val_loss': history_sentiment['val_loss'][-1] if history_sentiment.get('val_loss') else None,
    }

    # Calculate improvements
    print("\n" + "=" * 60)
    print("COMPARISON RESULTS")
    print("=" * 60)

    improvements = {}
    for metric in metrics_baseline.keys():
        baseline_val = metrics_baseline[metric]
        sentiment_val = metrics_sentiment[metric]

        if metric == 'direction_accuracy':
            # Higher is better
            improvement = (sentiment_val - baseline_val) / baseline_val * 100
        else:
            # Lower is better
            improvement = (baseline_val - sentiment_val) / baseline_val * 100

        improvements[metric] = improvement

        indicator = "✓" if improvement > 0 else "✗"
        print(f"{metric}:")
        print(f"  Baseline:  {baseline_val:.6f}")
        print(f"  Sentiment: {sentiment_val:.6f}")
        print(f"  Change:    {improvement:+.2f}% {indicator}")
        print()

    results['improvements'] = improvements

    # Save comparison results
    comparison_file = save_dir / f"{pair}_{model_type}_comparison.json"
    with open(comparison_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nComparison results saved to {comparison_file}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    positive_improvements = sum(1 for v in improvements.values() if v > 0)
    total_metrics = len(improvements)

    print(f"Metrics improved: {positive_improvements}/{total_metrics}")
    print(f"Additional features: {results['sentiment']['n_sentiment_features']}")

    avg_improvement = np.mean(list(improvements.values()))
    if avg_improvement > 0:
        print(f"\n✓ SENTIMENT INTEGRATION RECOMMENDED (avg improvement: {avg_improvement:+.2f}%)")
    else:
        print(f"\n✗ SENTIMENT INTEGRATION NOT RECOMMENDED (avg change: {avg_improvement:+.2f}%)")

    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Train sentiment-enhanced technical analysis models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Train with sentiment
    python scripts/train_with_sentiment.py --pair EURUSD --model short_term --sentiment

    # Train baseline (no sentiment)
    python scripts/train_with_sentiment.py --pair EURUSD --model short_term --no-sentiment

    # Compare both
    python scripts/train_with_sentiment.py --pair EURUSD --model short_term --compare

    # Quick test with limited data
    python scripts/train_with_sentiment.py --pair EURUSD --model short_term --compare --max-rows 50000 --epochs 10
        """,
    )

    parser.add_argument(
        '--pair',
        type=str,
        required=True,
        help='Trading pair (e.g., EURUSD, BTCUSDT)',
    )
    parser.add_argument(
        '--model',
        type=str,
        default='short_term',
        choices=['short_term', 'medium_term', 'long_term'],
        help='Model type to train',
    )
    parser.add_argument(
        '--sentiment',
        action='store_true',
        default=True,
        help='Include sentiment features (default)',
    )
    parser.add_argument(
        '--no-sentiment',
        dest='sentiment',
        action='store_false',
        help='Exclude sentiment features (baseline)',
    )
    parser.add_argument(
        '--compare',
        action='store_true',
        help='Train both with and without sentiment and compare',
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Number of training epochs',
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=64,
        help='Batch size',
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=1e-4,
        help='Learning rate',
    )
    parser.add_argument(
        '--sequence-length',
        type=int,
        default=168,
        help='Input sequence length',
    )
    parser.add_argument(
        '--save-dir',
        type=str,
        default='models/trained',
        help='Directory to save trained models',
    )
    parser.add_argument(
        '--max-rows',
        type=int,
        default=None,
        help='Maximum rows to use (for faster testing)',
    )
    parser.add_argument(
        '--resample',
        type=str,
        default=None,
        help='Resample timeframe (e.g., 1h, 4h, 1d)',
    )

    args = parser.parse_args()

    print("=" * 60)
    print("SENTIMENT-ENHANCED MODEL TRAINING")
    print("=" * 60)
    print(f"Pair: {args.pair}")
    print(f"Model: {args.model}")
    print(f"Mode: {'Compare' if args.compare else ('With Sentiment' if args.sentiment else 'Baseline')}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Learning Rate: {args.lr}")
    print(f"Sequence Length: {args.sequence_length}")
    if args.max_rows:
        print(f"Max Rows: {args.max_rows}")
    if args.resample:
        print(f"Resample: {args.resample}")
    print("=" * 60)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if args.compare:
        # Run comparison
        run_comparison(
            pair=args.pair,
            model_type=args.model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            save_dir=save_dir,
            sequence_length=args.sequence_length,
            max_rows=args.max_rows,
            resample_timeframe=args.resample,
        )
    else:
        # Train single model
        data = prepare_training_data(
            pair=args.pair,
            include_sentiment=args.sentiment,
            sequence_length=args.sequence_length,
            max_rows=args.max_rows,
            resample_timeframe=args.resample,
        )

        model, history, metrics = train_model(
            model_type=args.model,
            data=data,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
        )

        # Save model
        suffix = "_sentiment" if args.sentiment else "_baseline"
        save_model(
            model=model,
            save_path=save_dir / f"{args.pair}_{args.model}{suffix}",
            data=data,
            metrics=metrics,
        )

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    main()
