#!/usr/bin/env python3
"""
A/B Testing Script for Sentiment Integration.

This script performs rigorous statistical comparison between models
trained with and without sentiment features. It runs multiple trials
to assess the statistical significance of any improvements.

Usage:
    # Full A/B test with 5 runs
    python scripts/ab_test_sentiment.py --pair EURUSD --model short_term --runs 5

    # Quick test with 3 runs and limited data
    python scripts/ab_test_sentiment.py --pair EURUSD --runs 3 --max-rows 50000 --epochs 20
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import warnings

import numpy as np
import pandas as pd

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def run_ab_test(
    pair: str,
    model_type: str = 'short_term',
    n_runs: int = 5,
    epochs: int = 50,
    batch_size: int = 64,
    learning_rate: float = 1e-4,
    sequence_length: int = 168,
    max_rows: Optional[int] = None,
    resample_timeframe: str = '1h',
    save_dir: Path = Path('models/ab_test'),
) -> Dict:
    """
    Run A/B test comparing sentiment vs no-sentiment models.

    Args:
        pair: Trading pair to test
        model_type: Model architecture ('short_term', 'medium_term', 'long_term')
        n_runs: Number of test runs for statistical significance
        epochs: Training epochs per run
        batch_size: Batch size
        learning_rate: Learning rate
        sequence_length: Input sequence length
        max_rows: Maximum data rows to use
        resample_timeframe: Timeframe to resample to
        save_dir: Directory to save results

    Returns:
        Dictionary with comprehensive test results
    """
    from scripts.train_with_sentiment import prepare_training_data, train_model

    results = {
        'baseline': [],
        'sentiment': [],
    }

    print("\n" + "=" * 70)
    print("A/B TEST: SENTIMENT VS BASELINE")
    print("=" * 70)
    print(f"Pair: {pair}")
    print(f"Model: {model_type}")
    print(f"Runs: {n_runs}")
    print(f"Epochs per run: {epochs}")
    print(f"Resample: {resample_timeframe}")
    print("=" * 70)

    for run in range(n_runs):
        print(f"\n{'='*70}")
        print(f"RUN {run + 1}/{n_runs}")
        print(f"{'='*70}")

        # Set different random seeds for each run
        seed = 42 + run * 100
        np.random.seed(seed)

        # Train WITHOUT sentiment (baseline)
        print(f"\n[Run {run + 1}] Training BASELINE...")
        try:
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

            metrics_baseline['run'] = run + 1
            metrics_baseline['n_features'] = data_baseline['n_features']
            metrics_baseline['final_train_loss'] = history_baseline['train_loss'][-1]
            metrics_baseline['final_val_loss'] = history_baseline['val_loss'][-1] if history_baseline.get('val_loss') else None

            results['baseline'].append(metrics_baseline)
            print(f"  Baseline - Direction Accuracy: {metrics_baseline['direction_accuracy']:.4f}")

        except Exception as e:
            print(f"  Baseline failed: {e}")
            continue

        # Train WITH sentiment
        print(f"\n[Run {run + 1}] Training WITH SENTIMENT...")
        try:
            # Reset seed for fair comparison
            np.random.seed(seed)

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

            metrics_sentiment['run'] = run + 1
            metrics_sentiment['n_features'] = data_sentiment['n_features']
            metrics_sentiment['n_sentiment_features'] = data_sentiment['n_sentiment_features']
            metrics_sentiment['final_train_loss'] = history_sentiment['train_loss'][-1]
            metrics_sentiment['final_val_loss'] = history_sentiment['val_loss'][-1] if history_sentiment.get('val_loss') else None

            results['sentiment'].append(metrics_sentiment)
            print(f"  Sentiment - Direction Accuracy: {metrics_sentiment['direction_accuracy']:.4f}")

        except Exception as e:
            print(f"  Sentiment failed: {e}")
            continue

    # Calculate statistics
    if len(results['baseline']) > 0 and len(results['sentiment']) > 0:
        stats = calculate_statistics(results)
        print_statistics(stats)

        # Save results
        save_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        results_file = save_dir / f"ab_test_{pair}_{model_type}_{timestamp}.json"
        full_results = {
            'config': {
                'pair': pair,
                'model_type': model_type,
                'n_runs': n_runs,
                'epochs': epochs,
                'batch_size': batch_size,
                'learning_rate': learning_rate,
                'sequence_length': sequence_length,
                'max_rows': max_rows,
                'resample_timeframe': resample_timeframe,
                'timestamp': timestamp,
            },
            'results': results,
            'statistics': stats,
        }

        with open(results_file, 'w') as f:
            json.dump(full_results, f, indent=2)

        print(f"\nResults saved to: {results_file}")

        # Generate recommendation
        generate_recommendation(stats)

        return full_results

    else:
        print("\nInsufficient successful runs for statistical analysis.")
        return {'error': 'Insufficient runs'}


def calculate_statistics(results: Dict) -> Dict:
    """Calculate statistical comparison between baseline and sentiment models."""
    stats = {}

    baseline = results['baseline']
    sentiment = results['sentiment']

    metrics = ['mae', 'mse', 'rmse', 'mape', 'direction_accuracy']

    for metric in metrics:
        baseline_vals = [r[metric] for r in baseline if metric in r]
        sentiment_vals = [r[metric] for r in sentiment if metric in r]

        if len(baseline_vals) < 2 or len(sentiment_vals) < 2:
            continue

        baseline_mean = np.mean(baseline_vals)
        baseline_std = np.std(baseline_vals)
        sentiment_mean = np.mean(sentiment_vals)
        sentiment_std = np.std(sentiment_vals)

        # For direction_accuracy, higher is better; for others, lower is better
        if metric == 'direction_accuracy':
            improvement = (sentiment_mean - baseline_mean) / baseline_mean * 100
            is_better = sentiment_mean > baseline_mean
        else:
            improvement = (baseline_mean - sentiment_mean) / baseline_mean * 100
            is_better = sentiment_mean < baseline_mean

        # T-test for statistical significance
        try:
            from scipy import stats as scipy_stats
            t_stat, p_value = scipy_stats.ttest_ind(sentiment_vals, baseline_vals)
        except ImportError:
            # If scipy not available, use simple comparison
            t_stat = (sentiment_mean - baseline_mean) / np.sqrt(baseline_std**2/len(baseline_vals) + sentiment_std**2/len(sentiment_vals))
            p_value = None  # Can't calculate without scipy

        stats[metric] = {
            'baseline_mean': float(baseline_mean),
            'baseline_std': float(baseline_std),
            'sentiment_mean': float(sentiment_mean),
            'sentiment_std': float(sentiment_std),
            'improvement_pct': float(improvement),
            'is_better': bool(is_better),
            't_statistic': float(t_stat),
            'p_value': float(p_value) if p_value is not None else None,
            'significant': p_value < 0.05 if p_value is not None else None,
        }

    return stats


def print_statistics(stats: Dict) -> None:
    """Print formatted statistical results."""
    print("\n" + "=" * 70)
    print("STATISTICAL ANALYSIS")
    print("=" * 70)

    for metric, s in stats.items():
        print(f"\n{metric.upper()}:")
        print(f"  Baseline:    {s['baseline_mean']:.6f} (+/- {s['baseline_std']:.6f})")
        print(f"  Sentiment:   {s['sentiment_mean']:.6f} (+/- {s['sentiment_std']:.6f})")
        print(f"  Change:      {s['improvement_pct']:+.2f}%")

        if s['p_value'] is not None:
            sig = "YES" if s['significant'] else "NO"
            print(f"  p-value:     {s['p_value']:.4f} (Significant: {sig})")

        indicator = "✓ BETTER" if s['is_better'] else "✗ WORSE"
        print(f"  Result:      {indicator}")


def generate_recommendation(stats: Dict) -> None:
    """Generate final recommendation based on statistics."""
    print("\n" + "=" * 70)
    print("RECOMMENDATION")
    print("=" * 70)

    # Count improvements
    better_metrics = sum(1 for s in stats.values() if s['is_better'])
    significant_improvements = sum(
        1 for s in stats.values()
        if s['is_better'] and s.get('significant', False)
    )
    total_metrics = len(stats)

    # Calculate weighted score (direction accuracy is most important)
    score = 0
    weights = {
        'direction_accuracy': 3.0,  # Most important for trading
        'mae': 1.0,
        'mse': 0.5,
        'rmse': 1.0,
        'mape': 1.0,
    }

    total_weight = 0
    for metric, s in stats.items():
        weight = weights.get(metric, 1.0)
        total_weight += weight
        if s['is_better']:
            score += weight

    normalized_score = score / total_weight if total_weight > 0 else 0

    print(f"\nMetrics improved: {better_metrics}/{total_metrics}")
    print(f"Statistically significant: {significant_improvements}/{total_metrics}")
    print(f"Weighted score: {normalized_score:.2f}")

    # Direction accuracy specific check
    dir_acc = stats.get('direction_accuracy', {})
    dir_improved = dir_acc.get('is_better', False)
    dir_significant = dir_acc.get('significant', False)
    dir_improvement = dir_acc.get('improvement_pct', 0)

    print(f"\nDirection Accuracy (most critical):")
    print(f"  Improvement: {dir_improvement:+.2f}%")
    print(f"  Significant: {'YES' if dir_significant else 'NO'}")

    # Final recommendation
    print("\n" + "-" * 70)

    if dir_improved and dir_significant and dir_improvement > 2:
        print("✓✓ STRONG RECOMMENDATION: Merge sentiment integration to main branch")
        print("   Direction accuracy significantly improved.")
    elif dir_improved and dir_improvement > 1:
        print("✓ RECOMMENDATION: Consider merging with caution")
        print("   Direction accuracy improved but may need more validation.")
    elif normalized_score >= 0.5:
        print("? NEUTRAL: Mixed results, more testing recommended")
        print("   Some metrics improved but overall inconclusive.")
    else:
        print("✗ NOT RECOMMENDED: Keep on feature branch")
        print("   Sentiment features did not improve predictions.")
        print("   Consider: more training epochs, different features, or data quality issues.")

    print("-" * 70)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="A/B test sentiment integration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        '--pair',
        type=str,
        default='EURUSD',
        help='Trading pair to test',
    )
    parser.add_argument(
        '--model',
        type=str,
        default='short_term',
        choices=['short_term', 'medium_term', 'long_term'],
        help='Model type',
    )
    parser.add_argument(
        '--runs',
        type=int,
        default=5,
        help='Number of test runs',
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='Training epochs per run',
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
        '--max-rows',
        type=int,
        default=None,
        help='Maximum data rows',
    )
    parser.add_argument(
        '--resample',
        type=str,
        default='1h',
        help='Resample timeframe',
    )
    parser.add_argument(
        '--save-dir',
        type=str,
        default='models/ab_test',
        help='Directory for results',
    )

    args = parser.parse_args()

    run_ab_test(
        pair=args.pair,
        model_type=args.model,
        n_runs=args.runs,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        sequence_length=args.sequence_length,
        max_rows=args.max_rows,
        resample_timeframe=args.resample,
        save_dir=Path(args.save_dir),
    )


if __name__ == '__main__':
    main()
