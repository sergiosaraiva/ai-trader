---
name: creating-cli-scripts
description: Creates Python CLI scripts with argparse, logging, and progress output for training, backtesting, and data processing. Use when adding standalone command-line tools, batch processing scripts, or workflow automation.
---

# Creating CLI Scripts

## Quick Reference

- Use `argparse` for command-line arguments
- Add project root to `sys.path` before imports
- Configure logging at module level
- Use print with separators for progress output
- Always include `if __name__ == "__main__":` guard

## When to Use

- Training ML models
- Running backtests
- Processing/downloading data
- Walk-forward optimization
- Batch operations

## When NOT to Use

- API endpoints (use FastAPI routes)
- Library functions (use src/ modules)
- One-off analysis (use notebooks)

## Implementation Guide

```
Is this a new standalone script?
├─ Yes → Create in scripts/ directory
│   └─ Add shebang and module docstring
│   └─ Add project root to sys.path
└─ No → Modify existing script

Does script need command-line arguments?
├─ Yes → Use argparse with help text
│   └─ Provide sensible defaults
└─ No → Use hardcoded configuration

Does script produce output files?
├─ Yes → Add --output argument
│   └─ Create output directory if needed
└─ No → Print results to stdout
```

## Examples

**Example 1: Script Header and Imports**

```python
# From: scripts/train_mtf_ensemble.py:1-38
#!/usr/bin/env python3
"""Train Multi-Timeframe Ensemble model.

This script trains 3 XGBoost models at different timeframes (1H, 4H, Daily)
and combines them into a weighted ensemble:
- 1H (Short-term): 60% weight - entry timing
- 4H (Medium-term): 30% weight - trend confirmation
- Daily (Long-term): 10% weight - regime context

The ensemble reduces noise through higher timeframe filtering.
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import warnings
warnings.filterwarnings("ignore")

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd

from src.models.multi_timeframe import (
    MTFEnsemble,
    MTFEnsembleConfig,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)
```

**Explanation**: Shebang for direct execution. Docstring with usage description. Standard imports first. Project root added to path before local imports. Logging configured at module level.

**Example 2: Data Loading Function**

```python
# From: scripts/train_mtf_ensemble.py:41-65
def load_data(data_path: Path) -> pd.DataFrame:
    """Load 5-minute OHLCV data."""
    logger.info(f"Loading data from {data_path}")

    if data_path.suffix == ".parquet":
        df = pd.read_parquet(data_path)
    else:
        df = pd.read_csv(data_path)

    df.columns = [c.lower() for c in df.columns]

    time_col = None
    for col in ["timestamp", "time", "date", "datetime"]:
        if col in df.columns:
            time_col = col
            break

    if time_col:
        df[time_col] = pd.to_datetime(df[time_col])
        df = df.set_index(time_col)

    df = df.sort_index()
    logger.info(f"Loaded {len(df)} bars from {df.index[0]} to {df.index[-1]}")

    return df
```

**Explanation**: Function with type hints. Log progress. Support multiple file formats. Handle column name variations. Sort by time index.

**Example 3: Validation Function**

```python
# From: scripts/train_mtf_ensemble.py:68-148
def validate_ensemble(
    ensemble: MTFEnsemble,
    df_5min: pd.DataFrame,
    test_start_idx: int,
) -> dict:
    """Validate ensemble on held-out data."""
    from src.features.technical.calculator import TechnicalIndicatorCalculator

    logger.info("Validating ensemble on test data...")

    # Prepare test data for each timeframe
    timeframes = ["1H", "4H", "D"]
    X_dict = {}
    y_dict = {}

    for tf in timeframes:
        model = ensemble.models[tf]
        config = ensemble.model_configs[tf]

        # Resample
        df_tf = ensemble.resample_data(df_5min, config.base_timeframe)

        # Get features and labels
        X, y, _ = model.prepare_data(df_tf, higher_tf_data)

        # Split
        n_total = len(X)
        n_train = int(n_total * 0.6)
        n_val = int(n_total * 0.2)
        test_start = n_train + n_val

        X_dict[tf] = X[test_start:]
        y_dict[tf] = y[test_start:]

        logger.info(f"{tf}: {len(X_dict[tf])} test samples")

    # Calculate metrics
    accuracy = (directions == y_test).mean()

    results = {
        "test_samples": len(directions),
        "accuracy": accuracy,
        "mean_confidence": confidences.mean(),
        "mean_agreement": agreement_scores.mean(),
    }

    # Accuracy at confidence levels
    for thresh in [0.55, 0.60, 0.65, 0.70]:
        mask = confidences >= thresh
        if mask.sum() > 0:
            acc = (directions[mask] == y_test[mask]).mean()
            results[f"acc_conf_{int(thresh*100)}"] = acc
            results[f"samples_conf_{int(thresh*100)}"] = int(mask.sum())

    logger.info(f"Ensemble test accuracy: {accuracy:.2%}")

    return results
```

**Explanation**: Type hints on parameters and return. Log progress. Return dict with all metrics. Calculate metrics at multiple thresholds.

**Example 4: Argument Parser Pattern**

```python
# Pattern from: scripts/train_mtf_ensemble.py
def main():
    parser = argparse.ArgumentParser(
        description="Train MTF Ensemble model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Data arguments
    parser.add_argument(
        "--data",
        type=str,
        default="data/forex/EURUSD_20200101_20251231_5min_combined.csv",
        help="Path to 5-minute OHLCV data",
    )

    # Output arguments
    parser.add_argument(
        "--output",
        type=str,
        default="models/mtf_ensemble",
        help="Output directory for trained models",
    )

    # Training arguments
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.6,
        help="Training data ratio (default: 0.6)",
    )

    # Feature flags
    parser.add_argument(
        "--sentiment",
        action="store_true",
        help="Enable sentiment features on Daily model",
    )

    args = parser.parse_args()
```

**Explanation**: Descriptive description. Group related arguments. Provide defaults. Use action="store_true" for flags. Include help text for all arguments.

**Example 5: Main Function with Progress Output**

```python
# Pattern from: scripts/train_mtf_ensemble.py
def main():
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("MTF ENSEMBLE TRAINING")
    print("=" * 70)
    print(f"Data:     {args.data}")
    print(f"Output:   {args.output}")
    print(f"Sentiment: {'Enabled (Daily only)' if args.sentiment else 'Disabled'}")
    print("=" * 70)

    # Load data
    df_5min = load_data(Path(args.data))

    # Create config
    config = MTFEnsembleConfig(
        weights={"1H": 0.6, "4H": 0.3, "D": 0.1},
        include_sentiment=args.sentiment,
    )

    # Train
    ensemble = MTFEnsemble(config=config)
    results = ensemble.train(df_5min)

    # Print results
    print("\n" + "-" * 70)
    print("TRAINING RESULTS")
    print("-" * 70)
    for tf, tf_results in results.items():
        print(f"\n{tf} Model:")
        print(f"  Train Accuracy: {tf_results['train_accuracy']:.2%}")
        print(f"  Val Accuracy:   {tf_results['val_accuracy']:.2%}")

    # Save
    ensemble.save(Path(args.output))
    print(f"\nModels saved to: {args.output}")
    print("\nTraining complete!")


if __name__ == "__main__":
    main()
```

**Explanation**: Parse args first. Print banner with configuration. Log progress with separators. Print results in readable format. Save outputs. Print completion message.

## Quality Checklist

- [ ] Shebang `#!/usr/bin/env python3` at top
- [ ] Docstring with usage description
- [ ] Pattern matches `scripts/train_mtf_ensemble.py:1-38`
- [ ] Project root added to sys.path
- [ ] Logging configured at module level
- [ ] argparse with help text for all arguments
- [ ] Progress output with separators
- [ ] `if __name__ == "__main__":` guard

## Common Mistakes

- **Missing sys.path setup**: ImportError for src modules
  - Wrong: `from src.models import MTFEnsemble` without path setup
  - Correct: Add `sys.path.insert(0, str(project_root))` before imports

- **No default arguments**: Script requires all args
  - Wrong: `parser.add_argument("--data", type=str, required=True)`
  - Correct: `parser.add_argument("--data", type=str, default="data/...")`

- **Silent progress**: User doesn't know what's happening
  - Wrong: No print statements during long operations
  - Correct: Print banners, progress, and completion messages

## Validation

- [ ] Pattern confirmed in `scripts/train_mtf_ensemble.py:1-38`
- [ ] Script runs with `python scripts/script_name.py --help`
- [ ] Default values work without arguments

## Related Skills

- `creating-python-services` - Services used by scripts
- `creating-pydantic-schemas` - Config objects for scripts
