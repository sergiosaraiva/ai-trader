"""RFECV configuration for feature selection.

This module defines the configuration for Recursive Feature Elimination with
Cross-Validation (RFECV) used in the MTF Ensemble feature selection.
"""

from dataclasses import dataclass


@dataclass
class RFECVConfig:
    """Configuration for RFECV feature selection.

    Attributes:
        step: Fraction of features to remove per iteration (0.0-1.0)
        min_features_to_select: Minimum number of features to keep
        cv: Number of cross-validation folds (TimeSeriesSplit)
        n_estimators: Number of trees in XGBoost estimator
        max_depth: Maximum tree depth for XGBoost
        learning_rate: Learning rate for XGBoost
        scoring: Scoring metric for cross-validation
        n_jobs: Number of parallel jobs (-1 for all cores)
        verbose: Verbosity level (0=silent, 1=progress, 2=detailed)
        cache_enabled: Whether to cache selected features to disk
        cache_dir: Directory for caching feature selections
        random_state: Random seed for reproducibility
    """

    # Feature elimination settings
    step: float = 0.1  # Remove 10% of features per iteration
    min_features_to_select: int = 20  # Minimum features to keep

    # Cross-validation settings
    cv: int = 5  # Number of CV folds (TimeSeriesSplit)

    # XGBoost estimator parameters
    n_estimators: int = 100
    max_depth: int = 4
    learning_rate: float = 0.1

    # Evaluation settings
    scoring: str = "accuracy"
    n_jobs: int = -1  # Use all available cores
    verbose: int = 1  # Show progress

    # Caching settings
    cache_enabled: bool = True
    cache_dir: str = "models/feature_selections"

    # Reproducibility
    random_state: int = 42
