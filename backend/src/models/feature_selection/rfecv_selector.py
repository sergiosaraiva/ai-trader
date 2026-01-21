"""RFECV selector for feature selection.

This module implements Recursive Feature Elimination with Cross-Validation
using XGBoost and TimeSeriesSplit to prevent data leakage.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.feature_selection import RFECV
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier

from .rfecv_config import RFECVConfig

logger = logging.getLogger(__name__)


class RFECVSelector:
    """RFECV-based feature selector for time series data.

    Uses Recursive Feature Elimination with Cross-Validation to select
    optimal features. Uses TimeSeriesSplit to prevent data leakage.

    Attributes:
        config: RFECV configuration
        selector: Fitted RFECV instance
        selected_features: List of selected feature names
        selected_indices: Array of selected feature indices
        cv_scores: Cross-validation scores per iteration
    """

    def __init__(self, config: Optional[RFECVConfig] = None):
        """Initialize RFECV selector.

        Args:
            config: RFECV configuration. Uses defaults if None.
        """
        self.config = config or RFECVConfig()
        self.selector: Optional[RFECV] = None
        self.selected_features: Optional[List[str]] = None
        self.selected_indices: Optional[np.ndarray] = None
        self.cv_scores: Optional[Dict] = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
    ) -> Tuple[List[str], np.ndarray]:
        """Fit RFECV and select features.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target labels (n_samples,)
            feature_names: List of feature names

        Returns:
            Tuple of (selected_feature_names, selected_indices)

        Raises:
            ValueError: If inputs are invalid
        """
        if len(feature_names) != X.shape[1]:
            raise ValueError(
                f"Feature names length ({len(feature_names)}) must match "
                f"X columns ({X.shape[1]})"
            )

        if len(X) != len(y):
            raise ValueError(
                f"X length ({len(X)}) must match y length ({len(y)})"
            )

        logger.info(f"Starting RFECV with {X.shape[1]} features")
        logger.info(f"Config: step={self.config.step}, "
                   f"min_features={self.config.min_features_to_select}, "
                   f"cv_folds={self.config.cv}")

        # Create XGBoost estimator
        estimator = XGBClassifier(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            learning_rate=self.config.learning_rate,
            random_state=self.config.random_state,
            n_jobs=self.config.n_jobs,
            eval_metric="logloss",
        )

        # Create TimeSeriesSplit for cross-validation
        # CRITICAL: No shuffle to prevent data leakage
        cv = TimeSeriesSplit(n_splits=self.config.cv)

        # Create RFECV selector
        self.selector = RFECV(
            estimator=estimator,
            step=self.config.step,
            min_features_to_select=self.config.min_features_to_select,
            cv=cv,
            scoring=self.config.scoring,
            n_jobs=self.config.n_jobs,
            verbose=self.config.verbose,
        )

        # Fit selector
        logger.info("Fitting RFECV (this may take a while)...")
        self.selector.fit(X, y)

        # Extract results
        self.selected_indices = np.where(self.selector.support_)[0]
        self.selected_features = [feature_names[i] for i in self.selected_indices]

        # Store CV scores
        self.cv_scores = {
            "cv_scores_mean": self.selector.cv_results_["mean_test_score"].tolist(),
            "cv_scores_std": self.selector.cv_results_["std_test_score"].tolist(),
            "n_features": list(range(
                self.config.min_features_to_select,
                len(feature_names) + 1,
                max(1, int(self.config.step * len(feature_names)))
            )),
            "optimal_n_features": self.selector.n_features_,
        }

        logger.info(f"RFECV complete: selected {len(self.selected_features)} features")
        logger.info(f"Optimal n_features: {self.selector.n_features_}")
        logger.info(f"Best CV score: {max(self.cv_scores['cv_scores_mean']):.4f}")

        return self.selected_features, self.selected_indices

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform features using selected indices.

        Args:
            X: Feature matrix (n_samples, n_features)

        Returns:
            Transformed feature matrix (n_samples, n_selected_features)

        Raises:
            RuntimeError: If selector not fitted
        """
        if self.selector is None:
            raise RuntimeError("Selector not fitted. Call fit() first.")

        return self.selector.transform(X)

    def get_selected_features(self) -> List[str]:
        """Get list of selected feature names.

        Returns:
            List of selected feature names

        Raises:
            RuntimeError: If selector not fitted
        """
        if self.selected_features is None:
            raise RuntimeError("Selector not fitted. Call fit() first.")

        return self.selected_features

    def get_feature_importances(self) -> Dict[str, float]:
        """Get feature importances from final estimator.

        Returns:
            Dictionary mapping feature names to importance scores

        Raises:
            RuntimeError: If selector not fitted
        """
        if self.selector is None or self.selected_features is None:
            raise RuntimeError("Selector not fitted. Call fit() first.")

        importances = self.selector.estimator_.feature_importances_
        return dict(zip(self.selected_features, importances.tolist()))
