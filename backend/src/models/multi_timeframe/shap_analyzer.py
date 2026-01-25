"""SHAP analyzer for feature importance explanation.

This module provides SHAP (SHapley Additive exPlanations) analysis for
understanding feature contributions in the MTF Ensemble models.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

# Handle optional SHAP dependency
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    shap = None

# Handle optional matplotlib (may not be available in headless environments)
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    plt = None

logger = logging.getLogger(__name__)


def _check_shap_available():
    """Check if SHAP is available, raise informative error if not."""
    if not SHAP_AVAILABLE:
        raise ImportError(
            "SHAP is not installed. Install it with: pip install shap>=0.43.0"
        )


def _check_matplotlib_available():
    """Check if matplotlib is available for plotting."""
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError(
            "matplotlib is not installed. Install it with: pip install matplotlib"
        )


class SHAPAnalyzer:
    """SHAP analyzer for XGBoost models.

    Uses TreeExplainer to compute SHAP values for feature importance
    and feature interaction analysis.

    Attributes:
        model: Trained XGBoost model
        explainer: SHAP TreeExplainer
        shap_values: Computed SHAP values
        feature_names: List of feature names
    """

    def __init__(self, model: XGBClassifier, feature_names: List[str]):
        """Initialize SHAP analyzer.

        Args:
            model: Trained XGBoost model
            feature_names: List of feature names

        Raises:
            ValueError: If model is not XGBoost or not trained
            ImportError: If SHAP is not installed
        """
        _check_shap_available()

        if not isinstance(model, XGBClassifier):
            raise ValueError("SHAP analyzer only supports XGBoost models")

        if not hasattr(model, "get_booster"):
            raise ValueError("Model must be trained before SHAP analysis")

        self.model = model
        self.feature_names = feature_names
        self.explainer: Optional[shap.TreeExplainer] = None
        self.shap_values: Optional[np.ndarray] = None
        self.base_value: Optional[float] = None

    def compute_shap_values(
        self,
        X: np.ndarray,
        check_additivity: bool = False,
    ) -> np.ndarray:
        """Compute SHAP values for feature contributions.

        Args:
            X: Feature matrix (n_samples, n_features)
            check_additivity: Whether to check SHAP additivity property

        Returns:
            SHAP values array (n_samples, n_features)

        Raises:
            ValueError: If X shape doesn't match feature_names
        """
        if X.shape[1] != len(self.feature_names):
            raise ValueError(
                f"X columns ({X.shape[1]}) must match feature_names ({len(self.feature_names)})"
            )

        logger.info(f"Computing SHAP values for {X.shape[0]} samples...")

        # Create TreeExplainer
        self.explainer = shap.TreeExplainer(self.model)

        # Compute SHAP values
        shap_explanation = self.explainer(X, check_additivity=check_additivity)

        # Extract values for binary classification (class 1)
        if len(shap_explanation.shape) == 3:
            # Multi-class output: (n_samples, n_features, n_classes)
            self.shap_values = shap_explanation.values[:, :, 1]
        else:
            # Binary output: (n_samples, n_features)
            self.shap_values = shap_explanation.values

        self.base_value = shap_explanation.base_values
        if isinstance(self.base_value, np.ndarray):
            self.base_value = float(self.base_value[0])

        logger.info(f"SHAP values computed: shape={self.shap_values.shape}")

        return self.shap_values

    def get_feature_importance(
        self,
        method: str = "mean_abs",
    ) -> pd.DataFrame:
        """Get feature importance from SHAP values.

        Args:
            method: Aggregation method ("mean_abs", "mean", "max_abs")

        Returns:
            DataFrame with features and importance scores

        Raises:
            RuntimeError: If SHAP values not computed
            ValueError: If method is invalid
        """
        if self.shap_values is None:
            raise RuntimeError("SHAP values not computed. Call compute_shap_values() first.")

        valid_methods = ["mean_abs", "mean", "max_abs"]
        if method not in valid_methods:
            raise ValueError(f"Invalid method: {method}. Use one of {valid_methods}")

        # Calculate importance based on method
        if method == "mean_abs":
            importance = np.abs(self.shap_values).mean(axis=0)
        elif method == "mean":
            importance = self.shap_values.mean(axis=0)
        elif method == "max_abs":
            importance = np.abs(self.shap_values).max(axis=0)

        # Create DataFrame
        df = pd.DataFrame({
            "feature": self.feature_names,
            "importance": importance,
        })

        # Sort by absolute importance
        df = df.sort_values("importance", key=abs, ascending=False)
        df = df.reset_index(drop=True)

        return df

    def plot_summary(
        self,
        X: np.ndarray,
        max_display: int = 20,
        output_path: Optional[Path] = None,
    ) -> None:
        """Plot SHAP summary plot.

        Args:
            X: Feature matrix used for SHAP computation
            max_display: Maximum number of features to display
            output_path: Path to save plot (shows if None)

        Raises:
            RuntimeError: If SHAP values not computed
        """
        if self.shap_values is None:
            raise RuntimeError("SHAP values not computed. Call compute_shap_values() first.")

        logger.info(f"Creating SHAP summary plot (top {max_display} features)...")

        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            self.shap_values,
            X,
            feature_names=self.feature_names,
            max_display=max_display,
            show=False,
        )

        if output_path:
            plt.savefig(output_path, bbox_inches="tight", dpi=150)
            logger.info(f"Saved SHAP summary plot to {output_path}")
            plt.close()
        else:
            plt.show()

    def plot_bar(
        self,
        max_display: int = 20,
        output_path: Optional[Path] = None,
    ) -> None:
        """Plot SHAP bar chart of feature importance.

        Args:
            max_display: Maximum number of features to display
            output_path: Path to save plot (shows if None)

        Raises:
            RuntimeError: If SHAP values not computed
        """
        if self.shap_values is None:
            raise RuntimeError("SHAP values not computed. Call compute_shap_values() first.")

        logger.info(f"Creating SHAP bar plot (top {max_display} features)...")

        plt.figure(figsize=(10, 8))
        shap.plots.bar(
            shap.Explanation(
                values=self.shap_values,
                base_values=self.base_value,
                feature_names=self.feature_names,
            ),
            max_display=max_display,
            show=False,
        )

        if output_path:
            plt.savefig(output_path, bbox_inches="tight", dpi=150)
            logger.info(f"Saved SHAP bar plot to {output_path}")
            plt.close()
        else:
            plt.show()

    def plot_waterfall(
        self,
        sample_idx: int,
        output_path: Optional[Path] = None,
    ) -> None:
        """Plot SHAP waterfall chart for a single prediction.

        Args:
            sample_idx: Index of sample to explain
            output_path: Path to save plot (shows if None)

        Raises:
            RuntimeError: If SHAP values not computed
            IndexError: If sample_idx out of range
        """
        if self.shap_values is None:
            raise RuntimeError("SHAP values not computed. Call compute_shap_values() first.")

        if sample_idx >= len(self.shap_values):
            raise IndexError(
                f"sample_idx {sample_idx} out of range (0-{len(self.shap_values)-1})"
            )

        logger.info(f"Creating SHAP waterfall plot for sample {sample_idx}...")

        plt.figure(figsize=(10, 8))
        shap.plots.waterfall(
            shap.Explanation(
                values=self.shap_values[sample_idx],
                base_values=self.base_value,
                feature_names=self.feature_names,
            ),
            show=False,
        )

        if output_path:
            plt.savefig(output_path, bbox_inches="tight", dpi=150)
            logger.info(f"Saved SHAP waterfall plot to {output_path}")
            plt.close()
        else:
            plt.show()

    def get_top_features(
        self,
        n: int = 20,
        method: str = "mean_abs",
    ) -> List[str]:
        """Get top N most important features.

        Args:
            n: Number of features to return
            method: Aggregation method ("mean_abs", "mean", "max_abs")

        Returns:
            List of top N feature names

        Raises:
            RuntimeError: If SHAP values not computed
        """
        importance_df = self.get_feature_importance(method=method)
        return importance_df["feature"].head(n).tolist()

    def get_bottom_features(
        self,
        n: int = 20,
        method: str = "mean_abs",
    ) -> List[str]:
        """Get bottom N least important features (candidates for removal).

        Args:
            n: Number of features to return
            method: Aggregation method ("mean_abs", "mean", "max_abs")

        Returns:
            List of bottom N feature names

        Raises:
            RuntimeError: If SHAP values not computed
        """
        importance_df = self.get_feature_importance(method=method)
        return importance_df["feature"].tail(n).tolist()

    def analyze_feature_interactions(
        self,
        X: np.ndarray,
        feature_pairs: Optional[List[Tuple[str, str]]] = None,
        max_display: int = 5,
    ) -> Dict[Tuple[str, str], float]:
        """Analyze feature interactions using SHAP interaction values.

        Args:
            X: Feature matrix
            feature_pairs: Specific pairs to analyze (None = top interactions)
            max_display: Number of top interactions to return

        Returns:
            Dictionary mapping feature pairs to interaction strength

        Raises:
            RuntimeError: If SHAP values not computed
        """
        if self.shap_values is None:
            raise RuntimeError("SHAP values not computed. Call compute_shap_values() first.")

        logger.info("Computing SHAP interaction values...")

        # Compute interaction values
        shap_interaction = self.explainer.shap_interaction_values(X)

        # Extract main effects (diagonal) vs interactions (off-diagonal)
        n_features = len(self.feature_names)
        interaction_strengths = {}

        if feature_pairs:
            # Analyze specific pairs
            for feat1, feat2 in feature_pairs:
                idx1 = self.feature_names.index(feat1)
                idx2 = self.feature_names.index(feat2)
                strength = np.abs(shap_interaction[:, idx1, idx2]).mean()
                interaction_strengths[(feat1, feat2)] = float(strength)
        else:
            # Find top interactions
            for i in range(n_features):
                for j in range(i + 1, n_features):
                    strength = np.abs(shap_interaction[:, i, j]).mean()
                    interaction_strengths[
                        (self.feature_names[i], self.feature_names[j])
                    ] = float(strength)

            # Sort and keep top
            interaction_strengths = dict(
                sorted(
                    interaction_strengths.items(),
                    key=lambda x: x[1],
                    reverse=True,
                )[:max_display]
            )

        logger.info(f"Found {len(interaction_strengths)} feature interactions")
        return interaction_strengths

    def save_analysis(
        self,
        output_dir: Path,
        X: np.ndarray,
        n_top_features: int = 50,
    ) -> None:
        """Save complete SHAP analysis to directory.

        Args:
            output_dir: Directory to save analysis
            X: Feature matrix used for SHAP computation
            n_top_features: Number of top features to save

        Raises:
            RuntimeError: If SHAP values not computed
        """
        if self.shap_values is None:
            raise RuntimeError("SHAP values not computed. Call compute_shap_values() first.")

        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving SHAP analysis to {output_dir}")

        # Save feature importance
        importance_df = self.get_feature_importance()
        importance_df.head(n_top_features).to_csv(
            output_dir / "feature_importance.csv",
            index=False,
        )
        logger.info(f"  - Saved top {n_top_features} features to feature_importance.csv")

        # Save plots
        self.plot_summary(X, max_display=20, output_path=output_dir / "shap_summary.png")
        self.plot_bar(max_display=20, output_path=output_dir / "shap_bar.png")

        # Save waterfall for first sample
        if len(self.shap_values) > 0:
            self.plot_waterfall(0, output_path=output_dir / "shap_waterfall_sample0.png")

        logger.info(f"SHAP analysis saved to {output_dir}")
