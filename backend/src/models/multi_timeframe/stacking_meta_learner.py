"""Stacking Meta-Learner for MTF Ensemble.

This module implements a stacking ensemble that learns the optimal way to combine
base model predictions instead of using fixed weighted averaging.

CRITICAL: Data leakage prevention is paramount. This implementation:
1. Uses time-series cross-validation (TimeSeriesSplit) for OOF predictions
2. Only uses features available at prediction time
3. Trains meta-model on out-of-fold predictions only
4. Never uses future data in any computation

Architecture:
    5-min Data ──┬──► 1H Model ──► prob_1h ────────┐
                 │                                  │
                 ├──► 4H Model ──► prob_4h ────────┼──► Meta-Model ──► Final Prediction
                 │                                  │   (XGBoost)
                 └──► Daily Model ──► prob_d ──────┘
                                                    │
    Additional Meta-Features:                       │
    - Model agreement score                         │
    - Confidence spread                            ─┘
    - Volatility state
"""

import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)


@dataclass
class StackingConfig:
    """Configuration for Stacking Meta-Learner."""

    # Time-series cross-validation settings
    n_folds: int = 5  # Number of CV folds for OOF predictions
    min_train_size: int = 500  # Minimum training samples per fold

    # Meta-model settings
    meta_model_type: str = "xgboost"  # "xgboost" or "logistic"
    n_estimators: int = 100
    max_depth: int = 4
    learning_rate: float = 0.05
    min_child_weight: int = 10
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_alpha: float = 0.1
    reg_lambda: float = 1.0

    # Meta-features to include
    use_agreement_features: bool = True
    use_confidence_features: bool = True
    use_volatility_features: bool = True  # Rolling volatility (no future data)

    # Blending with weighted average (0=pure stacking, 1=pure weighted avg)
    blend_with_weighted_avg: float = 0.0

    @classmethod
    def default(cls) -> "StackingConfig":
        """Default configuration."""
        return cls()

    @classmethod
    def conservative(cls) -> "StackingConfig":
        """Conservative config with blending."""
        return cls(blend_with_weighted_avg=0.3)


@dataclass
class StackingMetaFeatures:
    """Container for meta-features at a single timestep."""

    # Base model probabilities (probability of UP)
    prob_1h: float
    prob_4h: float
    prob_d: float

    # Agreement features
    agreement_ratio: float = 0.0  # Fraction of models agreeing
    direction_spread: float = 0.0  # Std of direction (0 or 1)
    confidence_spread: float = 0.0  # Std of confidences
    prob_range: float = 0.0  # Max prob - min prob

    # Volatility features (computed from past data only)
    volatility: float = 0.0  # Rolling ATR normalized
    volatility_regime: int = 0  # 0=low, 1=normal, 2=high

    def to_array(self, config: StackingConfig) -> np.ndarray:
        """Convert to feature array based on config."""
        features = [self.prob_1h, self.prob_4h, self.prob_d]

        if config.use_agreement_features:
            features.extend([
                self.agreement_ratio,
                self.direction_spread,
                self.prob_range,
            ])

        if config.use_confidence_features:
            features.extend([self.confidence_spread])

        if config.use_volatility_features:
            features.extend([
                self.volatility,
                self.volatility_regime,
            ])

        return np.array(features, dtype=np.float32)

    @staticmethod
    def get_feature_names(config: StackingConfig) -> List[str]:
        """Get feature names based on config."""
        names = ["prob_1h", "prob_4h", "prob_d"]

        if config.use_agreement_features:
            names.extend(["agreement_ratio", "direction_spread", "prob_range"])

        if config.use_confidence_features:
            names.extend(["confidence_spread"])

        if config.use_volatility_features:
            names.extend(["volatility", "volatility_regime"])

        return names


class StackingMetaLearner:
    """Stacking Meta-Learner that learns to combine base model predictions.

    This class implements a two-level stacking ensemble:
    1. Level 1: Base models (1H, 4H, Daily XGBoost models)
    2. Level 2: Meta-model that combines Level 1 predictions

    The meta-model is trained on out-of-fold (OOF) predictions to prevent
    data leakage. Time-series cross-validation ensures temporal ordering.
    """

    def __init__(self, config: Optional[StackingConfig] = None):
        """Initialize the stacking meta-learner.

        Args:
            config: Stacking configuration. Uses defaults if None.
        """
        self.config = config or StackingConfig.default()
        self.meta_model = None
        self.meta_scaler: Optional[StandardScaler] = None
        self.feature_names: List[str] = []
        self.is_trained: bool = False

        # Training metadata
        self.oof_accuracy: float = 0.0
        self.meta_train_accuracy: float = 0.0
        self.meta_val_accuracy: float = 0.0
        self.feature_importance: Dict[str, float] = {}

    def _create_meta_model(self):
        """Create the meta-model based on config."""
        if self.config.meta_model_type == "xgboost":
            return XGBClassifier(
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth,
                learning_rate=self.config.learning_rate,
                min_child_weight=self.config.min_child_weight,
                subsample=self.config.subsample,
                colsample_bytree=self.config.colsample_bytree,
                reg_alpha=self.config.reg_alpha,
                reg_lambda=self.config.reg_lambda,
                random_state=42,
                use_label_encoder=False,
                eval_metric="logloss",
                verbosity=0,
            )
        elif self.config.meta_model_type == "logistic":
            from sklearn.linear_model import LogisticRegression
            return LogisticRegression(
                C=1.0,
                max_iter=1000,
                random_state=42,
            )
        else:
            raise ValueError(f"Unknown meta_model_type: {self.config.meta_model_type}")

    def generate_oof_predictions(
        self,
        models: Dict[str, Any],  # Dict of ImprovedTimeframeModel
        X_dict: Dict[str, np.ndarray],  # Features per timeframe
        y: np.ndarray,  # Labels (use 1H labels as ground truth)
        volatility_data: Optional[np.ndarray] = None,  # Optional volatility for meta-features
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate out-of-fold predictions using time-series CV.

        CRITICAL: This method ensures no data leakage by using TimeSeriesSplit
        which respects temporal ordering. Each fold's predictions are generated
        using only data that comes before it chronologically.

        Args:
            models: Dict mapping timeframe ('1H', '4H', 'D') to trained model
            X_dict: Dict mapping timeframe to feature array (aligned indices)
            y: Ground truth labels (aligned with 1H timeframe)
            volatility_data: Optional rolling volatility for meta-features

        Returns:
            Tuple of (meta_features, oof_predictions, valid_indices)
        """
        n_samples = len(y)
        logger.info(f"Generating OOF predictions for {n_samples} samples with {self.config.n_folds} folds")

        # Initialize storage for OOF predictions
        oof_probs = {tf: np.full(n_samples, np.nan) for tf in models.keys()}
        oof_confs = {tf: np.full(n_samples, np.nan) for tf in models.keys()}
        valid_mask = np.zeros(n_samples, dtype=bool)

        # Time-series cross-validation
        tscv = TimeSeriesSplit(n_splits=self.config.n_folds)

        for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(y)):
            # Ensure minimum training size
            if len(train_idx) < self.config.min_train_size:
                logger.debug(f"Fold {fold_idx}: Skipping (train size {len(train_idx)} < {self.config.min_train_size})")
                continue

            logger.debug(f"Fold {fold_idx}: train=[{train_idx[0]}:{train_idx[-1]}], val=[{val_idx[0]}:{val_idx[-1]}]")

            # For each base model, train on fold's train set and predict on fold's val set
            for tf, model in models.items():
                X_tf = X_dict[tf]

                # Map indices to this timeframe's data
                # (assumes aligned indices; in practice, we use the same length)
                tf_train_idx = train_idx[train_idx < len(X_tf)]
                tf_val_idx = val_idx[val_idx < len(X_tf)]

                if len(tf_train_idx) < self.config.min_train_size or len(tf_val_idx) == 0:
                    continue

                X_train_fold = X_tf[tf_train_idx]
                y_train_fold = y[tf_train_idx]
                X_val_fold = X_tf[tf_val_idx]

                # Train a fresh model on this fold (to get truly OOF predictions)
                fold_model = model._create_model()
                fold_scaler = StandardScaler()

                X_train_scaled = fold_scaler.fit_transform(X_train_fold)
                X_val_scaled = fold_scaler.transform(X_val_fold)

                fold_model.fit(X_train_scaled, y_train_fold)

                # Get predictions (probability of UP)
                probs = fold_model.predict_proba(X_val_scaled)
                if probs.shape[1] == 2:
                    prob_up = probs[:, 1]  # Probability of class 1 (UP)
                else:
                    prob_up = probs[:, 0]

                conf = np.max(probs, axis=1)

                # Store OOF predictions
                for i, idx in enumerate(tf_val_idx):
                    oof_probs[tf][idx] = prob_up[i]
                    oof_confs[tf][idx] = conf[i]

            # Mark validation indices as valid
            valid_mask[val_idx] = True

        # Now create meta-features from OOF predictions
        meta_features_list = []
        valid_indices = []

        for i in range(n_samples):
            if not valid_mask[i]:
                continue

            # Check if we have predictions from all models
            if any(np.isnan(oof_probs[tf][i]) for tf in models.keys()):
                continue

            # Create meta-features
            prob_1h = oof_probs["1H"][i]
            prob_4h = oof_probs["4H"][i]
            prob_d = oof_probs["D"][i]

            probs = [prob_1h, prob_4h, prob_d]
            dirs = [1 if p > 0.5 else 0 for p in probs]
            confs = [oof_confs["1H"][i], oof_confs["4H"][i], oof_confs["D"][i]]

            # Majority direction
            majority_dir = 1 if sum(dirs) >= 2 else 0
            agreement_count = sum(1 for d in dirs if d == majority_dir)

            meta_feat = StackingMetaFeatures(
                prob_1h=prob_1h,
                prob_4h=prob_4h,
                prob_d=prob_d,
                agreement_ratio=agreement_count / 3.0,
                direction_spread=np.std(dirs),
                confidence_spread=np.std(confs),
                prob_range=max(probs) - min(probs),
                volatility=volatility_data[i] if volatility_data is not None else 0.0,
                volatility_regime=self._get_volatility_regime(volatility_data[i]) if volatility_data is not None else 1,
            )

            meta_features_list.append(meta_feat.to_array(self.config))
            valid_indices.append(i)

        if len(meta_features_list) == 0:
            raise ValueError("No valid OOF predictions generated. Check fold sizes and data alignment.")

        meta_features = np.array(meta_features_list)
        valid_indices = np.array(valid_indices)
        oof_labels = y[valid_indices]

        logger.info(f"Generated {len(valid_indices)} OOF predictions ({len(valid_indices)/n_samples*100:.1f}% of data)")

        return meta_features, oof_labels, valid_indices

    def _get_volatility_regime(self, volatility: float) -> int:
        """Classify volatility into regime (0=low, 1=normal, 2=high).

        Uses fixed thresholds for consistency.
        """
        if volatility < 0.3:
            return 0  # Low volatility
        elif volatility > 0.7:
            return 2  # High volatility
        else:
            return 1  # Normal

    def train(
        self,
        meta_features: np.ndarray,
        labels: np.ndarray,
        val_ratio: float = 0.2,
    ) -> Dict[str, float]:
        """Train the meta-model on OOF predictions.

        Args:
            meta_features: Array of meta-features from OOF predictions
            labels: Ground truth labels
            val_ratio: Fraction for validation (from end, temporal order)

        Returns:
            Dict of training metrics
        """
        self.feature_names = StackingMetaFeatures.get_feature_names(self.config)

        # Temporal split (no shuffling!)
        n_train = int(len(meta_features) * (1 - val_ratio))
        X_train = meta_features[:n_train]
        y_train = labels[:n_train]
        X_val = meta_features[n_train:]
        y_val = labels[n_train:]

        logger.info(f"Training meta-model: {len(X_train)} train, {len(X_val)} val samples")

        # Scale features
        self.meta_scaler = StandardScaler()
        X_train_scaled = self.meta_scaler.fit_transform(X_train)
        X_val_scaled = self.meta_scaler.transform(X_val)

        # Train meta-model
        self.meta_model = self._create_meta_model()

        if self.config.meta_model_type == "xgboost":
            self.meta_model.fit(
                X_train_scaled, y_train,
                eval_set=[(X_val_scaled, y_val)],
                verbose=False,
            )
        else:
            self.meta_model.fit(X_train_scaled, y_train)

        # Calculate metrics
        train_pred = self.meta_model.predict(X_train_scaled)
        val_pred = self.meta_model.predict(X_val_scaled)

        self.meta_train_accuracy = (train_pred == y_train).mean()
        self.meta_val_accuracy = (val_pred == y_val).mean()
        self.is_trained = True

        # Feature importance
        if hasattr(self.meta_model, "feature_importances_"):
            importance = self.meta_model.feature_importances_
            self.feature_importance = dict(zip(self.feature_names, importance))

            logger.info("Meta-model feature importance:")
            for feat, imp in sorted(self.feature_importance.items(), key=lambda x: -x[1]):
                logger.info(f"  {feat}: {imp:.4f}")

        # Calculate accuracy at different confidence levels
        val_probs = self.meta_model.predict_proba(X_val_scaled)
        val_conf = np.max(val_probs, axis=1)

        results = {
            "meta_train_accuracy": self.meta_train_accuracy,
            "meta_val_accuracy": self.meta_val_accuracy,
            "train_samples": len(X_train),
            "val_samples": len(X_val),
        }

        for thresh in [0.55, 0.60, 0.65, 0.70]:
            mask = val_conf >= thresh
            if mask.sum() > 0:
                acc = (val_pred[mask] == y_val[mask]).mean()
                results[f"meta_val_acc_conf_{int(thresh*100)}"] = acc
                results[f"meta_val_samples_conf_{int(thresh*100)}"] = int(mask.sum())

        logger.info(
            f"Meta-model trained: train_acc={self.meta_train_accuracy:.2%}, "
            f"val_acc={self.meta_val_accuracy:.2%}"
        )

        return results

    def predict(
        self,
        prob_1h: float,
        prob_4h: float,
        prob_d: float,
        conf_1h: float,
        conf_4h: float,
        conf_d: float,
        volatility: float = 0.0,
        weighted_avg_prob: Optional[float] = None,
        weights: Optional[Dict[str, float]] = None,
    ) -> Tuple[int, float, float, float]:
        """Make prediction using the meta-model.

        Args:
            prob_1h: Probability of UP from 1H model
            prob_4h: Probability of UP from 4H model
            prob_d: Probability of UP from Daily model
            conf_1h: Confidence from 1H model
            conf_4h: Confidence from 4H model
            conf_d: Confidence from Daily model
            volatility: Current volatility (normalized, rolling)
            weighted_avg_prob: Optional weighted average prob for blending
            weights: Base model weights (for calculating weighted avg if not provided)

        Returns:
            Tuple of (direction, confidence, prob_up, prob_down)
        """
        if not self.is_trained:
            raise RuntimeError("Meta-model not trained. Call train() first.")

        probs = [prob_1h, prob_4h, prob_d]
        dirs = [1 if p > 0.5 else 0 for p in probs]
        confs = [conf_1h, conf_4h, conf_d]

        # Majority direction for agreement calculation
        majority_dir = 1 if sum(dirs) >= 2 else 0
        agreement_count = sum(1 for d in dirs if d == majority_dir)

        meta_feat = StackingMetaFeatures(
            prob_1h=prob_1h,
            prob_4h=prob_4h,
            prob_d=prob_d,
            agreement_ratio=agreement_count / 3.0,
            direction_spread=np.std(dirs),
            confidence_spread=np.std(confs),
            prob_range=max(probs) - min(probs),
            volatility=volatility,
            volatility_regime=self._get_volatility_regime(volatility),
        )

        X = meta_feat.to_array(self.config).reshape(1, -1)
        X_scaled = self.meta_scaler.transform(X)

        # Get meta-model prediction
        meta_probs = self.meta_model.predict_proba(X_scaled)[0]
        if len(meta_probs) == 2:
            meta_prob_up = meta_probs[1]
        else:
            meta_prob_up = meta_probs[0]

        meta_confidence = max(meta_probs)

        # Optionally blend with weighted average
        if self.config.blend_with_weighted_avg > 0:
            if weighted_avg_prob is None and weights is not None:
                # Calculate weighted average
                w_1h = weights.get("1H", 0.6)
                w_4h = weights.get("4H", 0.3)
                w_d = weights.get("D", 0.1)
                w_total = w_1h + w_4h + w_d
                weighted_avg_prob = (w_1h * prob_1h + w_4h * prob_4h + w_d * prob_d) / w_total

            if weighted_avg_prob is not None:
                blend = self.config.blend_with_weighted_avg
                meta_prob_up = (1 - blend) * meta_prob_up + blend * weighted_avg_prob
                # Recalculate confidence after blending
                meta_confidence = abs(meta_prob_up - 0.5) * 2 + 0.5

        direction = 1 if meta_prob_up > 0.5 else 0
        prob_down = 1 - meta_prob_up

        return direction, meta_confidence, meta_prob_up, prob_down

    def predict_batch(
        self,
        probs_1h: np.ndarray,
        probs_4h: np.ndarray,
        probs_d: np.ndarray,
        confs_1h: np.ndarray,
        confs_4h: np.ndarray,
        confs_d: np.ndarray,
        volatility: Optional[np.ndarray] = None,
        weights: Optional[Dict[str, float]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Batch prediction using the meta-model.

        Args:
            probs_1h: Array of UP probabilities from 1H model
            probs_4h: Array of UP probabilities from 4H model
            probs_d: Array of UP probabilities from Daily model
            confs_1h: Array of confidences from 1H model
            confs_4h: Array of confidences from 4H model
            confs_d: Array of confidences from Daily model
            volatility: Array of volatility values
            weights: Base model weights for blending

        Returns:
            Tuple of (directions, confidences, agreement_scores)
        """
        if not self.is_trained:
            raise RuntimeError("Meta-model not trained. Call train() first.")

        n = len(probs_1h)
        if volatility is None:
            volatility = np.zeros(n)

        # Build meta-features array
        meta_features = []
        for i in range(n):
            probs = [probs_1h[i], probs_4h[i], probs_d[i]]
            dirs = [1 if p > 0.5 else 0 for p in probs]
            confs = [confs_1h[i], confs_4h[i], confs_d[i]]

            majority_dir = 1 if sum(dirs) >= 2 else 0
            agreement_count = sum(1 for d in dirs if d == majority_dir)

            meta_feat = StackingMetaFeatures(
                prob_1h=probs_1h[i],
                prob_4h=probs_4h[i],
                prob_d=probs_d[i],
                agreement_ratio=agreement_count / 3.0,
                direction_spread=np.std(dirs),
                confidence_spread=np.std(confs),
                prob_range=max(probs) - min(probs),
                volatility=volatility[i],
                volatility_regime=self._get_volatility_regime(volatility[i]),
            )
            meta_features.append(meta_feat.to_array(self.config))

        X = np.array(meta_features)
        X_scaled = self.meta_scaler.transform(X)

        # Get predictions
        meta_probs_all = self.meta_model.predict_proba(X_scaled)
        if meta_probs_all.shape[1] == 2:
            meta_probs_up = meta_probs_all[:, 1]
        else:
            meta_probs_up = meta_probs_all[:, 0]

        meta_confidences = np.max(meta_probs_all, axis=1)

        # Optionally blend with weighted average
        if self.config.blend_with_weighted_avg > 0 and weights is not None:
            w_1h = weights.get("1H", 0.6)
            w_4h = weights.get("4H", 0.3)
            w_d = weights.get("D", 0.1)
            w_total = w_1h + w_4h + w_d
            weighted_avg_probs = (w_1h * probs_1h + w_4h * probs_4h + w_d * probs_d) / w_total

            blend = self.config.blend_with_weighted_avg
            meta_probs_up = (1 - blend) * meta_probs_up + blend * weighted_avg_probs
            meta_confidences = np.abs(meta_probs_up - 0.5) * 2 + 0.5

        directions = (meta_probs_up > 0.5).astype(int)

        # Calculate agreement scores
        agreement_scores = np.zeros(n)
        for i in range(n):
            dirs = [1 if probs_1h[i] > 0.5 else 0,
                    1 if probs_4h[i] > 0.5 else 0,
                    1 if probs_d[i] > 0.5 else 0]
            agreement_count = sum(1 for d in dirs if d == directions[i])
            agreement_scores[i] = agreement_count / 3.0

        return directions, meta_confidences, agreement_scores

    def save(self, path: Path) -> None:
        """Save meta-learner to disk."""
        data = {
            "config": self.config,
            "meta_model": self.meta_model,
            "meta_scaler": self.meta_scaler,
            "feature_names": self.feature_names,
            "meta_train_accuracy": self.meta_train_accuracy,
            "meta_val_accuracy": self.meta_val_accuracy,
            "feature_importance": self.feature_importance,
            "is_trained": self.is_trained,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)
        logger.info(f"Saved meta-learner to {path}")

    def load(self, path: Path) -> None:
        """Load meta-learner from disk."""
        with open(path, "rb") as f:
            data = pickle.load(f)

        self.config = data["config"]
        self.meta_model = data["meta_model"]
        self.meta_scaler = data["meta_scaler"]
        self.feature_names = data["feature_names"]
        self.meta_train_accuracy = data["meta_train_accuracy"]
        self.meta_val_accuracy = data["meta_val_accuracy"]
        self.feature_importance = data.get("feature_importance", {})
        self.is_trained = data["is_trained"]
        logger.info(f"Loaded meta-learner from {path}")

    def summary(self) -> str:
        """Return summary of meta-learner."""
        lines = [
            "=" * 50,
            "STACKING META-LEARNER SUMMARY",
            "=" * 50,
            "",
            "Configuration:",
            f"  N Folds: {self.config.n_folds}",
            f"  Meta Model: {self.config.meta_model_type}",
            f"  Blend Factor: {self.config.blend_with_weighted_avg}",
            f"  Agreement Features: {self.config.use_agreement_features}",
            f"  Confidence Features: {self.config.use_confidence_features}",
            f"  Volatility Features: {self.config.use_volatility_features}",
            "",
        ]

        if self.is_trained:
            lines.extend([
                "Training Results:",
                f"  Train Accuracy: {self.meta_train_accuracy:.2%}",
                f"  Val Accuracy: {self.meta_val_accuracy:.2%}",
                "",
                "Feature Importance:",
            ])
            for feat, imp in sorted(self.feature_importance.items(), key=lambda x: -x[1]):
                lines.append(f"  {feat}: {imp:.4f}")
        else:
            lines.append("Status: Not trained")

        lines.append("=" * 50)
        return "\n".join(lines)
