"""Hybrid ensemble combining XGBoost and Sequence models.

This ensemble leverages:
- XGBoost: Feature-based patterns, indicator relationships
- Sequence Model: Temporal patterns from raw price action

The combination captures complementary information for better predictions.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .improved_model import ImprovedTimeframeModel, ImprovedModelConfig
from .sequence_model import (
    CNNTransformerModel,
    SequenceModelConfig,
    SequenceDataset,
    SequenceTrainer,
    SequencePredictor,
)

logger = logging.getLogger(__name__)


@dataclass
class HybridConfig:
    """Configuration for hybrid ensemble."""

    name: str = "1H"
    base_timeframe: str = "1H"

    # Component weights (must sum to 1.0)
    xgboost_weight: float = 0.6
    sequence_weight: float = 0.4

    # Ensemble method: "weighted_avg", "confidence_weighted", "meta_learner"
    ensemble_method: str = "confidence_weighted"

    # Confidence threshold for trading
    min_confidence: float = 0.55

    # Component configs
    xgboost_config: ImprovedModelConfig = None
    sequence_config: SequenceModelConfig = None

    def __post_init__(self):
        if self.xgboost_config is None:
            if self.base_timeframe == "1H":
                self.xgboost_config = ImprovedModelConfig.hourly_model()
            elif self.base_timeframe == "4H":
                self.xgboost_config = ImprovedModelConfig.four_hour_model()

        if self.sequence_config is None:
            if self.base_timeframe == "1H":
                self.sequence_config = SequenceModelConfig.hourly_model()
            elif self.base_timeframe == "4H":
                self.sequence_config = SequenceModelConfig.four_hour_model()

    @classmethod
    def hourly(cls) -> "HybridConfig":
        return cls(
            name="1H",
            base_timeframe="1H",
            xgboost_weight=0.6,
            sequence_weight=0.4,
        )

    @classmethod
    def four_hour(cls) -> "HybridConfig":
        return cls(
            name="4H",
            base_timeframe="4H",
            xgboost_weight=0.6,
            sequence_weight=0.4,
        )


class HybridEnsemble:
    """Ensemble combining XGBoost and Sequence models."""

    def __init__(
        self,
        config: HybridConfig,
        model_dir: Optional[Path] = None,
    ):
        self.config = config
        self.model_dir = model_dir or Path("models/hybrid")

        # Component models
        self.xgboost_model: Optional[ImprovedTimeframeModel] = None
        self.sequence_model: Optional[SequencePredictor] = None

        # Training state
        self.is_trained = False

        # Metrics
        self.train_metrics: Dict = {}
        self.val_metrics: Dict = {}

    def train(
        self,
        df_5min: pd.DataFrame,
        train_ratio: float = 0.6,
        val_ratio: float = 0.2,
        higher_tf_data: Optional[Dict[str, pd.DataFrame]] = None,
    ) -> Dict[str, float]:
        """Train both component models.

        Args:
            df_5min: 5-minute OHLCV data
            train_ratio: Proportion for training
            val_ratio: Proportion for validation
            higher_tf_data: Higher timeframe data for XGBoost cross-TF features

        Returns:
            Combined training metrics
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Training Hybrid Ensemble for {self.config.name}")
        logger.info(f"{'='*60}")

        # Resample to target timeframe
        df_tf = self._resample(df_5min, self.config.base_timeframe)
        logger.info(f"Resampled to {self.config.base_timeframe}: {len(df_tf)} bars")

        # Calculate split indices
        n_total = len(df_tf)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)

        # ==================== Train XGBoost ====================
        logger.info(f"\n{'='*40}")
        logger.info("Training XGBoost Model")
        logger.info(f"{'='*40}")

        self.xgboost_model = ImprovedTimeframeModel(self.config.xgboost_config)

        # Prepare XGBoost data
        X_xgb, y_xgb, feature_cols = self.xgboost_model.prepare_data(df_tf, higher_tf_data)

        # Split for XGBoost
        X_train_xgb = X_xgb[:n_train]
        y_train_xgb = y_xgb[:n_train]
        X_val_xgb = X_xgb[n_train:n_train + n_val]
        y_val_xgb = y_xgb[n_train:n_train + n_val]

        xgb_metrics = self.xgboost_model.train(
            X_train_xgb, y_train_xgb,
            X_val_xgb, y_val_xgb,
            feature_cols,
        )

        # ==================== Train Sequence Model ====================
        logger.info(f"\n{'='*40}")
        logger.info("Training Sequence Model")
        logger.info(f"{'='*40}")

        # Prepare sequence data
        seq_dataset = SequenceDataset(self.config.sequence_config)
        X_seq, y_seq = seq_dataset.prepare(df_tf)

        # Calculate split for sequence data (accounting for sequence length offset)
        seq_len = self.config.sequence_config.sequence_length
        seq_train_end = n_train - seq_len
        seq_val_end = n_train + n_val - seq_len

        X_train_seq = X_seq[:seq_train_end]
        y_train_seq = y_seq[:seq_train_end]
        X_val_seq = X_seq[seq_train_end:seq_val_end]
        y_val_seq = y_seq[seq_train_end:seq_val_end]

        logger.info(f"Sequence train: {len(X_train_seq)}, val: {len(X_val_seq)}")

        # Train sequence model
        trainer = SequenceTrainer(self.config.sequence_config)
        seq_model, seq_metrics = trainer.train(
            X_train_seq, y_train_seq,
            X_val_seq, y_val_seq,
        )

        self.sequence_model = SequencePredictor(
            seq_model,
            self.config.sequence_config,
        )

        # ==================== Evaluate Ensemble ====================
        logger.info(f"\n{'='*40}")
        logger.info("Evaluating Hybrid Ensemble")
        logger.info(f"{'='*40}")

        ensemble_metrics = self._evaluate_ensemble(
            df_tf, X_val_xgb, y_val_xgb, X_val_seq, y_val_seq
        )

        # Combine all metrics
        self.train_metrics = {
            'xgboost': xgb_metrics,
            'sequence': seq_metrics,
            'ensemble': ensemble_metrics,
        }

        self.is_trained = True

        # Summary
        logger.info(f"\n{'='*60}")
        logger.info("TRAINING SUMMARY")
        logger.info(f"{'='*60}")
        logger.info(f"XGBoost val accuracy:  {xgb_metrics['val_accuracy']:.2%}")
        logger.info(f"Sequence val accuracy: {seq_metrics['val_accuracy']:.2%}")
        logger.info(f"Ensemble val accuracy: {ensemble_metrics['val_accuracy']:.2%}")

        return {
            'xgboost_val_acc': xgb_metrics['val_accuracy'],
            'sequence_val_acc': seq_metrics['val_accuracy'],
            'ensemble_val_acc': ensemble_metrics['val_accuracy'],
            **{f'ensemble_{k}': v for k, v in ensemble_metrics.items()},
        }

    def _evaluate_ensemble(
        self,
        df_tf: pd.DataFrame,
        X_val_xgb: np.ndarray,
        y_val_xgb: np.ndarray,
        X_val_seq: np.ndarray,
        y_val_seq: np.ndarray,
    ) -> Dict[str, float]:
        """Evaluate ensemble performance on validation data."""

        # Get XGBoost predictions
        xgb_preds, xgb_confs = self.xgboost_model.predict_batch(X_val_xgb)

        # Get sequence predictions
        # Note: sequence data is offset by seq_length, so we need to align
        seq_preds, seq_confs = self.sequence_model.predict_batch(X_val_seq)

        # Align predictions (use the intersection)
        n_common = min(len(xgb_preds), len(seq_preds))
        xgb_preds = xgb_preds[:n_common]
        xgb_confs = xgb_confs[:n_common]
        seq_preds = seq_preds[:n_common]
        seq_confs = seq_confs[:n_common]
        y_true = y_val_xgb[:n_common]

        # Ensemble predictions based on method
        if self.config.ensemble_method == "weighted_avg":
            # Simple weighted average of probabilities
            ensemble_prob_up = (
                self.config.xgboost_weight * (xgb_preds * xgb_confs + (1 - xgb_preds) * (1 - xgb_confs)) +
                self.config.sequence_weight * (seq_preds * seq_confs + (1 - seq_preds) * (1 - seq_confs))
            )
            ensemble_preds = (ensemble_prob_up > 0.5).astype(int)
            ensemble_confs = np.abs(ensemble_prob_up - 0.5) * 2 + 0.5

        elif self.config.ensemble_method == "confidence_weighted":
            # Weight by confidence (higher confidence model gets more weight)
            total_conf = xgb_confs + seq_confs
            xgb_weight = xgb_confs / total_conf
            seq_weight = seq_confs / total_conf

            # Convert predictions to probabilities
            xgb_prob_up = xgb_preds * xgb_confs + (1 - xgb_preds) * (1 - xgb_confs)
            seq_prob_up = seq_preds * seq_confs + (1 - seq_preds) * (1 - seq_confs)

            ensemble_prob_up = xgb_weight * xgb_prob_up + seq_weight * seq_prob_up
            ensemble_preds = (ensemble_prob_up > 0.5).astype(int)
            ensemble_confs = np.abs(ensemble_prob_up - 0.5) * 2 + 0.5

        else:
            # Default: majority voting with agreement bonus
            ensemble_preds = ((xgb_preds + seq_preds) >= 1).astype(int)
            agreement = (xgb_preds == seq_preds).astype(float)
            ensemble_confs = (xgb_confs + seq_confs) / 2 + agreement * 0.1

        # Calculate metrics
        val_acc = (ensemble_preds == y_true).mean()

        metrics = {
            'val_accuracy': val_acc,
            'val_samples': len(y_true),
            'agreement_rate': (xgb_preds == seq_preds).mean(),
        }

        # Accuracy at different confidence levels
        for thresh in [0.55, 0.60, 0.65, 0.70]:
            mask = ensemble_confs >= thresh
            if mask.sum() > 0:
                acc = (ensemble_preds[mask] == y_true[mask]).mean()
                metrics[f'val_acc_conf_{int(thresh*100)}'] = acc
                metrics[f'val_samples_conf_{int(thresh*100)}'] = int(mask.sum())

        # Log component agreement
        logger.info(f"Model agreement rate: {metrics['agreement_rate']:.2%}")
        logger.info(f"Ensemble accuracy: {val_acc:.2%}")

        return metrics

    def predict(
        self,
        df_tf: pd.DataFrame,
        X_xgb: np.ndarray,
        sequence: np.ndarray,
    ) -> Tuple[int, float, float, float]:
        """Make ensemble prediction.

        Args:
            df_tf: Timeframe OHLCV for context
            X_xgb: Feature vector for XGBoost (1, n_features)
            sequence: Normalized sequence for CNN-Transformer (seq_len, 5)

        Returns:
            (direction, confidence, prob_up, prob_down)
        """
        if not self.is_trained:
            raise RuntimeError("Ensemble not trained")

        # XGBoost prediction
        xgb_pred, xgb_conf, xgb_prob_up, xgb_prob_down = self.xgboost_model.predict(X_xgb)

        # Sequence prediction
        seq_pred, seq_conf, seq_prob_up, seq_prob_down = self.sequence_model.predict(sequence)

        # Ensemble based on method
        if self.config.ensemble_method == "confidence_weighted":
            total_conf = xgb_conf + seq_conf
            xgb_w = xgb_conf / total_conf
            seq_w = seq_conf / total_conf

            prob_up = xgb_w * xgb_prob_up + seq_w * seq_prob_up
            prob_down = xgb_w * xgb_prob_down + seq_w * seq_prob_down
        else:
            prob_up = self.config.xgboost_weight * xgb_prob_up + self.config.sequence_weight * seq_prob_up
            prob_down = self.config.xgboost_weight * xgb_prob_down + self.config.sequence_weight * seq_prob_down

        direction = 1 if prob_up > prob_down else 0
        confidence = max(prob_up, prob_down)

        # Boost confidence if models agree
        if xgb_pred == seq_pred:
            confidence = min(confidence + 0.05, 1.0)

        return direction, confidence, prob_up, prob_down

    def predict_batch(
        self,
        X_xgb: np.ndarray,
        X_seq: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Batch prediction.

        Returns:
            (predictions, confidences)
        """
        # Get individual predictions
        xgb_preds, xgb_confs = self.xgboost_model.predict_batch(X_xgb)
        seq_preds, seq_confs = self.sequence_model.predict_batch(X_seq)

        # Align lengths
        n = min(len(xgb_preds), len(seq_preds))
        xgb_preds, xgb_confs = xgb_preds[:n], xgb_confs[:n]
        seq_preds, seq_confs = seq_preds[:n], seq_confs[:n]

        # Ensemble
        if self.config.ensemble_method == "confidence_weighted":
            total_conf = xgb_confs + seq_confs
            xgb_w = xgb_confs / total_conf
            seq_w = seq_confs / total_conf

            xgb_prob = xgb_preds * xgb_confs + (1 - xgb_preds) * (1 - xgb_confs)
            seq_prob = seq_preds * seq_confs + (1 - seq_preds) * (1 - seq_confs)

            ensemble_prob = xgb_w * xgb_prob + seq_w * seq_prob
            predictions = (ensemble_prob > 0.5).astype(int)
            confidences = np.abs(ensemble_prob - 0.5) * 2 + 0.5
        else:
            predictions = ((xgb_preds + seq_preds) >= 1).astype(int)
            confidences = (xgb_confs + seq_confs) / 2

        # Boost confidence when models agree
        agreement = (xgb_preds == seq_preds).astype(float)
        confidences = np.minimum(confidences + agreement * 0.05, 1.0)

        return predictions, confidences

    def save(self) -> None:
        """Save ensemble models."""
        self.model_dir.mkdir(parents=True, exist_ok=True)

        # Save XGBoost
        xgb_path = self.model_dir / f"{self.config.name}_xgboost.pkl"
        self.xgboost_model.save(xgb_path)

        # Save sequence model
        seq_path = self.model_dir / f"{self.config.name}_sequence.pt"
        self.sequence_model.save(seq_path)

        # Save config
        import json
        config_path = self.model_dir / f"{self.config.name}_config.json"
        with open(config_path, 'w') as f:
            json.dump({
                'name': self.config.name,
                'base_timeframe': self.config.base_timeframe,
                'xgboost_weight': self.config.xgboost_weight,
                'sequence_weight': self.config.sequence_weight,
                'ensemble_method': self.config.ensemble_method,
                'min_confidence': self.config.min_confidence,
            }, f, indent=2)

        logger.info(f"Saved hybrid ensemble to {self.model_dir}")

    def load(self) -> None:
        """Load ensemble models."""
        import json

        # Load config
        config_path = self.model_dir / f"{self.config.name}_config.json"
        if config_path.exists():
            with open(config_path) as f:
                cfg = json.load(f)
                self.config.xgboost_weight = cfg.get('xgboost_weight', 0.6)
                self.config.sequence_weight = cfg.get('sequence_weight', 0.4)
                self.config.ensemble_method = cfg.get('ensemble_method', 'confidence_weighted')

        # Load XGBoost
        xgb_path = self.model_dir / f"{self.config.name}_xgboost.pkl"
        self.xgboost_model = ImprovedTimeframeModel(self.config.xgboost_config)
        self.xgboost_model.load(xgb_path)

        # Load sequence model
        seq_path = self.model_dir / f"{self.config.name}_sequence.pt"
        self.sequence_model = SequencePredictor.load(seq_path)

        self.is_trained = True
        logger.info(f"Loaded hybrid ensemble from {self.model_dir}")

    def _resample(self, df_5min: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """Resample 5-minute data to target timeframe."""
        if timeframe == "5min":
            return df_5min.copy()

        resampled = df_5min.resample(timeframe).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum' if 'volume' in df_5min.columns else 'first',
        }).dropna()

        return resampled
