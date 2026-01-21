"""Improved multi-timeframe model with better labeling and features.

This model addresses the key issues identified:
1. Uses triple barrier labeling instead of next-bar prediction
2. Longer prediction horizons (1-4 hours instead of 5 minutes)
3. Enhanced features including time-of-day, ROC, and patterns
4. Cross-timeframe alignment features
"""

import logging
import pickle
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from .labeling import AdvancedLabeler, LabelingConfig, LabelMethod
from .enhanced_features import EnhancedFeatureEngine

if TYPE_CHECKING:
    from ..feature_selection import RFECVConfig

logger = logging.getLogger(__name__)


@dataclass
class ImprovedModelConfig:
    """Configuration for improved model."""

    name: str
    base_timeframe: str  # "1H", "4H", "D"

    # Labeling configuration
    label_method: LabelMethod = LabelMethod.TRIPLE_BARRIER
    tp_pips: float = 20.0
    sl_pips: float = 10.0
    max_holding_bars: int = 24  # ~24 hours for 1H

    # Model type
    model_type: str = "xgboost"  # "xgboost", "gbm", "rf"

    # XGBoost parameters
    n_estimators: int = 200
    max_depth: int = 6
    learning_rate: float = 0.05
    min_child_weight: int = 10
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_alpha: float = 0.1
    reg_lambda: float = 1.0

    # Feature settings
    include_time_features: bool = True
    include_roc_features: bool = True
    include_normalized_features: bool = True
    include_pattern_features: bool = True
    include_sentiment_features: bool = False
    us_only_sentiment: bool = True  # Use US-only sentiment (EPU + VIX) - recommended
    trading_pair: str = "EURUSD"
    sentiment_source: str = "epu"  # 'epu' (daily VIX/EPU), 'gdelt' (hourly), or 'both'

    # RFECV feature selection
    use_rfecv: bool = False  # Enable RFECV feature selection
    rfecv_config: Optional["RFECVConfig"] = None  # RFECV configuration (uses defaults if None)

    @classmethod
    def hourly_model(cls) -> "ImprovedModelConfig":
        """1-hour timeframe model for intraday trading."""
        return cls(
            name="1H",
            base_timeframe="1H",
            tp_pips=25.0,
            sl_pips=15.0,
            max_holding_bars=12,  # 12 hours max
            n_estimators=200,
            max_depth=6,
        )

    @classmethod
    def four_hour_model(cls) -> "ImprovedModelConfig":
        """4-hour timeframe model for swing trading."""
        return cls(
            name="4H",
            base_timeframe="4H",
            tp_pips=50.0,   # Original value - stable
            sl_pips=25.0,   # Original value - stable
            max_holding_bars=18,  # Original value - 3 days max
            n_estimators=150,
            max_depth=5,
        )

    @classmethod
    def daily_model(cls) -> "ImprovedModelConfig":
        """Daily timeframe model for position trading."""
        return cls(
            name="D",
            base_timeframe="D",
            tp_pips=150.0,  # Increased from 100 - wider target for position trades
            sl_pips=75.0,   # Increased from 50 - more room for daily volatility
            max_holding_bars=15,  # Increased from 10 - 3 weeks max
            n_estimators=100,
            max_depth=4,
        )


class ImprovedTimeframeModel:
    """Improved model for a single timeframe."""

    def __init__(self, config: ImprovedModelConfig):
        self.config = config
        self.model = None
        self.scaler: Optional[StandardScaler] = None
        self.feature_names: List[str] = []
        self.is_trained: bool = False

        # RFECV feature selection
        self.selected_features: Optional[List[str]] = None
        self.selected_indices: Optional[np.ndarray] = None
        self.rfecv_scores: Optional[Dict] = None

        # Labeler
        label_config = LabelingConfig(
            method=config.label_method,
            tp_pips=config.tp_pips,
            sl_pips=config.sl_pips,
            max_holding_bars=config.max_holding_bars,
        )
        self.labeler = AdvancedLabeler(label_config)

        # Feature engine
        self.feature_engine = EnhancedFeatureEngine(
            base_timeframe=config.base_timeframe,
            include_time_features=config.include_time_features,
            include_roc_features=config.include_roc_features,
            include_normalized_features=config.include_normalized_features,
            include_pattern_features=config.include_pattern_features,
            include_sentiment_features=config.include_sentiment_features,
            trading_pair=config.trading_pair,
            us_only_sentiment=config.us_only_sentiment,
            sentiment_source=config.sentiment_source,
        )

        # Metrics
        self.train_accuracy: float = 0.0
        self.val_accuracy: float = 0.0
        self.feature_importance: Dict[str, float] = {}

    def _create_model(self):
        """Create the ML model based on config."""
        if self.config.model_type == "xgboost":
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
        elif self.config.model_type == "gbm":
            return GradientBoostingClassifier(
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth,
                learning_rate=self.config.learning_rate,
                min_samples_split=20,
                min_samples_leaf=10,
                subsample=self.config.subsample,
                random_state=42,
            )
        elif self.config.model_type == "rf":
            return RandomForestClassifier(
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth,
                min_samples_split=20,
                min_samples_leaf=10,
                random_state=42,
                n_jobs=-1,
            )
        else:
            raise ValueError(f"Unknown model type: {self.config.model_type}")

    def prepare_data(
        self,
        df: pd.DataFrame,
        higher_tf_data: Optional[Dict[str, pd.DataFrame]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Prepare features and labels from raw OHLCV data.

        Args:
            df: OHLCV DataFrame at model's base timeframe
            higher_tf_data: Optional higher timeframe data for cross-TF features

        Returns:
            Tuple of (X, y, feature_names)
        """
        # Calculate base technical indicators
        from src.features.technical.calculator import TechnicalIndicatorCalculator
        calc = TechnicalIndicatorCalculator(model_type="short_term")
        df_features = calc.calculate(df)

        # Add enhanced features
        df_features = self.feature_engine.add_all_features(df_features, higher_tf_data)

        # Create labels
        labels, valid_mask = self.labeler.create_labels(df_features)

        # Get feature columns (exclude OHLCV and labels)
        exclude_cols = {"open", "high", "low", "close", "volume", "label"}
        feature_cols = [c for c in df_features.columns if c not in exclude_cols]

        # Prepare arrays
        X = df_features[feature_cols].values
        y = labels.values

        # Remove rows with NaN
        valid = ~np.isnan(X).any(axis=1) & valid_mask.values & ~np.isnan(y)
        X = X[valid]
        y = y[valid]

        logger.info(f"Prepared {len(X)} samples with {len(feature_cols)} features")
        logger.info(f"Label distribution: {y.mean():.1%} bullish, {1-y.mean():.1%} bearish")

        return X, y, feature_cols

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        feature_names: List[str],
    ) -> Dict[str, float]:
        """Train the model.

        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
            feature_names: List of feature names

        Returns:
            Dict of training metrics
        """
        self.feature_names = feature_names

        # RFECV feature selection if enabled
        if self.config.use_rfecv:
            from ..feature_selection import FeatureSelectionManager

            logger.info(f"Running RFECV for {self.config.name} model...")
            manager = FeatureSelectionManager(self.config.rfecv_config)

            # Run RFECV on training data
            self.selected_features, self.selected_indices, self.rfecv_scores = (
                manager.select_features(
                    timeframe=self.config.name,
                    X=X_train,
                    y=y_train,
                    feature_names=feature_names,
                )
            )

            # Filter features
            X_train = X_train[:, self.selected_indices]
            X_val = X_val[:, self.selected_indices]

            logger.info(
                f"RFECV selected {len(self.selected_features)} / {len(feature_names)} features"
            )
        else:
            # Use all features
            self.selected_features = feature_names
            self.selected_indices = np.arange(len(feature_names))

        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)

        # Create and train model
        self.model = self._create_model()

        logger.info(f"Training {self.config.name} model with {len(X_train)} samples...")

        if self.config.model_type == "xgboost":
            self.model.fit(
                X_train_scaled, y_train,
                eval_set=[(X_val_scaled, y_val)],
                verbose=False,
            )
        else:
            self.model.fit(X_train_scaled, y_train)

        # Calculate accuracies
        train_pred = self.model.predict(X_train_scaled)
        val_pred = self.model.predict(X_val_scaled)

        self.train_accuracy = (train_pred == y_train).mean()
        self.val_accuracy = (val_pred == y_val).mean()
        self.is_trained = True

        # Calculate accuracy at different confidence levels
        val_probs = self.model.predict_proba(X_val_scaled)
        val_conf = np.max(val_probs, axis=1)

        results = {
            "train_accuracy": self.train_accuracy,
            "val_accuracy": self.val_accuracy,
            "train_samples": len(X_train),
            "val_samples": len(X_val),
        }

        for thresh in [0.55, 0.60, 0.65, 0.70]:
            mask = val_conf >= thresh
            if mask.sum() > 0:
                acc = (val_pred[mask] == y_val[mask]).mean()
                results[f"val_acc_conf_{int(thresh*100)}"] = acc
                results[f"val_samples_conf_{int(thresh*100)}"] = int(mask.sum())

        # Feature importance (using selected features)
        if hasattr(self.model, "feature_importances_"):
            importance = self.model.feature_importances_
            self.feature_importance = dict(zip(self.selected_features, importance))

            # Log top features
            top_features = sorted(
                self.feature_importance.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]
            logger.info("Top 10 features:")
            for feat, imp in top_features:
                logger.info(f"  {feat}: {imp:.4f}")

        logger.info(
            f"{self.config.name} trained: "
            f"train_acc={self.train_accuracy:.2%}, "
            f"val_acc={self.val_accuracy:.2%}"
        )

        return results

    def predict(self, X: np.ndarray) -> Tuple[int, float, float, float]:
        """Make prediction.

        Returns:
            Tuple of (direction, confidence, prob_up, prob_down)
        """
        if not self.is_trained:
            raise RuntimeError(f"Model {self.config.name} is not trained")

        # Filter features if RFECV was used
        if self.config.use_rfecv and self.selected_indices is not None:
            X = X[self.selected_indices]

        X_scaled = self.scaler.transform(X.reshape(1, -1))
        probs = self.model.predict_proba(X_scaled)[0]
        pred = self.model.predict(X_scaled)[0]
        confidence = max(probs)

        if len(probs) == 2:
            prob_down, prob_up = probs[0], probs[1]
        else:
            prob_up = probs[0] if pred == 1 else 1 - probs[0]
            prob_down = 1 - prob_up

        return int(pred), confidence, prob_up, prob_down

    def predict_batch(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Batch prediction.

        Returns:
            Tuple of (predictions, confidences)
        """
        if not self.is_trained:
            raise RuntimeError(f"Model {self.config.name} is not trained")

        # Filter features if RFECV was used
        if self.config.use_rfecv and self.selected_indices is not None:
            X = X[:, self.selected_indices]

        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        probs = self.model.predict_proba(X_scaled)
        confidences = np.max(probs, axis=1)

        return predictions, confidences

    def save(self, path: Path) -> None:
        """Save model to disk."""
        data = {
            "config": self.config,
            "model": self.model,
            "scaler": self.scaler,
            "feature_names": self.feature_names,
            "train_accuracy": self.train_accuracy,
            "val_accuracy": self.val_accuracy,
            "feature_importance": self.feature_importance,
            "selected_features": self.selected_features,
            "selected_indices": self.selected_indices,
            "rfecv_scores": self.rfecv_scores,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)
        logger.info(f"Saved {self.config.name} model to {path}")

    def load(self, path: Path) -> None:
        """Load model from disk."""
        with open(path, "rb") as f:
            data = pickle.load(f)

        self.config = data["config"]
        self.model = data["model"]
        self.scaler = data["scaler"]
        self.feature_names = data["feature_names"]
        self.train_accuracy = data["train_accuracy"]
        self.val_accuracy = data["val_accuracy"]
        self.feature_importance = data.get("feature_importance", {})
        self.selected_features = data.get("selected_features")
        self.selected_indices = data.get("selected_indices")
        self.rfecv_scores = data.get("rfecv_scores")
        self.is_trained = True
        logger.info(f"Loaded {self.config.name} model from {path}")


class ImprovedMultiTimeframeModel:
    """Manages improved models across multiple timeframes."""

    def __init__(
        self,
        configs: Optional[List[ImprovedModelConfig]] = None,
        model_dir: Optional[Path] = None,
    ):
        if configs is None:
            configs = [
                ImprovedModelConfig.hourly_model(),
                ImprovedModelConfig.four_hour_model(),
            ]

        self.configs = {cfg.name: cfg for cfg in configs}
        self.models: Dict[str, ImprovedTimeframeModel] = {}
        self.model_dir = model_dir or Path("models/improved_mtf")

        for cfg in configs:
            self.models[cfg.name] = ImprovedTimeframeModel(cfg)

    def resample_data(self, df_5min: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """Resample 5-minute data to target timeframe."""
        rule_map = {
            "5min": "5min",
            "15min": "15min",
            "30min": "30min",
            "1H": "1H",
            "4H": "4H",
            "D": "D",
        }
        rule = rule_map.get(timeframe, timeframe)

        if rule == "5min":
            return df_5min.copy()

        resampled = df_5min.resample(rule).agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum" if "volume" in df_5min.columns else "first",
        }).dropna()

        return resampled

    def train_all(
        self,
        df_5min: pd.DataFrame,
        train_ratio: float = 0.6,
        val_ratio: float = 0.2,
    ) -> Dict[str, Dict[str, float]]:
        """Train all timeframe models."""
        results = {}

        for name, model in self.models.items():
            logger.info(f"\n{'=' * 60}")
            logger.info(f"Training {name} model")
            logger.info(f"{'=' * 60}")

            config = self.configs[name]

            # Resample to target timeframe
            df_tf = self.resample_data(df_5min, config.base_timeframe)
            logger.info(f"Resampled to {config.base_timeframe}: {len(df_tf)} bars")

            # Prepare higher timeframe data for cross-TF features
            higher_tf_data = {}
            if config.base_timeframe == "1H":
                higher_tf_data["4H"] = self.resample_data(df_5min, "4H")
                higher_tf_data["D"] = self.resample_data(df_5min, "D")
            elif config.base_timeframe == "4H":
                higher_tf_data["D"] = self.resample_data(df_5min, "D")

            # Calculate technical indicators for higher TFs
            from src.features.technical.calculator import TechnicalIndicatorCalculator
            calc = TechnicalIndicatorCalculator(model_type="short_term")
            for tf_name, tf_df in higher_tf_data.items():
                higher_tf_data[tf_name] = calc.calculate(tf_df)

            # Prepare data
            X, y, feature_cols = model.prepare_data(df_tf, higher_tf_data)

            # Split chronologically
            n_train = int(len(X) * train_ratio)
            n_val = int(len(X) * val_ratio)

            X_train = X[:n_train]
            y_train = y[:n_train]
            X_val = X[n_train:n_train + n_val]
            y_val = y[n_train:n_train + n_val]

            logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X) - n_train - n_val}")

            # Train
            tf_results = model.train(X_train, y_train, X_val, y_val, feature_cols)
            results[name] = tf_results

        return results

    def save_all(self) -> None:
        """Save all models."""
        self.model_dir.mkdir(parents=True, exist_ok=True)

        for name, model in self.models.items():
            path = self.model_dir / f"{name}_improved_model.pkl"
            model.save(path)

    def load_all(self) -> None:
        """Load all models."""
        for name, model in self.models.items():
            path = self.model_dir / f"{name}_improved_model.pkl"
            if path.exists():
                model.load(path)
            else:
                logger.warning(f"Model not found: {path}")

    def predict(
        self,
        df_5min: pd.DataFrame,
        timeframe: str,
    ) -> Tuple[int, float, float, float]:
        """Make prediction for a timeframe.

        Args:
            df_5min: Recent 5-minute OHLCV data
            timeframe: Which model to use

        Returns:
            Tuple of (direction, confidence, prob_up, prob_down)
        """
        model = self.models.get(timeframe)
        if model is None or not model.is_trained:
            raise RuntimeError(f"Model {timeframe} not available or not trained")

        config = self.configs[timeframe]

        # Resample
        df_tf = self.resample_data(df_5min, config.base_timeframe)

        # Prepare features
        from src.features.technical.calculator import TechnicalIndicatorCalculator
        calc = TechnicalIndicatorCalculator(model_type="short_term")
        df_features = calc.calculate(df_tf)

        # Add enhanced features
        df_features = model.feature_engine.add_all_features(df_features)

        # Get latest valid row
        feature_cols = model.feature_names
        available_cols = [c for c in feature_cols if c in df_features.columns]

        if len(available_cols) < len(feature_cols):
            missing = set(feature_cols) - set(available_cols)
            logger.warning(f"Missing features: {missing}")

        X = df_features[available_cols].iloc[-1:].values

        if np.isnan(X).any():
            logger.warning("NaN in features, returning neutral prediction")
            return 0, 0.5, 0.5, 0.5

        return model.predict(X[0])
