"""Multi-timeframe model manager.

Manages individual models for each timeframe in the scalper system.
"""

import logging
import pickle
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


@dataclass
class TimeframeConfig:
    """Configuration for a single timeframe model."""

    name: str  # e.g., "5min", "15min", "30min"
    resample_rule: str  # pandas resample rule: "5min", "15min", "30min"
    label_threshold: float  # Price move threshold for labeling
    min_confidence: float  # Minimum confidence to generate signal
    lookback_bars: int  # Bars needed for regime detection
    role: str  # "primary", "confirmation", "trend"

    # Model hyperparameters
    n_estimators: int = 100
    learning_rate: float = 0.05
    max_depth: int = 4
    min_samples_split: int = 20
    min_samples_leaf: int = 10
    subsample: float = 0.8

    @classmethod
    def scalper_5min(cls) -> "TimeframeConfig":
        """5-minute config for primary signals."""
        return cls(
            name="5min",
            resample_rule="5min",
            label_threshold=0.0001,  # 1 pip
            min_confidence=0.65,
            lookback_bars=50,
            role="primary",
            n_estimators=150,
            max_depth=5,
        )

    @classmethod
    def scalper_15min(cls) -> "TimeframeConfig":
        """15-minute config for confirmation."""
        return cls(
            name="15min",
            resample_rule="15min",
            label_threshold=0.0002,  # 2 pips
            min_confidence=0.55,
            lookback_bars=50,
            role="confirmation",
            n_estimators=100,
            max_depth=4,
        )

    @classmethod
    def scalper_30min(cls) -> "TimeframeConfig":
        """30-minute config for trend filter."""
        return cls(
            name="30min",
            resample_rule="30min",
            label_threshold=0.0003,  # 3 pips
            min_confidence=0.55,
            lookback_bars=50,
            role="trend",
            n_estimators=100,
            max_depth=4,
        )


@dataclass
class TimeframePrediction:
    """Prediction from a single timeframe model."""

    timeframe: str
    direction: int  # 1 = UP, 0 = DOWN
    confidence: float  # 0.5 to 1.0
    probability_up: float
    probability_down: float
    regime: str
    timestamp: datetime
    is_valid: bool = True  # False if insufficient data

    @property
    def is_bullish(self) -> bool:
        return self.direction == 1

    @property
    def is_bearish(self) -> bool:
        return self.direction == 0

    @property
    def meets_confidence(self) -> bool:
        """Check if confidence meets threshold."""
        return self.is_valid and self.confidence >= 0.55


class TimeframeModel:
    """Model for a single timeframe."""

    def __init__(self, config: TimeframeConfig):
        self.config = config
        self.model: Optional[GradientBoostingClassifier] = None
        self.scaler: Optional[StandardScaler] = None
        self.feature_names: List[str] = []
        self.is_trained: bool = False
        self.training_accuracy: float = 0.0
        self.validation_accuracy: float = 0.0

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        feature_names: List[str],
    ) -> Dict[str, float]:
        """Train the model."""
        self.feature_names = feature_names

        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)

        # Create and train model
        self.model = GradientBoostingClassifier(
            n_estimators=self.config.n_estimators,
            learning_rate=self.config.learning_rate,
            max_depth=self.config.max_depth,
            min_samples_split=self.config.min_samples_split,
            min_samples_leaf=self.config.min_samples_leaf,
            subsample=self.config.subsample,
            random_state=42,
        )

        logger.info(f"Training {self.config.name} model with {len(X_train)} samples...")
        self.model.fit(X_train_scaled, y_train)

        # Calculate accuracies
        train_pred = self.model.predict(X_train_scaled)
        val_pred = self.model.predict(X_val_scaled)

        self.training_accuracy = (train_pred == y_train).mean()
        self.validation_accuracy = (val_pred == y_val).mean()
        self.is_trained = True

        # Calculate accuracy at different confidence levels
        val_probs = self.model.predict_proba(X_val_scaled)
        val_conf = np.max(val_probs, axis=1)

        results = {
            "train_accuracy": self.training_accuracy,
            "val_accuracy": self.validation_accuracy,
        }

        for thresh in [0.55, 0.60, 0.65, 0.70]:
            mask = val_conf >= thresh
            if mask.sum() > 0:
                acc = (val_pred[mask] == y_val[mask]).mean()
                results[f"val_acc_conf_{int(thresh*100)}"] = acc
                results[f"val_samples_conf_{int(thresh*100)}"] = int(mask.sum())

        logger.info(
            f"{self.config.name} trained: "
            f"train_acc={self.training_accuracy:.2%}, "
            f"val_acc={self.validation_accuracy:.2%}"
        )

        return results

    def predict(self, X: np.ndarray) -> Tuple[int, float, float, float]:
        """Make prediction.

        Returns:
            Tuple of (direction, confidence, prob_up, prob_down)
        """
        if not self.is_trained or self.model is None or self.scaler is None:
            raise RuntimeError(f"Model {self.config.name} is not trained")

        X_scaled = self.scaler.transform(X.reshape(1, -1))
        probs = self.model.predict_proba(X_scaled)[0]
        pred = self.model.predict(X_scaled)[0]
        confidence = max(probs)

        # Handle case where model only predicts one class
        if len(probs) == 2:
            prob_down, prob_up = probs[0], probs[1]
        else:
            prob_up = probs[0] if pred == 1 else 1 - probs[0]
            prob_down = 1 - prob_up

        return int(pred), confidence, prob_up, prob_down

    def save(self, path: Path) -> None:
        """Save model to disk."""
        data = {
            "config": self.config,
            "model": self.model,
            "scaler": self.scaler,
            "feature_names": self.feature_names,
            "training_accuracy": self.training_accuracy,
            "validation_accuracy": self.validation_accuracy,
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
        self.training_accuracy = data["training_accuracy"]
        self.validation_accuracy = data["validation_accuracy"]
        self.is_trained = True
        logger.info(f"Loaded {self.config.name} model from {path}")


class MultiTimeframeModel:
    """Manages models across multiple timeframes.

    This is the main class for the multi-timeframe trading system.
    It handles:
    - Model management for 5min, 15min, 30min
    - Data resampling and feature calculation
    - Coordinated predictions across timeframes
    """

    def __init__(
        self,
        configs: Optional[List[TimeframeConfig]] = None,
        model_dir: Optional[Path] = None,
    ):
        """Initialize multi-timeframe model.

        Args:
            configs: List of timeframe configurations. If None, uses default scalper configs.
            model_dir: Directory for saving/loading models.
        """
        if configs is None:
            configs = [
                TimeframeConfig.scalper_5min(),
                TimeframeConfig.scalper_15min(),
                TimeframeConfig.scalper_30min(),
            ]

        self.configs = {cfg.name: cfg for cfg in configs}
        self.models: Dict[str, TimeframeModel] = {}
        self.model_dir = model_dir or Path("models/mtf")

        # Initialize models
        for cfg in configs:
            self.models[cfg.name] = TimeframeModel(cfg)

        # Feature calculator (lazy loaded)
        self._feature_calculator = None

    @property
    def feature_calculator(self):
        """Lazy load feature calculator."""
        if self._feature_calculator is None:
            from src.features.technical.calculator import TechnicalIndicatorCalculator
            self._feature_calculator = TechnicalIndicatorCalculator(model_type="short_term")
        return self._feature_calculator

    @property
    def regime_filter(self):
        """Get regime filter (lazy loaded)."""
        if not hasattr(self, "_regime_filter"):
            from src.trading.filters import RegimeFilter
            self._regime_filter = RegimeFilter(timeframe="5min")
        return self._regime_filter

    def resample_data(self, df_5min: pd.DataFrame, rule: str) -> pd.DataFrame:
        """Resample 5-minute data to target timeframe."""
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

    def calculate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical features for a dataframe."""
        return self.feature_calculator.calculate(df)

    def create_labels(
        self, df: pd.DataFrame, threshold: float
    ) -> Tuple[pd.Series, pd.Series]:
        """Create binary labels and validity mask.

        Returns:
            Tuple of (labels, valid_mask)
        """
        returns = df["close"].pct_change(1).shift(-1)

        labels = pd.Series(index=df.index, dtype=float)
        labels[returns > threshold] = 1
        labels[returns < -threshold] = 0

        valid_mask = ~labels.isna()

        return labels, valid_mask

    def prepare_training_data(
        self, df_5min: pd.DataFrame, timeframe: str
    ) -> Tuple[np.ndarray, np.ndarray, List[str], pd.DataFrame]:
        """Prepare training data for a timeframe.

        Returns:
            Tuple of (X, y, feature_names, df_features)
        """
        config = self.configs[timeframe]

        # Resample to target timeframe
        df_resampled = self.resample_data(df_5min, config.resample_rule)

        # Calculate features
        df_features = self.calculate_features(df_resampled)

        # Create labels
        labels, valid_mask = self.create_labels(df_features, config.label_threshold)

        # Get feature columns
        feature_cols = [
            c for c in df_features.columns
            if c not in ["open", "high", "low", "close", "volume"]
        ]

        # Prepare arrays
        X = df_features[feature_cols].values
        y = labels.values

        # Remove NaN rows
        valid = ~np.isnan(X).any(axis=1) & valid_mask.values
        X = X[valid]
        y = y[valid]
        df_valid = df_features[valid].copy()

        return X, y, feature_cols, df_valid

    def train_all(
        self,
        df_5min: pd.DataFrame,
        train_ratio: float = 0.6,
        val_ratio: float = 0.2,
    ) -> Dict[str, Dict[str, float]]:
        """Train all timeframe models.

        Args:
            df_5min: 5-minute OHLCV data
            train_ratio: Fraction for training
            val_ratio: Fraction for validation

        Returns:
            Dict mapping timeframe to training results
        """
        results = {}

        for timeframe, model in self.models.items():
            logger.info(f"\n{'=' * 50}")
            logger.info(f"Training {timeframe} model")
            logger.info(f"{'=' * 50}")

            # Prepare data
            X, y, feature_cols, df_features = self.prepare_training_data(
                df_5min, timeframe
            )

            logger.info(f"Prepared {len(X)} samples with {len(feature_cols)} features")
            logger.info(f"Label distribution: UP {y.mean():.1%}, DOWN {(1-y).mean():.1%}")

            # Split data chronologically
            n_train = int(len(X) * train_ratio)
            n_val = int(len(X) * val_ratio)

            X_train = X[:n_train]
            y_train = y[:n_train]
            X_val = X[n_train:n_train + n_val]
            y_val = y[n_train:n_train + n_val]

            logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}")

            # Train model
            tf_results = model.train(X_train, y_train, X_val, y_val, feature_cols)
            results[timeframe] = tf_results

        return results

    def save_all(self) -> None:
        """Save all models to disk."""
        self.model_dir.mkdir(parents=True, exist_ok=True)

        for timeframe, model in self.models.items():
            path = self.model_dir / f"{timeframe}_model.pkl"
            model.save(path)

    def load_all(self) -> None:
        """Load all models from disk."""
        for timeframe, model in self.models.items():
            path = self.model_dir / f"{timeframe}_model.pkl"
            if path.exists():
                model.load(path)
            else:
                logger.warning(f"Model file not found: {path}")

    def predict_timeframe(
        self,
        df: pd.DataFrame,
        timeframe: str,
    ) -> TimeframePrediction:
        """Make prediction for a single timeframe.

        Args:
            df: OHLCV data (should be at the target timeframe resolution)
            timeframe: Which timeframe model to use

        Returns:
            TimeframePrediction with direction and confidence
        """
        model = self.models[timeframe]
        config = self.configs[timeframe]

        if not model.is_trained:
            return TimeframePrediction(
                timeframe=timeframe,
                direction=0,
                confidence=0.5,
                probability_up=0.5,
                probability_down=0.5,
                regime="unknown",
                timestamp=datetime.now(),
                is_valid=False,
            )

        # Need at least 50 bars for feature calculation
        if len(df) < 50:
            return TimeframePrediction(
                timeframe=timeframe,
                direction=0,
                confidence=0.5,
                probability_up=0.5,
                probability_down=0.5,
                regime="unknown",
                timestamp=df.index[-1] if len(df) > 0 else datetime.now(),
                is_valid=False,
            )

        # Calculate features
        df_features = self.calculate_features(df)

        # Check if we have valid features
        if len(df_features) == 0:
            return TimeframePrediction(
                timeframe=timeframe,
                direction=0,
                confidence=0.5,
                probability_up=0.5,
                probability_down=0.5,
                regime="unknown",
                timestamp=df.index[-1] if len(df) > 0 else datetime.now(),
                is_valid=False,
            )

        # Get feature values for latest bar
        feature_cols = model.feature_names

        # Check if all feature columns exist
        missing_cols = [c for c in feature_cols if c not in df_features.columns]
        if missing_cols:
            return TimeframePrediction(
                timeframe=timeframe,
                direction=0,
                confidence=0.5,
                probability_up=0.5,
                probability_down=0.5,
                regime="unknown",
                timestamp=df.index[-1],
                is_valid=False,
            )

        X = df_features[feature_cols].iloc[-1].values

        if np.isnan(X).any():
            return TimeframePrediction(
                timeframe=timeframe,
                direction=0,
                confidence=0.5,
                probability_up=0.5,
                probability_down=0.5,
                regime="unknown",
                timestamp=df.index[-1],
                is_valid=False,
            )

        # Make prediction
        direction, confidence, prob_up, prob_down = model.predict(X)

        # Detect regime
        regime = "unknown"
        if len(df) >= config.lookback_bars:
            analysis = self.regime_filter.analyze(df.tail(config.lookback_bars + 1))
            regime = analysis.regime.value

        return TimeframePrediction(
            timeframe=timeframe,
            direction=direction,
            confidence=confidence,
            probability_up=prob_up,
            probability_down=prob_down,
            regime=regime,
            timestamp=df.index[-1],
            is_valid=True,
        )

    def predict_all(
        self,
        df_5min: pd.DataFrame,
    ) -> Dict[str, TimeframePrediction]:
        """Make predictions for all timeframes.

        Args:
            df_5min: Recent 5-minute OHLCV data (need enough for 30min resampling)

        Returns:
            Dict mapping timeframe to prediction
        """
        predictions = {}

        for timeframe, config in self.configs.items():
            # Resample to target timeframe
            df_tf = self.resample_data(df_5min, config.resample_rule)

            # Make prediction
            pred = self.predict_timeframe(df_tf, timeframe)
            predictions[timeframe] = pred

        return predictions
