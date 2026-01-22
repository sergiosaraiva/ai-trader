"""Multi-Timeframe Ensemble combining 1H, 4H, and Daily models.

This ensemble implements the weighted combination approach:
- Short-term (1H): 60% weight - dominant for entry timing
- Medium-term (4H): 30% weight - trend confirmation
- Long-term (Daily): 10% weight - regime context

The ensemble provides noise reduction through higher timeframe filtering.
"""

import json
import logging
import pickle
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd

from .improved_model import ImprovedModelConfig, ImprovedTimeframeModel
from .stacking_meta_learner import StackingMetaLearner, StackingConfig

logger = logging.getLogger(__name__)


@dataclass
class MTFEnsembleConfig:
    """Configuration for Multi-Timeframe Ensemble."""

    # Model weights
    weights: Dict[str, float] = field(default_factory=lambda: {
        "1H": 0.6,   # Short-term: dominant for entry timing
        "4H": 0.3,   # Medium-term: trend confirmation
        "D": 0.1,    # Long-term: regime context
    })

    # Agreement bonus
    agreement_bonus: float = 0.05  # +5% confidence when all models agree

    # Regime-based weight adjustments
    use_regime_adjustment: bool = True
    regime_weights: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        "trending": {"1H": 0.5, "4H": 0.35, "D": 0.15},
        "ranging": {"1H": 0.7, "4H": 0.25, "D": 0.05},
        "volatile": {"1H": 0.6, "4H": 0.35, "D": 0.05},
    })

    # Dynamic weight adjustment
    use_dynamic_weights: bool = False  # Disabled by default for backward compatibility
    dynamic_weight_window: int = 50  # Number of recent predictions to consider
    dynamic_weight_min: float = 0.1  # Minimum weight for any model
    dynamic_weight_blend: float = 0.5  # Blend factor: 0=all base, 1=all dynamic

    # Minimum confidence thresholds
    min_confidence: float = 0.55
    min_agreement: float = 0.5  # At least 2 of 3 must agree (>50%)

    # Sentiment features - NOW TIMEFRAME-SPECIFIC
    # Based on research: monthly EPU data only useful for Daily model
    include_sentiment: bool = False
    trading_pair: str = "EURUSD"

    # Sentiment source: 'epu' (daily VIX/EPU), 'gdelt' (hourly), or 'both'
    # GDELT is recommended for intraday (1H, 4H) since it has hourly resolution
    # EPU/VIX is appropriate for Daily model (daily resolution matches)
    sentiment_source: str = "epu"

    # Per-timeframe sentiment settings (key finding from investigation)
    # Monthly EPU data is useless for intraday, marginal for swing, useful for position
    sentiment_by_timeframe: Dict[str, bool] = field(default_factory=lambda: {
        "1H": False,  # DISABLED - monthly data adds noise to intraday
        "4H": False,  # DISABLED - monthly data marginal for swing
        "D": True,    # ENABLED - monthly EPU appropriate for position trading
    })

    # Stacking Meta-Learner settings
    use_stacking: bool = False  # Disabled by default for backward compatibility
    stacking_config: Optional[StackingConfig] = None  # Uses defaults if None

    # RFECV feature selection settings
    use_rfecv: bool = False  # Disabled by default for backward compatibility
    rfecv_config: Optional["RFECVConfig"] = None  # Uses defaults if None

    # Probability calibration settings
    use_calibration: bool = False  # Disabled by default for backward compatibility

    # Optimized hyperparameters (from Optuna optimization)
    optimized_hyperparams: Optional[Dict[str, Dict[str, Any]]] = None  # Dict[timeframe, hyperparams]

    @classmethod
    def default(cls) -> "MTFEnsembleConfig":
        """Default configuration with 60/30/10 weights."""
        return cls()

    @classmethod
    def equal_weights(cls) -> "MTFEnsembleConfig":
        """Equal weight configuration for comparison."""
        return cls(weights={"1H": 0.33, "4H": 0.34, "D": 0.33})

    @classmethod
    def with_sentiment(cls, trading_pair: str = "EURUSD") -> "MTFEnsembleConfig":
        """Config with sentiment for Daily model only (research-based).

        Based on investigation findings:
        - Monthly EPU data is useless for 1H (adds noise)
        - Monthly EPU data is marginal for 4H
        - Monthly EPU data is useful for Daily position trading
        """
        return cls(
            include_sentiment=True,
            trading_pair=trading_pair,
            sentiment_by_timeframe={"1H": False, "4H": False, "D": True}
        )

    @classmethod
    def with_full_sentiment(cls, trading_pair: str = "EURUSD") -> "MTFEnsembleConfig":
        """Config with sentiment enabled for ALL timeframes (for testing)."""
        return cls(
            include_sentiment=True,
            trading_pair=trading_pair,
            sentiment_by_timeframe={"1H": True, "4H": True, "D": True}
        )

    @classmethod
    def with_gdelt_sentiment(cls, trading_pair: str = "EURUSD") -> "MTFEnsembleConfig":
        """Config with GDELT hourly sentiment for ALL timeframes (recommended for intraday).

        GDELT provides hourly news sentiment which is properly aggregated:
        - 1H: Raw hourly values (perfect match)
        - 4H: Aggregated avg, std, trend (4 hours → 3 features per region)
        - Daily: Aggregated avg, std, trend (24 hours → 3 features per region)
        """
        return cls(
            include_sentiment=True,
            trading_pair=trading_pair,
            sentiment_source="gdelt",
            sentiment_by_timeframe={"1H": True, "4H": True, "D": True}
        )

    @classmethod
    def with_stacking(cls, stacking_config: Optional[StackingConfig] = None) -> "MTFEnsembleConfig":
        """Config with stacking meta-learner enabled.

        The stacking meta-learner learns to optimally combine base model predictions
        instead of using fixed weighted averaging.
        """
        return cls(
            use_stacking=True,
            stacking_config=stacking_config or StackingConfig.default(),
        )

    @classmethod
    def with_stacking_and_sentiment(
        cls,
        trading_pair: str = "EURUSD",
        stacking_config: Optional[StackingConfig] = None,
    ) -> "MTFEnsembleConfig":
        """Config with both stacking and sentiment for Daily model."""
        return cls(
            include_sentiment=True,
            trading_pair=trading_pair,
            sentiment_by_timeframe={"1H": False, "4H": False, "D": True},
            use_stacking=True,
            stacking_config=stacking_config or StackingConfig.default(),
        )


@dataclass
class MTFPrediction:
    """Prediction from MTF Ensemble."""

    direction: int  # 0 = down, 1 = up
    confidence: float
    prob_up: float
    prob_down: float

    # Component predictions
    component_directions: Dict[str, int] = field(default_factory=dict)
    component_confidences: Dict[str, float] = field(default_factory=dict)
    component_weights: Dict[str, float] = field(default_factory=dict)

    # Agreement info
    agreement_count: int = 0  # How many models agree with final direction
    agreement_score: float = 0.0  # Fraction of models agreeing

    # Regime info
    market_regime: str = "unknown"

    @property
    def should_trade(self) -> bool:
        """Whether confidence is high enough to trade."""
        return self.confidence >= 0.55

    @property
    def all_agree(self) -> bool:
        """Whether all models agree on direction."""
        return self.agreement_count == len(self.component_directions)


class MTFEnsemble:
    """Multi-Timeframe Ensemble combining 1H, 4H, and Daily models.

    This ensemble:
    1. Loads/trains individual models at each timeframe
    2. Combines their predictions using weighted averaging
    3. Boosts confidence when models agree
    4. Optionally adjusts weights based on market regime
    """

    def __init__(
        self,
        config: Optional[MTFEnsembleConfig] = None,
        model_dir: Optional[Path] = None,
    ):
        self.config = config or MTFEnsembleConfig.default()
        self.model_dir = Path(model_dir) if model_dir else Path("models/mtf_ensemble")

        # Initialize models with sentiment support if enabled
        self.models: Dict[str, ImprovedTimeframeModel] = {}
        self.model_configs = {
            "1H": ImprovedModelConfig.hourly_model(),
            "4H": ImprovedModelConfig.four_hour_model(),
            "D": ImprovedModelConfig.daily_model(),
        }

        # Apply TIMEFRAME-SPECIFIC sentiment settings (research-based)
        # Key finding: Monthly EPU data only useful for Daily model
        if self.config.include_sentiment:
            for tf, cfg in self.model_configs.items():
                # Check per-timeframe sentiment setting
                tf_sentiment_enabled = self.config.sentiment_by_timeframe.get(tf, False)
                cfg.include_sentiment_features = tf_sentiment_enabled
                cfg.trading_pair = self.config.trading_pair
                cfg.sentiment_source = self.config.sentiment_source  # Pass sentiment source
                if tf_sentiment_enabled:
                    source_info = f"({self.config.sentiment_source})"
                    logger.info(f"{tf} model: Sentiment ENABLED {source_info}")
                else:
                    logger.info(f"{tf} model: Sentiment DISABLED (research-based)")

        # Apply RFECV settings if enabled
        if self.config.use_rfecv:
            for tf, cfg in self.model_configs.items():
                cfg.use_rfecv = True
                cfg.rfecv_config = self.config.rfecv_config
            logger.info("RFECV feature selection enabled for all timeframes")

        # Apply calibration settings if enabled
        if self.config.use_calibration:
            for tf, cfg in self.model_configs.items():
                cfg.use_calibration = True
            logger.info("Probability calibration (isotonic regression) enabled for all timeframes")

        # Apply optimized hyperparameters if provided
        if self.config.optimized_hyperparams:
            for tf, cfg in self.model_configs.items():
                if tf in self.config.optimized_hyperparams:
                    cfg.hyperparams = self.config.optimized_hyperparams[tf]
                    logger.info(f"{tf} model: Using optimized hyperparameters from Optuna")

        # Create model instances
        for tf, cfg in self.model_configs.items():
            self.models[tf] = ImprovedTimeframeModel(cfg)

        # Current weights (may be adjusted for regime or dynamic adjustment)
        self.current_weights = self.config.weights.copy()

        # Prediction history for dynamic weight adjustment
        self.prediction_history: List[Dict[str, Any]] = []
        self._history_lock = Lock()  # Thread-safe access to prediction_history

        # Training metadata
        self.is_trained = False
        self.training_results: Dict[str, Dict] = {}

        # Stacking meta-learner (optional)
        self.stacking_meta_learner: Optional[StackingMetaLearner] = None
        if self.config.use_stacking:
            stacking_config = self.config.stacking_config or StackingConfig.default()
            self.stacking_meta_learner = StackingMetaLearner(stacking_config)
            logger.info("Stacking meta-learner initialized")

    def _normalize_weights(self, weights: Dict[str, float]) -> Dict[str, float]:
        """Normalize weights to sum to 1."""
        total = sum(weights.values())
        if total > 0:
            return {k: v / total for k, v in weights.items()}
        return weights

    def record_outcome(self, prediction: MTFPrediction, actual_direction: int) -> None:
        """Record prediction outcome for dynamic weight adjustment.

        Call this after each prediction when the actual outcome is known.
        Only uses historical data - no leakage.

        Args:
            prediction: The MTFPrediction that was made
            actual_direction: The actual market direction (0=down, 1=up)
        """
        record = {
            "timestamp": datetime.now(),
            "actual": actual_direction,
        }

        # Record each model's prediction and whether it was correct
        for tf in prediction.component_directions:
            pred = prediction.component_directions[tf]
            record[f"{tf}_pred"] = pred
            record[f"{tf}_correct"] = int(pred == actual_direction)

        # Thread-safe update of prediction history
        with self._history_lock:
            self.prediction_history.append(record)

            # Keep only recent history (2x window size for buffer)
            if len(self.prediction_history) > self.config.dynamic_weight_window * 2:
                self.prediction_history = self.prediction_history[-self.config.dynamic_weight_window:]

            history_size = len(self.prediction_history)

        logger.debug(
            f"Recorded outcome: actual={actual_direction}, "
            f"history_size={history_size}"
        )

    def _calculate_dynamic_weights(self) -> Dict[str, float]:
        """Calculate weights based on recent model accuracy.

        Uses inverse error weighting: models with higher accuracy get more weight.
        Only uses past predictions - no data leakage.

        Returns:
            Dict mapping timeframe to dynamic weight
        """
        # Thread-safe access to prediction history
        with self._history_lock:
            if len(self.prediction_history) < 10:  # Need minimum history
                logger.debug("Insufficient history for dynamic weights, using base weights")
                return self.config.weights.copy()

            # Get recent predictions within the configured window
            recent = self.prediction_history[-self.config.dynamic_weight_window:].copy()

        # Calculate accuracy for each model
        accuracies = {}
        for tf in self.models:
            correct = sum(r.get(f"{tf}_correct", 0) for r in recent)
            total = sum(1 for r in recent if f"{tf}_correct" in r)
            accuracies[tf] = correct / total if total > 0 else 0.5

        logger.debug(f"Recent accuracies (window={len(recent)}): {accuracies}")

        # Inverse error weighting
        weights = {}
        for tf, acc in accuracies.items():
            # Clamp accuracy to avoid extreme weights (40% to 80%)
            acc_clamped = max(0.4, min(0.8, acc))
            error = 1 - acc_clamped
            # Inverse error: higher accuracy = lower error = higher weight
            weights[tf] = 1 / (error + 0.01)  # Add small constant to avoid division by zero

        # Normalize to sum to 1
        total = sum(weights.values())
        weights = {k: v / total for k, v in weights.items()}

        # Apply minimum weight constraint
        for tf in weights:
            weights[tf] = max(self.config.dynamic_weight_min, weights[tf])

        # Re-normalize after applying minimums
        total = sum(weights.values())
        weights = {k: v / total for k, v in weights.items()}

        # Blend with base weights (allows smooth transition)
        blend = self.config.dynamic_weight_blend
        blended = {}
        for tf in self.models:
            base = self.config.weights.get(tf, 0.33)
            dynamic = weights.get(tf, 0.33)
            blended[tf] = (1 - blend) * base + blend * dynamic

        # Final normalization
        blended = self._normalize_weights(blended)

        logger.debug(
            f"Dynamic weights: raw={weights}, blended={blended} "
            f"(blend={blend})"
        )

        return blended

    def resample_data(self, df_5min: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """Resample 5-minute data to target timeframe."""
        if timeframe == "5min":
            return df_5min.copy()

        # Map timeframe to pandas resample string (use lowercase 'h' for pandas 2.0+)
        tf_map = {"1H": "1h", "4H": "4h", "D": "D"}
        resample_tf = tf_map.get(timeframe, timeframe)

        resampled = df_5min.resample(resample_tf).agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum" if "volume" in df_5min.columns else "first",
        }).dropna()

        return resampled

    def prepare_higher_tf_data(
        self,
        df_5min: pd.DataFrame,
        base_timeframe: str,
    ) -> Dict[str, pd.DataFrame]:
        """Prepare higher timeframe data for cross-TF features.

        This method calculates both technical indicators AND enhanced features
        for higher timeframes, ensuring feature parity between training and
        prediction. Enhanced features include time, ROC, normalized, pattern,
        and lag features - but NOT cross-TF features (to avoid recursion).
        """
        from src.features.technical.calculator import TechnicalIndicatorCalculator
        from .enhanced_features import EnhancedFeatureEngine

        calc = TechnicalIndicatorCalculator(model_type="short_term")

        # Create feature engine for HTF data - NO cross-TF features to avoid recursion
        htf_feature_engine = EnhancedFeatureEngine(
            include_time_features=True,
            include_roc_features=True,
            include_normalized_features=True,
            include_pattern_features=True,
            include_lag_features=True,
            include_sentiment_features=False,  # Sentiment handled at target TF level
        )

        higher_tf_data = {}

        # Normalize timeframe for comparison (handle both "1H" and "1h")
        tf_upper = base_timeframe.upper()

        if tf_upper == "1H":
            # Prepare 4H data with enhanced features
            df_4h = calc.calculate(self.resample_data(df_5min, "4H"))
            higher_tf_data["4H"] = htf_feature_engine.add_all_features(
                df_4h, higher_tf_data=None  # No cross-TF for HTF
            )

            # Prepare Daily data with enhanced features
            df_d = calc.calculate(self.resample_data(df_5min, "D"))
            higher_tf_data["D"] = htf_feature_engine.add_all_features(
                df_d, higher_tf_data=None  # No cross-TF for HTF
            )

        elif tf_upper == "4H":
            # Prepare Daily data with enhanced features
            df_d = calc.calculate(self.resample_data(df_5min, "D"))
            higher_tf_data["D"] = htf_feature_engine.add_all_features(
                df_d, higher_tf_data=None  # No cross-TF for HTF
            )
        # Daily doesn't need higher TF data

        return higher_tf_data

    def train(
        self,
        df_5min: pd.DataFrame,
        train_ratio: float = 0.6,
        val_ratio: float = 0.2,
        timeframes: Optional[List[str]] = None,
    ) -> Dict[str, Dict]:
        """Train all timeframe models.

        Args:
            df_5min: 5-minute OHLCV data
            train_ratio: Fraction of data for training
            val_ratio: Fraction of data for validation
            timeframes: Which timeframes to train (default: all)

        Returns:
            Dict mapping timeframe to training metrics
        """
        if timeframes is None:
            timeframes = list(self.models.keys())

        results = {}

        for tf in timeframes:
            if tf not in self.models:
                logger.warning(f"Unknown timeframe: {tf}")
                continue

            model = self.models[tf]
            config = self.model_configs[tf]

            logger.info(f"\n{'=' * 60}")
            logger.info(f"Training {tf} model")
            logger.info(f"{'=' * 60}")

            # Resample to target timeframe
            df_tf = self.resample_data(df_5min, config.base_timeframe)
            logger.info(f"Resampled to {config.base_timeframe}: {len(df_tf)} bars")

            # Prepare higher timeframe data
            higher_tf_data = self.prepare_higher_tf_data(df_5min, config.base_timeframe)

            # Prepare features and labels
            X, y, feature_cols = model.prepare_data(df_tf, higher_tf_data)

            # Split chronologically
            n_train = int(len(X) * train_ratio)
            n_val = int(len(X) * val_ratio)

            X_train = X[:n_train]
            y_train = y[:n_train]
            X_val = X[n_train:n_train + n_val]
            y_val = y[n_train:n_train + n_val]

            logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X) - n_train - n_val}")

            # Fit calibrator if enabled (CRITICAL: using chronologically held-out data)
            if self.config.use_calibration:
                # Split training data: first 90% for model training, last 10% for calibration
                # This ensures calibration set is chronologically AFTER training set
                # and the model has NEVER seen the calibration data
                n_train_for_model = int(len(X_train) * 0.9)

                # Train model on first 90% only
                X_train_model = X_train[:n_train_for_model]
                y_train_model = y_train[:n_train_for_model]

                # Reserve last 10% for calibration (truly held-out)
                X_calib = X_train[n_train_for_model:]
                y_calib = y_train[n_train_for_model:]

                logger.info(f"Train (for model): {len(X_train_model)}, Calibration: {len(X_calib)}")

                # Train model on reduced training set
                tf_results = model.train(X_train_model, y_train_model, X_val, y_val, feature_cols)
                results[tf] = tf_results

                # Fit calibrator on truly held-out data (model never saw this)
                logger.info(f"Fitting calibrator for {tf} with {len(X_calib)} samples (truly held-out)")
                model.fit_calibrator(X_calib, y_calib)
            else:
                # Train on full training set when calibration disabled
                tf_results = model.train(X_train, y_train, X_val, y_val, feature_cols)
                results[tf] = tf_results

        self.training_results = results
        self.is_trained = all(m.is_trained for m in self.models.values())

        # Train stacking meta-learner if enabled
        if self.config.use_stacking and self.is_trained:
            stacking_results = self.train_stacking(df_5min, train_ratio, val_ratio)
            results["stacking"] = stacking_results

        return results

    def train_stacking(
        self,
        df_5min: pd.DataFrame,
        train_ratio: float = 0.6,
        val_ratio: float = 0.2,
    ) -> Dict[str, Any]:
        """Train the stacking meta-learner on OOF predictions.

        CRITICAL: This method uses time-series cross-validation to generate
        out-of-fold predictions, ensuring no data leakage.

        Args:
            df_5min: 5-minute OHLCV data
            train_ratio: Fraction used for training (for split reference)
            val_ratio: Fraction used for validation

        Returns:
            Dict of stacking training metrics
        """
        if self.stacking_meta_learner is None:
            raise RuntimeError("Stacking not enabled. Set use_stacking=True in config.")

        if not self.is_trained:
            raise RuntimeError("Base models not trained. Call train() first.")

        logger.info("\n" + "=" * 60)
        logger.info("TRAINING STACKING META-LEARNER")
        logger.info("=" * 60)

        from src.features.technical.calculator import TechnicalIndicatorCalculator
        calc = TechnicalIndicatorCalculator(model_type="short_term")

        # Prepare features for each timeframe
        X_dict = {}
        y_dict = {}
        volatility_data = None

        for tf in ["1H", "4H", "D"]:
            model = self.models[tf]
            config = self.model_configs[tf]

            # Resample
            df_tf = self.resample_data(df_5min, config.base_timeframe)

            # Prepare higher TF data
            higher_tf_data = self.prepare_higher_tf_data(df_5min, config.base_timeframe)

            # Get features and labels
            X, y, _ = model.prepare_data(df_tf, higher_tf_data)

            # Use full data for meta-training (splits handled inside OOF generation)
            n_train_val = int(len(X) * (train_ratio + val_ratio))
            X_dict[tf] = X[:n_train_val]
            y_dict[tf] = y[:n_train_val]

            logger.info(f"{tf}: {len(X_dict[tf])} samples for meta-training")

            # Extract volatility for meta-features (use 1H as reference)
            if tf == "1H" and "atr_14" in df_tf.columns and "close" in df_tf.columns:
                df_tf_features = calc.calculate(df_tf)
                if "atr_14" in df_tf_features.columns:
                    vol = df_tf_features["atr_14"] / df_tf_features["close"]
                    # Normalize to 0-1 range using rolling percentile (no future data!)
                    vol_normalized = vol.rolling(window=100, min_periods=20).apply(
                        lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min() + 1e-8)
                    )
                    vol_normalized = vol_normalized.fillna(0.5)
                    volatility_data = vol_normalized.values[:n_train_val]

        # Align data to shortest timeframe (use minimum length)
        min_len = min(len(X_dict[tf]) for tf in X_dict)
        for tf in X_dict:
            X_dict[tf] = X_dict[tf][:min_len]
            y_dict[tf] = y_dict[tf][:min_len]

        if volatility_data is not None:
            volatility_data = volatility_data[:min_len]

        # Use 1H labels as ground truth (primary trading timeframe)
        y_labels = y_dict["1H"]

        logger.info(f"Aligned data: {min_len} samples, generating OOF predictions...")

        # Generate OOF predictions
        meta_features, oof_labels, valid_indices = self.stacking_meta_learner.generate_oof_predictions(
            self.models,
            X_dict,
            y_labels,
            volatility_data=volatility_data,
        )

        logger.info(f"Generated {len(valid_indices)} OOF predictions, training meta-model...")

        # Train meta-model
        results = self.stacking_meta_learner.train(
            meta_features,
            oof_labels,
            val_ratio=0.2,  # Use 20% of OOF data for validation
        )

        logger.info(self.stacking_meta_learner.summary())

        return results

    def predict(
        self,
        df_5min: pd.DataFrame,
        use_regime_adjustment: bool = None,
    ) -> MTFPrediction:
        """Make ensemble prediction.

        Args:
            df_5min: Recent 5-minute OHLCV data (enough for all timeframes)
            use_regime_adjustment: Whether to adjust weights for market regime

        Returns:
            MTFPrediction with direction, confidence, and component info
        """
        if not self.is_trained:
            raise RuntimeError("Ensemble not trained. Call train() first.")

        use_regime = use_regime_adjustment if use_regime_adjustment is not None else self.config.use_regime_adjustment

        # Get predictions from each model
        predictions = {}
        confidences = {}
        probs_up = {}

        from src.features.technical.calculator import TechnicalIndicatorCalculator
        calc = TechnicalIndicatorCalculator(model_type="short_term")

        for tf, model in self.models.items():
            config = self.model_configs[tf]

            # Resample
            df_tf = self.resample_data(df_5min, config.base_timeframe)

            # Calculate indicators
            df_features = calc.calculate(df_tf)

            # Add enhanced features
            higher_tf_data = self.prepare_higher_tf_data(df_5min, config.base_timeframe)
            df_features = model.feature_engine.add_all_features(df_features, higher_tf_data)

            # Get features for latest bar
            feature_cols = model.feature_names
            available_cols = [c for c in feature_cols if c in df_features.columns]

            if len(available_cols) < len(feature_cols):
                missing = set(feature_cols) - set(available_cols)
                logger.warning(f"{tf}: Missing {len(missing)} features")

            X = df_features[available_cols].iloc[-1:].values

            # Handle NaN values by filling with 0 (neutral for standardized features)
            nan_count = np.isnan(X).sum()
            if nan_count > 0:
                nan_pct = nan_count / X.size * 100
                if nan_pct > 20:  # More than 20% NaN - use neutral
                    logger.warning(f"{tf}: {nan_pct:.1f}% NaN in features, using neutral prediction")
                    predictions[tf] = 0
                    confidences[tf] = 0.5
                    probs_up[tf] = 0.5
                    continue
                else:
                    logger.debug(f"{tf}: Filling {nan_count} NaN values with 0")
                    X = np.nan_to_num(X, nan=0.0)

            pred, conf, prob_up, prob_down = model.predict(X[0])
            predictions[tf] = pred
            confidences[tf] = conf
            probs_up[tf] = prob_up

        # Detect market regime (optional)
        market_regime = "unknown"
        if use_regime:
            market_regime = self._detect_regime(df_5min)
            if market_regime in self.config.regime_weights:
                self.current_weights = self._normalize_weights(
                    self.config.regime_weights[market_regime]
                )
            else:
                self.current_weights = self._normalize_weights(self.config.weights)
        else:
            self.current_weights = self._normalize_weights(self.config.weights)

        # Combine predictions
        return self._combine_predictions(
            predictions, confidences, probs_up, market_regime
        )

    def predict_from_features(
        self,
        df_1h: pd.DataFrame,
        df_4h: pd.DataFrame,
        df_daily: pd.DataFrame,
        use_regime_adjustment: bool = None,
    ) -> MTFPrediction:
        """Make ensemble prediction using pre-computed feature DataFrames.

        This is a fast path that skips resampling and feature calculation.
        Use when features have already been computed by the pipeline service.

        Args:
            df_1h: 1-hour DataFrame with pre-computed features
            df_4h: 4-hour DataFrame with pre-computed features
            df_daily: Daily DataFrame with pre-computed features
            use_regime_adjustment: Whether to adjust weights for market regime

        Returns:
            MTFPrediction with direction, confidence, and component info
        """
        if not self.is_trained:
            raise RuntimeError("Ensemble not trained. Call train() first.")

        use_regime = (
            use_regime_adjustment
            if use_regime_adjustment is not None
            else self.config.use_regime_adjustment
        )

        # Map timeframes to DataFrames
        tf_data = {
            "1H": df_1h,
            "4H": df_4h,
            "D": df_daily,
        }

        # Get predictions from each model using pre-computed features
        predictions = {}
        confidences = {}
        probs_up = {}

        for tf, model in self.models.items():
            df_features = tf_data.get(tf)

            if df_features is None or df_features.empty:
                logger.warning(f"{tf}: No feature data available, using neutral")
                predictions[tf] = 0
                confidences[tf] = 0.5
                probs_up[tf] = 0.5
                continue

            # Get features for latest bar
            feature_cols = model.feature_names
            available_cols = [c for c in feature_cols if c in df_features.columns]

            if len(available_cols) < len(feature_cols) * 0.8:  # Need at least 80%
                missing = set(feature_cols) - set(available_cols)
                logger.warning(
                    f"{tf}: Missing {len(missing)} features ({len(available_cols)}/{len(feature_cols)} available)"
                )
                predictions[tf] = 0
                confidences[tf] = 0.5
                probs_up[tf] = 0.5
                continue

            X = df_features[available_cols].iloc[-1:].values

            # Handle NaN values
            nan_count = np.isnan(X).sum()
            if nan_count > 0:
                nan_pct = nan_count / X.size * 100
                if nan_pct > 20:
                    logger.warning(
                        f"{tf}: {nan_pct:.1f}% NaN in features, using neutral"
                    )
                    predictions[tf] = 0
                    confidences[tf] = 0.5
                    probs_up[tf] = 0.5
                    continue
                else:
                    X = np.nan_to_num(X, nan=0.0)

            pred, conf, prob_up, prob_down = model.predict(X[0])
            predictions[tf] = pred
            confidences[tf] = conf
            probs_up[tf] = prob_up

        # Detect market regime from 1H data (optional)
        market_regime = "unknown"
        if use_regime and df_1h is not None and not df_1h.empty:
            # Simple regime detection from volatility
            if "atr_14" in df_1h.columns and "close" in df_1h.columns:
                recent = df_1h.tail(20)
                atr = recent["atr_14"].iloc[-1] if "atr_14" in recent else 0
                close = recent["close"].iloc[-1] if "close" in recent else 1
                volatility = atr / close if close > 0 else 0

                if volatility > 0.01:
                    market_regime = "trending_high_vol"
                elif volatility < 0.005:
                    market_regime = "ranging_low_vol"
                else:
                    market_regime = "ranging"

            if market_regime in self.config.regime_weights:
                self.current_weights = self._normalize_weights(
                    self.config.regime_weights[market_regime]
                )
            else:
                self.current_weights = self._normalize_weights(self.config.weights)
        else:
            self.current_weights = self._normalize_weights(self.config.weights)

        # Combine predictions
        return self._combine_predictions(
            predictions, confidences, probs_up, market_regime
        )

    def _combine_predictions(
        self,
        predictions: Dict[str, int],
        confidences: Dict[str, float],
        probs_up: Dict[str, float],
        market_regime: str,
        volatility: float = 0.0,
    ) -> MTFPrediction:
        """Combine individual predictions into ensemble prediction.

        If stacking is enabled and trained, uses the meta-learner to combine predictions.
        Otherwise, uses weighted averaging with optional dynamic weights.
        """

        # Use stacking meta-learner if available and trained
        if (self.config.use_stacking and
            self.stacking_meta_learner is not None and
            self.stacking_meta_learner.is_trained):

            # Get probabilities for stacking
            prob_1h = probs_up.get("1H", 0.5)
            prob_4h = probs_up.get("4H", 0.5)
            prob_d = probs_up.get("D", 0.5)
            conf_1h = confidences.get("1H", 0.5)
            conf_4h = confidences.get("4H", 0.5)
            conf_d = confidences.get("D", 0.5)

            # Use meta-learner for prediction
            direction, confidence, prob_up_final, prob_down_final = self.stacking_meta_learner.predict(
                prob_1h=prob_1h,
                prob_4h=prob_4h,
                prob_d=prob_d,
                conf_1h=conf_1h,
                conf_4h=conf_4h,
                conf_d=conf_d,
                volatility=volatility,
                weights=self.current_weights,
            )

            # Calculate agreement
            agreement_count = sum(1 for p in predictions.values() if p == direction)
            agreement_score = agreement_count / len(predictions)

            return MTFPrediction(
                direction=direction,
                confidence=confidence,
                prob_up=prob_up_final,
                prob_down=prob_down_final,
                component_directions=predictions.copy(),
                component_confidences=confidences.copy(),
                component_weights=self.current_weights.copy(),
                agreement_count=agreement_count,
                agreement_score=agreement_score,
                market_regime=market_regime,
            )

        # Fallback to weighted averaging

        # Apply dynamic weights if enabled
        if self.config.use_dynamic_weights and len(self.prediction_history) >= 10:
            dynamic_weights = self._calculate_dynamic_weights()

            # Blend dynamic with regime weights if both are used
            if self.config.use_regime_adjustment and market_regime in self.config.regime_weights:
                regime_w = self.config.regime_weights[market_regime]
                blend = self.config.dynamic_weight_blend

                # Blend regime weights with dynamic weights
                for tf in dynamic_weights:
                    base_regime = regime_w.get(tf, 0.33)
                    dynamic = dynamic_weights[tf]
                    self.current_weights[tf] = (1 - blend) * base_regime + blend * dynamic

                self.current_weights = self._normalize_weights(self.current_weights)
                logger.info(
                    f"Applied dynamic+regime weights for {market_regime}: {self.current_weights}"
                )
            else:
                # Use dynamic weights only
                self.current_weights = dynamic_weights
                logger.info(f"Applied dynamic weights: {self.current_weights}")

        # Weighted probability combination
        # Convert prediction + confidence to probability
        weighted_prob_up = 0.0
        weighted_confidence = 0.0
        for tf in predictions:
            pred = predictions[tf]
            conf = confidences[tf]
            weight = self.current_weights.get(tf, 0)

            # If prediction is up (1), prob_up = confidence
            # If prediction is down (0), prob_up = 1 - confidence
            if pred == 1:
                prob_up = conf
            else:
                prob_up = 1 - conf

            weighted_prob_up += weight * prob_up
            weighted_confidence += weight * conf

        # Final direction
        direction = 1 if weighted_prob_up > 0.5 else 0

        # Base confidence is the weighted average of component confidences
        # This reflects the actual model confidence, not an inflated value
        base_confidence = weighted_confidence

        # Agreement calculation
        agreement_count = sum(1 for p in predictions.values() if p == direction)
        agreement_score = agreement_count / len(predictions)

        # Agreement bonus (small boost when all models agree)
        if agreement_count == len(predictions):
            confidence = min(base_confidence + self.config.agreement_bonus, 1.0)
        else:
            # Reduce confidence when models disagree
            confidence = base_confidence * agreement_score

        return MTFPrediction(
            direction=direction,
            confidence=confidence,
            prob_up=weighted_prob_up,
            prob_down=1 - weighted_prob_up,
            component_directions=predictions.copy(),
            component_confidences=confidences.copy(),
            component_weights=self.current_weights.copy(),
            agreement_count=agreement_count,
            agreement_score=agreement_score,
            market_regime=market_regime,
        )

    def predict_batch(
        self,
        X_dict: Dict[str, np.ndarray],
        price_data: Optional[pd.DataFrame] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Batch prediction for backtesting.

        Args:
            X_dict: Dict mapping timeframe to feature arrays (aligned by index)
            price_data: Optional OHLC data for enhanced meta-features

        Returns:
            Tuple of (directions, confidences, agreement_scores)
        """
        if not self.is_trained:
            raise RuntimeError("Ensemble not trained")

        # Get predictions from each model
        all_preds = {}
        all_confs = {}

        for tf, model in self.models.items():
            if tf not in X_dict:
                raise ValueError(f"Missing data for timeframe: {tf}")

            preds, confs = model.predict_batch(X_dict[tf])
            all_preds[tf] = preds
            all_confs[tf] = confs

        # Get consistent length
        n_samples = min(len(p) for p in all_preds.values())

        # Convert predictions to probabilities
        all_probs_up = {}
        for tf in self.models:
            preds = all_preds[tf][:n_samples]
            confs = all_confs[tf][:n_samples]
            # prob_up = conf if pred==1, else 1-conf
            all_probs_up[tf] = np.where(preds == 1, confs, 1 - confs)

        weights = self._normalize_weights(self.config.weights)

        # Use stacking if enabled and trained
        if (self.config.use_stacking and
            self.stacking_meta_learner is not None and
            self.stacking_meta_learner.is_trained):

            # Use stacking meta-learner for batch prediction
            directions, confidences, agreement_scores = self.stacking_meta_learner.predict_batch(
                probs_1h=all_probs_up["1H"],
                probs_4h=all_probs_up["4H"],
                probs_d=all_probs_up["D"],
                confs_1h=all_confs["1H"][:n_samples],
                confs_4h=all_confs["4H"][:n_samples],
                confs_d=all_confs["D"][:n_samples],
                volatility=None,  # Not available in batch mode
                weights=weights,
                price_data=price_data,  # For enhanced meta-features
            )

            return directions, confidences, agreement_scores

        # Fallback to weighted averaging
        directions = np.zeros(n_samples, dtype=int)
        confidences = np.zeros(n_samples)
        agreement_scores = np.zeros(n_samples)

        for i in range(n_samples):
            weighted_prob_up = 0.0
            weighted_conf = 0.0

            for tf in self.models:
                prob_up = all_probs_up[tf][i]
                conf = all_confs[tf][i]
                weight = weights.get(tf, 0)

                weighted_prob_up += weight * prob_up
                weighted_conf += weight * conf

            # Direction
            direction = 1 if weighted_prob_up > 0.5 else 0
            directions[i] = direction

            # Base confidence is weighted average of component confidences
            base_conf = weighted_conf

            # Agreement
            agreement_count = sum(
                1 for tf in self.models if all_preds[tf][i] == direction
            )
            agreement_score = agreement_count / len(self.models)
            agreement_scores[i] = agreement_score

            # Confidence with agreement bonus or penalty
            if agreement_count == len(self.models):
                confidences[i] = min(base_conf + self.config.agreement_bonus, 1.0)
            else:
                # Reduce confidence when models disagree
                confidences[i] = base_conf * agreement_score

        return directions, confidences, agreement_scores

    def _detect_regime(self, df_5min: pd.DataFrame) -> str:
        """Detect market regime from price data.

        Simple regime detection based on recent price action.
        """
        df_daily = self.resample_data(df_5min, "D")

        if len(df_daily) < 20:
            return "unknown"

        recent = df_daily.tail(20)

        # Calculate metrics
        returns = recent["close"].pct_change().dropna()
        volatility = returns.std()
        trend = (recent["close"].iloc[-1] - recent["close"].iloc[0]) / recent["close"].iloc[0]

        # ATR-based volatility
        high_low = (recent["high"] - recent["low"]) / recent["close"]
        avg_range = high_low.mean()

        # Classify regime
        if volatility > 0.015 or avg_range > 0.012:
            return "volatile"
        elif abs(trend) > 0.02:
            return "trending"
        else:
            return "ranging"

    def save(self, path: Optional[Path] = None) -> None:
        """Save ensemble to disk."""
        save_dir = Path(path) if path else self.model_dir
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save each model
        for tf, model in self.models.items():
            model_path = save_dir / f"{tf}_model.pkl"
            model.save(model_path)

        # Save ensemble config (including dynamic weight settings and stacking)
        config_data = {
            "weights": self.config.weights,
            "agreement_bonus": self.config.agreement_bonus,
            "use_regime_adjustment": self.config.use_regime_adjustment,
            "regime_weights": self.config.regime_weights,
            "use_dynamic_weights": self.config.use_dynamic_weights,
            "dynamic_weight_window": self.config.dynamic_weight_window,
            "dynamic_weight_min": self.config.dynamic_weight_min,
            "dynamic_weight_blend": self.config.dynamic_weight_blend,
            "min_confidence": self.config.min_confidence,
            "min_agreement": self.config.min_agreement,
            "training_results": self.training_results,
            "use_stacking": self.config.use_stacking,
        }

        with open(save_dir / "ensemble_config.json", "w") as f:
            json.dump(config_data, f, indent=2, default=str)

        # Save prediction history for dynamic weights
        if self.config.use_dynamic_weights and self.prediction_history:
            history_path = save_dir / "prediction_history.pkl"
            with open(history_path, "wb") as f:
                pickle.dump(self.prediction_history, f)
            logger.info(f"Saved prediction history: {len(self.prediction_history)} records")

        # Save stacking meta-learner if enabled and trained
        if self.config.use_stacking and self.stacking_meta_learner is not None:
            if self.stacking_meta_learner.is_trained:
                stacking_path = save_dir / "stacking_meta_learner.pkl"
                self.stacking_meta_learner.save(stacking_path)
                logger.info("Saved stacking meta-learner")

        logger.info(f"Saved MTF Ensemble to {save_dir}")

    def load(self, path: Optional[Path] = None) -> None:
        """Load ensemble from disk."""
        load_dir = Path(path) if path else self.model_dir

        # Load each model
        for tf, model in self.models.items():
            model_path = load_dir / f"{tf}_model.pkl"
            if model_path.exists():
                model.load(model_path)
            else:
                logger.warning(f"Model not found: {model_path}")

        # Load ensemble config
        config_path = load_dir / "ensemble_config.json"
        if config_path.exists():
            with open(config_path) as f:
                config_data = json.load(f)

            self.config.weights = config_data.get("weights", self.config.weights)
            self.config.agreement_bonus = config_data.get("agreement_bonus", 0.05)
            self.config.use_regime_adjustment = config_data.get("use_regime_adjustment", True)
            self.config.regime_weights = config_data.get("regime_weights", self.config.regime_weights)
            self.config.use_dynamic_weights = config_data.get("use_dynamic_weights", False)
            self.config.dynamic_weight_window = config_data.get("dynamic_weight_window", 50)
            self.config.dynamic_weight_min = config_data.get("dynamic_weight_min", 0.1)
            self.config.dynamic_weight_blend = config_data.get("dynamic_weight_blend", 0.5)
            self.training_results = config_data.get("training_results", {})
            self.config.use_stacking = config_data.get("use_stacking", False)

        # Load prediction history for dynamic weights
        history_path = load_dir / "prediction_history.pkl"
        if history_path.exists():
            with open(history_path, "rb") as f:
                self.prediction_history = pickle.load(f)
            logger.info(f"Loaded prediction history: {len(self.prediction_history)} records")
        else:
            self.prediction_history = []

        # Load stacking meta-learner if enabled
        stacking_path = load_dir / "stacking_meta_learner.pkl"
        if stacking_path.exists():
            self.config.use_stacking = True
            if self.stacking_meta_learner is None:
                self.stacking_meta_learner = StackingMetaLearner()
            self.stacking_meta_learner.load(stacking_path)
            logger.info("Loaded stacking meta-learner")

        self.current_weights = self._normalize_weights(self.config.weights)
        self.is_trained = all(m.is_trained for m in self.models.values())

        logger.info(f"Loaded MTF Ensemble from {load_dir}")

    def summary(self) -> str:
        """Return summary of ensemble configuration and training."""
        lines = [
            "=" * 60,
            "MTF ENSEMBLE SUMMARY",
            "=" * 60,
            "",
            "Configuration:",
            f"  Weights: {self.config.weights}",
            f"  Current Weights: {self.current_weights}",
            f"  Agreement Bonus: {self.config.agreement_bonus}",
            f"  Regime Adjustment: {self.config.use_regime_adjustment}",
            f"  Dynamic Weights: {self.config.use_dynamic_weights}",
        ]

        if self.config.use_dynamic_weights:
            lines.extend([
                f"    Window: {self.config.dynamic_weight_window}",
                f"    Min Weight: {self.config.dynamic_weight_min}",
                f"    Blend Factor: {self.config.dynamic_weight_blend}",
                f"    History Size: {len(self.prediction_history)}",
            ])

        lines.extend([
            f"  Min Confidence: {self.config.min_confidence}",
            f"  Stacking: {self.config.use_stacking}",
            "",
            "Models:",
        ])

        for tf, model in self.models.items():
            status = "trained" if model.is_trained else "not trained"
            if model.is_trained:
                lines.append(f"  {tf}: {status}, val_acc={model.val_accuracy:.2%}")
            else:
                lines.append(f"  {tf}: {status}")

        if self.training_results:
            lines.extend([
                "",
                "Training Results:",
            ])
            for tf, results in self.training_results.items():
                if tf != "stacking":  # Show stacking separately
                    lines.append(f"  {tf}: val_acc={results.get('val_accuracy', 0):.2%}")

        # Stacking meta-learner info
        if self.config.use_stacking and self.stacking_meta_learner is not None:
            lines.extend([
                "",
                "Stacking Meta-Learner:",
            ])
            if self.stacking_meta_learner.is_trained:
                lines.extend([
                    f"  Status: trained",
                    f"  Meta Val Acc: {self.stacking_meta_learner.meta_val_accuracy:.2%}",
                ])
            else:
                lines.append("  Status: not trained")

        lines.append("=" * 60)

        return "\n".join(lines)
