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
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd

from .improved_model import ImprovedModelConfig, ImprovedTimeframeModel

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

        # Create model instances
        for tf, cfg in self.model_configs.items():
            self.models[tf] = ImprovedTimeframeModel(cfg)

        # Current weights (may be adjusted for regime)
        self.current_weights = self.config.weights.copy()

        # Training metadata
        self.is_trained = False
        self.training_results: Dict[str, Dict] = {}

    def _normalize_weights(self, weights: Dict[str, float]) -> Dict[str, float]:
        """Normalize weights to sum to 1."""
        total = sum(weights.values())
        if total > 0:
            return {k: v / total for k, v in weights.items()}
        return weights

    def resample_data(self, df_5min: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """Resample 5-minute data to target timeframe."""
        if timeframe == "5min":
            return df_5min.copy()

        resampled = df_5min.resample(timeframe).agg({
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
        """Prepare higher timeframe data for cross-TF features."""
        from src.features.technical.calculator import TechnicalIndicatorCalculator

        calc = TechnicalIndicatorCalculator(model_type="short_term")
        higher_tf_data = {}

        if base_timeframe == "1H":
            higher_tf_data["4H"] = calc.calculate(self.resample_data(df_5min, "4H"))
            higher_tf_data["D"] = calc.calculate(self.resample_data(df_5min, "D"))
        elif base_timeframe == "4H":
            higher_tf_data["D"] = calc.calculate(self.resample_data(df_5min, "D"))
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

            # Train
            tf_results = model.train(X_train, y_train, X_val, y_val, feature_cols)
            results[tf] = tf_results

        self.training_results = results
        self.is_trained = all(m.is_trained for m in self.models.values())

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

            if np.isnan(X).any():
                logger.warning(f"{tf}: NaN in features, using neutral prediction")
                predictions[tf] = 0
                confidences[tf] = 0.5
                probs_up[tf] = 0.5
                continue

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

    def _combine_predictions(
        self,
        predictions: Dict[str, int],
        confidences: Dict[str, float],
        probs_up: Dict[str, float],
        market_regime: str,
    ) -> MTFPrediction:
        """Combine individual predictions into ensemble prediction."""

        # Weighted probability combination
        # Convert prediction + confidence to probability
        weighted_prob_up = 0.0
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

        # Final direction
        direction = 1 if weighted_prob_up > 0.5 else 0

        # Base confidence from distance to 0.5
        base_confidence = abs(weighted_prob_up - 0.5) * 2 + 0.5

        # Agreement calculation
        agreement_count = sum(1 for p in predictions.values() if p == direction)
        agreement_score = agreement_count / len(predictions)

        # Agreement bonus
        if agreement_count == len(predictions):
            confidence = min(base_confidence + self.config.agreement_bonus, 1.0)
        else:
            confidence = base_confidence

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
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Batch prediction for backtesting.

        Args:
            X_dict: Dict mapping timeframe to feature arrays (aligned by index)

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

        # Combine predictions
        directions = np.zeros(n_samples, dtype=int)
        confidences = np.zeros(n_samples)
        agreement_scores = np.zeros(n_samples)

        weights = self._normalize_weights(self.config.weights)

        for i in range(n_samples):
            weighted_prob_up = 0.0

            for tf in self.models:
                pred = all_preds[tf][i]
                conf = all_confs[tf][i]
                weight = weights.get(tf, 0)

                if pred == 1:
                    prob_up = conf
                else:
                    prob_up = 1 - conf

                weighted_prob_up += weight * prob_up

            # Direction
            direction = 1 if weighted_prob_up > 0.5 else 0
            directions[i] = direction

            # Base confidence
            base_conf = abs(weighted_prob_up - 0.5) * 2 + 0.5

            # Agreement
            agreement_count = sum(
                1 for tf in self.models if all_preds[tf][i] == direction
            )
            agreement_score = agreement_count / len(self.models)
            agreement_scores[i] = agreement_score

            # Confidence with agreement bonus
            if agreement_count == len(self.models):
                confidences[i] = min(base_conf + self.config.agreement_bonus, 1.0)
            else:
                confidences[i] = base_conf

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

        # Save ensemble config
        config_data = {
            "weights": self.config.weights,
            "agreement_bonus": self.config.agreement_bonus,
            "use_regime_adjustment": self.config.use_regime_adjustment,
            "regime_weights": self.config.regime_weights,
            "min_confidence": self.config.min_confidence,
            "min_agreement": self.config.min_agreement,
            "training_results": self.training_results,
        }

        with open(save_dir / "ensemble_config.json", "w") as f:
            json.dump(config_data, f, indent=2, default=str)

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
            self.training_results = config_data.get("training_results", {})

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
            f"  Agreement Bonus: {self.config.agreement_bonus}",
            f"  Regime Adjustment: {self.config.use_regime_adjustment}",
            f"  Min Confidence: {self.config.min_confidence}",
            "",
            "Models:",
        ]

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
                lines.append(f"  {tf}: val_acc={results.get('val_accuracy', 0):.2%}")

        lines.append("=" * 60)

        return "\n".join(lines)
