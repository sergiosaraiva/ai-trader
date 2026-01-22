"""Model service for loading and running MTF Ensemble predictions.

This service provides:
- Singleton pattern for model loading (load once at startup)
- Thread-safe prediction interface
- Prediction caching (1-minute TTL)
- Warm-up on startup
"""

import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from threading import Lock
from typing import Dict, Optional, Any

import numpy as np
import pandas as pd

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)


class ModelService:
    """Service for MTF Ensemble model loading and prediction.

    Uses singleton pattern - model is loaded once and shared across requests.
    Provides thread-safe prediction with caching.
    """

    # Default model directory
    DEFAULT_MODEL_DIR = PROJECT_ROOT / "models" / "mtf_ensemble"

    # Cache TTL
    PREDICTION_CACHE_TTL = timedelta(minutes=1)

    # Cache size limit (prevent unbounded memory growth)
    MAX_CACHE_SIZE = 100

    def __init__(self, model_dir: Optional[Path] = None):
        self._lock = Lock()
        self._model_dir = Path(model_dir) if model_dir else self.DEFAULT_MODEL_DIR

        # Model instance (lazy loaded)
        self._ensemble = None
        self._config = None

        # Prediction cache
        self._cache: Dict[str, Dict] = {}
        self._cache_timestamp: Optional[datetime] = None

        # Status
        self._initialized = False
        self._initialization_error: Optional[str] = None

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._ensemble is not None and self._ensemble.is_trained

    @property
    def is_initialized(self) -> bool:
        """Check if service is initialized."""
        return self._initialized

    def initialize(self, warm_up: bool = True) -> bool:
        """Initialize model service by loading the ensemble.

        Args:
            warm_up: Whether to run a warm-up prediction

        Returns:
            True if successful, False otherwise
        """
        if self._initialized:
            return True

        logger.info("Initializing ModelService...")

        try:
            self._load_model()

            if warm_up:
                self._warm_up()

            self._initialized = True
            logger.info("ModelService initialized successfully")
            return True

        except Exception as e:
            self._initialization_error = str(e)
            logger.error(f"Failed to initialize ModelService: {e}")
            return False

    def _load_model(self) -> None:
        """Load the MTF Ensemble from disk.

        Configuration is loaded from training_metadata.json saved with the model.
        This ensures the service uses the exact configuration the model was trained with.
        """
        import json
        from src.models.multi_timeframe.mtf_ensemble import (
            MTFEnsemble,
            MTFEnsembleConfig,
        )
        from src.models.multi_timeframe.stacking_meta_learner import StackingConfig

        logger.info(f"Loading MTF Ensemble from {self._model_dir}")

        # Load configuration from training metadata
        metadata_path = self._model_dir / "training_metadata.json"
        if metadata_path.exists():
            with open(metadata_path) as f:
                metadata = json.load(f)
            logger.info("Loaded configuration from training_metadata.json")
        else:
            logger.warning("No training_metadata.json found, using defaults")
            metadata = {}

        # Extract configuration from metadata (with defaults for backwards compatibility)
        weights = metadata.get("weights", {"1H": 0.6, "4H": 0.3, "D": 0.1})
        include_sentiment = metadata.get("include_sentiment", True)
        sentiment_source = metadata.get("sentiment_source", "epu")
        sentiment_by_timeframe = metadata.get(
            "sentiment_by_timeframe", {"1H": False, "4H": False, "D": True}
        )
        trading_pair = metadata.get("trading_pair", "EURUSD")
        use_stacking = metadata.get("use_stacking", False)
        stacking_blend = metadata.get("stacking_blend", 0.0)

        # Create stacking config if stacking is enabled
        stacking_config = None
        if use_stacking:
            # Load use_enhanced_meta_features from the saved meta-learner config
            use_enhanced_meta_features = False
            stacking_pkl_path = self._model_dir / "stacking_meta_learner.pkl"
            if stacking_pkl_path.exists():
                import pickle
                try:
                    with open(stacking_pkl_path, "rb") as f:
                        saved_meta_learner = pickle.load(f)
                    if isinstance(saved_meta_learner, dict) and "config" in saved_meta_learner:
                        saved_config = saved_meta_learner["config"]
                        use_enhanced_meta_features = getattr(saved_config, "use_enhanced_meta_features", False)
                        logger.info(f"Loaded use_enhanced_meta_features={use_enhanced_meta_features} from saved model")
                except Exception as e:
                    logger.warning(f"Could not load stacking config from pickle: {e}")

            stacking_config = StackingConfig(
                blend_with_weighted_avg=stacking_blend,
                use_enhanced_meta_features=use_enhanced_meta_features,
            )
            logger.info(f"Stacking enabled with blend={stacking_blend}, enhanced_features={use_enhanced_meta_features}")

        # Create config from loaded metadata
        self._config = MTFEnsembleConfig(
            weights=weights,
            agreement_bonus=0.05,
            use_regime_adjustment=True,
            include_sentiment=include_sentiment,
            sentiment_source=sentiment_source,
            sentiment_by_timeframe=sentiment_by_timeframe,
            trading_pair=trading_pair,
            use_stacking=use_stacking,
            stacking_config=stacking_config,
        )

        logger.info(f"Config: weights={weights}, stacking={use_stacking}, sentiment={include_sentiment}")

        # Create and load ensemble
        self._ensemble = MTFEnsemble(
            config=self._config,
            model_dir=self._model_dir,
        )
        self._ensemble.load()

        if not self._ensemble.is_trained:
            raise RuntimeError("Failed to load trained models")

        logger.info("MTF Ensemble loaded successfully")
        logger.info(self._ensemble.summary())

    def _warm_up(self) -> None:
        """Run a warm-up prediction to initialize all internal state.

        The first prediction is typically slow due to lazy initialization.
        """
        logger.info("Running model warm-up...")

        # Create dummy data for warm-up
        # We need at least 500 bars of 5-min data for proper feature calculation
        dates = pd.date_range(
            end=datetime.now(),
            periods=1000,
            freq="5min",
        )

        dummy_df = pd.DataFrame({
            "open": np.random.uniform(1.05, 1.10, len(dates)),
            "high": np.random.uniform(1.06, 1.11, len(dates)),
            "low": np.random.uniform(1.04, 1.09, len(dates)),
            "close": np.random.uniform(1.05, 1.10, len(dates)),
            "volume": np.random.uniform(1000, 10000, len(dates)),
        }, index=dates)

        # Ensure OHLC relationships are valid
        dummy_df["high"] = dummy_df[["open", "high", "close"]].max(axis=1) + 0.001
        dummy_df["low"] = dummy_df[["open", "low", "close"]].min(axis=1) - 0.001

        try:
            # Run prediction (result doesn't matter)
            _ = self._ensemble.predict(dummy_df)
            logger.info("Model warm-up complete")
        except Exception as e:
            logger.warning(f"Warm-up failed (non-critical): {e}")

    def predict(
        self,
        df_5min: pd.DataFrame,
        use_cache: bool = True,
        symbol: str = "EURUSD",
    ) -> Dict[str, Any]:
        """Make a prediction using the MTF Ensemble.

        Args:
            df_5min: 5-minute OHLCV DataFrame with sufficient history
            use_cache: Whether to use cached predictions
            symbol: Trading symbol (default: EURUSD)

        Returns:
            Dict with prediction details including symbol
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call initialize() first.")

        # Generate cache key from latest timestamp
        if df_5min is not None and len(df_5min) > 0:
            cache_key = str(df_5min.index[-1])
        else:
            cache_key = "unknown"

        # Check cache
        with self._lock:
            if use_cache and cache_key in self._cache:
                cached = self._cache[cache_key]
                cache_age = datetime.now() - cached["cached_at"]
                if cache_age < self.PREDICTION_CACHE_TTL:
                    logger.debug(f"Returning cached prediction ({cache_age.seconds}s old)")
                    return cached["prediction"]

        # Make prediction
        try:
            with self._lock:
                prediction = self._ensemble.predict(df_5min)

            # Convert to dict (clamp values to [0, 1] range for API validation)
            result = {
                "direction": "long" if prediction.direction == 1 else "short",
                "confidence": float(min(1.0, max(0.0, prediction.confidence))),
                "prob_up": float(min(1.0, max(0.0, prediction.prob_up))),
                "prob_down": float(min(1.0, max(0.0, prediction.prob_down))),
                "should_trade": prediction.confidence >= 0.70,  # 70% threshold
                "agreement_count": prediction.agreement_count,
                "agreement_score": float(min(1.0, max(0.0, prediction.agreement_score))),
                "all_agree": prediction.all_agree,
                "market_regime": prediction.market_regime,
                "component_directions": {
                    k: int(v) for k, v in prediction.component_directions.items()
                },
                "component_confidences": {
                    k: float(min(1.0, max(0.0, v))) for k, v in prediction.component_confidences.items()
                },
                "component_weights": {
                    k: float(v) for k, v in prediction.component_weights.items()
                },
                "timestamp": datetime.now().isoformat(),
                "symbol": symbol,
            }

            # Cache result (with size limit to prevent memory leak)
            with self._lock:
                # Evict expired and oldest entries if cache is full
                self._cleanup_cache_if_needed()
                self._cache[cache_key] = {
                    "prediction": result,
                    "cached_at": datetime.now(),
                }

            return result

        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        if not self.is_loaded:
            return {
                "loaded": False,
                "error": self._initialization_error,
            }

        return {
            "loaded": True,
            "model_dir": str(self._model_dir),
            "weights": self._config.weights if self._config else None,
            "agreement_bonus": self._config.agreement_bonus if self._config else None,
            "sentiment_enabled": self._config.include_sentiment if self._config else False,
            "sentiment_by_timeframe": (
                self._config.sentiment_by_timeframe if self._config else {}
            ),
            "use_stacking": self._config.use_stacking if self._config else False,
            "stacking_blend": (
                self._config.stacking_config.blend_with_weighted_avg
                if self._config and self._config.use_stacking and self._config.stacking_config
                else None
            ),
            "models": {
                tf: {
                    "trained": model.is_trained,
                    "val_accuracy": (
                        float(model.val_accuracy) if model.is_trained else None
                    ),
                }
                for tf, model in self._ensemble.models.items()
            } if self._ensemble else {},
            "initialized_at": datetime.now().isoformat(),
        }

    def get_status(self) -> Dict[str, Any]:
        """Get service status."""
        return {
            "initialized": self._initialized,
            "loaded": self.is_loaded,
            "model_dir": str(self._model_dir),
            "cache_entries": len(self._cache),
            "error": self._initialization_error,
        }

    def clear_cache(self) -> None:
        """Clear prediction cache."""
        with self._lock:
            self._cache.clear()
        logger.info("Model prediction cache cleared")

    def _cleanup_cache_if_needed(self) -> None:
        """Clean up expired entries and enforce cache size limit.

        Must be called while holding self._lock.
        """
        now = datetime.now()

        # First, remove expired entries
        expired_keys = [
            k for k, v in self._cache.items()
            if now - v["cached_at"] > self.PREDICTION_CACHE_TTL
        ]
        for key in expired_keys:
            del self._cache[key]

        # Then enforce size limit by removing oldest entries
        if len(self._cache) >= self.MAX_CACHE_SIZE:
            # Remove 20% of oldest entries
            entries_to_remove = max(1, len(self._cache) // 5)
            sorted_keys = sorted(
                self._cache.keys(),
                key=lambda k: self._cache[k]["cached_at"]
            )
            for key in sorted_keys[:entries_to_remove]:
                del self._cache[key]

    def predict_from_pipeline(
        self, use_cache: bool = True, symbol: str = "EURUSD"
    ) -> Dict[str, Any]:
        """Make a prediction using pre-processed pipeline data.

        This method uses the cached processed data from the pipeline service
        instead of resampling and calculating features on the fly.

        The pipeline provides:
        - 1H data with technical indicators and features
        - 4H data with technical indicators and features
        - Daily data with technical indicators, features, and sentiment

        Args:
            use_cache: Whether to use cached predictions
            symbol: Trading symbol (default: EURUSD)

        Returns:
            Dict with prediction details
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call initialize() first.")

        # Import pipeline service
        from .pipeline_service import pipeline_service

        # Get latest processed data from pipeline
        df_1h = pipeline_service.get_processed_data("1h")
        df_4h = pipeline_service.get_processed_data("4h")
        df_daily = pipeline_service.get_processed_data("D")

        if df_1h is None or df_1h.empty:
            logger.warning("No 1H data from pipeline, falling back to standard predict")
            from .data_service import data_service
            df = data_service.get_data_for_prediction()
            if df is not None:
                return self.predict(df, use_cache=use_cache)
            raise RuntimeError("No data available for prediction")

        # Generate cache key from latest 1H timestamp
        cache_key = f"pipeline_{df_1h.index[-1]}"

        # Check cache
        with self._lock:
            if use_cache and cache_key in self._cache:
                cached = self._cache[cache_key]
                cache_age = datetime.now() - cached["cached_at"]
                if cache_age < self.PREDICTION_CACHE_TTL:
                    logger.debug(f"Returning cached pipeline prediction ({cache_age.seconds}s old)")
                    return cached["prediction"]

        # Make predictions using pipeline data
        try:
            with self._lock:
                # Use ensemble's predict_from_features method if available,
                # otherwise fall back to standard prediction with 1H data
                if hasattr(self._ensemble, 'predict_from_features'):
                    prediction = self._ensemble.predict_from_features(
                        df_1h=df_1h,
                        df_4h=df_4h,
                        df_daily=df_daily,
                    )
                else:
                    # Fallback: use 1H data and let ensemble do its thing
                    # The 1H model will use the pre-calculated features
                    prediction = self._ensemble.predict(df_1h)

            # Extract data timestamp safely (handle NaT or missing isoformat)
            data_timestamp = None
            try:
                last_bar = df_1h.index[-1]
                if hasattr(last_bar, 'isoformat') and not pd.isna(last_bar):
                    data_timestamp = last_bar.isoformat()
            except (IndexError, AttributeError):
                logger.warning("Could not extract data timestamp from 1H data")

            # Convert to dict (clamp values to [0, 1] range for API validation)
            result = {
                "direction": "long" if prediction.direction == 1 else "short",
                "confidence": float(min(1.0, max(0.0, prediction.confidence))),
                "prob_up": float(min(1.0, max(0.0, prediction.prob_up))),
                "prob_down": float(min(1.0, max(0.0, prediction.prob_down))),
                "should_trade": prediction.confidence >= 0.70,  # 70% threshold
                "agreement_count": prediction.agreement_count,
                "agreement_score": float(min(1.0, max(0.0, prediction.agreement_score))),
                "all_agree": prediction.all_agree,
                "market_regime": prediction.market_regime,
                "component_directions": {
                    k: int(v) for k, v in prediction.component_directions.items()
                },
                "component_confidences": {
                    k: float(min(1.0, max(0.0, v))) for k, v in prediction.component_confidences.items()
                },
                "component_weights": {
                    k: float(v) for k, v in prediction.component_weights.items()
                },
                "timestamp": datetime.now().isoformat(),
                "symbol": symbol,
                "data_source": "pipeline",
                "data_timestamp": data_timestamp,
            }

            # Cache result (with size limit to prevent memory leak)
            with self._lock:
                # Evict expired and oldest entries if cache is full
                self._cleanup_cache_if_needed()
                self._cache[cache_key] = {
                    "prediction": result,
                    "cached_at": datetime.now(),
                }

            return result

        except Exception as e:
            logger.error(f"Pipeline prediction failed: {e}")
            # Try fallback to standard prediction
            try:
                from .data_service import data_service
                df = data_service.get_data_for_prediction()
                if df is not None:
                    logger.info("Falling back to standard prediction")
                    return self.predict(df, use_cache=use_cache)
            except Exception:
                pass
            raise


# Singleton instance
model_service = ModelService()
