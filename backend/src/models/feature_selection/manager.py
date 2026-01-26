"""Feature selection manager for multi-timeframe ensemble.

This module manages feature selection across multiple timeframes with caching
to avoid recomputing selections.
"""

import hashlib
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .rfecv_config import RFECVConfig
from .rfecv_selector import RFECVSelector

logger = logging.getLogger(__name__)

# Length of MD5 hash for cache file names - short hash sufficient for config differentiation
CACHE_HASH_LENGTH = 8


class FeatureSelectionManager:
    """Manager for feature selection across multiple timeframes.

    Handles feature selection for each timeframe model with caching support.
    Computes config hash for cache invalidation when settings change.

    Attributes:
        config: RFECV configuration
        cache_dir: Directory for caching selections
        selections: Dictionary mapping timeframe to selection results
    """

    def __init__(self, config: Optional[RFECVConfig] = None):
        """Initialize feature selection manager.

        Args:
            config: RFECV configuration. Uses defaults if None.

        Raises:
            ValueError: If cache_dir is an absolute path outside project structure
        """
        self.config = config or RFECVConfig()
        self.cache_dir = Path(self.config.cache_dir)
        self.selections: Dict[str, Dict] = {}

        # Validate cache directory path for safety
        if self.cache_dir.is_absolute():
            logger.warning(
                f"Cache directory is absolute path: {self.cache_dir}. "
                "Consider using relative path for portability."
            )

        # Create cache directory if needed
        if self.config.cache_enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _compute_config_hash(self) -> str:
        """Compute hash of config for cache invalidation.

        Returns:
            MD5 hash of configuration
        """
        config_dict = {
            "step": self.config.step,
            "min_features_to_select": self.config.min_features_to_select,
            "cv": self.config.cv,
            "n_estimators": self.config.n_estimators,
            "max_depth": self.config.max_depth,
            "learning_rate": self.config.learning_rate,
            "scoring": self.config.scoring,
        }
        config_str = json.dumps(config_dict, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:CACHE_HASH_LENGTH]

    def _get_cache_path(self, timeframe: str) -> Path:
        """Get cache file path for timeframe.

        Args:
            timeframe: Timeframe identifier (e.g., "1H", "4H", "D")

        Returns:
            Path to cache file
        """
        config_hash = self._compute_config_hash()
        return self.cache_dir / f"{timeframe}_rfecv_{config_hash}.json"

    def _load_from_cache(self, timeframe: str, n_features: int) -> Optional[Dict]:
        """Load selection from cache if available and valid.

        Args:
            timeframe: Timeframe identifier
            n_features: Current number of features (for validation)

        Returns:
            Cached selection dict or None if not found/invalid
        """
        if not self.config.cache_enabled:
            return None

        cache_path = self._get_cache_path(timeframe)
        if not cache_path.exists():
            return None

        try:
            with open(cache_path, "r") as f:
                cached = json.load(f)

            # CRITICAL: Validate that cached selection matches current feature count
            cached_n_features = cached.get("n_original_features", 0)
            if cached_n_features != n_features:
                logger.warning(
                    f"Cache invalidated for {timeframe}: cached n_features={cached_n_features}, "
                    f"current n_features={n_features}. Recomputing selection."
                )
                # Delete invalid cache file
                cache_path.unlink()
                return None

            logger.info(f"Loaded cached selection for {timeframe} from {cache_path}")
            return cached
        except Exception as e:
            logger.warning(f"Failed to load cache for {timeframe}: {e}")
            return None

    def _save_to_cache(self, timeframe: str, selection: Dict) -> None:
        """Save selection to cache.

        Args:
            timeframe: Timeframe identifier
            selection: Selection results to cache
        """
        if not self.config.cache_enabled:
            return

        cache_path = self._get_cache_path(timeframe)
        try:
            with open(cache_path, "w") as f:
                json.dump(selection, f, indent=2)
            logger.info(f"Cached selection for {timeframe} to {cache_path}")
        except Exception as e:
            logger.warning(f"Failed to save cache for {timeframe}: {e}")

    def select_features(
        self,
        timeframe: str,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        force_recompute: bool = False,
    ) -> Tuple[List[str], np.ndarray, Dict]:
        """Select features for a timeframe model.

        Args:
            timeframe: Timeframe identifier (e.g., "1H", "4H", "D")
            X: Feature matrix (n_samples, n_features)
            y: Target labels (n_samples,)
            feature_names: List of feature names
            force_recompute: If True, ignore cache and recompute

        Returns:
            Tuple of (selected_features, selected_indices, cv_scores)
        """
        logger.info(f"Selecting features for {timeframe} timeframe")
        n_features = len(feature_names)

        # Check cache first (with feature count validation)
        if not force_recompute:
            cached = self._load_from_cache(timeframe, n_features)
            if cached is not None:
                selected_features = cached["selected_features"]
                selected_indices = np.array(cached["selected_indices"])
                cv_scores = cached["cv_scores"]

                logger.info(f"Using cached selection: {len(selected_features)} features")
                self.selections[timeframe] = cached
                return selected_features, selected_indices, cv_scores

        # Run RFECV
        logger.info(f"Running RFECV for {timeframe} (no cache available)")
        selector = RFECVSelector(self.config)
        selected_features, selected_indices = selector.fit(X, y, feature_names)
        cv_scores = selector.cv_scores

        # Store results (convert numpy types for JSON serialization)
        def convert_for_json(obj):
            """Convert numpy types to JSON-serializable Python types."""
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, list):
                return [convert_for_json(item) for item in obj]
            elif isinstance(obj, dict):
                return {str(k): convert_for_json(v) for k, v in obj.items()}
            return obj

        selection = {
            "timeframe": timeframe,
            "selected_features": selected_features,
            "selected_indices": [int(i) for i in selected_indices.tolist()],
            "cv_scores": convert_for_json(cv_scores),
            "config_hash": self._compute_config_hash(),
            "n_original_features": int(len(feature_names)),
            "n_selected_features": int(len(selected_features)),
        }

        self.selections[timeframe] = selection

        # Cache results
        self._save_to_cache(timeframe, selection)

        return selected_features, selected_indices, cv_scores

    def get_selection(self, timeframe: str) -> Optional[Dict]:
        """Get selection results for a timeframe.

        Args:
            timeframe: Timeframe identifier

        Returns:
            Selection dict or None if not available
        """
        return self.selections.get(timeframe)

    def clear_cache(self, timeframe: Optional[str] = None) -> None:
        """Clear cached selections.

        Args:
            timeframe: If specified, clear only this timeframe. Otherwise clear all.
        """
        if not self.config.cache_enabled:
            logger.warning("Caching disabled, nothing to clear")
            return

        if timeframe:
            cache_path = self._get_cache_path(timeframe)
            if cache_path.exists():
                cache_path.unlink()
                logger.info(f"Cleared cache for {timeframe}")
        else:
            # Clear all cache files
            for cache_file in self.cache_dir.glob("*_rfecv_*.json"):
                cache_file.unlink()
            logger.info("Cleared all cached selections")
