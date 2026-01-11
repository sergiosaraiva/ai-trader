"""Feature Store for caching and managing computed features.

This module provides a caching layer for computed technical indicators and
features, avoiding redundant calculations and enabling efficient feature
retrieval for model training and inference.
"""

import hashlib
import json
import logging
import pickle
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .technical.calculator import TechnicalIndicatorCalculator


logger = logging.getLogger(__name__)


class FeatureStoreError(Exception):
    """Exception raised for feature store errors."""

    pass


@dataclass
class FeatureMetadata:
    """Metadata for stored features.

    Attributes:
        symbol: Trading symbol.
        timeframe: Data timeframe.
        model_type: Model type used for feature calculation.
        feature_names: List of feature column names.
        start_date: Data start date.
        end_date: Data end date.
        row_count: Number of rows.
        created_at: Creation timestamp.
        config_hash: Hash of configuration used.
        version: Feature store version.
    """

    symbol: str
    timeframe: str
    model_type: str
    feature_names: List[str]
    start_date: datetime
    end_date: datetime
    row_count: int
    created_at: datetime = field(default_factory=datetime.now)
    config_hash: str = ""
    version: str = "1.0"


class FeatureStore:
    """Store and manage computed features with caching.

    The FeatureStore provides:
    - Persistent storage of computed features in Parquet format
    - In-memory caching for frequently accessed features
    - Automatic invalidation when source data changes
    - Version tracking for feature configurations

    File organization:
        store_path/
        ├── EURUSD/
        │   ├── 1H/
        │   │   ├── short_term/
        │   │   │   ├── features.parquet
        │   │   │   └── metadata.json
        │   │   ├── medium_term/
        │   │   └── long_term/
        │   └── 4H/
        └── GBPUSD/

    Example:
        ```python
        store = FeatureStore("data/features")

        # Compute and store features
        store.compute_and_store(df_ohlcv, "EURUSD", "1H", "medium_term")

        # Retrieve cached features
        features = store.get_features("EURUSD", "1H", "medium_term")

        # Check if features exist and are valid
        if store.is_valid("EURUSD", "1H", "medium_term"):
            features = store.get_features("EURUSD", "1H", "medium_term")
        ```
    """

    def __init__(
        self,
        store_path: Union[str, Path] = "data/features",
        *,
        cache_enabled: bool = True,
        max_cache_entries: int = 100,
        compression: str = "snappy",
    ):
        """Initialize feature store.

        Args:
            store_path: Base directory for feature storage.
            cache_enabled: Enable in-memory caching.
            max_cache_entries: Maximum cache entries before eviction.
            compression: Parquet compression algorithm.
        """
        self.store_path = Path(store_path)
        self.store_path.mkdir(parents=True, exist_ok=True)

        self.cache_enabled = cache_enabled
        self.max_cache_entries = max_cache_entries
        self.compression = compression

        # In-memory cache
        self._cache: Dict[str, pd.DataFrame] = {}
        self._cache_order: List[str] = []  # For LRU eviction
        self._metadata_cache: Dict[str, FeatureMetadata] = {}

        # Calculator instances per model type
        self._calculators: Dict[str, TechnicalIndicatorCalculator] = {}

    def _get_storage_path(
        self,
        symbol: str,
        timeframe: str,
        model_type: str,
    ) -> Path:
        """Get path to feature storage directory."""
        return self.store_path / symbol.upper() / timeframe.upper() / model_type

    def _get_features_path(
        self,
        symbol: str,
        timeframe: str,
        model_type: str,
    ) -> Path:
        """Get path to features parquet file."""
        return self._get_storage_path(symbol, timeframe, model_type) / "features.parquet"

    def _get_metadata_path(
        self,
        symbol: str,
        timeframe: str,
        model_type: str,
    ) -> Path:
        """Get path to metadata file."""
        return self._get_storage_path(symbol, timeframe, model_type) / "metadata.json"

    def _get_cache_key(
        self,
        symbol: str,
        timeframe: str,
        model_type: str,
    ) -> str:
        """Generate cache key."""
        return f"{symbol.upper()}_{timeframe.upper()}_{model_type}"

    def _get_calculator(self, model_type: str) -> TechnicalIndicatorCalculator:
        """Get or create calculator for model type."""
        if model_type not in self._calculators:
            self._calculators[model_type] = TechnicalIndicatorCalculator(
                model_type=model_type
            )
        return self._calculators[model_type]

    def _compute_config_hash(self, calculator: TechnicalIndicatorCalculator) -> str:
        """Compute hash of calculator configuration."""
        config = calculator.get_config()
        config_str = json.dumps(config, sort_keys=True, default=str)
        return hashlib.md5(config_str.encode()).hexdigest()[:8]

    def compute_and_store(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str,
        model_type: str,
        *,
        force: bool = False,
    ) -> pd.DataFrame:
        """Compute features and store them.

        Args:
            df: Source OHLCV DataFrame.
            symbol: Trading symbol.
            timeframe: Data timeframe.
            model_type: Model type for indicator selection.
            force: Force recomputation even if cached.

        Returns:
            DataFrame with computed features.

        Raises:
            FeatureStoreError: If computation or storage fails.
        """
        symbol = symbol.upper()
        timeframe = timeframe.upper()
        cache_key = self._get_cache_key(symbol, timeframe, model_type)

        # Check cache if not forcing
        if not force and self.cache_enabled and cache_key in self._cache:
            logger.debug(f"Returning cached features for {cache_key}")
            return self._cache[cache_key].copy()

        # Get calculator and compute
        calculator = self._get_calculator(model_type)

        try:
            features = calculator.calculate(df)
        except Exception as e:
            raise FeatureStoreError(f"Feature computation failed: {e}") from e

        # Create metadata
        metadata = FeatureMetadata(
            symbol=symbol,
            timeframe=timeframe,
            model_type=model_type,
            feature_names=calculator.get_feature_names(),
            start_date=features.index.min().to_pydatetime(),
            end_date=features.index.max().to_pydatetime(),
            row_count=len(features),
            config_hash=self._compute_config_hash(calculator),
        )

        # Store to disk
        self._store_features(features, metadata)

        # Update cache
        if self.cache_enabled:
            self._update_cache(cache_key, features, metadata)

        logger.info(
            f"Computed and stored {len(metadata.feature_names)} features "
            f"for {symbol}/{timeframe}/{model_type}"
        )

        return features

    def _store_features(
        self,
        features: pd.DataFrame,
        metadata: FeatureMetadata,
    ) -> None:
        """Store features to disk."""
        storage_path = self._get_storage_path(
            metadata.symbol, metadata.timeframe, metadata.model_type
        )
        storage_path.mkdir(parents=True, exist_ok=True)

        # Store features as parquet
        features_path = self._get_features_path(
            metadata.symbol, metadata.timeframe, metadata.model_type
        )

        # Reset index to store timestamp as column
        df_to_store = features.reset_index()
        df_to_store.rename(columns={df_to_store.columns[0]: "timestamp"}, inplace=True)

        table = pa.Table.from_pandas(df_to_store, preserve_index=False)
        pq.write_table(table, features_path, compression=self.compression)

        # Store metadata
        metadata_path = self._get_metadata_path(
            metadata.symbol, metadata.timeframe, metadata.model_type
        )

        metadata_dict = {
            "symbol": metadata.symbol,
            "timeframe": metadata.timeframe,
            "model_type": metadata.model_type,
            "feature_names": metadata.feature_names,
            "start_date": metadata.start_date.isoformat(),
            "end_date": metadata.end_date.isoformat(),
            "row_count": metadata.row_count,
            "created_at": metadata.created_at.isoformat(),
            "config_hash": metadata.config_hash,
            "version": metadata.version,
        }

        with open(metadata_path, "w") as f:
            json.dump(metadata_dict, f, indent=2)

    def _update_cache(
        self,
        cache_key: str,
        features: pd.DataFrame,
        metadata: FeatureMetadata,
    ) -> None:
        """Update in-memory cache with LRU eviction."""
        # Evict if at capacity
        while len(self._cache) >= self.max_cache_entries:
            oldest_key = self._cache_order.pop(0)
            self._cache.pop(oldest_key, None)
            self._metadata_cache.pop(oldest_key, None)

        # Add to cache
        self._cache[cache_key] = features.copy()
        self._metadata_cache[cache_key] = metadata

        # Update LRU order
        if cache_key in self._cache_order:
            self._cache_order.remove(cache_key)
        self._cache_order.append(cache_key)

    def get_features(
        self,
        symbol: str,
        timeframe: str,
        model_type: str,
        *,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """Retrieve stored features.

        Args:
            symbol: Trading symbol.
            timeframe: Data timeframe.
            model_type: Model type.
            start_date: Filter start date.
            end_date: Filter end date.

        Returns:
            DataFrame with features.

        Raises:
            FeatureStoreError: If features not found.
        """
        symbol = symbol.upper()
        timeframe = timeframe.upper()
        cache_key = self._get_cache_key(symbol, timeframe, model_type)

        # Check cache first
        if self.cache_enabled and cache_key in self._cache:
            features = self._cache[cache_key].copy()

            # Update LRU order
            if cache_key in self._cache_order:
                self._cache_order.remove(cache_key)
            self._cache_order.append(cache_key)
        else:
            # Load from disk
            features = self._load_features(symbol, timeframe, model_type)

            # Update cache
            if self.cache_enabled:
                metadata = self._load_metadata(symbol, timeframe, model_type)
                if metadata:
                    self._update_cache(cache_key, features, metadata)

        # Apply date filters
        if start_date:
            features = features[features.index >= pd.Timestamp(start_date)]
        if end_date:
            features = features[features.index <= pd.Timestamp(end_date)]

        return features

    def _load_features(
        self,
        symbol: str,
        timeframe: str,
        model_type: str,
    ) -> pd.DataFrame:
        """Load features from disk."""
        features_path = self._get_features_path(symbol, timeframe, model_type)

        if not features_path.exists():
            raise FeatureStoreError(
                f"Features not found for {symbol}/{timeframe}/{model_type}"
            )

        try:
            table = pq.read_table(features_path)
            df = table.to_pandas()

            # Set timestamp as index
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                df = df.set_index("timestamp")

            return df

        except Exception as e:
            raise FeatureStoreError(f"Failed to load features: {e}") from e

    def _load_metadata(
        self,
        symbol: str,
        timeframe: str,
        model_type: str,
    ) -> Optional[FeatureMetadata]:
        """Load metadata from disk."""
        metadata_path = self._get_metadata_path(symbol, timeframe, model_type)

        if not metadata_path.exists():
            return None

        try:
            with open(metadata_path, "r") as f:
                data = json.load(f)

            return FeatureMetadata(
                symbol=data["symbol"],
                timeframe=data["timeframe"],
                model_type=data["model_type"],
                feature_names=data["feature_names"],
                start_date=datetime.fromisoformat(data["start_date"]),
                end_date=datetime.fromisoformat(data["end_date"]),
                row_count=data["row_count"],
                created_at=datetime.fromisoformat(data["created_at"]),
                config_hash=data.get("config_hash", ""),
                version=data.get("version", "1.0"),
            )

        except Exception as e:
            logger.warning(f"Failed to load metadata: {e}")
            return None

    def exists(self, symbol: str, timeframe: str, model_type: str) -> bool:
        """Check if features exist for symbol/timeframe/model."""
        features_path = self._get_features_path(
            symbol.upper(), timeframe.upper(), model_type
        )
        return features_path.exists()

    def is_valid(
        self,
        symbol: str,
        timeframe: str,
        model_type: str,
        *,
        source_end_date: Optional[datetime] = None,
    ) -> bool:
        """Check if cached features are valid.

        Args:
            symbol: Trading symbol.
            timeframe: Data timeframe.
            model_type: Model type.
            source_end_date: End date of source data for freshness check.

        Returns:
            True if features exist and are up-to-date.
        """
        if not self.exists(symbol, timeframe, model_type):
            return False

        metadata = self._load_metadata(symbol, timeframe, model_type)
        if metadata is None:
            return False

        # Check if source data is newer
        if source_end_date and metadata.end_date < source_end_date:
            return False

        # Check config hash matches current calculator config
        calculator = self._get_calculator(model_type)
        current_hash = self._compute_config_hash(calculator)
        if metadata.config_hash and metadata.config_hash != current_hash:
            logger.debug(f"Config hash mismatch for {symbol}/{timeframe}/{model_type}")
            return False

        return True

    def get_feature_names(
        self,
        symbol: str,
        timeframe: str,
        model_type: str,
    ) -> List[str]:
        """Get list of feature names without loading data.

        Args:
            symbol: Trading symbol.
            timeframe: Data timeframe.
            model_type: Model type.

        Returns:
            List of feature column names.
        """
        metadata = self._load_metadata(symbol.upper(), timeframe.upper(), model_type)
        if metadata:
            return metadata.feature_names
        return []

    def get_metadata(
        self,
        symbol: str,
        timeframe: str,
        model_type: str,
    ) -> Optional[FeatureMetadata]:
        """Get metadata for stored features.

        Args:
            symbol: Trading symbol.
            timeframe: Data timeframe.
            model_type: Model type.

        Returns:
            FeatureMetadata or None if not found.
        """
        cache_key = self._get_cache_key(symbol, timeframe, model_type)

        # Check cache first
        if cache_key in self._metadata_cache:
            return self._metadata_cache[cache_key]

        return self._load_metadata(symbol.upper(), timeframe.upper(), model_type)

    def invalidate(
        self,
        symbol: str,
        timeframe: str,
        model_type: str,
    ) -> bool:
        """Invalidate cached features.

        Args:
            symbol: Trading symbol.
            timeframe: Data timeframe.
            model_type: Model type.

        Returns:
            True if features were invalidated.
        """
        symbol = symbol.upper()
        timeframe = timeframe.upper()
        cache_key = self._get_cache_key(symbol, timeframe, model_type)

        # Remove from cache
        self._cache.pop(cache_key, None)
        self._metadata_cache.pop(cache_key, None)
        if cache_key in self._cache_order:
            self._cache_order.remove(cache_key)

        # Remove from disk
        features_path = self._get_features_path(symbol, timeframe, model_type)
        metadata_path = self._get_metadata_path(symbol, timeframe, model_type)

        invalidated = False
        if features_path.exists():
            features_path.unlink()
            invalidated = True
        if metadata_path.exists():
            metadata_path.unlink()
            invalidated = True

        return invalidated

    def clear_cache(self) -> None:
        """Clear in-memory cache."""
        self._cache.clear()
        self._metadata_cache.clear()
        self._cache_order.clear()
        logger.debug("Feature store cache cleared")

    def list_stored_features(self) -> List[Dict[str, str]]:
        """List all stored feature sets.

        Returns:
            List of dictionaries with symbol, timeframe, model_type.
        """
        stored = []

        for symbol_dir in self.store_path.iterdir():
            if not symbol_dir.is_dir() or symbol_dir.name.startswith("."):
                continue

            for tf_dir in symbol_dir.iterdir():
                if not tf_dir.is_dir():
                    continue

                for model_dir in tf_dir.iterdir():
                    if not model_dir.is_dir():
                        continue

                    features_path = model_dir / "features.parquet"
                    if features_path.exists():
                        stored.append({
                            "symbol": symbol_dir.name,
                            "timeframe": tf_dir.name,
                            "model_type": model_dir.name,
                        })

        return stored

    def get_storage_info(self) -> Dict[str, Any]:
        """Get storage statistics.

        Returns:
            Dictionary with storage info.
        """
        stored = self.list_stored_features()

        total_size = 0
        for item in stored:
            features_path = self._get_features_path(
                item["symbol"], item["timeframe"], item["model_type"]
            )
            if features_path.exists():
                total_size += features_path.stat().st_size

        return {
            "store_path": str(self.store_path),
            "total_feature_sets": len(stored),
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "cache_entries": len(self._cache),
            "cache_enabled": self.cache_enabled,
        }
