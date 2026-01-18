"""Unit tests for FeatureStore."""

import json
import tempfile
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest
import numpy as np

from src.features.store import (
    FeatureStore,
    FeatureMetadata,
    FeatureStoreError,
)


@pytest.fixture
def sample_ohlcv_data():
    """Create sample OHLCV data."""
    dates = pd.date_range("2024-01-01", periods=200, freq="1h")
    np.random.seed(42)
    base_price = 1.1
    prices = base_price + np.cumsum(np.random.randn(200) * 0.001)

    return pd.DataFrame(
        {
            "open": prices,
            "high": prices + np.random.rand(200) * 0.005,
            "low": prices - np.random.rand(200) * 0.005,
            "close": prices + np.random.randn(200) * 0.002,
            "volume": np.random.randint(100, 1000, 200).astype(float),
        },
        index=dates,
    )


@pytest.fixture
def temp_store_path():
    """Create temporary directory for feature store."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def feature_store(temp_store_path):
    """Create FeatureStore instance."""
    return FeatureStore(store_path=temp_store_path)


class TestFeatureMetadata:
    """Tests for FeatureMetadata."""

    def test_metadata_creation(self):
        """Test metadata creation with required fields."""
        metadata = FeatureMetadata(
            symbol="EURUSD",
            timeframe="1H",
            model_type="medium_term",
            feature_names=["rsi_14", "sma_20"],
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 6, 1),
            row_count=1000,
        )
        assert metadata.symbol == "EURUSD"
        assert metadata.timeframe == "1H"
        assert len(metadata.feature_names) == 2

    def test_metadata_defaults(self):
        """Test metadata default values."""
        metadata = FeatureMetadata(
            symbol="EURUSD",
            timeframe="1H",
            model_type="medium_term",
            feature_names=[],
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 6, 1),
            row_count=0,
        )
        assert metadata.version == "1.0"
        assert metadata.config_hash == ""
        assert metadata.created_at is not None


class TestFeatureStore:
    """Tests for FeatureStore."""

    def test_init_creates_directory(self, temp_store_path):
        """Test store directory is created on init."""
        store = FeatureStore(store_path=temp_store_path / "new_store")
        assert store.store_path.exists()

    def test_init_with_cache_disabled(self, temp_store_path):
        """Test initialization with cache disabled."""
        store = FeatureStore(store_path=temp_store_path, cache_enabled=False)
        assert store.cache_enabled is False

    def test_compute_and_store(self, feature_store, sample_ohlcv_data):
        """Test computing and storing features."""
        result = feature_store.compute_and_store(
            sample_ohlcv_data,
            symbol="EURUSD",
            timeframe="1H",
            model_type="medium_term",
        )
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
        assert feature_store.exists("EURUSD", "1H", "medium_term")

    def test_compute_and_store_creates_files(self, feature_store, sample_ohlcv_data):
        """Test that files are created on disk."""
        feature_store.compute_and_store(
            sample_ohlcv_data,
            symbol="EURUSD",
            timeframe="1H",
            model_type="medium_term",
        )

        features_path = feature_store._get_features_path("EURUSD", "1H", "medium_term")
        metadata_path = feature_store._get_metadata_path("EURUSD", "1H", "medium_term")

        assert features_path.exists()
        assert metadata_path.exists()

    def test_compute_and_store_uses_cache(self, feature_store, sample_ohlcv_data):
        """Test that compute_and_store uses cache."""
        feature_store.compute_and_store(
            sample_ohlcv_data,
            symbol="EURUSD",
            timeframe="1H",
            model_type="medium_term",
        )

        cache_key = feature_store._get_cache_key("EURUSD", "1H", "medium_term")
        assert cache_key in feature_store._cache

    def test_compute_and_store_force_recompute(self, feature_store, sample_ohlcv_data):
        """Test force recomputation."""
        result1 = feature_store.compute_and_store(
            sample_ohlcv_data,
            symbol="EURUSD",
            timeframe="1H",
            model_type="medium_term",
        )

        result2 = feature_store.compute_and_store(
            sample_ohlcv_data,
            symbol="EURUSD",
            timeframe="1H",
            model_type="medium_term",
            force=True,
        )

        # Both should work
        assert len(result1) > 0
        assert len(result2) > 0

    def test_get_features(self, feature_store, sample_ohlcv_data):
        """Test retrieving stored features."""
        feature_store.compute_and_store(
            sample_ohlcv_data,
            symbol="EURUSD",
            timeframe="1H",
            model_type="medium_term",
        )

        result = feature_store.get_features("EURUSD", "1H", "medium_term")
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

    def test_get_features_with_date_filter(self, feature_store, sample_ohlcv_data):
        """Test retrieving features with date filter."""
        feature_store.compute_and_store(
            sample_ohlcv_data,
            symbol="EURUSD",
            timeframe="1H",
            model_type="medium_term",
        )

        start_date = datetime(2024, 1, 3)
        result = feature_store.get_features(
            "EURUSD", "1H", "medium_term", start_date=start_date
        )

        assert isinstance(result, pd.DataFrame)
        assert result.index.min() >= pd.Timestamp(start_date)

    def test_get_features_not_found(self, feature_store):
        """Test error when features not found."""
        with pytest.raises(FeatureStoreError, match="not found"):
            feature_store.get_features("UNKNOWN", "1H", "medium_term")

    def test_exists(self, feature_store, sample_ohlcv_data):
        """Test exists method."""
        assert not feature_store.exists("EURUSD", "1H", "medium_term")

        feature_store.compute_and_store(
            sample_ohlcv_data,
            symbol="EURUSD",
            timeframe="1H",
            model_type="medium_term",
        )

        assert feature_store.exists("EURUSD", "1H", "medium_term")

    def test_is_valid(self, feature_store, sample_ohlcv_data):
        """Test is_valid method."""
        assert not feature_store.is_valid("EURUSD", "1H", "medium_term")

        feature_store.compute_and_store(
            sample_ohlcv_data,
            symbol="EURUSD",
            timeframe="1H",
            model_type="medium_term",
        )

        assert feature_store.is_valid("EURUSD", "1H", "medium_term")

    def test_is_valid_with_source_end_date(self, feature_store, sample_ohlcv_data):
        """Test is_valid with source data freshness check."""
        feature_store.compute_and_store(
            sample_ohlcv_data,
            symbol="EURUSD",
            timeframe="1H",
            model_type="medium_term",
        )

        # Should be valid with earlier source date
        assert feature_store.is_valid(
            "EURUSD", "1H", "medium_term", source_end_date=datetime(2024, 1, 1)
        )

        # Should be invalid with later source date
        assert not feature_store.is_valid(
            "EURUSD", "1H", "medium_term", source_end_date=datetime(2025, 1, 1)
        )

    def test_get_feature_names(self, feature_store, sample_ohlcv_data):
        """Test getting feature names without loading data."""
        feature_store.compute_and_store(
            sample_ohlcv_data,
            symbol="EURUSD",
            timeframe="1H",
            model_type="medium_term",
        )

        names = feature_store.get_feature_names("EURUSD", "1H", "medium_term")
        assert isinstance(names, list)
        assert len(names) > 0

    def test_get_metadata(self, feature_store, sample_ohlcv_data):
        """Test getting metadata."""
        feature_store.compute_and_store(
            sample_ohlcv_data,
            symbol="EURUSD",
            timeframe="1H",
            model_type="medium_term",
        )

        metadata = feature_store.get_metadata("EURUSD", "1H", "medium_term")
        assert metadata is not None
        assert metadata.symbol == "EURUSD"
        assert metadata.timeframe == "1H"
        assert metadata.row_count > 0

    def test_invalidate(self, feature_store, sample_ohlcv_data):
        """Test invalidating cached features."""
        feature_store.compute_and_store(
            sample_ohlcv_data,
            symbol="EURUSD",
            timeframe="1H",
            model_type="medium_term",
        )

        assert feature_store.exists("EURUSD", "1H", "medium_term")

        result = feature_store.invalidate("EURUSD", "1H", "medium_term")
        assert result is True
        assert not feature_store.exists("EURUSD", "1H", "medium_term")

    def test_clear_cache(self, feature_store, sample_ohlcv_data):
        """Test clearing in-memory cache."""
        feature_store.compute_and_store(
            sample_ohlcv_data,
            symbol="EURUSD",
            timeframe="1H",
            model_type="medium_term",
        )

        assert len(feature_store._cache) > 0

        feature_store.clear_cache()
        assert len(feature_store._cache) == 0

    def test_list_stored_features(self, feature_store, sample_ohlcv_data):
        """Test listing all stored features."""
        feature_store.compute_and_store(
            sample_ohlcv_data,
            symbol="EURUSD",
            timeframe="1H",
            model_type="medium_term",
        )
        feature_store.compute_and_store(
            sample_ohlcv_data,
            symbol="GBPUSD",
            timeframe="4H",
            model_type="short_term",
        )

        stored = feature_store.list_stored_features()
        assert len(stored) == 2
        symbols = [s["symbol"] for s in stored]
        assert "EURUSD" in symbols
        assert "GBPUSD" in symbols

    def test_get_storage_info(self, feature_store, sample_ohlcv_data):
        """Test getting storage statistics."""
        feature_store.compute_and_store(
            sample_ohlcv_data,
            symbol="EURUSD",
            timeframe="1H",
            model_type="medium_term",
        )

        info = feature_store.get_storage_info()
        assert info["total_feature_sets"] == 1
        assert info["total_size_bytes"] > 0
        assert info["cache_entries"] == 1

    def test_lru_cache_eviction(self, temp_store_path, sample_ohlcv_data):
        """Test LRU cache eviction when at capacity."""
        store = FeatureStore(store_path=temp_store_path, max_cache_entries=2)

        # Add 3 items to trigger eviction
        store.compute_and_store(sample_ohlcv_data, "SYM1", "1H", "medium_term")
        store.compute_and_store(sample_ohlcv_data, "SYM2", "1H", "medium_term")
        store.compute_and_store(sample_ohlcv_data, "SYM3", "1H", "medium_term")

        # Should only have 2 items in cache
        assert len(store._cache) == 2

    def test_case_insensitive_symbol(self, feature_store, sample_ohlcv_data):
        """Test symbol handling is case insensitive."""
        feature_store.compute_and_store(
            sample_ohlcv_data,
            symbol="eurusd",  # lowercase
            timeframe="1h",  # lowercase
            model_type="medium_term",
        )

        # Should find with uppercase
        assert feature_store.exists("EURUSD", "1H", "medium_term")

    def test_metadata_json_roundtrip(self, feature_store, sample_ohlcv_data):
        """Test metadata is correctly saved and loaded."""
        feature_store.compute_and_store(
            sample_ohlcv_data,
            symbol="EURUSD",
            timeframe="1H",
            model_type="medium_term",
        )

        # Clear cache to force reload from disk
        feature_store.clear_cache()

        metadata = feature_store.get_metadata("EURUSD", "1H", "medium_term")
        assert metadata is not None
        assert isinstance(metadata.start_date, datetime)
        assert isinstance(metadata.end_date, datetime)
        assert isinstance(metadata.created_at, datetime)
