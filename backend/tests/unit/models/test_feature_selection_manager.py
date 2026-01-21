"""Tests for FeatureSelectionManager.

Tests caching, config hash computation, and multi-timeframe management.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.models.feature_selection.manager import FeatureSelectionManager
from src.models.feature_selection.rfecv_config import RFECVConfig


@pytest.fixture
def minimal_data():
    """Generate minimal data for testing."""
    np.random.seed(42)
    n_samples = 50
    n_features = 10

    X = np.random.randn(n_samples, n_features)
    y = np.random.choice([0, 1], size=n_samples)
    feature_names = [f"feat_{i}" for i in range(n_features)]

    return X, y, feature_names


@pytest.fixture
def temp_cache_dir():
    """Create temporary directory for caching tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


class TestFeatureSelectionManager:
    """Tests for FeatureSelectionManager."""

    def test_initialization_default_config(self):
        """Test manager initialization with default config."""
        manager = FeatureSelectionManager()

        assert manager.config is not None
        assert isinstance(manager.config, RFECVConfig)
        assert isinstance(manager.cache_dir, Path)
        assert manager.selections == {}

    def test_initialization_custom_config(self, temp_cache_dir):
        """Test manager initialization with custom config."""
        config = RFECVConfig(
            cache_dir=temp_cache_dir,
            cache_enabled=True,
        )
        manager = FeatureSelectionManager(config)

        assert manager.config.cache_dir == temp_cache_dir
        assert manager.config.cache_enabled is True
        assert manager.cache_dir == Path(temp_cache_dir)

    def test_cache_directory_creation(self, temp_cache_dir):
        """Test that cache directory is created if enabled."""
        cache_path = Path(temp_cache_dir) / "feature_selections"
        config = RFECVConfig(
            cache_dir=str(cache_path),
            cache_enabled=True,
        )

        # Cache dir should not exist yet
        assert not cache_path.exists()

        # Initialize manager
        manager = FeatureSelectionManager(config)

        # Cache dir should now exist
        assert cache_path.exists()
        assert cache_path.is_dir()

    def test_cache_directory_not_created_when_disabled(self, temp_cache_dir):
        """Test that cache directory is not created if caching disabled."""
        cache_path = Path(temp_cache_dir) / "feature_selections"
        config = RFECVConfig(
            cache_dir=str(cache_path),
            cache_enabled=False,
        )

        # Initialize manager with caching disabled
        manager = FeatureSelectionManager(config)

        # Cache dir should not be created
        assert not cache_path.exists()

    def test_compute_config_hash_deterministic(self):
        """Test that config hash is deterministic."""
        config = RFECVConfig(
            step=0.1,
            min_features_to_select=20,
            cv=5,
        )
        manager = FeatureSelectionManager(config)

        hash1 = manager._compute_config_hash()
        hash2 = manager._compute_config_hash()

        # Same config should produce same hash
        assert hash1 == hash2
        assert len(hash1) == 8  # MD5 truncated to 8 chars

    def test_compute_config_hash_different_for_different_configs(self):
        """Test that different configs produce different hashes."""
        config1 = RFECVConfig(step=0.1)
        config2 = RFECVConfig(step=0.2)

        manager1 = FeatureSelectionManager(config1)
        manager2 = FeatureSelectionManager(config2)

        hash1 = manager1._compute_config_hash()
        hash2 = manager2._compute_config_hash()

        # Different configs should produce different hashes
        assert hash1 != hash2

    def test_get_cache_path(self, temp_cache_dir):
        """Test cache path generation."""
        config = RFECVConfig(cache_dir=temp_cache_dir)
        manager = FeatureSelectionManager(config)

        cache_path = manager._get_cache_path("1H")

        # Verify path format
        assert cache_path.parent == Path(temp_cache_dir)
        assert cache_path.name.startswith("1H_rfecv_")
        assert cache_path.suffix == ".json"

    def test_get_cache_path_different_timeframes(self, temp_cache_dir):
        """Test that different timeframes get different cache paths."""
        config = RFECVConfig(cache_dir=temp_cache_dir)
        manager = FeatureSelectionManager(config)

        cache_path_1h = manager._get_cache_path("1H")
        cache_path_4h = manager._get_cache_path("4H")
        cache_path_d = manager._get_cache_path("D")

        # All paths should be different
        assert cache_path_1h != cache_path_4h
        assert cache_path_1h != cache_path_d
        assert cache_path_4h != cache_path_d

    @patch("src.models.feature_selection.manager.RFECVSelector")
    def test_select_features_saves_to_cache(
        self, mock_selector_class, minimal_data, temp_cache_dir
    ):
        """Test that selection results are saved to cache."""
        X, y, feature_names = minimal_data

        # Mock RFECVSelector
        mock_selector = MagicMock()
        mock_selector.fit.return_value = (
            ["feat_0", "feat_2", "feat_4"],
            np.array([0, 2, 4]),
        )
        mock_selector.cv_scores = {
            "cv_scores_mean": [0.7],
            "cv_scores_std": [0.05],
            "n_features": [3],
            "optimal_n_features": 3,
        }
        mock_selector_class.return_value = mock_selector

        # Run selection with caching enabled
        config = RFECVConfig(cache_dir=temp_cache_dir, cache_enabled=True)
        manager = FeatureSelectionManager(config)

        manager.select_features("1H", X, y, feature_names)

        # Verify cache file was created
        cache_path = manager._get_cache_path("1H")
        assert cache_path.exists()

        # Verify cache content
        with open(cache_path, "r") as f:
            cached = json.load(f)

        assert cached["timeframe"] == "1H"
        assert cached["selected_features"] == ["feat_0", "feat_2", "feat_4"]
        assert cached["selected_indices"] == [0, 2, 4]
        assert cached["n_original_features"] == len(feature_names)
        assert cached["n_selected_features"] == 3

    @patch("src.models.feature_selection.manager.RFECVSelector")
    def test_select_features_loads_from_cache(
        self, mock_selector_class, minimal_data, temp_cache_dir
    ):
        """Test that cached selections are loaded instead of recomputing."""
        X, y, feature_names = minimal_data

        # Create cache file manually
        config = RFECVConfig(cache_dir=temp_cache_dir, cache_enabled=True)
        manager = FeatureSelectionManager(config)

        cache_path = manager._get_cache_path("1H")
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        cached_data = {
            "timeframe": "1H",
            "selected_features": ["feat_0", "feat_2", "feat_4"],
            "selected_indices": [0, 2, 4],
            "cv_scores": {
                "cv_scores_mean": [0.7],
                "cv_scores_std": [0.05],
                "n_features": [3],
                "optimal_n_features": 3,
            },
            "config_hash": manager._compute_config_hash(),
            "n_original_features": 10,
            "n_selected_features": 3,
        }

        with open(cache_path, "w") as f:
            json.dump(cached_data, f)

        # Run selection - should load from cache
        selected_features, selected_indices, cv_scores = manager.select_features(
            "1H", X, y, feature_names
        )

        # Verify selector was NOT called (cache was used)
        assert not mock_selector_class.called

        # Verify cached data was returned
        assert selected_features == ["feat_0", "feat_2", "feat_4"]
        assert list(selected_indices) == [0, 2, 4]
        assert cv_scores["optimal_n_features"] == 3

    @patch("src.models.feature_selection.manager.RFECVSelector")
    def test_select_features_force_recompute(
        self, mock_selector_class, minimal_data, temp_cache_dir
    ):
        """Test that force_recompute ignores cache."""
        X, y, feature_names = minimal_data

        # Create cache file
        config = RFECVConfig(cache_dir=temp_cache_dir, cache_enabled=True)
        manager = FeatureSelectionManager(config)

        cache_path = manager._get_cache_path("1H")
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        cached_data = {
            "timeframe": "1H",
            "selected_features": ["feat_0"],
            "selected_indices": [0],
            "cv_scores": {"optimal_n_features": 1},
            "config_hash": manager._compute_config_hash(),
            "n_original_features": 10,
            "n_selected_features": 1,
        }

        with open(cache_path, "w") as f:
            json.dump(cached_data, f)

        # Mock selector with different result
        mock_selector = MagicMock()
        mock_selector.fit.return_value = (
            ["feat_1", "feat_2", "feat_3"],
            np.array([1, 2, 3]),
        )
        mock_selector.cv_scores = {
            "cv_scores_mean": [0.8],
            "cv_scores_std": [0.03],
            "n_features": [3],
            "optimal_n_features": 3,
        }
        mock_selector_class.return_value = mock_selector

        # Run with force_recompute=True
        selected_features, selected_indices, cv_scores = manager.select_features(
            "1H", X, y, feature_names, force_recompute=True
        )

        # Verify selector WAS called (cache was ignored)
        assert mock_selector_class.called

        # Verify new result was returned (not cached)
        assert selected_features == ["feat_1", "feat_2", "feat_3"]
        assert list(selected_indices) == [1, 2, 3]

    @patch("src.models.feature_selection.manager.RFECVSelector")
    def test_select_features_no_cache_when_disabled(
        self, mock_selector_class, minimal_data, temp_cache_dir
    ):
        """Test that caching is skipped when disabled."""
        X, y, feature_names = minimal_data

        # Mock selector
        mock_selector = MagicMock()
        mock_selector.fit.return_value = (
            ["feat_0", "feat_2"],
            np.array([0, 2]),
        )
        mock_selector.cv_scores = {"optimal_n_features": 2}
        mock_selector_class.return_value = mock_selector

        # Run with caching disabled
        config = RFECVConfig(cache_dir=temp_cache_dir, cache_enabled=False)
        manager = FeatureSelectionManager(config)

        manager.select_features("1H", X, y, feature_names)

        # Verify cache file was NOT created
        cache_path = manager._get_cache_path("1H")
        assert not cache_path.exists()

    @patch("src.models.feature_selection.manager.RFECVSelector")
    def test_get_selection(self, mock_selector_class, minimal_data, temp_cache_dir):
        """Test get_selection retrieves stored selection."""
        X, y, feature_names = minimal_data

        # Mock selector
        mock_selector = MagicMock()
        mock_selector.fit.return_value = (
            ["feat_0"],
            np.array([0]),
        )
        mock_selector.cv_scores = {"optimal_n_features": 1}
        mock_selector_class.return_value = mock_selector

        # Run selection
        config = RFECVConfig(cache_dir=temp_cache_dir, cache_enabled=False)
        manager = FeatureSelectionManager(config)
        manager.select_features("1H", X, y, feature_names)

        # Get selection
        selection = manager.get_selection("1H")

        assert selection is not None
        assert selection["timeframe"] == "1H"
        assert selection["selected_features"] == ["feat_0"]

    def test_get_selection_not_found(self):
        """Test get_selection returns None for non-existent timeframe."""
        manager = FeatureSelectionManager()

        selection = manager.get_selection("1H")

        assert selection is None

    def test_clear_cache_specific_timeframe(self, temp_cache_dir):
        """Test clearing cache for specific timeframe."""
        config = RFECVConfig(cache_dir=temp_cache_dir, cache_enabled=True)
        manager = FeatureSelectionManager(config)

        # Create cache files
        cache_path_1h = manager._get_cache_path("1H")
        cache_path_4h = manager._get_cache_path("4H")

        cache_path_1h.parent.mkdir(parents=True, exist_ok=True)
        cache_path_1h.write_text("{}")
        cache_path_4h.write_text("{}")

        assert cache_path_1h.exists()
        assert cache_path_4h.exists()

        # Clear only 1H cache
        manager.clear_cache("1H")

        # Verify 1H was cleared but 4H remains
        assert not cache_path_1h.exists()
        assert cache_path_4h.exists()

    def test_clear_cache_all_timeframes(self, temp_cache_dir):
        """Test clearing all cached timeframes."""
        config = RFECVConfig(cache_dir=temp_cache_dir, cache_enabled=True)
        manager = FeatureSelectionManager(config)

        # Create multiple cache files
        cache_path_1h = manager._get_cache_path("1H")
        cache_path_4h = manager._get_cache_path("4H")
        cache_path_d = manager._get_cache_path("D")

        cache_path_1h.parent.mkdir(parents=True, exist_ok=True)
        cache_path_1h.write_text("{}")
        cache_path_4h.write_text("{}")
        cache_path_d.write_text("{}")

        # Clear all caches
        manager.clear_cache()

        # Verify all were cleared
        assert not cache_path_1h.exists()
        assert not cache_path_4h.exists()
        assert not cache_path_d.exists()

    def test_clear_cache_disabled_warning(self, temp_cache_dir):
        """Test that clearing cache with caching disabled shows warning."""
        config = RFECVConfig(cache_dir=temp_cache_dir, cache_enabled=False)
        manager = FeatureSelectionManager(config)

        # Should not raise error, just log warning
        manager.clear_cache()

    @patch("src.models.feature_selection.manager.RFECVSelector")
    def test_select_features_multiple_timeframes(
        self, mock_selector_class, minimal_data, temp_cache_dir
    ):
        """Test selecting features for multiple timeframes."""
        X, y, feature_names = minimal_data

        # Mock selector with different results per timeframe
        def mock_fit_side_effect(X, y, feature_names):
            # Return different features based on call count
            if mock_selector_class.call_count == 1:
                return ["feat_0", "feat_1"], np.array([0, 1])
            elif mock_selector_class.call_count == 2:
                return ["feat_2", "feat_3"], np.array([2, 3])
            else:
                return ["feat_4", "feat_5"], np.array([4, 5])

        mock_selector = MagicMock()
        mock_selector.fit.side_effect = mock_fit_side_effect
        mock_selector.cv_scores = {"optimal_n_features": 2}
        mock_selector_class.return_value = mock_selector

        # Run for multiple timeframes
        config = RFECVConfig(cache_dir=temp_cache_dir, cache_enabled=True)
        manager = FeatureSelectionManager(config)

        manager.select_features("1H", X, y, feature_names)
        manager.select_features("4H", X, y, feature_names)
        manager.select_features("D", X, y, feature_names)

        # Verify all selections are stored
        assert "1H" in manager.selections
        assert "4H" in manager.selections
        assert "D" in manager.selections

        # Verify all have different cache files
        cache_1h = manager._get_cache_path("1H")
        cache_4h = manager._get_cache_path("4H")
        cache_d = manager._get_cache_path("D")

        assert cache_1h.exists()
        assert cache_4h.exists()
        assert cache_d.exists()
        assert cache_1h != cache_4h != cache_d

    def test_cache_invalidation_on_config_change(self, temp_cache_dir):
        """Test that cache is invalidated when config changes."""
        # Create cache with config1
        config1 = RFECVConfig(cache_dir=temp_cache_dir, step=0.1, cache_enabled=True)
        manager1 = FeatureSelectionManager(config1)
        cache_path1 = manager1._get_cache_path("1H")
        cache_path1.parent.mkdir(parents=True, exist_ok=True)
        cache_path1.write_text("{}")

        # Create manager with config2 (different step)
        config2 = RFECVConfig(cache_dir=temp_cache_dir, step=0.2, cache_enabled=True)
        manager2 = FeatureSelectionManager(config2)
        cache_path2 = manager2._get_cache_path("1H")

        # Cache paths should be different due to different config hash
        assert cache_path1 != cache_path2
        assert cache_path1.exists()
        assert not cache_path2.exists()

    @patch("src.models.feature_selection.manager.RFECVSelector")
    def test_select_features_stores_in_selections_dict(
        self, mock_selector_class, minimal_data, temp_cache_dir
    ):
        """Test that selections are stored in manager.selections dict."""
        X, y, feature_names = minimal_data

        # Mock selector
        mock_selector = MagicMock()
        mock_selector.fit.return_value = (
            ["feat_0", "feat_2"],
            np.array([0, 2]),
        )
        mock_selector.cv_scores = {
            "cv_scores_mean": [0.75],
            "cv_scores_std": [0.04],
            "n_features": [2],
            "optimal_n_features": 2,
        }
        mock_selector_class.return_value = mock_selector

        # Run selection
        config = RFECVConfig(cache_dir=temp_cache_dir, cache_enabled=False)
        manager = FeatureSelectionManager(config)

        manager.select_features("1H", X, y, feature_names)

        # Verify stored in selections dict
        assert "1H" in manager.selections
        selection = manager.selections["1H"]
        assert selection["timeframe"] == "1H"
        assert selection["selected_features"] == ["feat_0", "feat_2"]
        assert selection["n_selected_features"] == 2
