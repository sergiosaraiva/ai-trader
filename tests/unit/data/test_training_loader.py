"""Unit tests for TrainingDataLoader."""

import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

from src.data.loaders.training_loader import (
    TrainingDataLoader,
    TradingDataset,
    LabelGenerator,
    DataLoaderConfig,
    LabelMethod,
    create_dataloaders,
)


@pytest.fixture
def sample_feature_data():
    """Create sample feature data with OHLCV and derived features."""
    dates = pd.date_range("2024-01-01", periods=500, freq="1h")
    np.random.seed(42)
    base_price = 1.1
    prices = base_price + np.cumsum(np.random.randn(500) * 0.001)

    return pd.DataFrame(
        {
            "open": prices,
            "high": prices + np.random.rand(500) * 0.005,
            "low": prices - np.random.rand(500) * 0.005,
            "close": prices + np.random.randn(500) * 0.002,
            "volume": np.random.randint(100, 1000, 500).astype(float),
            "rsi_14": np.random.uniform(20, 80, 500),
            "sma_20": prices + np.random.randn(500) * 0.001,
            "macd": np.random.randn(500) * 0.001,
            "atr_14": np.random.rand(500) * 0.01,
        },
        index=dates,
    )


@pytest.fixture
def default_config():
    """Create default DataLoaderConfig."""
    return DataLoaderConfig(
        sequence_length=50,
        prediction_horizon=1,
        batch_size=16,
        train_ratio=0.7,
        val_ratio=0.15,
    )


@pytest.fixture
def training_loader(default_config):
    """Create TrainingDataLoader instance."""
    return TrainingDataLoader(default_config)


class TestDataLoaderConfig:
    """Tests for DataLoaderConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = DataLoaderConfig()
        assert config.sequence_length == 100
        assert config.prediction_horizon == 1
        assert config.batch_size == 64
        assert config.train_ratio == 0.7
        assert config.val_ratio == 0.15
        assert config.label_method == LabelMethod.DIRECTION

    def test_custom_config(self):
        """Test custom configuration."""
        config = DataLoaderConfig(
            sequence_length=200,
            batch_size=128,
            label_method=LabelMethod.TRIPLE_BARRIER,
        )
        assert config.sequence_length == 200
        assert config.batch_size == 128
        assert config.label_method == LabelMethod.TRIPLE_BARRIER


class TestLabelMethod:
    """Tests for LabelMethod enum."""

    def test_label_methods_exist(self):
        """Test all label methods are defined."""
        assert LabelMethod.DIRECTION.value == "direction"
        assert LabelMethod.DIRECTION_THREE.value == "direction_three"
        assert LabelMethod.RETURNS.value == "returns"
        assert LabelMethod.LOG_RETURNS.value == "log_returns"
        assert LabelMethod.TRIPLE_BARRIER.value == "triple_barrier"


class TestLabelGenerator:
    """Tests for LabelGenerator."""

    def test_direction_labels(self, sample_feature_data, default_config):
        """Test binary direction label generation."""
        generator = LabelGenerator(default_config)
        labels = generator.generate(sample_feature_data, LabelMethod.DIRECTION)

        assert len(labels) == len(sample_feature_data)
        assert labels.isin([0.0, 1.0, np.nan]).all()

    def test_direction_three_labels(self, sample_feature_data, default_config):
        """Test ternary direction label generation."""
        default_config.target_threshold = 0.001
        generator = LabelGenerator(default_config)
        labels = generator.generate(sample_feature_data, LabelMethod.DIRECTION_THREE)

        assert len(labels) == len(sample_feature_data)
        valid_labels = labels.dropna()
        assert valid_labels.isin([-1, 0, 1]).all()

    def test_return_labels(self, sample_feature_data, default_config):
        """Test continuous return label generation."""
        generator = LabelGenerator(default_config)
        labels = generator.generate(sample_feature_data, LabelMethod.RETURNS)

        assert len(labels) == len(sample_feature_data)
        # Returns should be continuous values
        non_nan = labels.dropna()
        assert len(non_nan) > 0

    def test_log_return_labels(self, sample_feature_data, default_config):
        """Test log return label generation."""
        generator = LabelGenerator(default_config)
        labels = generator.generate(sample_feature_data, LabelMethod.LOG_RETURNS)

        assert len(labels) == len(sample_feature_data)

    def test_triple_barrier_labels(self, sample_feature_data, default_config):
        """Test triple barrier label generation."""
        default_config.prediction_horizon = 10
        generator = LabelGenerator(default_config)
        labels = generator.generate(sample_feature_data, LabelMethod.TRIPLE_BARRIER)

        assert len(labels) == len(sample_feature_data)
        assert labels.isin([-1, 0, 1]).all()

    def test_missing_label_column(self, sample_feature_data, default_config):
        """Test error when label column is missing."""
        default_config.label_column = "nonexistent"
        generator = LabelGenerator(default_config)

        with pytest.raises(ValueError, match="not in DataFrame"):
            generator.generate(sample_feature_data)

    def test_direction_with_threshold(self, sample_feature_data, default_config):
        """Test direction labels with threshold."""
        default_config.target_threshold = 0.01  # 1%
        generator = LabelGenerator(default_config)
        labels = generator.generate(sample_feature_data, LabelMethod.DIRECTION)

        # With high threshold, should have more 0s
        assert len(labels.dropna()) > 0


class TestTradingDataset:
    """Tests for TradingDataset."""

    def test_dataset_creation(self):
        """Test dataset creation."""
        features = np.random.randn(100, 10)
        labels = np.random.randint(0, 2, 100).astype(float)

        dataset = TradingDataset(features, labels, sequence_length=10)

        assert len(dataset) == 90  # 100 - 10

    def test_getitem(self):
        """Test getting individual samples."""
        features = np.random.randn(100, 10)
        labels = np.random.randint(0, 2, 100).astype(float)

        dataset = TradingDataset(features, labels, sequence_length=10)

        sequence, label = dataset[0]

        assert isinstance(sequence, torch.Tensor)
        assert isinstance(label, torch.Tensor)
        assert sequence.shape == (10, 10)  # (seq_len, n_features)

    def test_tensor_types(self):
        """Test that outputs are float tensors."""
        features = np.random.randn(100, 10)
        labels = np.random.randint(0, 2, 100).astype(float)

        dataset = TradingDataset(features, labels, sequence_length=10)
        sequence, label = dataset[0]

        assert sequence.dtype == torch.float32
        assert label.dtype == torch.float32


class TestTrainingDataLoader:
    """Tests for TrainingDataLoader."""

    def test_init_default(self):
        """Test default initialization."""
        loader = TrainingDataLoader()
        assert loader.config.sequence_length == 100

    def test_init_with_config(self, default_config):
        """Test initialization with config."""
        loader = TrainingDataLoader(default_config)
        assert loader.config.sequence_length == 50

    def test_create_dataloaders(self, training_loader, sample_feature_data):
        """Test creating DataLoaders."""
        train, val, test = training_loader.create_dataloaders(sample_feature_data)

        assert isinstance(train, DataLoader)
        assert isinstance(val, DataLoader)
        assert isinstance(test, DataLoader)

    def test_create_dataloaders_returns_datasets(self, training_loader, sample_feature_data):
        """Test creating DataLoaders with datasets."""
        train, val, test, train_ds, val_ds, test_ds = training_loader.create_dataloaders(
            sample_feature_data, return_datasets=True
        )

        assert isinstance(train_ds, TradingDataset)
        assert isinstance(val_ds, TradingDataset)
        assert isinstance(test_ds, TradingDataset)

    def test_chronological_split(self, training_loader, sample_feature_data):
        """Test that splits are chronological (no data leakage)."""
        training_loader.create_dataloaders(sample_feature_data)

        dates = training_loader.split_dates
        assert dates["train_end"] < dates["val_start"]
        assert dates["val_end"] < dates["test_start"]

    def test_split_ratios(self, training_loader, sample_feature_data):
        """Test that split ratios are approximately correct."""
        training_loader.create_dataloaders(sample_feature_data)

        total = training_loader.train_size + training_loader.val_size + training_loader.test_size
        # Check ratios are roughly correct (accounting for sequence_length reduction)
        assert training_loader.train_size > training_loader.val_size
        assert training_loader.train_size > training_loader.test_size

    def test_batch_shapes(self, training_loader, sample_feature_data):
        """Test batch shapes from DataLoader."""
        train_loader, _, _ = training_loader.create_dataloaders(sample_feature_data)

        batch_x, batch_y = next(iter(train_loader))

        assert batch_x.shape[0] == training_loader.config.batch_size
        assert batch_x.shape[1] == training_loader.config.sequence_length
        assert batch_y.shape[0] == training_loader.config.batch_size

    def test_normalization_zscore(self, sample_feature_data):
        """Test z-score normalization."""
        config = DataLoaderConfig(
            sequence_length=20,
            batch_size=16,
            normalization="zscore",
        )
        loader = TrainingDataLoader(config)
        loader.create_dataloaders(sample_feature_data)

        assert "mean" in loader.scalers
        assert "std" in loader.scalers

    def test_normalization_minmax(self, sample_feature_data):
        """Test min-max normalization."""
        config = DataLoaderConfig(
            sequence_length=20,
            batch_size=16,
            normalization="minmax",
        )
        loader = TrainingDataLoader(config)
        loader.create_dataloaders(sample_feature_data)

        assert "min" in loader.scalers
        assert "max" in loader.scalers

    def test_normalization_robust(self, sample_feature_data):
        """Test robust normalization."""
        config = DataLoaderConfig(
            sequence_length=20,
            batch_size=16,
            normalization="robust",
        )
        loader = TrainingDataLoader(config)
        loader.create_dataloaders(sample_feature_data)

        assert "median" in loader.scalers
        assert "iqr" in loader.scalers

    def test_normalize_method(self, training_loader, sample_feature_data):
        """Test normalize method after fitting."""
        training_loader.create_dataloaders(sample_feature_data)

        features = np.random.randn(10, len(training_loader.feature_names))
        normalized = training_loader.normalize(features)

        assert normalized.shape == features.shape

    def test_denormalize_method(self, training_loader, sample_feature_data):
        """Test denormalize method."""
        training_loader.create_dataloaders(sample_feature_data)

        features = np.random.randn(10, len(training_loader.feature_names))
        normalized = training_loader.normalize(features)
        denormalized = training_loader.denormalize(normalized)

        np.testing.assert_array_almost_equal(features, denormalized, decimal=5)

    def test_save_and_load_scalers(self, training_loader, sample_feature_data):
        """Test saving and loading scalers."""
        training_loader.create_dataloaders(sample_feature_data)

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            path = Path(f.name)

        try:
            training_loader.save_scalers(path)
            assert path.exists()

            # Create new loader and load scalers
            new_loader = TrainingDataLoader(DataLoaderConfig(sequence_length=50))
            new_loader.load_scalers(path)

            assert new_loader.scalers == training_loader.scalers
            assert new_loader.feature_names == training_loader.feature_names
        finally:
            path.unlink()

    def test_get_info(self, training_loader, sample_feature_data):
        """Test get_info method."""
        training_loader.create_dataloaders(sample_feature_data)

        info = training_loader.get_info()

        assert info["sequence_length"] == 50
        assert info["batch_size"] == 16
        assert info["train_size"] > 0
        assert len(info["feature_names"]) > 0

    def test_feature_column_selection(self, sample_feature_data):
        """Test selecting specific feature columns."""
        config = DataLoaderConfig(
            sequence_length=20,
            batch_size=16,
            feature_columns=["rsi_14", "sma_20", "macd"],
        )
        loader = TrainingDataLoader(config)
        loader.create_dataloaders(sample_feature_data)

        assert len(loader.feature_names) == 3

    def test_drop_ohlcv(self, sample_feature_data):
        """Test dropping OHLCV columns."""
        config = DataLoaderConfig(
            sequence_length=20,
            batch_size=16,
            drop_ohlcv=True,
        )
        loader = TrainingDataLoader(config)
        loader.create_dataloaders(sample_feature_data)

        assert "open" not in loader.feature_names
        assert "close" not in loader.feature_names
        assert "volume" not in loader.feature_names


class TestConvenienceFunction:
    """Tests for create_dataloaders convenience function."""

    def test_create_dataloaders_function(self, sample_feature_data):
        """Test convenience function."""
        train, val, test, loader = create_dataloaders(
            sample_feature_data,
            sequence_length=30,
            batch_size=16,
        )

        assert isinstance(train, DataLoader)
        assert isinstance(val, DataLoader)
        assert isinstance(test, DataLoader)
        assert isinstance(loader, TrainingDataLoader)

    def test_create_dataloaders_with_label_method(self, sample_feature_data):
        """Test convenience function with label method."""
        train, val, test, loader = create_dataloaders(
            sample_feature_data,
            sequence_length=30,
            batch_size=16,
            label_method="returns",
        )

        assert loader.config.label_method == LabelMethod.RETURNS
