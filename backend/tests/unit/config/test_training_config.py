"""Unit tests for training configuration."""

import pytest
from src.config.training_config import (
    DataSplitParameters,
    StackingParameters,
    EarlyStoppingParameters,
    TrainingParameters,
)


class TestDataSplitParameters:
    """Test data split parameters."""

    def test_defaults(self):
        """Test data split parameters load with correct defaults."""
        config = DataSplitParameters()

        assert config.train_ratio == 0.6
        assert config.validation_ratio == 0.2
        assert config.test_ratio == 0.2
        assert config.enforce_chronological is True

    def test_override(self):
        """Test data split parameters can be overridden."""
        config = DataSplitParameters()
        config.train_ratio = 0.7
        config.validation_ratio = 0.15

        assert config.train_ratio == 0.7
        assert config.validation_ratio == 0.15

    def test_to_dict(self):
        """Test conversion to dictionary."""
        config = DataSplitParameters()
        result = config.to_dict()

        assert isinstance(result, dict)
        assert "train_ratio" in result
        assert result["train_ratio"] == 0.6


class TestStackingParameters:
    """Test stacking meta-learner parameters."""

    def test_defaults(self):
        """Test stacking parameters load with correct defaults."""
        config = StackingParameters()

        assert config.n_folds == 5
        assert config.min_train_size == 500
        assert config.shuffle is False
        assert config.stratified is True
        assert config.use_base_hyperparams is True
        assert config.custom_hyperparams == {}

    def test_override(self):
        """Test stacking parameters can be overridden."""
        config = StackingParameters()
        config.n_folds = 10
        config.min_train_size = 1000

        assert config.n_folds == 10
        assert config.min_train_size == 1000

    def test_custom_hyperparams(self):
        """Test custom hyperparameters can be set."""
        config = StackingParameters()
        config.custom_hyperparams = {"max_depth": 4, "learning_rate": 0.01}

        assert config.custom_hyperparams["max_depth"] == 4
        assert config.custom_hyperparams["learning_rate"] == 0.01

    def test_to_dict(self):
        """Test conversion to dictionary."""
        config = StackingParameters()
        result = config.to_dict()

        assert isinstance(result, dict)
        assert "n_folds" in result
        assert result["n_folds"] == 5


class TestEarlyStoppingParameters:
    """Test early stopping parameters."""

    def test_defaults(self):
        """Test early stopping parameters load with correct defaults."""
        config = EarlyStoppingParameters()

        assert config.enabled is True
        assert config.stopping_rounds == 10
        assert config.eval_metric == "logloss"
        assert config.verbose is False

    def test_override(self):
        """Test early stopping parameters can be overridden."""
        config = EarlyStoppingParameters()
        config.enabled = False
        config.stopping_rounds = 20

        assert config.enabled is False
        assert config.stopping_rounds == 20

    def test_to_dict(self):
        """Test conversion to dictionary."""
        config = EarlyStoppingParameters()
        result = config.to_dict()

        assert isinstance(result, dict)
        assert "enabled" in result
        assert result["stopping_rounds"] == 10


class TestTrainingParameters:
    """Test complete training parameters wrapper."""

    def test_defaults(self):
        """Test training parameters load with correct structure."""
        config = TrainingParameters()

        assert isinstance(config.splits, DataSplitParameters)
        assert isinstance(config.stacking, StackingParameters)
        assert isinstance(config.early_stopping, EarlyStoppingParameters)

    def test_nested_access(self):
        """Test nested parameter access works."""
        config = TrainingParameters()

        assert config.splits.train_ratio == 0.6
        assert config.stacking.n_folds == 5
        assert config.early_stopping.stopping_rounds == 10

    def test_to_dict(self):
        """Test conversion to dictionary."""
        config = TrainingParameters()
        result = config.to_dict()

        assert isinstance(result, dict)
        assert "splits" in result
        assert "stacking" in result
        assert "early_stopping" in result
        assert isinstance(result["splits"], dict)
