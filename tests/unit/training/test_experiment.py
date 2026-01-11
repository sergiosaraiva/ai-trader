"""Tests for experiment management module."""

import pytest
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.training.experiment import (
    ExperimentConfig,
    ExperimentResult,
    ExperimentManager,
)
from src.training.config import TrainingConfig
from src.training.trainer import Trainer


class TestExperimentConfig:
    """Tests for ExperimentConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ExperimentConfig()

        assert config.name == "default_experiment"
        assert config.description == ""
        assert config.tracking_uri == "mlruns"
        assert config.seed == 42
        assert config.tags == {}
        assert config.log_artifacts is True
        assert config.log_models is True

    def test_custom_config(self):
        """Test custom configuration values."""
        config = ExperimentConfig(
            name="my_experiment",
            description="Test experiment",
            tracking_uri="http://localhost:5000",
            seed=123,
            tags={"version": "1.0"},
            log_artifacts=False,
        )

        assert config.name == "my_experiment"
        assert config.description == "Test experiment"
        assert config.tracking_uri == "http://localhost:5000"
        assert config.seed == 123
        assert config.tags == {"version": "1.0"}
        assert config.log_artifacts is False


class TestExperimentResult:
    """Tests for ExperimentResult dataclass."""

    def test_full_result(self):
        """Test result with all fields."""
        result = ExperimentResult(
            run_id="abc123",
            experiment_id="exp1",
            status="completed",
            metrics={"loss": 0.1, "accuracy": 0.9},
            best_epoch=10,
            best_val_loss=0.1,
            model_path="/models/test",
            duration_seconds=120.5,
            config={"lr": 0.001},
        )

        assert result.run_id == "abc123"
        assert result.experiment_id == "exp1"
        assert result.status == "completed"
        assert result.metrics["loss"] == 0.1
        assert result.best_epoch == 10
        assert result.best_val_loss == 0.1
        assert result.model_path == "/models/test"
        assert result.duration_seconds == 120.5

    def test_to_dict(self):
        """Test conversion to dictionary."""
        result = ExperimentResult(
            run_id="run1",
            experiment_id="exp1",
            status="completed",
            metrics={"loss": 0.5},
            best_epoch=5,
            best_val_loss=0.5,
        )

        data = result.to_dict()

        assert data["run_id"] == "run1"
        assert data["experiment_id"] == "exp1"
        assert data["status"] == "completed"
        assert data["metrics"]["loss"] == 0.5
        assert data["best_epoch"] == 5


class TestExperimentManager:
    """Tests for ExperimentManager class."""

    @pytest.fixture
    def mock_trainer(self):
        """Create a mock trainer."""
        trainer = MagicMock(spec=Trainer)
        trainer.fit.return_value = {
            "best_val_loss": 0.1,
            "final_train_loss": 0.05,
            "best_epoch": 8,
        }
        trainer.model = MagicMock()
        trainer.model.get_num_parameters.return_value = 1000
        trainer.config = TrainingConfig(epochs=10, batch_size=32)
        return trainer

    @pytest.fixture
    def sample_loaders(self):
        """Create sample data loaders."""
        X = torch.randn(100, 50, 10)  # (samples, seq_len, features)
        y = torch.randint(0, 2, (100,)).float()

        dataset = TensorDataset(X, y)
        train_loader = DataLoader(dataset, batch_size=32)
        val_loader = DataLoader(dataset, batch_size=32)

        return train_loader, val_loader

    def test_init_without_mlflow(self):
        """Test initialization without MLflow."""
        manager = ExperimentManager(use_mlflow=False)

        assert manager.use_mlflow is False
        assert manager.config is not None
        assert manager.config.name == "default_experiment"

    def test_init_with_config(self):
        """Test initialization with custom config."""
        config = ExperimentConfig(
            name="test_exp",
            seed=123,
        )
        manager = ExperimentManager(config=config, use_mlflow=False)

        assert manager.config.name == "test_exp"
        assert manager.config.seed == 123

    def test_run_experiment_without_mlflow(self, mock_trainer, sample_loaders):
        """Test running experiment without MLflow tracking."""
        manager = ExperimentManager(use_mlflow=False)
        train_loader, val_loader = sample_loaders

        result = manager.run_experiment(
            trainer=mock_trainer,
            train_loader=train_loader,
            val_loader=val_loader,
            training_config=TrainingConfig(),
            run_name="test_run",
        )

        assert isinstance(result, ExperimentResult)
        assert result.status == "completed"
        assert "best_val_loss" in result.metrics
        mock_trainer.fit.assert_called_once()

    def test_run_experiment_failed(self, sample_loaders):
        """Test experiment that fails."""
        manager = ExperimentManager(use_mlflow=False)
        train_loader, val_loader = sample_loaders

        # Create a trainer that raises an exception
        mock_trainer = MagicMock(spec=Trainer)
        mock_trainer.fit.side_effect = RuntimeError("Training failed")
        mock_trainer.model = MagicMock()
        mock_trainer.model.get_num_parameters.return_value = 100
        mock_trainer.config = TrainingConfig()

        result = manager.run_experiment(
            trainer=mock_trainer,
            train_loader=train_loader,
            val_loader=val_loader,
            training_config=TrainingConfig(),
        )

        assert result.status == "failed"

    def test_set_seeds(self):
        """Test seed setting for reproducibility."""
        manager = ExperimentManager(use_mlflow=False)

        # Set seeds
        manager._set_seeds(42)

        # Generate some random numbers
        np_val1 = np.random.rand()
        torch_val1 = torch.rand(1).item()

        # Reset seeds and regenerate
        manager._set_seeds(42)
        np_val2 = np.random.rand()
        torch_val2 = torch.rand(1).item()

        assert np_val1 == np_val2
        assert torch_val1 == torch_val2

    def test_generate_run_name(self):
        """Test run name generation."""
        manager = ExperimentManager(use_mlflow=False)
        config = TrainingConfig(name="test_model")

        run_name = manager._generate_run_name(config)

        assert "test_model" in run_name
        # Should contain timestamp pattern
        assert "_20" in run_name  # Year prefix


class TestExperimentManagerIntegration:
    """Integration tests for ExperimentManager."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def simple_model(self):
        """Create a simple model for testing."""
        class SimpleModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = torch.nn.Linear(10, 1)

            def forward(self, x):
                # x: (batch, seq, features)
                return {"price": self.fc(x[:, -1, :])}

            def get_num_parameters(self):
                return sum(p.numel() for p in self.parameters())

        return SimpleModel()

    def test_full_experiment_flow(self, temp_dir, simple_model):
        """Test complete experiment workflow without MLflow."""
        # Create sample data
        X = torch.randn(64, 20, 10)
        y = torch.randn(64, 1)
        dataset = TensorDataset(X, y)
        train_loader = DataLoader(dataset, batch_size=16)
        val_loader = DataLoader(dataset, batch_size=16)

        # Create mock trainer
        trainer = MagicMock()
        trainer.model = simple_model
        trainer.fit.return_value = {"best_val_loss": 0.5, "best_epoch": 2}
        trainer.config = TrainingConfig(epochs=2)

        # Run experiment
        config = ExperimentConfig(
            name="integration_test",
            seed=42,
        )
        manager = ExperimentManager(config=config, use_mlflow=False)

        result = manager.run_experiment(
            trainer=trainer,
            train_loader=train_loader,
            val_loader=val_loader,
            training_config=TrainingConfig(epochs=2),
        )

        assert result.status == "completed"
        assert result.metrics["best_val_loss"] == 0.5

    def test_reproducibility_with_seeds(self, simple_model):
        """Test that experiments are reproducible with same seed."""
        # Create sample data
        X = torch.randn(32, 10, 10)
        y = torch.randn(32, 1)
        dataset = TensorDataset(X, y)
        train_loader = DataLoader(dataset, batch_size=8)
        val_loader = DataLoader(dataset, batch_size=8)

        config = ExperimentConfig(name="repro_test", seed=42)
        manager = ExperimentManager(config=config, use_mlflow=False)

        # Run two experiments with same seed
        results = []
        for _ in range(2):
            trainer = MagicMock()
            trainer.model = simple_model
            trainer.fit.return_value = {"best_val_loss": 0.3, "best_epoch": 1}
            trainer.config = TrainingConfig(epochs=1)

            result = manager.run_experiment(
                trainer=trainer,
                train_loader=train_loader,
                val_loader=val_loader,
                training_config=TrainingConfig(epochs=1, seed=42),
            )
            results.append(result)

        # Both should complete successfully
        assert all(r.status == "completed" for r in results)
