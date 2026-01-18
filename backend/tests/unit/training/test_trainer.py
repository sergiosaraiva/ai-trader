"""Unit tests for training orchestrator."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.training.config import (
    EarlyStoppingConfig,
    OptimizerConfig,
    OptimizerType,
    SchedulerConfig,
    SchedulerType,
    TrainingConfig,
)
from src.training.trainer import MultiTaskLoss, Trainer


@pytest.fixture
def sample_data():
    """Create sample training data."""
    np.random.seed(42)
    torch.manual_seed(42)

    seq_length = 50
    input_dim = 30
    num_samples = 64

    # Generate random data
    X = np.random.randn(num_samples, seq_length, input_dim).astype(np.float32)
    # Single target value per sample (simpler for testing)
    y = np.random.randn(num_samples, 1).astype(np.float32)

    return X, y


@pytest.fixture
def data_loaders(sample_data):
    """Create train and validation data loaders."""
    X, y = sample_data

    # Split into train/val
    train_size = int(0.8 * len(X))

    train_dataset = TensorDataset(
        torch.FloatTensor(X[:train_size]),
        torch.FloatTensor(y[:train_size]),
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X[train_size:]),
        torch.FloatTensor(y[train_size:]),
    )

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    return train_loader, val_loader


@pytest.fixture
def small_config():
    """Create small config for fast testing."""
    return TrainingConfig(
        name="test_run",
        batch_size=8,
        epochs=3,
        verbose=0,
        device="cpu",
        early_stopping=EarlyStoppingConfig(enabled=False),
        optimizer=OptimizerConfig(learning_rate=0.001),
        scheduler=SchedulerConfig(scheduler_type=SchedulerType.NONE),
    )


class TestMultiTaskLoss:
    """Tests for MultiTaskLoss."""

    def test_initialization(self):
        """Test initialization."""
        loss_fn = MultiTaskLoss()
        assert loss_fn.price_weight == 1.0
        assert loss_fn.direction_weight == 1.0
        assert loss_fn.confidence_weight == 0.5

    def test_custom_weights(self):
        """Test custom weight initialization."""
        loss_fn = MultiTaskLoss(
            price_weight=2.0,
            direction_weight=0.5,
            confidence_weight=1.0,
        )
        assert loss_fn.price_weight == 2.0
        assert loss_fn.direction_weight == 0.5
        assert loss_fn.confidence_weight == 1.0

    def test_price_loss_only(self):
        """Test loss with only price predictions."""
        loss_fn = MultiTaskLoss(use_uncertainty_weighting=False)

        predictions = {"price": torch.randn(8, 2)}
        targets = {"price": torch.randn(8, 2)}

        total_loss, components = loss_fn(predictions, targets)

        assert isinstance(total_loss, torch.Tensor)
        assert "price" in components
        assert "total" in components

    def test_direction_loss(self):
        """Test loss with direction predictions."""
        loss_fn = MultiTaskLoss(use_uncertainty_weighting=False)

        predictions = {
            "price": torch.randn(8, 2),
            "direction_logits": torch.randn(8, 2, 3),  # 2 horizons, 3 classes
        }
        targets = {
            "price": torch.randn(8, 2),
            "direction": torch.randint(0, 3, (8, 2)),
        }

        total_loss, components = loss_fn(predictions, targets)

        assert "direction" in components
        assert components["direction"] > 0

    def test_confidence_loss(self):
        """Test loss with confidence (Beta) predictions."""
        loss_fn = MultiTaskLoss(use_uncertainty_weighting=False)

        predictions = {
            "price": torch.randn(8, 2),
            "alpha": torch.ones(8, 2) * 2,
            "beta": torch.ones(8, 2) * 2,
        }
        targets = {
            "price": torch.randn(8, 2),
            "direction": torch.randint(-1, 2, (8,)),
        }

        total_loss, components = loss_fn(predictions, targets)

        assert "confidence" in components

    def test_uncertainty_weighting(self):
        """Test uncertainty-weighted loss."""
        loss_fn = MultiTaskLoss(use_uncertainty_weighting=True)

        predictions = {
            "price": torch.randn(8, 2),
            "direction_logits": torch.randn(8, 2, 3),
            "alpha": torch.ones(8, 2) * 2,
            "beta": torch.ones(8, 2) * 2,
        }
        targets = {
            "price": torch.randn(8, 2),
            "direction": torch.randint(0, 3, (8, 2)),
        }

        total_loss, components = loss_fn(predictions, targets)

        assert isinstance(total_loss, torch.Tensor)
        assert total_loss.requires_grad


class TestTrainer:
    """Tests for Trainer."""

    def test_initialization_with_string(self, small_config):
        """Test initialization with architecture name."""
        trainer = Trainer(
            architecture="cnn_lstm_attention",
            config=small_config,
            input_dim=30,
            sequence_length=50,
            hidden_dim=32,
            prediction_horizons=[1],
        )

        assert trainer.model is not None
        assert trainer.optimizer is not None
        assert trainer.device == torch.device("cpu")

    def test_initialization_with_model(self, small_config):
        """Test initialization with model instance."""
        from src.training.architectures import ArchitectureRegistry

        model = ArchitectureRegistry.create(
            "cnn_lstm_attention",
            input_dim=30,
            sequence_length=50,
            hidden_dim=32,
        )

        trainer = Trainer(
            architecture=model,
            config=small_config,
        )

        assert trainer.model is model

    def test_optimizer_creation(self, small_config):
        """Test different optimizers are created."""
        for opt_type in [OptimizerType.ADAM, OptimizerType.ADAMW, OptimizerType.SGD]:
            config = TrainingConfig(
                **{**small_config.to_dict(), "optimizer": {"optimizer_type": opt_type}}
            )
            config.early_stopping.enabled = False

            trainer = Trainer(
                architecture="cnn_lstm_attention",
                config=config,
                input_dim=30,
                sequence_length=50,
                hidden_dim=32,
            )

            assert trainer.optimizer is not None

    def test_scheduler_creation(self, small_config):
        """Test scheduler creation."""
        config = TrainingConfig(
            **{
                **small_config.to_dict(),
                "scheduler": {"scheduler_type": SchedulerType.STEP, "step_size": 5},
            }
        )
        config.early_stopping.enabled = False

        trainer = Trainer(
            architecture="cnn_lstm_attention",
            config=config,
            input_dim=30,
            sequence_length=50,
            hidden_dim=32,
        )

        assert trainer.scheduler is not None
        assert isinstance(trainer.scheduler, torch.optim.lr_scheduler.StepLR)

    def test_fit_basic(self, small_config, data_loaders):
        """Test basic training loop."""
        train_loader, val_loader = data_loaders

        trainer = Trainer(
            architecture="cnn_lstm_attention",
            config=small_config,
            input_dim=30,
            sequence_length=50,
            hidden_dim=32,
            prediction_horizons=[1],
        )

        results = trainer.fit(train_loader, val_loader)

        assert "status" in results
        assert "current_epoch" in results
        assert results["current_epoch"] >= 1

    def test_fit_without_validation(self, small_config, data_loaders):
        """Test training without validation data."""
        train_loader, _ = data_loaders

        trainer = Trainer(
            architecture="cnn_lstm_attention",
            config=small_config,
            input_dim=30,
            sequence_length=50,
            hidden_dim=32,
            prediction_horizons=[1],
        )

        results = trainer.fit(train_loader)

        assert "status" in results

    def test_predict(self, small_config, data_loaders):
        """Test prediction."""
        train_loader, val_loader = data_loaders

        trainer = Trainer(
            architecture="cnn_lstm_attention",
            config=small_config,
            input_dim=30,
            sequence_length=50,
            hidden_dim=32,
            prediction_horizons=[1],
        )

        # Train briefly
        trainer.fit(train_loader, val_loader)

        # Predict with numpy array
        X = np.random.randn(4, 50, 30).astype(np.float32)
        predictions = trainer.predict(X)

        assert isinstance(predictions, dict)
        assert "price" in predictions
        assert predictions["price"].shape[0] == 4

    def test_predict_with_tensor(self, small_config, data_loaders):
        """Test prediction with tensor input."""
        train_loader, _ = data_loaders

        trainer = Trainer(
            architecture="cnn_lstm_attention",
            config=small_config,
            input_dim=30,
            sequence_length=50,
            hidden_dim=32,
            prediction_horizons=[1],
        )

        trainer.fit(train_loader)

        # Predict with tensor
        X = torch.randn(4, 50, 30)
        predictions = trainer.predict(X)

        assert isinstance(predictions, dict)
        assert all(isinstance(v, np.ndarray) for v in predictions.values())

    def test_gradient_clipping(self, small_config, data_loaders):
        """Test gradient clipping is applied."""
        train_loader, _ = data_loaders

        config = TrainingConfig(
            **{**small_config.to_dict(), "gradient_clip": 1.0}
        )
        config.early_stopping.enabled = False

        trainer = Trainer(
            architecture="cnn_lstm_attention",
            config=config,
            input_dim=30,
            sequence_length=50,
            hidden_dim=32,
        )

        # Train should work without errors
        results = trainer.fit(train_loader)
        assert results is not None

    def test_callbacks_are_called(self, small_config, data_loaders):
        """Test that callbacks are called during training."""
        train_loader, val_loader = data_loaders

        # Create mock callback
        mock_callback = MagicMock()
        mock_callback.on_epoch_end.return_value = False

        trainer = Trainer(
            architecture="cnn_lstm_attention",
            config=small_config,
            input_dim=30,
            sequence_length=50,
            hidden_dim=32,
            callbacks=[mock_callback],
        )

        trainer.fit(train_loader, val_loader)

        mock_callback.on_train_begin.assert_called()
        mock_callback.on_epoch_begin.assert_called()
        mock_callback.on_epoch_end.assert_called()
        mock_callback.on_train_end.assert_called()


class TestTrainerEarlyStopping:
    """Tests for early stopping in Trainer."""

    def test_early_stopping_triggers(self, data_loaders):
        """Test that early stopping triggers when val_loss doesn't improve."""
        train_loader, val_loader = data_loaders

        config = TrainingConfig(
            name="early_stop_test",
            batch_size=8,
            epochs=100,  # Long enough to trigger early stopping
            verbose=0,
            device="cpu",
            early_stopping=EarlyStoppingConfig(
                enabled=True,
                patience=2,
                min_delta=1e-10,  # Very small to ensure "no improvement"
            ),
            scheduler=SchedulerConfig(scheduler_type=SchedulerType.NONE),
        )

        trainer = Trainer(
            architecture="cnn_lstm_attention",
            config=config,
            input_dim=30,
            sequence_length=50,
            hidden_dim=16,  # Small model
            prediction_horizons=[1],
        )

        results = trainer.fit(train_loader, val_loader)

        # Should stop early (not complete all 100 epochs)
        assert results["current_epoch"] < 100

    def test_early_stopping_disabled(self, data_loaders):
        """Test training completes when early stopping is disabled."""
        train_loader, val_loader = data_loaders

        config = TrainingConfig(
            name="no_early_stop",
            batch_size=8,
            epochs=3,
            verbose=0,
            device="cpu",
            early_stopping=EarlyStoppingConfig(enabled=False),
            scheduler=SchedulerConfig(scheduler_type=SchedulerType.NONE),
        )

        trainer = Trainer(
            architecture="cnn_lstm_attention",
            config=config,
            input_dim=30,
            sequence_length=50,
            hidden_dim=16,
        )

        results = trainer.fit(train_loader, val_loader)

        assert results["current_epoch"] == 3


class TestTrainerSaveLoad:
    """Tests for Trainer save/load functionality."""

    def test_save_and_load(self, small_config, data_loaders):
        """Test saving and loading trainer."""
        train_loader, val_loader = data_loaders

        trainer = Trainer(
            architecture="cnn_lstm_attention",
            config=small_config,
            input_dim=30,
            sequence_length=50,
            hidden_dim=32,
            prediction_horizons=[1],
        )

        trainer.fit(train_loader, val_loader)

        # Get prediction before save
        X = np.random.randn(4, 50, 30).astype(np.float32)
        pred_before = trainer.predict(X)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "model"
            trainer.save(save_path)

            # Load in new trainer
            loaded_trainer = Trainer.load(save_path)

            # Predictions should be identical
            pred_after = loaded_trainer.predict(X)

            assert np.allclose(
                pred_before["price"],
                pred_after["price"],
                rtol=1e-5,
            )


class TestTrainerIntegration:
    """Integration tests for complete training workflows."""

    def test_full_training_workflow(self):
        """Test complete training workflow from scratch."""
        # Generate data
        np.random.seed(42)
        torch.manual_seed(42)

        n_samples = 100
        seq_length = 50
        input_dim = 30

        X = np.random.randn(n_samples, seq_length, input_dim).astype(np.float32)
        y = np.random.randn(n_samples, 1).astype(np.float32)  # Single target

        # Create data loaders
        train_dataset = TensorDataset(
            torch.FloatTensor(X[:80]),
            torch.FloatTensor(y[:80]),
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(X[80:]),
            torch.FloatTensor(y[80:]),
        )

        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

        # Create config
        config = TrainingConfig(
            name="integration_test",
            batch_size=16,
            epochs=5,
            verbose=0,
            device="cpu",
            early_stopping=EarlyStoppingConfig(enabled=True, patience=3),
            optimizer=OptimizerConfig(learning_rate=0.001),
            scheduler=SchedulerConfig(scheduler_type=SchedulerType.COSINE, T_max=5),
        )

        # Create and train
        trainer = Trainer(
            architecture="cnn_lstm_attention",
            config=config,
            input_dim=input_dim,
            sequence_length=seq_length,
            hidden_dim=32,
            prediction_horizons=[1],
        )

        results = trainer.fit(train_loader, val_loader)

        # Verify results
        assert results["status"] in ["completed", "stopped_early"]
        assert results["best_epoch"] >= 1
        assert "best_val_loss" in results

        # Verify predictions work
        predictions = trainer.predict(X[:4])
        assert "price" in predictions
        assert predictions["price"].shape == (4, 1)  # 4 samples, 1 horizon

    def test_training_with_different_architectures(self, data_loaders):
        """Test training with different architectures."""
        train_loader, val_loader = data_loaders

        config = TrainingConfig(
            name="multi_arch_test",
            batch_size=8,
            epochs=2,
            verbose=0,
            device="cpu",
            early_stopping=EarlyStoppingConfig(enabled=False),
            scheduler=SchedulerConfig(scheduler_type=SchedulerType.NONE),
        )

        for arch_name in ["cnn_lstm_attention", "tft", "nbeats"]:
            trainer = Trainer(
                architecture=arch_name,
                config=config,
                input_dim=30,
                sequence_length=50,
                hidden_dim=32,
                prediction_horizons=[1],
            )

            results = trainer.fit(train_loader, val_loader)
            assert results["current_epoch"] == 2
