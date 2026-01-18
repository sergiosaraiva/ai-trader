"""Unit tests for training configuration classes."""

import tempfile
from pathlib import Path

import pytest

from src.training.config import (
    CheckpointConfig,
    EarlyStoppingConfig,
    OptimizerConfig,
    OptimizerType,
    SchedulerConfig,
    SchedulerType,
    TrainingConfig,
)


class TestOptimizerConfig:
    """Tests for OptimizerConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = OptimizerConfig()
        assert config.optimizer_type == OptimizerType.ADAMW
        assert config.learning_rate == 1e-4
        assert config.weight_decay == 1e-5
        assert config.betas == (0.9, 0.999)
        assert config.momentum == 0.9
        assert config.eps == 1e-8

    def test_custom_values(self):
        """Test custom configuration."""
        config = OptimizerConfig(
            optimizer_type=OptimizerType.SGD,
            learning_rate=0.01,
            weight_decay=0.001,
            momentum=0.95,
        )
        assert config.optimizer_type == OptimizerType.SGD
        assert config.learning_rate == 0.01
        assert config.weight_decay == 0.001
        assert config.momentum == 0.95

    def test_to_dict(self):
        """Test conversion to dictionary."""
        config = OptimizerConfig()
        d = config.to_dict()
        assert d["optimizer_type"] == "adamw"
        assert d["learning_rate"] == 1e-4
        assert "betas" in d
        assert "momentum" in d


class TestSchedulerConfig:
    """Tests for SchedulerConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = SchedulerConfig()
        assert config.scheduler_type == SchedulerType.COSINE
        assert config.step_size == 10
        assert config.gamma == 0.1
        assert config.T_max == 100
        assert config.eta_min == 1e-7
        assert config.patience == 10

    def test_custom_values(self):
        """Test custom configuration."""
        config = SchedulerConfig(
            scheduler_type=SchedulerType.REDUCE_ON_PLATEAU,
            patience=5,
            factor=0.1,
        )
        assert config.scheduler_type == SchedulerType.REDUCE_ON_PLATEAU
        assert config.patience == 5
        assert config.factor == 0.1

    def test_one_cycle_config(self):
        """Test OneCycle-specific parameters."""
        config = SchedulerConfig(
            scheduler_type=SchedulerType.ONE_CYCLE,
            pct_start=0.2,
            div_factor=10.0,
            final_div_factor=1000.0,
        )
        assert config.scheduler_type == SchedulerType.ONE_CYCLE
        assert config.pct_start == 0.2
        assert config.div_factor == 10.0
        assert config.final_div_factor == 1000.0

    def test_to_dict(self):
        """Test conversion to dictionary."""
        config = SchedulerConfig()
        d = config.to_dict()
        assert d["scheduler_type"] == "cosine"
        assert "T_max" in d
        assert "eta_min" in d


class TestEarlyStoppingConfig:
    """Tests for EarlyStoppingConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = EarlyStoppingConfig()
        assert config.enabled is True
        assert config.patience == 15
        assert config.min_delta == 1e-5
        assert config.monitor == "val_loss"
        assert config.mode == "min"
        assert config.restore_best_weights is True
        assert config.baseline is None
        assert config.start_from_epoch == 0

    def test_custom_values(self):
        """Test custom configuration."""
        config = EarlyStoppingConfig(
            enabled=False,
            patience=10,
            min_delta=0.001,
            monitor="val_accuracy",
            mode="max",
        )
        assert config.enabled is False
        assert config.patience == 10
        assert config.min_delta == 0.001
        assert config.monitor == "val_accuracy"
        assert config.mode == "max"

    def test_baseline_config(self):
        """Test baseline configuration."""
        config = EarlyStoppingConfig(
            baseline=0.5,
            start_from_epoch=5,
        )
        assert config.baseline == 0.5
        assert config.start_from_epoch == 5

    def test_to_dict(self):
        """Test conversion to dictionary."""
        config = EarlyStoppingConfig()
        d = config.to_dict()
        assert d["enabled"] is True
        assert d["patience"] == 15
        assert d["monitor"] == "val_loss"


class TestCheckpointConfig:
    """Tests for CheckpointConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = CheckpointConfig()
        assert config.enabled is True
        assert config.save_dir == "checkpoints"
        assert config.save_best_only is True
        assert config.save_freq == "epoch"
        assert config.monitor == "val_loss"
        assert config.mode == "min"
        assert config.max_to_keep == 3
        assert config.save_weights_only is False

    def test_custom_values(self):
        """Test custom configuration."""
        config = CheckpointConfig(
            enabled=True,
            save_dir="/tmp/models",
            save_best_only=False,
            save_freq=5,
            max_to_keep=10,
        )
        assert config.save_dir == "/tmp/models"
        assert config.save_best_only is False
        assert config.save_freq == 5
        assert config.max_to_keep == 10

    def test_to_dict(self):
        """Test conversion to dictionary."""
        config = CheckpointConfig()
        d = config.to_dict()
        assert d["enabled"] is True
        assert "save_dir" in d
        assert "max_to_keep" in d


class TestTrainingConfig:
    """Tests for TrainingConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = TrainingConfig()
        assert config.name == "training_run"
        assert config.batch_size == 64
        assert config.epochs == 100
        assert config.gradient_clip == 1.0
        assert config.accumulation_steps == 1
        assert config.mixed_precision is False
        assert config.seed == 42
        assert config.device == "auto"
        assert config.verbose == 1

    def test_custom_values(self):
        """Test custom configuration."""
        config = TrainingConfig(
            name="test_run",
            batch_size=32,
            epochs=50,
            gradient_clip=0.5,
            device="cpu",
        )
        assert config.name == "test_run"
        assert config.batch_size == 32
        assert config.epochs == 50
        assert config.gradient_clip == 0.5
        assert config.device == "cpu"

    def test_nested_configs(self):
        """Test nested configuration objects."""
        config = TrainingConfig(
            optimizer=OptimizerConfig(learning_rate=0.001),
            scheduler=SchedulerConfig(scheduler_type=SchedulerType.STEP),
            early_stopping=EarlyStoppingConfig(patience=5),
            checkpoint=CheckpointConfig(max_to_keep=5),
        )
        assert config.optimizer.learning_rate == 0.001
        assert config.scheduler.scheduler_type == SchedulerType.STEP
        assert config.early_stopping.patience == 5
        assert config.checkpoint.max_to_keep == 5

    def test_post_init_dict_conversion(self):
        """Test that dictionaries are converted to config objects."""
        config = TrainingConfig(
            optimizer={"learning_rate": 0.001},
            scheduler={"scheduler_type": SchedulerType.STEP},
            early_stopping={"patience": 5},
            checkpoint={"max_to_keep": 5},
        )
        assert isinstance(config.optimizer, OptimizerConfig)
        assert isinstance(config.scheduler, SchedulerConfig)
        assert isinstance(config.early_stopping, EarlyStoppingConfig)
        assert isinstance(config.checkpoint, CheckpointConfig)

    def test_to_dict(self):
        """Test conversion to dictionary."""
        config = TrainingConfig()
        d = config.to_dict()
        assert d["name"] == "training_run"
        assert d["batch_size"] == 64
        assert isinstance(d["optimizer"], dict)
        assert isinstance(d["scheduler"], dict)
        assert isinstance(d["early_stopping"], dict)
        assert isinstance(d["checkpoint"], dict)

    def test_from_dict(self):
        """Test creation from dictionary."""
        data = {
            "name": "from_dict_test",
            "batch_size": 128,
            "epochs": 200,
            "optimizer": {"learning_rate": 0.01},
            "scheduler": {"scheduler_type": "step"},
        }
        config = TrainingConfig.from_dict(data)
        assert config.name == "from_dict_test"
        assert config.batch_size == 128
        assert config.epochs == 200

    def test_yaml_round_trip(self):
        """Test saving and loading from YAML."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_path = Path(tmpdir) / "config.yaml"

            # Save
            config = TrainingConfig(
                name="yaml_test",
                batch_size=128,
                epochs=50,
            )
            config.save_yaml(yaml_path)

            # Load
            loaded = TrainingConfig.from_yaml(yaml_path)
            assert loaded.name == "yaml_test"
            assert loaded.batch_size == 128
            assert loaded.epochs == 50

    def test_get_device_cpu(self):
        """Test device resolution for CPU."""
        config = TrainingConfig(device="cpu")
        assert config.get_device() == "cpu"

    def test_get_device_auto(self):
        """Test auto device resolution."""
        config = TrainingConfig(device="auto")
        device = config.get_device()
        # Should return one of the valid devices
        assert device in ["cpu", "cuda", "mps"]

    def test_architecture_config(self):
        """Test architecture-specific configuration."""
        config = TrainingConfig(
            architecture_config={
                "hidden_dim": 256,
                "num_layers": 4,
                "attention_heads": 8,
            }
        )
        assert config.architecture_config["hidden_dim"] == 256
        assert config.architecture_config["num_layers"] == 4


class TestSchedulerTypes:
    """Tests for scheduler type enum."""

    def test_all_scheduler_types(self):
        """Test all scheduler types exist."""
        assert SchedulerType.NONE.value == "none"
        assert SchedulerType.STEP.value == "step"
        assert SchedulerType.COSINE.value == "cosine"
        assert SchedulerType.COSINE_WARM_RESTARTS.value == "cosine_warm_restarts"
        assert SchedulerType.ONE_CYCLE.value == "one_cycle"
        assert SchedulerType.REDUCE_ON_PLATEAU.value == "reduce_on_plateau"
        assert SchedulerType.EXPONENTIAL.value == "exponential"


class TestOptimizerTypes:
    """Tests for optimizer type enum."""

    def test_all_optimizer_types(self):
        """Test all optimizer types exist."""
        assert OptimizerType.ADAM.value == "adam"
        assert OptimizerType.ADAMW.value == "adamw"
        assert OptimizerType.SGD.value == "sgd"
        assert OptimizerType.RMSPROP.value == "rmsprop"
        assert OptimizerType.RADAM.value == "radam"
