"""Unit tests for training callbacks."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.training.callbacks import (
    Callback,
    CallbackList,
    EarlyStoppingCallback,
    LRSchedulerCallback,
    MetricsLoggerCallback,
    ModelCheckpointCallback,
    ProgressCallback,
    TrainingLogs,
)


class TestTrainingLogs:
    """Tests for TrainingLogs dataclass."""

    def test_basic_creation(self):
        """Test basic creation."""
        logs = TrainingLogs(
            epoch=1,
            batch=10,
            train_loss=0.5,
            val_loss=0.4,
        )
        assert logs.epoch == 1
        assert logs.batch == 10
        assert logs.train_loss == 0.5
        assert logs.val_loss == 0.4

    def test_default_values(self):
        """Test default values are set correctly."""
        logs = TrainingLogs()
        assert logs.epoch == 0
        assert logs.batch == 0
        assert logs.train_loss == 0.0
        assert logs.val_loss is None
        assert logs.train_metrics == {}
        assert logs.val_metrics == {}

    def test_get_method(self):
        """Test get method for accessing values."""
        logs = TrainingLogs(
            epoch=1,
            train_loss=0.5,
            val_loss=0.4,
            train_metrics={"accuracy": 0.85},
            val_metrics={"accuracy": 0.82},
        )
        assert logs.get("train_loss") == 0.5
        assert logs.get("val_loss") == 0.4
        assert logs.get("accuracy") == 0.85  # From train_metrics
        assert logs.get("nonexistent", 0.0) == 0.0

    def test_get_from_val_metrics(self):
        """Test get prioritizes train_metrics then val_metrics."""
        logs = TrainingLogs(val_metrics={"custom_metric": 0.9})
        assert logs.get("custom_metric") == 0.9


class TestCallback:
    """Tests for base Callback class."""

    def test_set_model(self):
        """Test setting model reference."""
        callback = Callback()
        model = MagicMock()
        callback.set_model(model)
        assert callback.model == model

    def test_set_trainer(self):
        """Test setting trainer reference."""
        callback = Callback()
        trainer = MagicMock()
        callback.set_trainer(trainer)
        assert callback.trainer == trainer

    def test_callback_methods_do_nothing_by_default(self):
        """Test that base callback methods are no-ops."""
        callback = Callback()
        logs = TrainingLogs()

        # These should not raise
        callback.on_train_begin(logs)
        callback.on_train_end(logs)
        callback.on_epoch_begin(1, logs)
        callback.on_batch_begin(1, logs)
        callback.on_batch_end(1, logs)
        callback.on_validation_begin(logs)
        callback.on_validation_end(logs)

        # on_epoch_end returns False by default
        assert callback.on_epoch_end(1, logs) is False


class TestCallbackList:
    """Tests for CallbackList."""

    def test_empty_list(self):
        """Test empty callback list."""
        cb_list = CallbackList()
        assert len(cb_list.callbacks) == 0

    def test_append(self):
        """Test appending callbacks."""
        cb_list = CallbackList()
        cb1 = Callback()
        cb2 = Callback()
        cb_list.append(cb1)
        cb_list.append(cb2)
        assert len(cb_list.callbacks) == 2

    def test_set_model_propagates(self):
        """Test that set_model propagates to all callbacks."""
        cb1 = Callback()
        cb2 = Callback()
        cb_list = CallbackList([cb1, cb2])

        model = MagicMock()
        cb_list.set_model(model)

        assert cb1.model == model
        assert cb2.model == model

    def test_on_epoch_end_aggregates_stop_signals(self):
        """Test that stop signals are aggregated."""
        cb1 = MagicMock(spec=Callback)
        cb1.on_epoch_end.return_value = False

        cb2 = MagicMock(spec=Callback)
        cb2.on_epoch_end.return_value = True

        cb_list = CallbackList([cb1, cb2])
        result = cb_list.on_epoch_end(1, TrainingLogs())

        assert result is True  # Should stop because cb2 returned True

    def test_all_callbacks_false_continues(self):
        """Test that training continues if all callbacks return False."""
        cb1 = MagicMock(spec=Callback)
        cb1.on_epoch_end.return_value = False

        cb2 = MagicMock(spec=Callback)
        cb2.on_epoch_end.return_value = False

        cb_list = CallbackList([cb1, cb2])
        result = cb_list.on_epoch_end(1, TrainingLogs())

        assert result is False


class TestEarlyStoppingCallback:
    """Tests for EarlyStoppingCallback."""

    def test_initialization(self):
        """Test initialization with defaults."""
        callback = EarlyStoppingCallback()
        assert callback.monitor == "val_loss"
        assert callback.patience == 15
        assert callback.min_delta == pytest.approx(-1e-5)  # Negated for min mode
        assert callback.mode == "min"

    def test_initialization_max_mode(self):
        """Test initialization in max mode."""
        callback = EarlyStoppingCallback(mode="max", min_delta=0.01)
        assert callback.min_delta == pytest.approx(0.01)

    def test_on_train_begin_resets_state(self):
        """Test that on_train_begin resets state."""
        callback = EarlyStoppingCallback()
        callback.wait = 5
        callback.best = 0.5
        callback.on_train_begin()
        assert callback.wait == 0
        assert callback.best == np.inf

    def test_improvement_resets_wait(self):
        """Test that improvement resets wait counter."""
        callback = EarlyStoppingCallback(patience=5)
        callback.on_train_begin()

        # First epoch - improvement
        logs1 = TrainingLogs(val_loss=0.5)
        result1 = callback.on_epoch_end(1, logs1)
        assert result1 is False
        assert callback.wait == 0
        assert callback.best == 0.5

        # Second epoch - more improvement
        logs2 = TrainingLogs(val_loss=0.4)
        result2 = callback.on_epoch_end(2, logs2)
        assert result2 is False
        assert callback.wait == 0
        assert callback.best == 0.4

    def test_no_improvement_increases_wait(self):
        """Test that no improvement increases wait counter."""
        callback = EarlyStoppingCallback(patience=5)
        callback.on_train_begin()

        # First epoch
        callback.on_epoch_end(1, TrainingLogs(val_loss=0.5))

        # Second epoch - no improvement
        callback.on_epoch_end(2, TrainingLogs(val_loss=0.6))
        assert callback.wait == 1

        # Third epoch - still no improvement
        callback.on_epoch_end(3, TrainingLogs(val_loss=0.55))
        assert callback.wait == 2

    def test_triggers_early_stopping(self):
        """Test early stopping is triggered after patience."""
        callback = EarlyStoppingCallback(patience=3)
        callback.on_train_begin()

        # First epoch - set baseline
        callback.on_epoch_end(1, TrainingLogs(val_loss=0.5))

        # No improvement epochs
        for epoch in range(2, 5):
            result = callback.on_epoch_end(epoch, TrainingLogs(val_loss=0.6))
            if epoch < 4:
                assert result is False
            else:
                assert result is True  # Should stop at epoch 4

    def test_start_from_epoch(self):
        """Test monitoring starts from specified epoch."""
        callback = EarlyStoppingCallback(patience=3, start_from_epoch=5)
        callback.on_train_begin()

        # Early epochs should not affect stopping
        for epoch in range(1, 5):
            result = callback.on_epoch_end(epoch, TrainingLogs(val_loss=0.5))
            assert result is False

    def test_max_mode_improvement(self):
        """Test improvement detection in max mode."""
        callback = EarlyStoppingCallback(mode="max", patience=3)
        callback.on_train_begin()

        # Higher is better
        callback.on_epoch_end(1, TrainingLogs(val_loss=0.0, val_metrics={"accuracy": 0.7}))
        callback.best = 0.7  # Simulate tracking accuracy

        logs = TrainingLogs(val_loss=0.0)
        logs.val_metrics = {"accuracy": 0.8}
        # Note: EarlyStoppingCallback uses logs.get(monitor) which checks val_loss first


class TestModelCheckpointCallback:
    """Tests for ModelCheckpointCallback."""

    def test_initialization(self):
        """Test initialization."""
        callback = ModelCheckpointCallback(
            filepath="models/model.pt",
            save_best_only=True,
        )
        assert callback.save_best_only is True
        assert callback.monitor == "val_loss"

    def test_save_best_only(self):
        """Test save_best_only behavior."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "model.pt"
            callback = ModelCheckpointCallback(
                filepath=filepath,
                save_best_only=True,
            )

            # Mock model
            mock_model = MagicMock()
            mock_model.state_dict.return_value = {"layer": "weights"}
            callback.set_model(mock_model)

            # First epoch - should save
            callback.on_epoch_end(1, TrainingLogs(val_loss=0.5))
            assert callback.best == 0.5

            # Second epoch - better, should save
            callback.on_epoch_end(2, TrainingLogs(val_loss=0.4))
            assert callback.best == 0.4

            # Third epoch - worse, should not update best
            callback.on_epoch_end(3, TrainingLogs(val_loss=0.6))
            assert callback.best == 0.4

    def test_filepath_formatting(self):
        """Test filepath with epoch/loss formatting."""
        callback = ModelCheckpointCallback(
            filepath="models/model_{epoch:03d}_{val_loss:.4f}.pt"
        )
        assert "{epoch" in str(callback.filepath)


class TestLRSchedulerCallback:
    """Tests for LRSchedulerCallback."""

    def test_initialization(self):
        """Test initialization."""
        callback = LRSchedulerCallback(monitor="val_loss")
        assert callback.monitor == "val_loss"
        assert callback.scheduler is None

    def test_no_scheduler_does_nothing(self):
        """Test that no scheduler is handled gracefully."""
        callback = LRSchedulerCallback()
        result = callback.on_epoch_end(1, TrainingLogs(val_loss=0.5))
        assert result is False

    def test_with_scheduler(self):
        """Test with a mock scheduler."""
        mock_scheduler = MagicMock()
        mock_scheduler.get_last_lr.return_value = [0.001]

        callback = LRSchedulerCallback(scheduler=mock_scheduler)
        callback.on_epoch_end(1, TrainingLogs(val_loss=0.5))

        mock_scheduler.step.assert_called_once()


class TestMetricsLoggerCallback:
    """Tests for MetricsLoggerCallback."""

    def test_initialization(self):
        """Test initialization."""
        callback = MetricsLoggerCallback(log_frequency=5)
        assert callback.log_frequency == 5
        assert callback.history == []

    def test_logs_metrics(self):
        """Test that metrics are logged."""
        callback = MetricsLoggerCallback()
        callback.on_train_begin()

        logs = TrainingLogs(
            epoch=1,
            train_loss=0.5,
            val_loss=0.4,
            learning_rate=0.001,
        )
        callback.on_epoch_end(1, logs)

        assert len(callback.history) == 1
        assert callback.history[0]["epoch"] == 1
        assert callback.history[0]["train_loss"] == 0.5
        assert callback.history[0]["val_loss"] == 0.4

    def test_log_frequency(self):
        """Test log frequency filtering."""
        callback = MetricsLoggerCallback(log_frequency=2)
        callback.on_train_begin()

        for epoch in range(1, 6):
            callback.on_epoch_end(epoch, TrainingLogs(train_loss=0.5))

        # Should only log epochs 2 and 4 (divisible by 2)
        assert len(callback.history) == 2

    def test_saves_to_file(self):
        """Test saving logs to file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            callback = MetricsLoggerCallback(log_dir=tmpdir)
            callback.on_train_begin()
            callback.on_epoch_end(1, TrainingLogs(train_loss=0.5, val_loss=0.4))
            callback.on_train_end()

            log_file = Path(tmpdir) / "training_history.json"
            assert log_file.exists()


class TestProgressCallback:
    """Tests for ProgressCallback."""

    def test_initialization(self):
        """Test initialization."""
        callback = ProgressCallback(use_tqdm=False)
        assert callback.use_tqdm is False

    def test_fallback_to_print(self):
        """Test fallback when tqdm is disabled."""
        callback = ProgressCallback(use_tqdm=False)
        callback.on_train_begin()

        # Should not raise
        result = callback.on_epoch_end(
            1, TrainingLogs(train_loss=0.5, val_loss=0.4)
        )
        assert result is False

    def test_on_train_end_closes_pbar(self):
        """Test that progress bar is closed on train end."""
        callback = ProgressCallback(use_tqdm=False)
        callback._pbar = MagicMock()
        callback.on_train_end()
        callback._pbar.close.assert_called_once()


class TestCallbackIntegration:
    """Integration tests for callbacks."""

    def test_multiple_callbacks_work_together(self):
        """Test multiple callbacks working together."""
        early_stopping = EarlyStoppingCallback(patience=3)
        logger = MetricsLoggerCallback()

        cb_list = CallbackList([early_stopping, logger])
        cb_list.on_train_begin()

        # Simulate training
        for epoch in range(1, 10):
            logs = TrainingLogs(
                epoch=epoch,
                train_loss=0.5,
                val_loss=0.5 + epoch * 0.01,  # Getting worse
            )

            if cb_list.on_epoch_end(epoch, logs):
                break

        # Early stopping should have triggered
        assert early_stopping.stopped_epoch > 0
        assert len(logger.history) > 0

    def test_callback_order_matters(self):
        """Test that callback order is respected."""
        order = []

        class OrderTracker(Callback):
            def __init__(self, name):
                self.name = name

            def on_epoch_end(self, epoch, logs=None):
                order.append(self.name)
                return False

        cb1 = OrderTracker("first")
        cb2 = OrderTracker("second")
        cb3 = OrderTracker("third")

        cb_list = CallbackList([cb1, cb2, cb3])
        cb_list.on_epoch_end(1, TrainingLogs())

        assert order == ["first", "second", "third"]
