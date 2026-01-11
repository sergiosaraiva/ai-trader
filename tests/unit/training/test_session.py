"""Unit tests for training session management."""

import json
import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from src.training.session import (
    EpochMetrics,
    SessionStatus,
    TrainingSession,
    TrainingState,
)


class TestEpochMetrics:
    """Tests for EpochMetrics."""

    def test_basic_creation(self):
        """Test basic creation of epoch metrics."""
        metrics = EpochMetrics(
            epoch=1,
            train_loss=0.5,
            val_loss=0.4,
        )
        assert metrics.epoch == 1
        assert metrics.train_loss == 0.5
        assert metrics.val_loss == 0.4

    def test_default_values(self):
        """Test default values."""
        metrics = EpochMetrics(epoch=1, train_loss=0.5)
        assert metrics.val_loss is None
        assert metrics.train_metrics == {}
        assert metrics.val_metrics == {}
        assert metrics.learning_rate == 0.0
        assert metrics.duration_seconds == 0.0
        assert isinstance(metrics.timestamp, datetime)

    def test_with_additional_metrics(self):
        """Test with additional metrics."""
        metrics = EpochMetrics(
            epoch=1,
            train_loss=0.5,
            val_loss=0.4,
            train_metrics={"accuracy": 0.85},
            val_metrics={"accuracy": 0.82},
            learning_rate=0.001,
            duration_seconds=10.5,
        )
        assert metrics.train_metrics["accuracy"] == 0.85
        assert metrics.val_metrics["accuracy"] == 0.82
        assert metrics.learning_rate == 0.001
        assert metrics.duration_seconds == 10.5

    def test_to_dict(self):
        """Test conversion to dictionary."""
        metrics = EpochMetrics(
            epoch=1,
            train_loss=0.5,
            val_loss=0.4,
        )
        d = metrics.to_dict()
        assert d["epoch"] == 1
        assert d["train_loss"] == 0.5
        assert d["val_loss"] == 0.4
        assert "timestamp" in d

    def test_from_dict(self):
        """Test creation from dictionary."""
        data = {
            "epoch": 1,
            "train_loss": 0.5,
            "val_loss": 0.4,
            "train_metrics": {},
            "val_metrics": {},
            "learning_rate": 0.001,
            "duration_seconds": 10.0,
            "timestamp": "2024-01-01T00:00:00",
        }
        metrics = EpochMetrics.from_dict(data)
        assert metrics.epoch == 1
        assert metrics.train_loss == 0.5
        assert isinstance(metrics.timestamp, datetime)


class TestTrainingState:
    """Tests for TrainingState."""

    def test_default_values(self):
        """Test default values."""
        state = TrainingState()
        assert state.current_epoch == 0
        assert state.total_epochs == 100
        assert state.best_epoch == 0
        assert state.best_val_loss == float("inf")
        assert state.epochs_without_improvement == 0
        assert state.global_step == 0
        assert state.history == []
        assert state.status == SessionStatus.PENDING

    def test_update_best_improvement(self):
        """Test update_best with improvement."""
        state = TrainingState()
        result = state.update_best(epoch=1, val_loss=0.5)
        assert result is True
        assert state.best_epoch == 1
        assert state.best_val_loss == 0.5
        assert state.epochs_without_improvement == 0

    def test_update_best_no_improvement(self):
        """Test update_best without improvement."""
        state = TrainingState()
        state.update_best(epoch=1, val_loss=0.5)
        result = state.update_best(epoch=2, val_loss=0.6)
        assert result is False
        assert state.best_epoch == 1
        assert state.best_val_loss == 0.5
        assert state.epochs_without_improvement == 1

    def test_update_best_max_mode(self):
        """Test update_best in max mode."""
        state = TrainingState()
        state.best_val_metric = -float("inf")
        result = state.update_best(epoch=1, val_loss=0.0, val_metric=0.8, mode="max")
        assert result is True
        assert state.best_val_metric == 0.8

        result = state.update_best(epoch=2, val_loss=0.0, val_metric=0.7, mode="max")
        assert result is False

    def test_should_stop_early(self):
        """Test early stopping check."""
        state = TrainingState()
        state.epochs_without_improvement = 5
        assert state.should_stop_early(patience=10) is False
        assert state.should_stop_early(patience=5) is True
        assert state.should_stop_early(patience=3) is True

    def test_get_metric_history(self):
        """Test getting metric history."""
        state = TrainingState()
        state.history = [
            EpochMetrics(epoch=1, train_loss=0.5, val_loss=0.4),
            EpochMetrics(epoch=2, train_loss=0.4, val_loss=0.35),
            EpochMetrics(epoch=3, train_loss=0.35, val_loss=0.3),
        ]
        train_losses = state.get_metric_history("train_loss")
        assert train_losses == [0.5, 0.4, 0.35]
        val_losses = state.get_metric_history("val_loss")
        assert val_losses == [0.4, 0.35, 0.3]

    def test_get_best_metrics(self):
        """Test getting best epoch metrics."""
        state = TrainingState()
        state.history = [
            EpochMetrics(epoch=1, train_loss=0.5, val_loss=0.4),
            EpochMetrics(epoch=2, train_loss=0.4, val_loss=0.35),
        ]
        state.best_epoch = 2
        best = state.get_best_metrics()
        assert best["epoch"] == 2
        assert best["val_loss"] == 0.35

    def test_to_dict(self):
        """Test conversion to dictionary."""
        state = TrainingState()
        state.current_epoch = 5
        state.best_epoch = 3
        d = state.to_dict()
        assert d["current_epoch"] == 5
        assert d["best_epoch"] == 3
        assert d["status"] == "pending"

    def test_from_dict(self):
        """Test creation from dictionary."""
        data = {
            "current_epoch": 5,
            "total_epochs": 50,
            "best_epoch": 3,
            "best_val_loss": 0.3,
            "best_val_metric": 0.3,
            "monitor_metric": "val_loss",
            "epochs_without_improvement": 2,
            "global_step": 100,
            "history": [],
            "status": "running",
            "stop_reason": None,
            "started_at": "2024-01-01T00:00:00",
            "ended_at": None,
            "total_duration_seconds": 0.0,
        }
        state = TrainingState.from_dict(data)
        assert state.current_epoch == 5
        assert state.total_epochs == 50
        assert state.status == SessionStatus.RUNNING


class TestSessionStatus:
    """Tests for SessionStatus enum."""

    def test_all_status_values(self):
        """Test all status values exist."""
        assert SessionStatus.PENDING.value == "pending"
        assert SessionStatus.RUNNING.value == "running"
        assert SessionStatus.PAUSED.value == "paused"
        assert SessionStatus.COMPLETED.value == "completed"
        assert SessionStatus.STOPPED_EARLY.value == "stopped_early"
        assert SessionStatus.FAILED.value == "failed"


class TestTrainingSession:
    """Tests for TrainingSession."""

    def test_initialization(self):
        """Test basic initialization."""
        session = TrainingSession(
            name="test_session",
            total_epochs=50,
        )
        assert session.name == "test_session"
        assert session.state.total_epochs == 50
        assert session.state.status == SessionStatus.PENDING

    def test_start_session(self):
        """Test starting a session."""
        session = TrainingSession(name="test")
        session.start()
        assert session.state.status == SessionStatus.RUNNING
        assert session.state.started_at is not None

    def test_end_session_completed(self):
        """Test ending a session normally."""
        session = TrainingSession(name="test")
        session.start()
        session.end()
        assert session.state.status == SessionStatus.COMPLETED
        assert session.state.ended_at is not None

    def test_end_session_early_stopped(self):
        """Test ending a session with early stopping."""
        session = TrainingSession(name="test")
        session.start()
        session.end(reason="Early stopped after 10 epochs")
        assert session.state.status == SessionStatus.STOPPED_EARLY
        assert "Early stopped" in session.state.stop_reason

    def test_record_epoch(self):
        """Test recording epoch metrics."""
        session = TrainingSession(name="test")
        session.start()
        session.start_epoch()
        is_best = session.record_epoch(
            epoch=1,
            train_loss=0.5,
            val_loss=0.4,
            learning_rate=0.001,
        )
        assert is_best is True
        assert len(session.state.history) == 1
        assert session.state.current_epoch == 1

    def test_record_epoch_improvement_tracking(self):
        """Test improvement tracking across epochs."""
        session = TrainingSession(name="test")
        session.start()

        # First epoch - improvement
        session.record_epoch(epoch=1, train_loss=0.5, val_loss=0.5)
        assert session.state.best_epoch == 1

        # Second epoch - improvement
        session.record_epoch(epoch=2, train_loss=0.4, val_loss=0.4)
        assert session.state.best_epoch == 2

        # Third epoch - no improvement
        session.record_epoch(epoch=3, train_loss=0.35, val_loss=0.45)
        assert session.state.best_epoch == 2
        assert session.state.epochs_without_improvement == 1

    def test_should_stop_max_epochs(self):
        """Test stopping at max epochs."""
        session = TrainingSession(name="test", total_epochs=5)
        session.state.current_epoch = 5
        assert session.should_stop(patience=10) is True

    def test_should_stop_early_stopping(self):
        """Test early stopping trigger."""
        session = TrainingSession(name="test", total_epochs=100)
        session.state.epochs_without_improvement = 15
        assert session.should_stop(patience=15) is True
        assert session.should_stop(patience=20) is False

    def test_get_stop_reason_completed(self):
        """Test stop reason for completed training."""
        session = TrainingSession(name="test", total_epochs=10)
        session.state.current_epoch = 10
        reason = session.get_stop_reason(patience=15)
        assert "Completed all 10 epochs" in reason

    def test_get_stop_reason_early_stopped(self):
        """Test stop reason for early stopping."""
        session = TrainingSession(name="test", total_epochs=100)
        session.state.epochs_without_improvement = 15
        session.state.best_epoch = 10
        reason = session.get_stop_reason(patience=15)
        assert "Early stopped" in reason
        assert "Best epoch: 10" in reason

    def test_save_and_load_state(self):
        """Test saving and loading session state."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create and populate session
            session = TrainingSession(
                name="test_session",
                total_epochs=50,
                checkpoint_dir=tmpdir,
            )
            session.start()
            session.record_epoch(epoch=1, train_loss=0.5, val_loss=0.4)
            session.record_epoch(epoch=2, train_loss=0.4, val_loss=0.35)

            # Save state
            state_path = session.save_state()
            assert state_path.exists()

            # Load in new session
            new_session = TrainingSession(
                name="new_session",
                total_epochs=50,
                resume_from=state_path,
            )
            assert new_session.state.current_epoch == 2
            assert len(new_session.state.history) == 2

    def test_get_summary(self):
        """Test getting session summary."""
        session = TrainingSession(name="test", total_epochs=100)
        session.start()
        session.record_epoch(epoch=1, train_loss=0.5, val_loss=0.4)

        summary = session.get_summary()
        assert summary["name"] == "test"
        assert summary["current_epoch"] == 1
        assert summary["total_epochs"] == 100
        assert summary["best_epoch"] == 1
        assert summary["final_train_loss"] == 0.5
        assert summary["final_val_loss"] == 0.4

    def test_monitor_max_mode(self):
        """Test monitoring in max mode (for accuracy)."""
        session = TrainingSession(
            name="test",
            total_epochs=100,
            monitor="val_accuracy",
            mode="max",
        )
        session.start()

        # Higher is better in max mode
        session.record_epoch(
            epoch=1,
            train_loss=0.5,
            val_loss=0.4,
            val_metrics={"val_accuracy": 0.8},
        )
        session.record_epoch(
            epoch=2,
            train_loss=0.4,
            val_loss=0.35,
            val_metrics={"val_accuracy": 0.85},
        )
        assert session.state.best_epoch == 2


class TestTrainingSessionIntegration:
    """Integration tests for TrainingSession."""

    def test_full_training_loop_simulation(self):
        """Simulate a complete training loop."""
        with tempfile.TemporaryDirectory() as tmpdir:
            session = TrainingSession(
                name="integration_test",
                total_epochs=10,
                checkpoint_dir=tmpdir,
            )
            session.start()

            # Simulate training loop
            train_losses = [0.5, 0.4, 0.35, 0.32, 0.30, 0.29, 0.29, 0.29, 0.29, 0.29]
            val_losses = [0.45, 0.38, 0.33, 0.31, 0.30, 0.30, 0.31, 0.32, 0.33, 0.34]

            for epoch, (train_loss, val_loss) in enumerate(
                zip(train_losses, val_losses), 1
            ):
                session.start_epoch()
                session.record_epoch(
                    epoch=epoch,
                    train_loss=train_loss,
                    val_loss=val_loss,
                    learning_rate=0.001 * (0.9 ** epoch),
                )

                if session.should_stop(patience=3):
                    break

            session.end()

            # Verify session state
            assert session.state.best_epoch == 5
            assert session.state.best_val_loss == pytest.approx(0.30, abs=0.01)
            assert session.state.status in [
                SessionStatus.COMPLETED,
                SessionStatus.STOPPED_EARLY,
            ]

    def test_resume_training(self):
        """Test resuming training from checkpoint."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Initial training
            session1 = TrainingSession(
                name="resume_test",
                total_epochs=20,
                checkpoint_dir=tmpdir,
            )
            session1.start()
            for epoch in range(1, 6):
                session1.record_epoch(
                    epoch=epoch,
                    train_loss=0.5 - epoch * 0.05,
                    val_loss=0.45 - epoch * 0.04,
                )
            state_path = session1.save_state()

            # Resume training
            session2 = TrainingSession(
                name="resume_test_2",
                total_epochs=20,
                checkpoint_dir=tmpdir,
                resume_from=state_path,
            )

            assert session2.state.current_epoch == 5
            assert len(session2.state.history) == 5

            # Continue training
            for epoch in range(6, 11):
                session2.record_epoch(
                    epoch=epoch,
                    train_loss=0.5 - epoch * 0.05,
                    val_loss=0.45 - epoch * 0.04,
                )

            assert session2.state.current_epoch == 10
            assert len(session2.state.history) == 10
