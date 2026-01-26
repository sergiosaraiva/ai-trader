"""Unit tests for MTF Ensemble deque implementation (Fix #2).

This test suite validates the deque bug fix:

FIX #2 - Deque Implementation:
- prediction_history is now a deque with proper maxlen for automatic FIFO
- Adding items beyond maxlen automatically removes oldest (no manual cleanup needed)
- _calculate_dynamic_weights() correctly handles deque to list conversion
- Save/load cycle preserves prediction history correctly
"""

import pytest
import pickle
import tempfile
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.models.multi_timeframe import MTFEnsemble, MTFEnsembleConfig, MTFPrediction


class TestDequeImplementation:
    """Tests for Fix #2: Deque implementation for prediction_history."""

    @pytest.fixture
    def ensemble(self):
        """Create a basic MTF ensemble with dynamic weights enabled."""
        config = MTFEnsembleConfig.default()
        config.use_dynamic_weights = True
        config.dynamic_weight_window = 50

        ensemble = MTFEnsemble(config=config)
        return ensemble

    def test_prediction_history_is_deque(self, ensemble):
        """Test that prediction_history is initialized as a deque."""
        assert isinstance(ensemble.prediction_history, deque)

    def test_prediction_history_has_maxlen(self, ensemble):
        """Test that prediction_history deque has proper maxlen set."""
        # maxlen should be dynamic_weight_window * 2
        expected_maxlen = ensemble.config.dynamic_weight_window * 2
        assert ensemble.prediction_history.maxlen == expected_maxlen
        assert ensemble.prediction_history.maxlen == 100  # 50 * 2

    def test_prediction_history_starts_empty(self, ensemble):
        """Test that prediction_history starts empty."""
        assert len(ensemble.prediction_history) == 0

    def test_add_single_record(self, ensemble):
        """Test adding a single record to prediction_history."""
        prediction = MTFPrediction(
            direction=1,
            confidence=0.75,
            prob_up=0.75,
            prob_down=0.25,
            component_directions={"1H": 1, "4H": 1, "D": 1},
            component_confidences={"1H": 0.7, "4H": 0.75, "D": 0.8},
            component_weights={"1H": 0.6, "4H": 0.3, "D": 0.1},
        )

        ensemble.record_outcome(prediction, actual_direction=1)

        assert len(ensemble.prediction_history) == 1

    def test_deque_automatic_fifo_behavior(self, ensemble):
        """Test that deque automatically removes oldest items when maxlen is reached."""
        maxlen = ensemble.prediction_history.maxlen

        prediction = MTFPrediction(
            direction=1,
            confidence=0.75,
            prob_up=0.75,
            prob_down=0.25,
            component_directions={"1H": 1, "4H": 1, "D": 1},
            component_confidences={"1H": 0.7, "4H": 0.75, "D": 0.8},
            component_weights={"1H": 0.6, "4H": 0.3, "D": 0.1},
        )

        # Add more items than maxlen
        for i in range(maxlen + 10):
            ensemble.record_outcome(prediction, actual_direction=i % 2)

        # Should be capped at maxlen (oldest items automatically removed)
        assert len(ensemble.prediction_history) == maxlen

    def test_deque_fifo_order_is_maintained(self, ensemble):
        """Test that deque maintains FIFO order (oldest items removed first)."""
        prediction = MTFPrediction(
            direction=1,
            confidence=0.75,
            prob_up=0.75,
            prob_down=0.25,
            component_directions={"1H": 1, "4H": 1, "D": 1},
            component_confidences={"1H": 0.7, "4H": 0.75, "D": 0.8},
            component_weights={"1H": 0.6, "4H": 0.3, "D": 0.1},
        )

        # Add 105 items (maxlen is 100)
        for i in range(105):
            pred_copy = MTFPrediction(
                direction=1,
                confidence=0.75,
                prob_up=0.75,
                prob_down=0.25,
                component_directions={"1H": i, "4H": 1, "D": 1},  # Use i to track order
                component_confidences={"1H": 0.7, "4H": 0.75, "D": 0.8},
                component_weights={"1H": 0.6, "4H": 0.3, "D": 0.1},
            )
            ensemble.record_outcome(pred_copy, actual_direction=1)

        # First item in deque should be from iteration 5 (items 0-4 were removed)
        first_record = list(ensemble.prediction_history)[0]
        assert first_record["1H_pred"] == 5  # Item 0-4 removed, 5 is now first

        # Last item should be from iteration 104
        last_record = list(ensemble.prediction_history)[-1]
        assert last_record["1H_pred"] == 104

    def test_calculate_dynamic_weights_with_deque(self, ensemble):
        """Test that _calculate_dynamic_weights() correctly handles deque to list conversion."""
        prediction = MTFPrediction(
            direction=1,
            confidence=0.75,
            prob_up=0.75,
            prob_down=0.25,
            component_directions={"1H": 1, "4H": 1, "D": 1},
            component_confidences={"1H": 0.7, "4H": 0.75, "D": 0.8},
            component_weights={"1H": 0.6, "4H": 0.3, "D": 0.1},
        )

        # Add enough records to trigger dynamic weight calculation (>= 10)
        for i in range(20):
            ensemble.record_outcome(prediction, actual_direction=1)

        # This should work without errors (deque is converted to list internally)
        weights = ensemble._calculate_dynamic_weights()

        assert isinstance(weights, dict)
        assert "1H" in weights
        assert "4H" in weights
        assert "D" in weights
        assert sum(weights.values()) == pytest.approx(1.0)

    def test_calculate_dynamic_weights_insufficient_history(self, ensemble):
        """Test that _calculate_dynamic_weights() returns base weights when history < 10."""
        prediction = MTFPrediction(
            direction=1,
            confidence=0.75,
            prob_up=0.75,
            prob_down=0.25,
            component_directions={"1H": 1, "4H": 1, "D": 1},
            component_confidences={"1H": 0.7, "4H": 0.75, "D": 0.8},
            component_weights={"1H": 0.6, "4H": 0.3, "D": 0.1},
        )

        # Add only 5 records (< 10 required)
        for i in range(5):
            ensemble.record_outcome(prediction, actual_direction=1)

        weights = ensemble._calculate_dynamic_weights()

        # Should return base weights
        assert weights == ensemble.config.weights

    def test_deque_thread_safe_access(self, ensemble):
        """Test that deque access is thread-safe (using lock)."""
        prediction = MTFPrediction(
            direction=1,
            confidence=0.75,
            prob_up=0.75,
            prob_down=0.25,
            component_directions={"1H": 1, "4H": 1, "D": 1},
            component_confidences={"1H": 0.7, "4H": 0.75, "D": 0.8},
            component_weights={"1H": 0.6, "4H": 0.3, "D": 0.1},
        )

        # Add records
        for i in range(15):
            ensemble.record_outcome(prediction, actual_direction=1)

        # Access should work without issues (lock is used internally)
        with ensemble._history_lock:
            history_len = len(ensemble.prediction_history)

        assert history_len == 15


class TestDequeSaveLoad:
    """Tests for Fix #2: Save/load cycle preserves prediction history correctly."""

    @pytest.fixture
    def ensemble_with_history(self):
        """Create ensemble with prediction history."""
        config = MTFEnsembleConfig.default()
        config.use_dynamic_weights = True
        config.dynamic_weight_window = 50

        ensemble = MTFEnsemble(config=config)

        # Add some prediction history
        prediction = MTFPrediction(
            direction=1,
            confidence=0.75,
            prob_up=0.75,
            prob_down=0.25,
            component_directions={"1H": 1, "4H": 1, "D": 1},
            component_confidences={"1H": 0.7, "4H": 0.75, "D": 0.8},
            component_weights={"1H": 0.6, "4H": 0.3, "D": 0.1},
        )

        for i in range(30):
            ensemble.record_outcome(prediction, actual_direction=i % 2)

        return ensemble

    def test_save_prediction_history_to_pickle(self, ensemble_with_history):
        """Test that save() creates prediction_history.pkl file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir)

            ensemble_with_history.save(save_path)

            history_file = save_path / "prediction_history.pkl"
            assert history_file.exists()

    def test_load_prediction_history_from_pickle(self, ensemble_with_history):
        """Test that load() restores prediction_history as deque."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir)

            # Save ensemble with history
            original_history_len = len(ensemble_with_history.prediction_history)
            ensemble_with_history.save(save_path)

            # Create new ensemble and load
            config = MTFEnsembleConfig.default()
            config.use_dynamic_weights = True
            new_ensemble = MTFEnsemble(config=config)

            new_ensemble.load(save_path)

            # Verify prediction_history is restored as deque
            assert isinstance(new_ensemble.prediction_history, deque)
            assert len(new_ensemble.prediction_history) == original_history_len
            assert new_ensemble.prediction_history.maxlen == ensemble_with_history.prediction_history.maxlen

    def test_save_load_preserves_history_content(self, ensemble_with_history):
        """Test that save/load cycle preserves history content exactly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir)

            # Get original history
            original_history = list(ensemble_with_history.prediction_history)

            # Save and load
            ensemble_with_history.save(save_path)

            config = MTFEnsembleConfig.default()
            config.use_dynamic_weights = True
            new_ensemble = MTFEnsemble(config=config)
            new_ensemble.load(save_path)

            # Compare histories
            loaded_history = list(new_ensemble.prediction_history)

            assert len(loaded_history) == len(original_history)

            # Compare first and last records
            assert loaded_history[0] == original_history[0]
            assert loaded_history[-1] == original_history[-1]

    def test_save_empty_history(self):
        """Test that save/load works with empty prediction history."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir)

            config = MTFEnsembleConfig.default()
            config.use_dynamic_weights = True
            ensemble = MTFEnsemble(config=config)

            # Save with empty history
            ensemble.save(save_path)

            # Load
            new_ensemble = MTFEnsemble(config=config)
            new_ensemble.load(save_path)

            # Should still be deque with correct maxlen
            assert isinstance(new_ensemble.prediction_history, deque)
            assert len(new_ensemble.prediction_history) == 0
            assert new_ensemble.prediction_history.maxlen == ensemble.prediction_history.maxlen

    def test_load_without_history_file_initializes_empty_deque(self):
        """Test that loading without history file creates empty deque."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir)
            save_path.mkdir(parents=True, exist_ok=True)

            # Create ensemble config file but no history file
            config_path = save_path / "ensemble_config.json"
            config_path.write_text('{"weights": {"1H": 0.6, "4H": 0.3, "D": 0.1}}')

            config = MTFEnsembleConfig.default()
            config.use_dynamic_weights = True
            ensemble = MTFEnsemble(config=config)

            # Load (no history file exists)
            ensemble.load(save_path)

            # Should have empty deque
            assert isinstance(ensemble.prediction_history, deque)
            assert len(ensemble.prediction_history) == 0
            assert ensemble.prediction_history.maxlen == 100  # dynamic_weight_window * 2

    def test_deque_maxlen_preserved_after_load(self, ensemble_with_history):
        """Test that deque maxlen is correctly set after loading."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir)

            original_maxlen = ensemble_with_history.prediction_history.maxlen

            ensemble_with_history.save(save_path)

            config = MTFEnsembleConfig.default()
            config.use_dynamic_weights = True
            config.dynamic_weight_window = 50  # Same as original
            new_ensemble = MTFEnsemble(config=config)
            new_ensemble.load(save_path)

            assert new_ensemble.prediction_history.maxlen == original_maxlen
            assert new_ensemble.prediction_history.maxlen == 100


class TestDequeRecordOutcome:
    """Tests for record_outcome() with deque."""

    @pytest.fixture
    def ensemble(self):
        """Create ensemble with dynamic weights enabled."""
        config = MTFEnsembleConfig.default()
        config.use_dynamic_weights = True
        config.dynamic_weight_window = 50
        return MTFEnsemble(config=config)

    def test_record_outcome_creates_correct_record(self, ensemble):
        """Test that record_outcome creates a properly formatted record."""
        prediction = MTFPrediction(
            direction=1,
            confidence=0.75,
            prob_up=0.75,
            prob_down=0.25,
            component_directions={"1H": 1, "4H": 0, "D": 1},
            component_confidences={"1H": 0.7, "4H": 0.65, "D": 0.8},
            component_weights={"1H": 0.6, "4H": 0.3, "D": 0.1},
        )

        ensemble.record_outcome(prediction, actual_direction=1)

        record = list(ensemble.prediction_history)[0]

        assert "timestamp" in record
        assert "actual" in record
        assert record["actual"] == 1
        assert "1H_pred" in record
        assert "4H_pred" in record
        assert "D_pred" in record
        assert "1H_correct" in record
        assert "4H_correct" in record
        assert "D_correct" in record

    def test_record_outcome_tracks_correct_predictions(self, ensemble):
        """Test that record_outcome correctly tracks which models were right."""
        prediction = MTFPrediction(
            direction=1,
            confidence=0.75,
            prob_up=0.75,
            prob_down=0.25,
            component_directions={"1H": 1, "4H": 0, "D": 1},  # 1H and D correct, 4H wrong
            component_confidences={"1H": 0.7, "4H": 0.65, "D": 0.8},
            component_weights={"1H": 0.6, "4H": 0.3, "D": 0.1},
        )

        ensemble.record_outcome(prediction, actual_direction=1)

        record = list(ensemble.prediction_history)[0]

        assert record["1H_correct"] == 1  # 1H predicted 1, actual 1 → correct
        assert record["4H_correct"] == 0  # 4H predicted 0, actual 1 → wrong
        assert record["D_correct"] == 1   # D predicted 1, actual 1 → correct

    def test_record_outcome_multiple_records(self, ensemble):
        """Test adding multiple records to deque."""
        prediction = MTFPrediction(
            direction=1,
            confidence=0.75,
            prob_up=0.75,
            prob_down=0.25,
            component_directions={"1H": 1, "4H": 1, "D": 1},
            component_confidences={"1H": 0.7, "4H": 0.75, "D": 0.8},
            component_weights={"1H": 0.6, "4H": 0.3, "D": 0.1},
        )

        # Add 50 records
        for i in range(50):
            ensemble.record_outcome(prediction, actual_direction=i % 2)

        assert len(ensemble.prediction_history) == 50


class TestDequePerformance:
    """Performance tests for deque implementation."""

    @pytest.fixture
    def ensemble(self):
        """Create ensemble."""
        config = MTFEnsembleConfig.default()
        config.use_dynamic_weights = True
        config.dynamic_weight_window = 50
        return MTFEnsemble(config=config)

    def test_deque_performance_at_maxlen(self, ensemble):
        """Test that deque performs well at maxlen (O(1) append/pop)."""
        import time

        prediction = MTFPrediction(
            direction=1,
            confidence=0.75,
            prob_up=0.75,
            prob_down=0.25,
            component_directions={"1H": 1, "4H": 1, "D": 1},
            component_confidences={"1H": 0.7, "4H": 0.75, "D": 0.8},
            component_weights={"1H": 0.6, "4H": 0.3, "D": 0.1},
        )

        # Fill deque to maxlen
        for i in range(ensemble.prediction_history.maxlen):
            ensemble.record_outcome(prediction, actual_direction=1)

        # Measure time for 1000 appends at maxlen (should be fast - O(1))
        start = time.perf_counter()
        for i in range(1000):
            ensemble.record_outcome(prediction, actual_direction=1)
        duration = time.perf_counter() - start

        # Should complete in less than 100ms
        assert duration < 0.1

    def test_deque_memory_bounded(self, ensemble):
        """Test that deque memory usage is bounded by maxlen."""
        prediction = MTFPrediction(
            direction=1,
            confidence=0.75,
            prob_up=0.75,
            prob_down=0.25,
            component_directions={"1H": 1, "4H": 1, "D": 1},
            component_confidences={"1H": 0.7, "4H": 0.75, "D": 0.8},
            component_weights={"1H": 0.6, "4H": 0.3, "D": 0.1},
        )

        maxlen = ensemble.prediction_history.maxlen

        # Add 10x maxlen items
        for i in range(maxlen * 10):
            ensemble.record_outcome(prediction, actual_direction=1)

        # Size should still be capped at maxlen
        assert len(ensemble.prediction_history) == maxlen


@pytest.mark.parametrize("dynamic_weight_window,expected_maxlen", [
    (10, 20),   # 10 * 2 = 20
    (25, 50),   # 25 * 2 = 50
    (50, 100),  # 50 * 2 = 100
    (100, 200), # 100 * 2 = 200
])
def test_deque_maxlen_parametrized(dynamic_weight_window, expected_maxlen):
    """Parametrized test for deque maxlen calculation."""
    config = MTFEnsembleConfig.default()
    config.use_dynamic_weights = True
    config.dynamic_weight_window = dynamic_weight_window

    ensemble = MTFEnsemble(config=config)

    assert ensemble.prediction_history.maxlen == expected_maxlen
