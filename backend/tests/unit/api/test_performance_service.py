"""Unit tests for PerformanceService.

This test suite validates:
1. Service initialization and singleton pattern
2. Metrics loading from training_metadata.json and backtest_results.json
3. Default metrics when files are missing
4. Highlight generation logic
5. Summary headline generation based on thresholds
6. Reload functionality
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from src.api.services.performance_service import (
    PerformanceService,
    DEFAULT_BASELINE_METRICS,
    DEFAULT_HIGH_CONF_METRICS,
    DEFAULT_WFO_METRICS,
    DEFAULT_AGREEMENT_METRICS,
    DEFAULT_REGIME_METRICS,
)


class TestPerformanceServiceInitialization:
    """Tests for service initialization."""

    def test_service_starts_uninitialized(self):
        """Test service starts in uninitialized state."""
        service = PerformanceService()
        assert not service.is_loaded
        assert not service._initialized
        assert service._metrics is None
        assert service._highlights is None
        assert service._summary is None

    def test_initialize_success(self):
        """Test successful initialization."""
        service = PerformanceService()

        # Mock file paths to use defaults
        with patch.object(service, "_load_metrics"), \
             patch.object(service, "_generate_highlights"), \
             patch.object(service, "_generate_summary"):
            success = service.initialize()

        assert success is True
        assert service.is_loaded is True
        assert service._initialized is True

    def test_initialize_already_initialized_returns_true(self):
        """Test initialize returns True if already initialized."""
        service = PerformanceService()
        service._initialized = True

        # Should not call loading methods
        with patch.object(service, "_load_metrics") as mock_load:
            success = service.initialize()

        assert success is True
        mock_load.assert_not_called()

    def test_initialize_handles_exception(self):
        """Test initialize handles exceptions gracefully."""
        service = PerformanceService()

        # Make _load_metrics raise an exception
        with patch.object(service, "_load_metrics", side_effect=Exception("Test error")):
            success = service.initialize()

        assert success is False
        assert not service.is_loaded


class TestMetricsLoading:
    """Tests for metrics loading from files."""

    @pytest.fixture
    def valid_training_metadata(self):
        """Create valid training metadata JSON."""
        return {
            "ensemble_results": {
                "accuracy": 0.59,
                "test_samples": 1100,
                "acc_conf_70": 0.63,
                "samples_conf_70": 970,
                "acc_full_agreement": 0.84,
                "samples_full_agreement": 52,
            },
            "individual_results": {
                "1H": {
                    "val_acc_conf_70": 0.63,
                    "val_samples_conf_70": 970,
                }
            },
        }

    @pytest.fixture
    def valid_backtest_results(self):
        """Create valid backtest results JSON."""
        return {
            "periods": {
                "5y": {
                    "total_pips": 8500.0,
                    "win_rate": 60.5,  # Stored as percentage
                    "profit_factor": 2.35,
                    "total_trades": 1150,
                }
            }
        }

    def test_load_metrics_with_valid_files(self, valid_training_metadata, valid_backtest_results):
        """Test loading metrics from valid JSON files."""
        service = PerformanceService()

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create mock directories
            model_dir = Path(tmpdir) / "models" / "mtf_ensemble"
            data_dir = Path(tmpdir) / "data"
            model_dir.mkdir(parents=True)
            data_dir.mkdir(parents=True)

            # Write files
            metadata_path = model_dir / "training_metadata.json"
            with open(metadata_path, "w") as f:
                json.dump(valid_training_metadata, f)

            backtest_path = data_dir / "backtest_results.json"
            with open(backtest_path, "w") as f:
                json.dump(valid_backtest_results, f)

            # Patch paths
            with patch.object(service, "DEFAULT_MODEL_DIR", model_dir), \
                 patch.object(service, "DEFAULT_DATA_DIR", data_dir):
                service._load_metrics()

        # Verify metrics loaded from files
        assert service._metrics is not None
        assert service._metrics["total_pips"] == 8500.0
        assert service._metrics["win_rate"] == 0.605  # Converted to decimal
        assert service._metrics["profit_factor"] == 2.35
        assert service._metrics["total_trades"] == 1150

        # Verify high confidence metrics
        assert service._metrics["high_confidence"]["win_rate"] == 0.63
        assert service._metrics["high_confidence"]["sample_size"] == 970

        # Verify full agreement metrics
        assert service._metrics["full_agreement"]["accuracy"] == 0.84
        assert service._metrics["full_agreement"]["sample_size"] == 52

    def test_load_metrics_missing_training_metadata(self):
        """Test loading metrics when training_metadata.json is missing."""
        service = PerformanceService()

        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir) / "models" / "mtf_ensemble"
            data_dir = Path(tmpdir) / "data"
            model_dir.mkdir(parents=True)
            data_dir.mkdir(parents=True)

            # Don't create training_metadata.json

            with patch.object(service, "DEFAULT_MODEL_DIR", model_dir), \
                 patch.object(service, "DEFAULT_DATA_DIR", data_dir):
                service._load_metrics()

        # Should use defaults
        assert service._metrics is not None
        assert service._metrics["total_pips"] == DEFAULT_BASELINE_METRICS["TOTAL_PIPS"]
        assert service._metrics["win_rate"] == DEFAULT_BASELINE_METRICS["WIN_RATE"]
        assert service._metrics["profit_factor"] == DEFAULT_BASELINE_METRICS["PROFIT_FACTOR"]

    def test_load_metrics_missing_backtest_results(self, valid_training_metadata):
        """Test loading metrics when backtest_results.json is missing."""
        service = PerformanceService()

        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir) / "models" / "mtf_ensemble"
            data_dir = Path(tmpdir) / "data"
            model_dir.mkdir(parents=True)
            data_dir.mkdir(parents=True)

            # Only create training_metadata.json
            metadata_path = model_dir / "training_metadata.json"
            with open(metadata_path, "w") as f:
                json.dump(valid_training_metadata, f)

            # Don't create backtest_results.json

            with patch.object(service, "DEFAULT_MODEL_DIR", model_dir), \
                 patch.object(service, "DEFAULT_DATA_DIR", data_dir):
                service._load_metrics()

        # Should use defaults for backtest metrics, training metadata for others
        assert service._metrics is not None
        assert service._metrics["total_pips"] == DEFAULT_BASELINE_METRICS["TOTAL_PIPS"]
        assert service._metrics["win_rate"] == 0.59  # From training metadata
        assert service._metrics["high_confidence"]["win_rate"] == 0.63  # From training metadata

    def test_get_default_metrics(self):
        """Test get_default_metrics returns correct structure."""
        service = PerformanceService()
        defaults = service._get_default_metrics()

        assert defaults is not None
        assert defaults["total_pips"] == DEFAULT_BASELINE_METRICS["TOTAL_PIPS"]
        assert defaults["win_rate"] == DEFAULT_BASELINE_METRICS["WIN_RATE"]
        assert defaults["profit_factor"] == DEFAULT_BASELINE_METRICS["PROFIT_FACTOR"]
        assert defaults["total_trades"] == DEFAULT_BASELINE_METRICS["TOTAL_TRADES"]

        # Verify nested structures
        assert "high_confidence" in defaults
        assert "full_agreement" in defaults
        assert "wfo_validation" in defaults
        assert "regime_performance" in defaults

        # Verify high confidence defaults
        assert defaults["high_confidence"]["threshold"] == DEFAULT_HIGH_CONF_METRICS["THRESHOLD"]
        assert defaults["high_confidence"]["win_rate"] == DEFAULT_HIGH_CONF_METRICS["WIN_RATE"]

        # Verify WFO defaults
        assert defaults["wfo_validation"]["consistency_score"] == DEFAULT_WFO_METRICS["CONSISTENCY_SCORE"]
        assert defaults["wfo_validation"]["total_windows"] == DEFAULT_WFO_METRICS["TOTAL_WINDOWS"]


class TestHighlightGeneration:
    """Tests for highlight generation logic."""

    def test_generate_highlights_with_valid_metrics(self):
        """Test highlight generation with valid metrics."""
        service = PerformanceService()
        service._metrics = {
            "high_confidence": {"win_rate": 0.621},
            "full_agreement": {"accuracy": 0.82},
            "wfo_validation": {"windows_profitable": 7, "total_windows": 7},
            "regime_performance": {"regimes_count": 6},
        }

        service._generate_highlights()

        assert service._highlights is not None
        assert len(service._highlights) == 4

        # Check highlight types
        types = [h["type"] for h in service._highlights]
        assert "confidence" in types
        assert "agreement" in types
        assert "validation" in types
        assert "robustness" in types

        # Check confidence highlight
        confidence_highlight = next(h for h in service._highlights if h["type"] == "confidence")
        assert confidence_highlight["title"] == "High-Confidence Trading"
        assert confidence_highlight["value"] == "62.1%"
        assert "70%" in confidence_highlight["description"]

        # Check agreement highlight
        agreement_highlight = next(h for h in service._highlights if h["type"] == "agreement")
        assert agreement_highlight["title"] == "Model Consensus"
        assert agreement_highlight["value"] == "82%"

        # Check validation highlight
        validation_highlight = next(h for h in service._highlights if h["type"] == "validation")
        assert validation_highlight["title"] == "Walk-Forward Validated"
        assert validation_highlight["value"] == "7/7"

        # Check robustness highlight
        robustness_highlight = next(h for h in service._highlights if h["type"] == "robustness")
        assert robustness_highlight["title"] == "All-Regime Profitable"
        assert robustness_highlight["value"] == "6/6"

    def test_generate_highlights_with_no_metrics(self):
        """Test highlight generation when metrics is None."""
        service = PerformanceService()
        service._metrics = None

        service._generate_highlights()

        assert service._highlights == []

    def test_generate_highlights_with_partial_metrics(self):
        """Test highlight generation with partial metrics."""
        service = PerformanceService()
        service._metrics = {
            "high_confidence": {"win_rate": 0.65},
            "full_agreement": {},  # Empty
            "wfo_validation": {"windows_profitable": 5, "total_windows": 7},
            "regime_performance": {"regimes_count": 6},
        }

        service._generate_highlights()

        assert service._highlights is not None
        assert len(service._highlights) == 4

        # Agreement should use 0 when missing
        agreement_highlight = next(h for h in service._highlights if h["type"] == "agreement")
        assert agreement_highlight["value"] == "0%"


class TestSummaryGeneration:
    """Tests for summary headline generation."""

    def test_generate_summary_excellent_performance(self):
        """Test summary generation for excellent performance."""
        service = PerformanceService()
        service._metrics = {
            "win_rate": 0.62,
            "profit_factor": 2.7,
            "total_pips": 8700,
            "high_confidence": {"win_rate": 0.63},
            "wfo_validation": {"consistency_score": 1.0},
        }

        service._generate_summary()

        assert service._summary is not None
        assert service._summary["headline"] == "Excellent Performance"
        assert "62.0%" in service._summary["description"]
        assert "2.70x" in service._summary["description"]
        assert "63.0%" in service._summary["description"]
        assert "100%" in service._summary["description"]

    def test_generate_summary_solid_performance(self):
        """Test summary generation for solid performance."""
        service = PerformanceService()
        service._metrics = {
            "win_rate": 0.57,
            "profit_factor": 2.1,
            "total_pips": 8100,
            "high_confidence": {"win_rate": 0.60},
            "wfo_validation": {"consistency_score": 0.85},
        }

        service._generate_summary()

        assert service._summary is not None
        assert service._summary["headline"] == "Solid Performance"
        assert "57.0%" in service._summary["description"]
        assert "2.10x" in service._summary["description"]

    def test_generate_summary_moderate_performance(self):
        """Test summary generation for moderate performance."""
        service = PerformanceService()
        service._metrics = {
            "win_rate": 0.52,
            "profit_factor": 1.5,
            "total_pips": 5000,
            "high_confidence": {"win_rate": 0.55},
            "wfo_validation": {"consistency_score": 0.70},
        }

        service._generate_summary()

        assert service._summary is not None
        assert service._summary["headline"] == "Moderate Performance"

    def test_generate_summary_developing_performance(self):
        """Test summary generation for developing performance."""
        service = PerformanceService()
        service._metrics = {
            "win_rate": 0.48,
            "profit_factor": 1.2,
            "total_pips": 2000,
            "high_confidence": {"win_rate": 0.50},
            "wfo_validation": {"consistency_score": 0.60},
        }

        service._generate_summary()

        assert service._summary is not None
        assert service._summary["headline"] == "Developing Performance"

    def test_generate_summary_with_no_metrics(self):
        """Test summary generation when metrics is None."""
        service = PerformanceService()
        service._metrics = None

        service._generate_summary()

        assert service._summary is not None
        assert service._summary["headline"] == "Model Performance"
        assert service._summary["description"] == "Performance metrics unavailable"


class TestGetPerformanceData:
    """Tests for get_performance_data method."""

    def test_get_performance_data_when_loaded(self):
        """Test get_performance_data returns data when loaded."""
        service = PerformanceService()
        service._initialized = True
        service._metrics = {"total_pips": 8135.6}
        service._highlights = [{"type": "confidence", "value": "62.1%"}]
        service._summary = {"headline": "Solid Performance"}

        data = service.get_performance_data()

        assert data is not None
        assert "metrics" in data
        assert "highlights" in data
        assert "summary" in data
        assert data["metrics"]["total_pips"] == 8135.6
        assert len(data["highlights"]) == 1
        assert data["summary"]["headline"] == "Solid Performance"

    def test_get_performance_data_when_not_loaded(self):
        """Test get_performance_data initializes if not loaded."""
        service = PerformanceService()
        service._initialized = False

        with patch.object(service, "initialize", return_value=True), \
             patch.object(service, "_metrics", {"total_pips": 8000}), \
             patch.object(service, "_highlights", []), \
             patch.object(service, "_summary", {"headline": "Test"}):
            data = service.get_performance_data()

        assert data is not None
        assert "metrics" in data
        assert "highlights" in data
        assert "summary" in data

    def test_get_performance_data_initialization_fails(self):
        """Test get_performance_data when initialization fails."""
        service = PerformanceService()
        service._initialized = False

        with patch.object(service, "initialize", return_value=False):
            data = service.get_performance_data()

        # Should return defaults
        assert data is not None
        assert "metrics" in data
        assert "highlights" in data
        assert "summary" in data
        assert data["summary"]["headline"] == "Performance Data Loading"


class TestReloadFunctionality:
    """Tests for reload method."""

    def test_reload_resets_state_and_reinitializes(self):
        """Test reload resets state and calls initialize."""
        service = PerformanceService()
        service._initialized = True
        service._metrics = {"test": "data"}
        service._highlights = [{"test": "highlight"}]
        service._summary = {"test": "summary"}

        with patch.object(service, "initialize", return_value=True) as mock_init:
            success = service.reload()

        # Should reset state
        mock_init.assert_called_once()
        assert success is True

    def test_reload_returns_false_on_failure(self):
        """Test reload returns False when initialization fails."""
        service = PerformanceService()
        service._initialized = True

        with patch.object(service, "initialize", return_value=False):
            success = service.reload()

        assert success is False


class TestDefaultConstants:
    """Tests to verify default constants are valid."""

    def test_default_baseline_metrics_structure(self):
        """Test DEFAULT_BASELINE_METRICS has correct structure."""
        assert "TOTAL_PIPS" in DEFAULT_BASELINE_METRICS
        assert "WIN_RATE" in DEFAULT_BASELINE_METRICS
        assert "PROFIT_FACTOR" in DEFAULT_BASELINE_METRICS
        assert "TOTAL_TRADES" in DEFAULT_BASELINE_METRICS

        # Verify types
        assert isinstance(DEFAULT_BASELINE_METRICS["TOTAL_PIPS"], (int, float))
        assert isinstance(DEFAULT_BASELINE_METRICS["WIN_RATE"], float)
        assert isinstance(DEFAULT_BASELINE_METRICS["PROFIT_FACTOR"], float)
        assert isinstance(DEFAULT_BASELINE_METRICS["TOTAL_TRADES"], int)

    def test_default_high_conf_metrics_structure(self):
        """Test DEFAULT_HIGH_CONF_METRICS has correct structure."""
        assert "THRESHOLD" in DEFAULT_HIGH_CONF_METRICS
        assert "WIN_RATE" in DEFAULT_HIGH_CONF_METRICS
        assert "PROFIT_FACTOR" in DEFAULT_HIGH_CONF_METRICS
        assert "TOTAL_PIPS" in DEFAULT_HIGH_CONF_METRICS
        assert "SAMPLE_SIZE" in DEFAULT_HIGH_CONF_METRICS

    def test_default_wfo_metrics_structure(self):
        """Test DEFAULT_WFO_METRICS has correct structure."""
        assert "WINDOWS_PROFITABLE" in DEFAULT_WFO_METRICS
        assert "TOTAL_WINDOWS" in DEFAULT_WFO_METRICS
        assert "TOTAL_PIPS" in DEFAULT_WFO_METRICS
        assert "CONSISTENCY_SCORE" in DEFAULT_WFO_METRICS

        # Verify consistency
        assert DEFAULT_WFO_METRICS["CONSISTENCY_SCORE"] == 1.0
        assert DEFAULT_WFO_METRICS["WINDOWS_PROFITABLE"] == DEFAULT_WFO_METRICS["TOTAL_WINDOWS"]

    def test_default_agreement_metrics_structure(self):
        """Test DEFAULT_AGREEMENT_METRICS has correct structure."""
        assert "ACCURACY" in DEFAULT_AGREEMENT_METRICS
        assert "SAMPLE_SIZE" in DEFAULT_AGREEMENT_METRICS

    def test_default_regime_metrics_structure(self):
        """Test DEFAULT_REGIME_METRICS has correct structure."""
        assert "ALL_PROFITABLE" in DEFAULT_REGIME_METRICS
        assert "REGIMES_COUNT" in DEFAULT_REGIME_METRICS
        assert DEFAULT_REGIME_METRICS["ALL_PROFITABLE"] is True
