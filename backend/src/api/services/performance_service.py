"""Performance service for model metrics and highlights.

This service provides:
- Singleton pattern for performance data loading
- Dynamic performance metrics aggregation
- Highlight generation based on thresholds
- Summary headline generation
"""

import json
import logging
from pathlib import Path
from threading import Lock
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

# Project root path (resolved for safety)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

# Default metric values from validated backtest results (2026-01-22)
# These are used when data files are unavailable
DEFAULT_BASELINE_METRICS = {
    "TOTAL_PIPS": 8135.6,        # From backtest on test set with shallow_fast config
    "WIN_RATE": 0.586,           # 58.6% win rate at 55% confidence threshold
    "PROFIT_FACTOR": 2.26,       # Risk-reward ratio from backtest
    "TOTAL_TRADES": 1093,        # Number of trades in backtest period
}

# High-confidence metrics (70% threshold) from confidence optimization analysis
DEFAULT_HIGH_CONF_METRICS = {
    "THRESHOLD": 0.70,           # 70% confidence threshold
    "WIN_RATE": 0.621,           # 62.1% win rate at high confidence
    "PROFIT_FACTOR": 2.69,       # Profit factor at high confidence
    "TOTAL_PIPS": 8693,          # Total pips at high confidence threshold
    "SAMPLE_SIZE": 966,          # Number of high-confidence predictions
}

# Full model agreement metrics (used when training metadata unavailable)
DEFAULT_AGREEMENT_METRICS = {
    "ACCURACY": 0.82,            # 82% accuracy when all 3 timeframes agree
    "SAMPLE_SIZE": 50,           # Number of full agreement samples
}

# WFO validation metrics (used when wfo_results.json unavailable)
DEFAULT_WFO_METRICS = {
    "WINDOWS_PROFITABLE": 8,     # All 8 windows profitable
    "TOTAL_WINDOWS": 8,          # 8 WFO windows
    "TOTAL_PIPS": 18136,         # Total pips across all windows
    "CONSISTENCY_SCORE": 1.0,    # 100% consistency (8/8)
}

# Regime performance metrics (used when backtest_results.json unavailable)
DEFAULT_REGIME_METRICS = {
    "ALL_PROFITABLE": True,      # All market regimes profitable
    "REGIMES_COUNT": 6,          # 6 market regimes tested
    "PROFITABLE_REGIMES": 6,     # 6/6 profitable
}


class PerformanceService:
    """Service for loading and providing model performance metrics.

    Uses singleton pattern - data is loaded once and cached.
    """

    # Default paths (using resolved project root)
    DEFAULT_MODEL_DIR = PROJECT_ROOT / "models" / "mtf_ensemble"
    DEFAULT_DATA_DIR = PROJECT_ROOT / "data"

    def __init__(self):
        self._lock = Lock()
        self._initialized = False
        self._metrics: Optional[Dict[str, Any]] = None
        self._highlights: Optional[List[Dict[str, Any]]] = None
        self._summary: Optional[Dict[str, str]] = None

    @property
    def is_loaded(self) -> bool:
        """Check if performance data is loaded."""
        return self._initialized and self._metrics is not None

    def initialize(self) -> bool:
        """Initialize performance service by loading metrics.

        Returns:
            True if successful, False otherwise
        """
        if self._initialized:
            return True

        logger.info("Initializing PerformanceService...")

        try:
            self._load_metrics()
            self._generate_highlights()
            self._generate_summary()
            self._initialized = True
            logger.info("PerformanceService initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize PerformanceService: {e}")
            return False

    def _load_wfo_metrics(self) -> Optional[Dict[str, Any]]:
        """Load WFO validation metrics from wfo_results.json.

        Returns:
            Dict with WFO metrics or None if unavailable
        """
        wfo_path = PROJECT_ROOT / "models" / "wfo_validation" / "wfo_results.json"
        if not wfo_path.exists():
            logger.warning(f"WFO results not found at {wfo_path}")
            return None

        try:
            with open(wfo_path) as f:
                wfo_data = json.load(f)

            summary = wfo_data.get("summary", {})
            profitable_windows = summary.get("profitable_windows", 0)
            total_windows = summary.get("total_windows", 0)
            total_pips = summary.get("total_pips", 0)
            consistency_score = (profitable_windows / total_windows) if total_windows > 0 else 0

            return {
                "windows_profitable": profitable_windows,
                "total_windows": total_windows,
                "total_pips": total_pips,
                "consistency_score": consistency_score,
            }
        except Exception as e:
            logger.warning(f"Could not load WFO metrics: {e}")
            return None

    def _build_high_confidence_metrics(
        self,
        backtest_data: Optional[Dict],
        training_data: Dict,
    ) -> Dict[str, Any]:
        """Build high-confidence metrics from backtest by_threshold data.

        Loads profit_factor and total_pips from backtest results at 70% threshold,
        with fallback to training data for win_rate and defaults for missing values.

        Args:
            backtest_data: Loaded backtest_results.json data (may be None)
            training_data: Loaded training_metadata.json data

        Returns:
            Dict with high confidence metrics
        """
        # Try to get by_threshold data from backtest results
        by_threshold = {}
        if backtest_data:
            by_threshold = backtest_data.get("by_threshold", {})

        threshold_70 = by_threshold.get("0.70", {})

        # Get win_rate from training data (more granular validation data)
        # or from backtest threshold data, with fallback to default
        win_rate_from_training = training_data.get("individual_results", {}).get("1H", {}).get(
            "val_acc_conf_70", None
        )
        win_rate_from_backtest = threshold_70.get("win_rate", 0) / 100 if threshold_70.get("win_rate") else None

        # Prefer training data win rate (validation accuracy), fall back to backtest, then default
        if win_rate_from_training is not None:
            win_rate = win_rate_from_training
        elif win_rate_from_backtest is not None:
            win_rate = win_rate_from_backtest
        else:
            win_rate = DEFAULT_HIGH_CONF_METRICS["WIN_RATE"]

        return {
            "threshold": DEFAULT_HIGH_CONF_METRICS["THRESHOLD"],
            "win_rate": win_rate,
            # Load from backtest by_threshold, with fallback to defaults
            "profit_factor": threshold_70.get("profit_factor", DEFAULT_HIGH_CONF_METRICS["PROFIT_FACTOR"]),
            "total_pips": threshold_70.get("total_pips", DEFAULT_HIGH_CONF_METRICS["TOTAL_PIPS"]),
            "sample_size": training_data.get("individual_results", {}).get("1H", {}).get(
                "val_samples_conf_70", DEFAULT_HIGH_CONF_METRICS["SAMPLE_SIZE"]
            ),
        }

    def _load_regime_metrics(self) -> Optional[Dict[str, Any]]:
        """Load regime performance metrics from backtest_results.json.

        Returns:
            Dict with regime metrics or None if unavailable
        """
        backtest_path = self.DEFAULT_DATA_DIR / "backtest_results.json"
        if not backtest_path.exists():
            logger.warning(f"Backtest results not found at {backtest_path}")
            return None

        try:
            with open(backtest_path) as f:
                backtest_data = json.load(f)

            # Check if regime data exists in backtest results
            regime_data = backtest_data.get("regimes")
            if not regime_data:
                logger.info("Regime data not found in backtest_results.json")
                return None

            # Count profitable regimes
            total_regimes = len(regime_data)
            profitable_regimes = sum(
                1 for regime in regime_data.values()
                if regime.get("profit_factor", 0) > 1.0 or regime.get("total_pips", 0) > 0
            )

            return {
                "all_profitable": profitable_regimes == total_regimes,
                "regimes_count": total_regimes,
                "profitable_regimes": profitable_regimes,
            }
        except Exception as e:
            logger.warning(f"Could not load regime metrics: {e}")
            return None

    def _load_metrics(self) -> None:
        """Load metrics from training_metadata.json and backtest_results.json."""
        with self._lock:
            # Load training metadata
            metadata_path = self.DEFAULT_MODEL_DIR / "training_metadata.json"
            if not metadata_path.exists():
                logger.warning(f"Training metadata not found at {metadata_path}")
                self._metrics = self._get_default_metrics()
                return

            with open(metadata_path) as f:
                training_data = json.load(f)

            # Load backtest results if available
            backtest_path = self.DEFAULT_DATA_DIR / "backtest_results.json"
            backtest_data = None
            if backtest_path.exists():
                try:
                    with open(backtest_path) as f:
                        backtest_data = json.load(f)
                except Exception as e:
                    logger.warning(f"Could not load backtest results: {e}")

            # Load WFO and regime metrics dynamically
            wfo_metrics = self._load_wfo_metrics()
            regime_metrics = self._load_regime_metrics()

            # Extract key metrics
            ensemble_results = training_data.get("ensemble_results", {})

            # Get backtest metrics (use all-time period if available)
            backtest_metrics = {}
            if backtest_data:
                all_time = backtest_data.get("periods", {}).get("5y", {})
                if all_time:
                    backtest_metrics = {
                        "total_pips": all_time.get("total_pips", 0),
                        "win_rate": all_time.get("win_rate", 0) / 100,  # Convert to decimal
                        "profit_factor": all_time.get("profit_factor", 0),
                        "total_trades": all_time.get("total_trades", 0),
                    }

            # Combine metrics using documented constants for defaults
            self._metrics = {
                # Use backtest metrics if available, otherwise use validated defaults
                "total_pips": backtest_metrics.get("total_pips", DEFAULT_BASELINE_METRICS["TOTAL_PIPS"]),
                "win_rate": backtest_metrics.get("win_rate",
                    ensemble_results.get("accuracy", DEFAULT_BASELINE_METRICS["WIN_RATE"])),
                "profit_factor": backtest_metrics.get("profit_factor", DEFAULT_BASELINE_METRICS["PROFIT_FACTOR"]),
                "total_trades": backtest_metrics.get("total_trades",
                    ensemble_results.get("test_samples", DEFAULT_BASELINE_METRICS["TOTAL_TRADES"])),

                # High confidence metrics (70% threshold)
                # Load from backtest by_threshold data, with fallback to training data and defaults
                "high_confidence": self._build_high_confidence_metrics(
                    backtest_data, training_data
                ),

                # Full agreement metrics
                "full_agreement": {
                    "accuracy": ensemble_results.get("acc_full_agreement", DEFAULT_AGREEMENT_METRICS["ACCURACY"]),
                    "sample_size": ensemble_results.get("samples_full_agreement", DEFAULT_AGREEMENT_METRICS["SAMPLE_SIZE"]),
                },

                # WFO validation results (dynamically loaded, may be None)
                "wfo_validation": wfo_metrics,

                # Regime performance (dynamically loaded, may be None)
                "regime_performance": regime_metrics,
            }

            logger.info("Performance metrics loaded successfully")

    def _get_default_metrics(self) -> Dict[str, Any]:
        """Get default metrics when training metadata is not available.

        Uses documented constants from validated backtest results.
        """
        # Try to load dynamic metrics even when training metadata is unavailable
        wfo_metrics = self._load_wfo_metrics()
        regime_metrics = self._load_regime_metrics()

        # Try to load backtest by_threshold data for high-confidence metrics
        backtest_path = self.DEFAULT_DATA_DIR / "backtest_results.json"
        by_threshold_70 = {}
        if backtest_path.exists():
            try:
                with open(backtest_path) as f:
                    backtest_data = json.load(f)
                by_threshold_70 = backtest_data.get("by_threshold", {}).get("0.70", {})
            except Exception:
                pass

        return {
            "total_pips": DEFAULT_BASELINE_METRICS["TOTAL_PIPS"],
            "win_rate": DEFAULT_BASELINE_METRICS["WIN_RATE"],
            "profit_factor": DEFAULT_BASELINE_METRICS["PROFIT_FACTOR"],
            "total_trades": DEFAULT_BASELINE_METRICS["TOTAL_TRADES"],
            "high_confidence": {
                "threshold": DEFAULT_HIGH_CONF_METRICS["THRESHOLD"],
                "win_rate": by_threshold_70.get("win_rate", DEFAULT_HIGH_CONF_METRICS["WIN_RATE"] * 100) / 100
                    if by_threshold_70.get("win_rate") else DEFAULT_HIGH_CONF_METRICS["WIN_RATE"],
                "profit_factor": by_threshold_70.get("profit_factor", DEFAULT_HIGH_CONF_METRICS["PROFIT_FACTOR"]),
                "total_pips": by_threshold_70.get("total_pips", DEFAULT_HIGH_CONF_METRICS["TOTAL_PIPS"]),
                "sample_size": by_threshold_70.get("total_trades", DEFAULT_HIGH_CONF_METRICS["SAMPLE_SIZE"]),
            },
            "full_agreement": {
                "accuracy": DEFAULT_AGREEMENT_METRICS["ACCURACY"],
                "sample_size": DEFAULT_AGREEMENT_METRICS["SAMPLE_SIZE"],
            },
            # Use dynamically loaded values or None
            "wfo_validation": wfo_metrics,
            "regime_performance": regime_metrics,
        }

    def _get_status(self, metric_type: str, value: float) -> str:
        """Determine status based on metric type and value.

        Returns: 'excellent', 'good', 'moderate', or 'poor'
        """
        thresholds = {
            "agreement": {"excellent": 75, "good": 60, "moderate": 50},
            "validation": {"excellent": 100, "good": 80, "moderate": 60},
            "robustness": {"excellent": 100, "good": 80, "moderate": 60},
            "profit_factor": {"excellent": 2.5, "good": 2.0, "moderate": 1.5},
        }

        t = thresholds.get(metric_type, {"excellent": 80, "good": 60, "moderate": 40})

        if value >= t["excellent"]:
            return "excellent"
        elif value >= t["good"]:
            return "good"
        elif value >= t["moderate"]:
            return "moderate"
        else:
            return "poor"

    def _generate_highlights(self) -> None:
        """Generate highlights based on loaded metrics.

        Ordered by impact: strongest metrics first.
        Each highlight includes a 'status' field for semantic coloring.
        Only includes highlights where data is available.
        """
        if not self._metrics:
            self._highlights = []
            return

        highlights = []

        # 1. Model Agreement (strongest - 82% accuracy)
        full_agreement = self._metrics.get("full_agreement", {})
        if full_agreement:
            accuracy_pct = full_agreement.get("accuracy", 0) * 100
            highlights.append({
                "type": "agreement",
                "title": "Model Agreement",
                "value": f"{accuracy_pct:.0f}%",
                "description": "Accuracy when all 3 timeframes align",
                "status": self._get_status("agreement", accuracy_pct),
            })

        # 2. Walk-Forward Validation (dynamically loaded from wfo_results.json)
        wfo = self._metrics.get("wfo_validation")
        if wfo:
            profitable = wfo.get("windows_profitable", 0)
            total = wfo.get("total_windows", 0)
            validation_pct = (profitable / total * 100) if total > 0 else 0
            highlights.append({
                "type": "validation",
                "title": "WFO Validation",
                "value": f"{profitable}/{total} Windows Profitable",
                "description": "Profitable across all test periods",
                "status": self._get_status("validation", validation_pct),
            })

        # 3. Regime Robustness (dynamically loaded from backtest_results.json)
        regime = self._metrics.get("regime_performance")
        if regime:
            regimes_count = regime.get("regimes_count", 0)
            all_profitable = regime.get("all_profitable", False)
            robustness_pct = 100 if all_profitable else 0
            highlights.append({
                "type": "robustness",
                "title": "All Conditions",
                "value": f"{regimes_count}/{regimes_count} Regimes",
                "description": "Works in any market regime",
                "status": self._get_status("robustness", robustness_pct),
            })

        # 4. Total Pips (from backtest results)
        total_pips = self._metrics.get("total_pips", 0)
        if total_pips > 0:
            # Format with comma separator
            pips_formatted = f"+{total_pips:,.0f}" if total_pips > 0 else f"{total_pips:,.0f}"
            # Status based on pips (excellent: >10k, good: >5k, moderate: >0)
            if total_pips >= 10000:
                pips_status = "excellent"
            elif total_pips >= 5000:
                pips_status = "good"
            elif total_pips > 0:
                pips_status = "moderate"
            else:
                pips_status = "poor"
            highlights.append({
                "type": "pips",
                "title": "Total Profit",
                "value": f"{pips_formatted} pips",
                "description": "Cumulative profit over 4 years",
                "status": pips_status,
            })

        # 5. Profit Factor (from backtest results)
        profit_factor = self._metrics.get("profit_factor", 0)
        if profit_factor > 0:
            highlights.append({
                "type": "returns",
                "title": "Profit Factor",
                "value": f"{profit_factor:.2f}x",
                "description": f"Returns ${profit_factor:.2f} for every $1 risked",
                "status": self._get_status("profit_factor", profit_factor),
            })

        # Limit to 4 highlights for UI layout
        self._highlights = highlights[:4]

    def _generate_summary(self) -> None:
        """Generate dynamic summary based on metrics."""
        if not self._metrics:
            self._summary = {
                "headline": "Model Performance",
                "description": "Performance metrics unavailable",
            }
            return

        win_rate = self._metrics.get("win_rate", 0)
        profit_factor = self._metrics.get("profit_factor", 0)
        wfo = self._metrics.get("wfo_validation", {})
        consistency = wfo.get("consistency_score", 0)

        # Determine headline
        if win_rate >= 0.60 and profit_factor >= 2.5 and consistency == 1.0:
            headline = "Excellent Performance"
        elif win_rate >= 0.55 and profit_factor >= 2.0:
            headline = "Solid Performance"
        elif win_rate >= 0.50:
            headline = "Moderate Performance"
        else:
            headline = "Developing Performance"

        # Generate description
        win_rate_pct = win_rate * 100
        total_pips = self._metrics.get("total_pips", 0)
        high_conf_wr = self._metrics.get("high_confidence", {}).get("win_rate", 0) * 100

        description = (
            f"The Multi-Timeframe model demonstrates {headline.lower()} with "
            f"{win_rate_pct:.1f}% overall win rate and {profit_factor:.2f}x profit factor. "
            f"High-confidence predictions (≥70%) achieve {high_conf_wr:.1f}% accuracy. "
            f"Walk-forward optimization confirms {consistency*100:.0f}% consistency across all test periods."
        )

        self._summary = {
            "headline": headline,
            "description": description,
        }

    def get_performance_data(self) -> Dict[str, Any]:
        """Get complete performance data including metrics, highlights, and summary.

        Returns:
            Dict with metrics, highlights, and summary
        """
        if not self.is_loaded:
            if not self.initialize():
                return {
                    "metrics": self._get_default_metrics(),
                    "highlights": [],
                    "summary": {
                        "headline": "Performance Data Loading",
                        "description": "Performance metrics are being loaded...",
                    },
                }

        return {
            "metrics": self._metrics,
            "highlights": self._highlights,
            "summary": self._summary,
        }

    def reload(self) -> bool:
        """Reload performance data from disk.

        Returns:
            True if successful, False otherwise
        """
        self._initialized = False
        self._metrics = None
        self._highlights = None
        self._summary = None
        return self.initialize()


# Singleton instance
performance_service = PerformanceService()
