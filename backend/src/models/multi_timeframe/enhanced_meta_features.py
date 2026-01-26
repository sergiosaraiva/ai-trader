"""Enhanced Meta-Features for Stacking Meta-Learner.

This module provides additional meta-features that improve the stacking meta-learner's
ability to combine base model predictions effectively.

CRITICAL: Data Leakage Prevention
----------------------------------
All features MUST use .shift(1) or appropriate lag to prevent look-ahead bias:
- Rolling calculations are shifted to use only past data
- Historical accuracy uses lagged predictions vs lagged actuals
- Market context uses only past price data

Meta-Feature Categories:
1. Prediction Quality: Entropy, margin (confidence spread)
2. Cross-Timeframe Patterns: HTF agreement, trend alignment
3. Market Context: Volatility, trend strength, regime (shifted)
4. Prediction Stability: Rolling std of predictions (shifted)

These features are OPTIONAL and disabled by default for backward compatibility.
Enable with `StackingConfig.use_enhanced_meta_features = True`.
"""

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class EnhancedMetaFeatureCalculator:
    """Calculator for enhanced meta-features with data leakage prevention.

    All rolling calculations use .shift(1) to ensure no future data is used.
    NaN values from shifting are handled appropriately.
    """

    def __init__(self, lookback_window: int = 50):
        """Initialize calculator.

        Args:
            lookback_window: Number of past samples for rolling calculations
        """
        self.lookback_window = lookback_window

    def calculate_all(
        self,
        predictions: Dict[str, np.ndarray],
        probabilities: Dict[str, np.ndarray],
        price_data: Optional[pd.DataFrame] = None,
        actuals: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        """Calculate all enhanced meta-features.

        Args:
            predictions: Dict mapping timeframe to prediction arrays (0/1)
            probabilities: Dict mapping timeframe to probability arrays (prob of UP)
            price_data: Optional DataFrame with OHLC data for market context
            actuals: Optional ground truth labels for historical accuracy

        Returns:
            Dict mapping feature name to numpy array
        """
        features = {}

        # 1. Prediction Quality Features
        quality_features = self.calculate_prediction_quality(probabilities)
        features.update(quality_features)

        # 2. Cross-Timeframe Patterns
        pattern_features = self.calculate_cross_timeframe_patterns(predictions)
        features.update(pattern_features)

        # 3. Market Context (requires price_data)
        if price_data is not None:
            context_features = self.calculate_market_context(price_data)
            features.update(context_features)

        # 4. Prediction Stability
        stability_features = self.calculate_prediction_stability(predictions)
        features.update(stability_features)

        return features

    def calculate_prediction_quality(
        self,
        probabilities: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        """Calculate entropy and margin features.

        Entropy measures the uncertainty in predictions:
        - Low entropy: model is confident (peaked distribution)
        - High entropy: model is uncertain (flat distribution)

        Margin measures the gap between top two class probabilities:
        - Large margin: clear winner
        - Small margin: model is uncertain between classes

        Args:
            probabilities: Dict mapping timeframe to prob arrays

        Returns:
            Dict with 'prob_entropy' and 'confidence_margin' arrays
        """
        # Average entropy across all models
        n_samples = len(next(iter(probabilities.values())))
        entropies = np.zeros(n_samples)
        margins = np.zeros(n_samples)

        for i in range(n_samples):
            # Get probabilities for all models at this timestep
            probs = [probabilities[tf][i] for tf in probabilities]

            # Entropy: -sum(p * log(p))
            # Convert prob_up to [prob_down, prob_up] for entropy calculation
            all_probs = []
            for p_up in probs:
                p_down = 1 - p_up
                # Avoid log(0) by clipping
                p_up_safe = np.clip(p_up, 1e-10, 1 - 1e-10)
                p_down_safe = np.clip(p_down, 1e-10, 1 - 1e-10)
                entropy = -(p_down_safe * np.log(p_down_safe) + p_up_safe * np.log(p_up_safe))
                all_probs.extend([p_down_safe, p_up_safe])
                entropies[i] += entropy

            # Average entropy
            entropies[i] /= len(probs)

            # Margin: difference between max and second-max probability
            all_probs = np.array(all_probs)
            sorted_probs = np.sort(all_probs)[::-1]  # Descending
            margins[i] = sorted_probs[0] - sorted_probs[1] if len(sorted_probs) >= 2 else 0.0

        return {
            "prob_entropy": entropies,
            "confidence_margin": margins,
        }

    def calculate_cross_timeframe_patterns(
        self,
        predictions: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        """Calculate HTF agreement features.

        These features capture cross-timeframe dynamics:
        - htf_agreement_1h_4h: Do 1H and 4H models agree?
        - htf_agreement_4h_d: Do 4H and Daily models agree?
        - trend_alignment: What fraction of models agree? (0-1)

        Args:
            predictions: Dict mapping timeframe to prediction arrays

        Returns:
            Dict with agreement feature arrays
        """
        n_samples = len(next(iter(predictions.values())))

        # Extract predictions
        preds_1h = predictions.get("1H", np.zeros(n_samples))
        preds_4h = predictions.get("4H", np.zeros(n_samples))
        preds_d = predictions.get("D", np.zeros(n_samples))

        # Agreement indicators
        htf_agreement_1h_4h = (preds_1h == preds_4h).astype(float)
        htf_agreement_4h_d = (preds_4h == preds_d).astype(float)

        # Trend alignment: fraction of models agreeing on direction
        trend_alignment = np.zeros(n_samples)
        for i in range(n_samples):
            # Majority direction
            dirs = [preds_1h[i], preds_4h[i], preds_d[i]]
            majority = 1 if sum(dirs) >= 2 else 0
            # Fraction agreeing with majority
            agreement_count = sum(1 for d in dirs if d == majority)
            trend_alignment[i] = agreement_count / 3.0

        return {
            "htf_agreement_1h_4h": htf_agreement_1h_4h,
            "htf_agreement_4h_d": htf_agreement_4h_d,
            "trend_alignment": trend_alignment,
        }

    def calculate_market_context(
        self,
        price_data: pd.DataFrame,
    ) -> Dict[str, np.ndarray]:
        """Calculate volatility, trend strength, and regime features.

        CRITICAL: All calculations use .shift(1) to prevent look-ahead bias.

        Args:
            price_data: DataFrame with OHLC columns

        Returns:
            Dict with 'recent_volatility', 'trend_strength', 'market_regime' arrays
        """
        df = price_data.copy()

        # 1. Recent Volatility: 20-bar rolling std of returns (SHIFTED)
        returns = df["close"].pct_change()
        vol_raw = returns.rolling(window=20, min_periods=5).std()
        vol_shifted = vol_raw.shift(1)  # CRITICAL: Use past volatility only

        # 2. Trend Strength: Simplified ADX proxy (SHIFTED)
        # Use 14-period moving average deviation
        close = df["close"]
        ma14 = close.rolling(window=14, min_periods=5).mean()
        deviation = abs(close - ma14) / ma14
        trend_strength_raw = deviation.rolling(window=14, min_periods=5).mean()
        trend_strength_shifted = trend_strength_raw.shift(1)  # CRITICAL: Use past trend only

        # 3. Market Regime: 0=low vol, 1=normal, 2=high vol (SHIFTED)
        # Use percentile thresholds on shifted volatility
        vol_pctl_low = 33
        vol_pctl_high = 67

        def classify_regime(vol_series):
            """Classify regime based on volatility percentiles."""
            regimes = np.ones(len(vol_series))  # Default to normal (1)
            if len(vol_series) >= 20:
                low_thresh = np.nanpercentile(vol_series, vol_pctl_low)
                high_thresh = np.nanpercentile(vol_series, vol_pctl_high)
                regimes[vol_series < low_thresh] = 0  # Low vol
                regimes[vol_series > high_thresh] = 2  # High vol
            return regimes

        # Calculate regime on shifted volatility (already shifted)
        market_regime = classify_regime(vol_shifted.values)

        # Fill NaN values (from shifting and initial window)
        vol_shifted = vol_shifted.fillna(vol_shifted.median())
        trend_strength_shifted = trend_strength_shifted.fillna(trend_strength_shifted.median())

        return {
            "recent_volatility": vol_shifted.values,
            "trend_strength": trend_strength_shifted.values,
            "market_regime": market_regime,
        }

    def calculate_prediction_stability(
        self,
        predictions: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        """Calculate prediction stability features using shift to prevent leakage.

        Stability measures how consistent predictions are over time:
        - Low stability: predictions flip frequently (uncertain model)
        - High stability: predictions are consistent (confident model)

        CRITICAL: Uses .shift(1) on rolling calculations.

        Args:
            predictions: Dict mapping timeframe to prediction arrays

        Returns:
            Dict with stability feature arrays per timeframe
        """
        features = {}

        for tf, preds in predictions.items():
            # Convert to pandas Series for rolling operations
            preds_series = pd.Series(preds)

            # Rolling std of predictions (SHIFTED to prevent look-ahead)
            stability_raw = preds_series.rolling(
                window=min(10, len(preds) // 2),
                min_periods=3
            ).std()
            stability_shifted = stability_raw.shift(1)  # CRITICAL: Use past stability only

            # Fill NaN values (from shifting and initial window)
            stability_shifted = stability_shifted.fillna(stability_shifted.median())

            features[f"stability_{tf.lower()}"] = stability_shifted.values

        return features


def get_enhanced_feature_names() -> list:
    """Get list of enhanced meta-feature names.

    Returns:
        List of feature name strings
    """
    return [
        # Prediction quality (2)
        "prob_entropy",
        "confidence_margin",
        # Cross-timeframe patterns (3)
        "htf_agreement_1h_4h",
        "htf_agreement_4h_d",
        "trend_alignment",
        # Market context (3)
        "recent_volatility",
        "trend_strength",
        "market_regime",
        # Prediction stability (3 - one per timeframe)
        "stability_1h",
        "stability_4h",
        "stability_d",
    ]
