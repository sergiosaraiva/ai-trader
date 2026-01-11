"""
Examples and demonstrations of the confidence system.

Run this file to see how confidence estimation works:
    python -m src.models.confidence.examples
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict

from .calibration import (
    TemperatureScaling,
    CalibrationMetrics,
    select_best_calibrator,
)
from .uncertainty import (
    MCDropoutUncertainty,
    EnsembleUncertainty,
    ConfidenceEstimator,
    entropy_uncertainty,
)
from .integration import (
    ConfidenceAwarePredictor,
    TradingPrediction,
)


class SimpleTradingModel(nn.Module):
    """Simple model for demonstration."""

    def __init__(self, input_dim: int = 50, hidden_dim: int = 64, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def demo_confidence_interpretation():
    """
    Demonstrate how sigmoid outputs translate to confidence.

    Key insight: The distance from 0.5 indicates confidence.
    """
    print("\n" + "=" * 60)
    print("DEMO: Understanding Sigmoid Confidence")
    print("=" * 60)

    # Example sigmoid outputs
    examples = [
        (0.001, "Market going DOWN - VERY HIGH confidence"),
        (0.10, "Market going DOWN - HIGH confidence"),
        (0.30, "Market going DOWN - MODERATE confidence"),
        (0.45, "Market going DOWN - LOW confidence"),
        (0.50, "Market UNCERTAIN - NO confidence (skip trade)"),
        (0.55, "Market going UP - LOW confidence"),
        (0.70, "Market going UP - MODERATE confidence"),
        (0.90, "Market going UP - HIGH confidence"),
        (0.999, "Market going UP - VERY HIGH confidence"),
    ]

    print("\nSigmoid Output → Direction + Confidence:")
    print("-" * 60)

    for sigmoid_value, interpretation in examples:
        direction = "UP" if sigmoid_value >= 0.5 else "DOWN"
        confidence = sigmoid_value if sigmoid_value >= 0.5 else (1 - sigmoid_value)
        distance = abs(sigmoid_value - 0.5) * 2  # Scale to [0, 1]

        print(f"  {sigmoid_value:.3f} → {direction:4s} | "
              f"Confidence: {confidence:.3f} | "
              f"Certainty: {distance:.1%}")

    print("\n💡 Key Insight:")
    print("   - Values near 0.5 = uncertain = DON'T TRADE")
    print("   - Values near 0 or 1 = confident = TRADE")
    print("   - Use confidence for POSITION SIZING")


def demo_calibration():
    """
    Demonstrate why calibration matters.

    Shows how raw model outputs are often overconfident.
    """
    print("\n" + "=" * 60)
    print("DEMO: Why Calibration Matters")
    print("=" * 60)

    # Simulate overconfident model outputs
    np.random.seed(42)

    # Generate synthetic data
    n_samples = 1000

    # True probabilities (what the model should output)
    true_probs = np.random.uniform(0, 1, n_samples)
    labels = (np.random.random(n_samples) < true_probs).astype(float)

    # Simulate overconfident model (pushes predictions toward 0 or 1)
    # This is typical of neural networks
    overconfidence_factor = 2.0
    centered = true_probs - 0.5
    overconfident_probs = 0.5 + np.tanh(centered * overconfidence_factor) / 2

    # Convert to logits for calibration
    eps = 1e-7
    logits = np.log((overconfident_probs + eps) / (1 - overconfident_probs + eps))

    # Compute metrics BEFORE calibration
    predictions = (overconfident_probs >= 0.5).astype(int)
    confidences = np.where(predictions == 1, overconfident_probs, 1 - overconfident_probs)
    metrics_before = CalibrationMetrics.compute(confidences, predictions, labels)

    print("\nBEFORE Calibration (typical neural network):")
    print(f"  Expected Calibration Error (ECE): {metrics_before.ece:.4f}")
    print(f"  Maximum Calibration Error (MCE): {metrics_before.mce:.4f}")
    print(f"  Brier Score: {metrics_before.brier_score:.4f}")

    # Apply temperature scaling
    temp_scaling = TemperatureScaling()
    temp_scaling.fit(torch.tensor(logits), torch.tensor(labels))

    calibrated_probs = temp_scaling(torch.tensor(logits)).detach().numpy()
    predictions_after = (calibrated_probs >= 0.5).astype(int)
    confidences_after = np.where(
        predictions_after == 1, calibrated_probs, 1 - calibrated_probs
    )
    metrics_after = CalibrationMetrics.compute(
        confidences_after, predictions_after, labels
    )

    print(f"\nAFTER Temperature Scaling (T = {temp_scaling.temperature.item():.3f}):")
    print(f"  Expected Calibration Error (ECE): {metrics_after.ece:.4f}")
    print(f"  Maximum Calibration Error (MCE): {metrics_after.mce:.4f}")
    print(f"  Brier Score: {metrics_after.brier_score:.4f}")

    print(f"\n📉 Improvement:")
    print(f"  ECE reduced by {(1 - metrics_after.ece/metrics_before.ece)*100:.1f}%")

    print("\n💡 Key Insight:")
    print("   - Neural networks are typically OVERCONFIDENT")
    print("   - Temperature scaling 'softens' predictions")
    print("   - Calibrated confidence = actual accuracy")


def demo_mc_dropout():
    """
    Demonstrate MC Dropout for uncertainty estimation.
    """
    print("\n" + "=" * 60)
    print("DEMO: MC Dropout Uncertainty")
    print("=" * 60)

    # Create a simple model
    model = SimpleTradingModel(input_dim=20, dropout=0.3)

    # Create test inputs
    # Input 1: Clear pattern (low uncertainty expected)
    x_clear = torch.randn(1, 20) * 0.1
    x_clear[:, 0] = 5.0  # Strong signal

    # Input 2: Noisy pattern (high uncertainty expected)
    x_noisy = torch.randn(1, 20) * 2.0

    mc_estimator = MCDropoutUncertainty(n_samples=50)

    print("\nEstimating uncertainty with 50 forward passes...")

    for name, x in [("Clear Signal", x_clear), ("Noisy Signal", x_noisy)]:
        estimate = mc_estimator.estimate(model, x, return_samples=True)

        # Handle scalar vs array output
        pred = estimate.prediction.item() if hasattr(estimate.prediction, 'item') else estimate.prediction
        conf = estimate.confidence.item() if hasattr(estimate.confidence, 'item') else estimate.confidence
        epist = estimate.epistemic.item() if hasattr(estimate.epistemic, 'item') else estimate.epistemic

        print(f"\n{name}:")
        print(f"  Mean Prediction: {pred}")
        print(f"  Confidence: {conf:.3f}")
        print(f"  Epistemic Uncertainty: {epist:.4f}")

        if hasattr(estimate, 'samples'):
            samples = estimate.samples.squeeze()
            print(f"  Prediction Range: [{samples.min():.3f}, {samples.max():.3f}]")

    print("\n💡 Key Insight:")
    print("   - High epistemic uncertainty = model is UNSURE")
    print("   - Could be: unusual data, ambiguous pattern, out-of-distribution")
    print("   - Use uncertainty to REDUCE position size or SKIP trade")


def demo_ensemble_disagreement():
    """
    Demonstrate ensemble disagreement as confidence measure.
    """
    print("\n" + "=" * 60)
    print("DEMO: Ensemble Disagreement")
    print("=" * 60)

    # Simulate predictions from 3 models (short, medium, long term)
    scenarios = {
        "All models AGREE (UP)": {
            'short': 0.85,
            'medium': 0.80,
            'long': 0.75,
        },
        "All models AGREE (DOWN)": {
            'short': 0.15,
            'medium': 0.20,
            'long': 0.25,
        },
        "Models DISAGREE (mixed)": {
            'short': 0.85,  # Strong up
            'medium': 0.45,  # Uncertain
            'long': 0.30,   # Down
        },
        "Models very UNCERTAIN": {
            'short': 0.52,
            'medium': 0.48,
            'long': 0.51,
        },
    }

    ensemble_estimator = EnsembleUncertainty()

    print("\nMulti-Timeframe Ensemble Analysis:")
    print("-" * 60)

    for scenario_name, predictions in scenarios.items():
        pred_array = [np.array([p]) for p in predictions.values()]
        weights = np.array([0.5, 0.3, 0.2])  # Short, Medium, Long

        estimate = ensemble_estimator.estimate(pred_array, weights)

        print(f"\n{scenario_name}:")
        print(f"  Short-term:  {predictions['short']:.2f}")
        print(f"  Medium-term: {predictions['medium']:.2f}")
        print(f"  Long-term:   {predictions['long']:.2f}")
        print(f"  ─────────────────────")
        print(f"  Final Prediction: {'UP' if estimate.prediction[0] == 1 else 'DOWN'}")
        print(f"  Confidence: {estimate.confidence[0]:.3f}")
        print(f"  Disagreement: {estimate.epistemic[0]:.3f}")

    print("\n💡 Key Insight:")
    print("   - When models agree → HIGH confidence")
    print("   - When models disagree → LOW confidence")
    print("   - Perfect for multi-timeframe trading!")


def demo_trading_signals():
    """
    Demonstrate how confidence translates to trading signals.
    """
    print("\n" + "=" * 60)
    print("DEMO: Confidence → Trading Signals")
    print("=" * 60)

    confidence_estimator = ConfidenceEstimator(
        confidence_thresholds={
            'very_high': 0.90,
            'high': 0.80,
            'moderate': 0.70,
            'low': 0.60,
            'minimum': 0.50
        }
    )

    from .uncertainty import UncertaintyEstimate

    print("\nConfidence Level → Position Sizing:")
    print("-" * 60)

    test_cases = [
        (0.95, 0.02, "Very confident UP"),
        (0.85, 0.05, "Confident UP"),
        (0.75, 0.10, "Moderate confidence UP"),
        (0.65, 0.12, "Low confidence UP"),
        (0.55, 0.15, "Very low confidence UP"),
        (0.50, 0.20, "No confidence (uncertain)"),
        (0.45, 0.15, "Very low confidence DOWN"),
        (0.25, 0.08, "Confident DOWN"),
    ]

    for prob, uncertainty, description in test_cases:
        prediction = 1 if prob >= 0.5 else 0
        confidence = prob if prob >= 0.5 else (1 - prob)

        estimate = UncertaintyEstimate(
            prediction=np.array([prediction]),
            confidence=np.array([confidence]),
            epistemic=np.array([uncertainty])
        )

        signal = confidence_estimator.get_trading_signal(estimate)

        dir_str = "↑" if signal['direction'][0] == 1 else "↓" if signal['direction'][0] == -1 else "─"
        print(f"  {description:30s} | {dir_str} | "
              f"Size: {signal['position_size'][0]:.0%} | "
              f"Level: {signal['confidence_level'][0]}")

    print("\n💡 Trading Rules:")
    print("   - Very High (>90%): Full position (100%)")
    print("   - High (80-90%): Large position (75%)")
    print("   - Moderate (70-80%): Medium position (50%)")
    print("   - Low (60-70%): Small position (25%)")
    print("   - Below 60%: NO TRADE")


def demo_full_pipeline():
    """
    Demonstrate the complete confidence-aware prediction pipeline.
    """
    print("\n" + "=" * 60)
    print("DEMO: Full Pipeline Integration")
    print("=" * 60)

    # Create three models for different timeframes
    short_model = SimpleTradingModel(input_dim=30, dropout=0.2)
    medium_model = SimpleTradingModel(input_dim=30, dropout=0.2)
    long_model = SimpleTradingModel(input_dim=30, dropout=0.2)

    # Create predictor
    predictor = ConfidenceAwarePredictor(
        models={
            'short': short_model,
            'medium': medium_model,
            'long': long_model,
        },
        model_weights={'short': 0.5, 'medium': 0.3, 'long': 0.2},
        min_confidence=0.60,
        use_mc_dropout=True,
        mc_samples=20  # Reduced for demo speed
    )

    # Generate some test features
    features = {
        'short': torch.randn(1, 30),
        'medium': torch.randn(1, 30),
        'long': torch.randn(1, 30),
    }

    print("\nMaking ensemble prediction with confidence estimation...")
    print("-" * 60)

    prediction = predictor.predict(features, symbol="EURUSD", timestamp="2024-01-08 10:30:00")

    print(f"\n📊 Prediction Results:")
    print(f"  Symbol: {prediction.symbol}")
    print(f"  Timestamp: {prediction.timestamp}")
    print(f"\n  Direction: {'UP ↑' if prediction.direction == 1 else 'DOWN ↓' if prediction.direction == -1 else 'NO TRADE ─'}")
    print(f"  Raw Probability: {prediction.raw_probability:.3f}")
    print(f"  Confidence: {prediction.confidence:.3f}")
    print(f"  Confidence Level: {prediction.confidence_level}")
    print(f"  Epistemic Uncertainty: {prediction.epistemic_uncertainty:.4f}")
    print(f"\n  Should Trade: {'YES ✓' if prediction.should_trade else 'NO ✗'}")
    print(f"  Position Size: {prediction.position_size:.0%}")

    print(f"\n  Individual Model Predictions:")
    for name, prob in prediction.model_predictions.items():
        conf = prediction.model_confidences.get(name, 0)
        print(f"    {name:8s}: {prob:.3f} (conf: {conf:.3f})")

    print("\n💡 Pipeline Summary:")
    print("   1. Each model makes independent prediction")
    print("   2. MC Dropout estimates per-model uncertainty")
    print("   3. Ensemble combines with weighted average")
    print("   4. Disagreement adds to total uncertainty")
    print("   5. Final confidence determines position size")


def run_all_demos():
    """Run all demonstration functions."""
    print("\n" + "=" * 60)
    print("   CONFIDENCE ESTIMATION SYSTEM - DEMONSTRATIONS")
    print("=" * 60)

    demo_confidence_interpretation()
    demo_calibration()
    demo_mc_dropout()
    demo_ensemble_disagreement()
    demo_trading_signals()
    demo_full_pipeline()

    print("\n" + "=" * 60)
    print("   ALL DEMOS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    run_all_demos()
