"""
Demonstration: Why Learned Uncertainty is Better Than Sigmoid

This script shows the key difference between:
1. Sigmoid approach: Model outputs number → we interpret as confidence
2. Beta approach: Model outputs distribution → confidence is learned

Run: python -m src.models.confidence.comparison_demo
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple

from .learned_uncertainty import (
    BetaOutputLayer,
    BetaPrediction,
    BetaNLLLoss,
    TradingModelWithUncertainty,
)


def demo_sigmoid_vs_beta():
    """
    Demonstrate the fundamental difference between sigmoid and Beta approaches.
    """
    print("\n" + "=" * 70)
    print("   SIGMOID vs BETA: Why Learned Uncertainty is Better")
    print("=" * 70)

    print("""
┌─────────────────────────────────────────────────────────────────────┐
│                     THE FUNDAMENTAL PROBLEM                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  SIGMOID APPROACH:                                                  │
│  ─────────────────                                                  │
│  Model → 0.85 → "85% probability of UP"                             │
│                                                                     │
│  But wait... does 0.85 mean:                                        │
│  A) "I'm 85% confident it will go UP"                               │
│  B) "I'm very confident it will go up by a small amount"            │
│  C) "I'm somewhat confident it will go up by a lot"                 │
│  D) "I have no idea, but 0.85 is my best guess"                     │
│                                                                     │
│  THE MODEL DOESN'T KNOW! It just outputs a number.                  │
│  We ASSUME confidence from distance to 0.5.                         │
│                                                                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  BETA APPROACH:                                                     │
│  ─────────────────                                                  │
│  Model → Beta(α=15, β=3) → "UP with HIGH confidence"                │
│  Model → Beta(α=3, β=1.5) → "UP with LOW confidence"                │
│  Model → Beta(α=1.1, β=1.1) → "I DON'T KNOW"                        │
│                                                                     │
│  The model EXPLICITLY tells us:                                     │
│  - WHAT it predicts (mean = α/(α+β))                                │
│  - HOW SURE it is (concentration = α+β)                             │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
""")

    # Create visual examples
    print("\n" + "─" * 70)
    print("VISUAL COMPARISON: Same mean probability, different confidence")
    print("─" * 70)

    examples = [
        (15.0, 5.0, "HIGH confidence UP (α=15, β=5)"),
        (3.0, 1.0, "LOW confidence UP (α=3, β=1)"),
        (1.5, 0.5, "VERY LOW confidence UP (α=1.5, β=0.5)"),
        (5.0, 15.0, "HIGH confidence DOWN (α=5, β=15)"),
        (1.0, 3.0, "LOW confidence DOWN (α=1, β=3)"),
        (1.1, 1.1, "NO IDEA - uncertain (α=1.1, β=1.1)"),
    ]

    for alpha, beta, description in examples:
        mean = alpha / (alpha + beta)
        concentration = alpha + beta

        # Confidence calculation
        concentration_factor = 1.0 - 2.0 / (concentration + 2.0)
        mean_factor = 2.0 * abs(mean - 0.5)
        confidence = 0.5 + 0.5 * concentration_factor * mean_factor

        # Sigmoid equivalent (just the mean)
        sigmoid_confidence = mean if mean >= 0.5 else (1 - mean)

        # Create simple ASCII visualization
        width = 40
        mean_pos = int(mean * width)
        bar = ["-"] * width
        bar[mean_pos] = "|"

        # Show distribution width based on concentration
        spread = max(1, int(10 / concentration))
        for i in range(max(0, mean_pos - spread), min(width, mean_pos + spread + 1)):
            bar[i] = "#"
        bar[mean_pos] = "█"

        print(f"\n{description}")
        print(f"  Mean: {mean:.2f}, Concentration: {concentration:.1f}")
        print(f"  [{''.join(bar)}]")
        print(f"   0                  0.5                  1")
        print(f"  Beta confidence:   {confidence:.3f}")
        print(f"  Sigmoid confidence: {sigmoid_confidence:.3f}")

        if abs(confidence - sigmoid_confidence) > 0.1:
            print(f"  ⚠️  DIFFERENCE: {abs(confidence - sigmoid_confidence):.3f}")


def demo_learning_uncertainty():
    """
    Show how the model LEARNS when to be confident.
    """
    print("\n" + "=" * 70)
    print("   DEMO: Model Learning When to Be Confident")
    print("=" * 70)

    # Create synthetic data with different "difficulty" levels
    np.random.seed(42)
    torch.manual_seed(42)

    n_samples = 500

    # Easy samples: clear pattern
    easy_x = torch.randn(n_samples // 2, 20)
    easy_x[:, 0] = torch.where(torch.rand(n_samples // 2) > 0.5,
                                torch.ones(n_samples // 2) * 3,
                                torch.ones(n_samples // 2) * -3)
    easy_y = (easy_x[:, 0] > 0).float()

    # Hard samples: noisy, ambiguous
    hard_x = torch.randn(n_samples // 2, 20) * 2
    hard_x[:, 0] = torch.randn(n_samples // 2) * 0.5  # Very noisy signal
    hard_y = (torch.rand(n_samples // 2) > 0.5).float()  # Random labels

    # Combine
    x = torch.cat([easy_x, hard_x])
    y = torch.cat([easy_y, hard_y])
    difficulty = torch.cat([torch.zeros(n_samples // 2), torch.ones(n_samples // 2)])

    # Shuffle
    perm = torch.randperm(n_samples)
    x, y, difficulty = x[perm], y[perm], difficulty[perm]

    # Train model
    print("\nTraining model on mixed easy/hard data...")

    model = TradingModelWithUncertainty(
        input_dim=20,
        hidden_dim=64,
        output_type='beta'
    )
    criterion = BetaNLLLoss(reg_weight=0.05)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

    model.train()
    for epoch in range(200):
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs['direction'], y)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 50 == 0:
            print(f"  Epoch {epoch+1}: Loss = {loss.item():.4f}")

    # Evaluate
    model.eval()
    with torch.no_grad():
        results = model.predict_with_confidence(x)

    easy_mask = difficulty == 0
    hard_mask = difficulty == 1

    print("\n" + "─" * 70)
    print("RESULTS: Does the model know when it should be confident?")
    print("─" * 70)

    print(f"\nEASY samples (clear patterns):")
    print(f"  Average confidence: {results['confidence'][easy_mask].mean():.3f}")
    print(f"  Concentration: {(results['alpha'][easy_mask] + results['beta'][easy_mask]).mean():.1f}")
    accuracy_easy = (results['direction'][easy_mask] == y[easy_mask]).float().mean()
    print(f"  Accuracy: {accuracy_easy:.1%}")

    print(f"\nHARD samples (random/noisy):")
    print(f"  Average confidence: {results['confidence'][hard_mask].mean():.3f}")
    print(f"  Concentration: {(results['alpha'][hard_mask] + results['beta'][hard_mask]).mean():.1f}")
    accuracy_hard = (results['direction'][hard_mask] == y[hard_mask]).float().mean()
    print(f"  Accuracy: {accuracy_hard:.1%}")

    conf_diff = results['confidence'][easy_mask].mean() - results['confidence'][hard_mask].mean()
    print(f"\n✓ Confidence difference: {conf_diff:.3f}")

    if conf_diff > 0.05:
        print("  ✅ Model correctly learned to be MORE confident on easy samples!")
    else:
        print("  ⚠️  Model needs more training or data to distinguish difficulty")


def demo_trading_decision():
    """
    Show how confidence affects trading decisions.
    """
    print("\n" + "=" * 70)
    print("   DEMO: Confidence-Based Trading Decisions")
    print("=" * 70)

    scenarios = [
        {
            'name': "Strong Bullish Signal",
            'alpha': 20.0, 'beta': 3.0,
            'expected_action': "FULL POSITION LONG"
        },
        {
            'name': "Weak Bullish Signal",
            'alpha': 2.5, 'beta': 1.5,
            'expected_action': "SMALL POSITION LONG"
        },
        {
            'name': "Uncertain Market",
            'alpha': 1.2, 'beta': 1.2,
            'expected_action': "NO TRADE"
        },
        {
            'name': "Weak Bearish Signal",
            'alpha': 1.5, 'beta': 2.5,
            'expected_action': "SMALL POSITION SHORT"
        },
        {
            'name': "Strong Bearish Signal",
            'alpha': 3.0, 'beta': 20.0,
            'expected_action': "FULL POSITION SHORT"
        },
    ]

    print("\n" + "─" * 70)
    print(f"{'Scenario':<25} {'Mean':<8} {'Conf':<8} {'Action':<25}")
    print("─" * 70)

    for s in scenarios:
        pred = BetaPrediction(
            alpha=torch.tensor(s['alpha']),
            beta=torch.tensor(s['beta'])
        )

        mean = pred.mean.item()
        conf = pred.confidence.item()
        direction = "LONG" if mean >= 0.5 else "SHORT"

        # Position sizing
        if conf >= 0.85:
            size = "100%"
            action = f"FULL {direction}"
        elif conf >= 0.70:
            size = "50%"
            action = f"MEDIUM {direction}"
        elif conf >= 0.55:
            size = "25%"
            action = f"SMALL {direction}"
        else:
            size = "0%"
            action = "NO TRADE"

        print(f"{s['name']:<25} {mean:<8.3f} {conf:<8.3f} {action:<25}")

    print("\n" + "─" * 70)
    print("""
KEY INSIGHT:
───────────
With Beta distribution, the model tells us BOTH:
1. DIRECTION: α > β means UP, α < β means DOWN
2. CONFIDENCE: High (α+β) means "I'm sure", Low means "I don't know"

With sigmoid, we only get direction. We GUESS confidence from
distance to 0.5, but the model doesn't actually KNOW if it's confident!
""")


def demo_comparison_table():
    """
    Summary comparison of all approaches.
    """
    print("\n" + "=" * 70)
    print("   COMPARISON: All Uncertainty Methods")
    print("=" * 70)

    print("""
┌────────────────┬─────────────────────┬─────────────────────┬────────────┐
│ Method         │ How Confidence      │ Key Advantage       │ Best For   │
│                │ is Obtained         │                     │            │
├────────────────┼─────────────────────┼─────────────────────┼────────────┤
│ Sigmoid        │ Distance from 0.5   │ Simple              │ Baseline   │
│ (current)      │ (DERIVED)           │                     │            │
├────────────────┼─────────────────────┼─────────────────────┼────────────┤
│ Sigmoid +      │ Post-hoc adjustment │ Better calibrated   │ Existing   │
│ Calibration    │ (CORRECTED)         │ probabilities       │ models     │
├────────────────┼─────────────────────┼─────────────────────┼────────────┤
│ MC Dropout     │ Variance over runs  │ No architecture     │ Quick      │
│                │ (ESTIMATED)         │ changes needed      │ estimate   │
├────────────────┼─────────────────────┼─────────────────────┼────────────┤
│ Deep Ensemble  │ Model disagreement  │ Very reliable       │ Production │
│                │ (COMPUTED)          │                     │ systems    │
├────────────────┼─────────────────────┼─────────────────────┼────────────┤
│ BETA OUTPUT    │ Concentration       │ Model LEARNS when   │ RECOMMENDED│
│ (recommended)  │ (LEARNED)           │ to be confident     │ for trading│
├────────────────┼─────────────────────┼─────────────────────┼────────────┤
│ Quantile       │ Interval width      │ Direct risk bounds  │ Risk       │
│ Regression     │ (LEARNED)           │ (10th/90th %)       │ management │
├────────────────┼─────────────────────┼─────────────────────┼────────────┤
│ Gaussian       │ Predicted variance  │ Best for            │ Price/     │
│ Output         │ (LEARNED)           │ continuous targets  │ return     │
└────────────────┴─────────────────────┴─────────────────────┴────────────┘

RECOMMENDATION FOR TRADING:
───────────────────────────
Use BETA OUTPUT for direction (up/down) predictions because:

1. Model explicitly learns "how sure am I about this prediction?"
2. Single forward pass (efficient for real-time trading)
3. Natural interpretation: α=evidence for UP, β=evidence for DOWN
4. Concentration (α+β) directly indicates confidence level
5. Can say "I don't know" (low concentration) instead of guessing

Optionally add QUANTILE OUTPUT for risk bounds:
- 10th percentile: worst case
- 90th percentile: best case
- Width = uncertainty for position sizing
""")


def run_all_demos():
    """Run all demonstration functions."""
    demo_sigmoid_vs_beta()
    demo_learning_uncertainty()
    demo_trading_decision()
    demo_comparison_table()

    print("\n" + "=" * 70)
    print("   CONCLUSION")
    print("=" * 70)
    print("""
The BEST approach for your trading system:

1. Replace sigmoid output with BETA OUTPUT LAYER
   - Model learns to output Beta(α, β) distribution
   - Direction = mean = α/(α+β)
   - Confidence = function of concentration (α+β)

2. Use BETA NLL LOSS for training
   - Encourages high confidence on correct predictions
   - Low confidence when uncertain

3. Position sizing based on learned confidence:
   - High confidence (>85%): Full position
   - Medium confidence (70-85%): Half position
   - Low confidence (<70%): No trade

This is BETTER than sigmoid because the model LEARNS when
it should be confident vs uncertain, rather than us guessing
from the output value.
""")


if __name__ == "__main__":
    run_all_demos()
