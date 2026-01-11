# Confidence & Uncertainty System

## 1. Overview

The Confidence & Uncertainty System is a critical component of the AI Assets Trader that enables the model to express not just WHAT it predicts, but HOW SURE it is about that prediction. This enables intelligent position sizing and risk management.

**Key Insight**: Traditional sigmoid outputs only provide a probability, but not confidence. The model might output 0.75 meaning "75% probability of UP", but it doesn't tell us if it's confident in that prediction or just guessing.

**Our Solution**: Use **Beta distribution outputs** that learn both direction AND confidence simultaneously.

## 2. The Problem with Sigmoid Confidence

### 2.1 What Sigmoid Provides

```
Model → sigmoid → 0.75 → "75% probability UP"

Typical interpretation:
- Distance from 0.5 = confidence
- 0.75 → 25% distance from 0.5 → "75% confident in UP"
```

### 2.2 Why This Is Wrong

The distance from 0.5 is **derived**, not **learned**. Consider these scenarios:

| Output | Derived Confidence | But Could Mean |
|--------|-------------------|----------------|
| 0.75 | 75% | "I'm very sure it goes UP slightly" |
| 0.75 | 75% | "I'm somewhat sure it goes UP a lot" |
| 0.75 | 75% | "I have no idea, 0.75 is just noise" |

**The model has no mechanism to distinguish these cases!**

### 2.3 Real-World Consequences

```
Scenario A: Clear bullish pattern after consolidation
  → Model outputs 0.80
  → We interpret: "80% confident UP"
  → Correct interpretation

Scenario B: Random noise, unclear market
  → Model outputs 0.80 (because that's what it learned to output)
  → We interpret: "80% confident UP"
  → WRONG! Model is just guessing

Result: We take the same position size for both scenarios,
        but Scenario B should be a NO TRADE.
```

## 3. The Beta Distribution Solution

### 3.1 What Beta Distribution Provides

Instead of a single value, the model outputs two parameters (α, β) that define a probability distribution:

```
Model → Beta(α, β) → Distribution over [0, 1]

Two separate pieces of information:
1. DIRECTION: mean = α/(α+β)
   - If mean > 0.5: Prediction is UP
   - If mean < 0.5: Prediction is DOWN

2. CONFIDENCE: concentration = α + β
   - High concentration (α+β > 10): "I'm very sure"
   - Low concentration (α+β < 4): "I'm not sure"
   - α ≈ β ≈ 1: "I have no idea" (uniform distribution)
```

### 3.2 Visual Comparison

```
SIGMOID (old approach):
────────────────────────
Output: 0.75
We ASSUME: 75% confident UP
Reality: Could be confident or uncertain - we don't know!

[.........|.....█.....]
0        0.5   ↑      1
               0.75

BETA (recommended approach):
────────────────────────
Output: Beta(α=15, β=5)  →  Mean: 0.75, Concentration: 20
Model TELLS us: "I'm confident it's UP"

[............█████.....]  ← Narrow peak = HIGH confidence
0        0.5   ↑      1
               0.75

Output: Beta(α=3, β=1)  →  Mean: 0.75, Concentration: 4
Model TELLS us: "I think UP, but I'm not sure"

[.....###########......]  ← Wide spread = LOW confidence
0        0.5   ↑      1
               0.75

Output: Beta(α=1.1, β=1.1)  →  Mean: 0.50, Concentration: 2.2
Model TELLS us: "I have NO IDEA"

[######################]  ← Uniform = UNCERTAIN
0        0.5         1
```

### 3.3 Why This Works

The model **learns when to be confident** through the loss function:

1. **Correct prediction + high concentration** → Low loss (rewarded)
2. **Correct prediction + low concentration** → Medium loss (not fully rewarded)
3. **Wrong prediction + high concentration** → Very high loss (heavily penalized)
4. **Wrong prediction + low concentration** → Medium loss (somewhat forgiven)

This incentivizes the model to:
- Be confident when it has good evidence
- Be uncertain when the data is ambiguous
- Say "I don't know" rather than guess

## 4. Implementation Details

### 4.1 Beta Output Layer

```python
# src/models/confidence/learned_uncertainty.py

class BetaOutputLayer(nn.Module):
    """
    Outputs Beta distribution parameters for binary classification
    with learned uncertainty.
    """
    def __init__(self, input_dim: int, min_concentration: float = 1.0):
        super().__init__()
        self.alpha_layer = nn.Linear(input_dim, 1)
        self.beta_layer = nn.Linear(input_dim, 1)
        self.min_concentration = min_concentration

    def forward(self, x: torch.Tensor) -> BetaPrediction:
        # Softplus ensures positive values
        alpha = F.softplus(self.alpha_layer(x)) + self.min_concentration
        beta = F.softplus(self.beta_layer(x)) + self.min_concentration
        return BetaPrediction(alpha=alpha.squeeze(-1), beta=beta.squeeze(-1))
```

### 4.2 Beta Prediction Dataclass

```python
@dataclass
class BetaPrediction:
    """Prediction from Beta output layer."""
    alpha: torch.Tensor  # Evidence for class 1 (UP)
    beta: torch.Tensor   # Evidence for class 0 (DOWN)

    @property
    def mean(self) -> torch.Tensor:
        """Expected probability of UP."""
        return self.alpha / (self.alpha + self.beta)

    @property
    def concentration(self) -> torch.Tensor:
        """Total evidence (α + β). Higher = more confident."""
        return self.alpha + self.beta

    @property
    def confidence(self) -> torch.Tensor:
        """
        Confidence score [0.5, 1.0].

        Combines:
        - Distance of mean from 0.5 (direction strength)
        - Concentration (certainty level)
        """
        concentration_factor = 1.0 - 2.0 / (self.concentration + 2.0)
        mean_factor = 2.0 * torch.abs(self.mean - 0.5)
        return 0.5 + 0.5 * concentration_factor * mean_factor

    @property
    def direction(self) -> torch.Tensor:
        """Binary direction: 1 = UP, 0 = DOWN."""
        return (self.mean >= 0.5).long()
```

### 4.3 Beta NLL Loss Function

```python
class BetaNLLLoss(nn.Module):
    """
    Negative log-likelihood loss for Beta distribution.

    This loss function:
    - Rewards high concentration for correct predictions
    - Penalizes high concentration for incorrect predictions
    - Allows model to learn "I don't know" (low concentration)
    """
    def __init__(self, reg_weight: float = 0.05):
        super().__init__()
        self.reg_weight = reg_weight

    def forward(self, prediction: BetaPrediction, target: torch.Tensor) -> torch.Tensor:
        alpha = prediction.alpha
        beta = prediction.beta

        # Negative log-likelihood of Beta distribution
        nll = -torch.lgamma(alpha + beta) + torch.lgamma(alpha) + torch.lgamma(beta)
        nll -= (alpha - 1) * torch.log(target + 1e-8)
        nll -= (beta - 1) * torch.log(1 - target + 1e-8)

        # Optional regularization to prevent extreme concentrations
        reg = self.reg_weight * (alpha + beta)

        return nll.mean() + reg.mean()
```

## 5. Complete Model with Uncertainty

### 5.1 TradingModelWithUncertainty

```python
class TradingModelWithUncertainty(nn.Module):
    """
    Complete trading model with learned uncertainty outputs.

    Supports multiple output types:
    - 'beta': For binary direction prediction (UP/DOWN) - RECOMMENDED
    - 'gaussian': For continuous price prediction
    - 'dirichlet': For multi-class regime classification
    - 'quantile': For risk bounds (10th, 90th percentile)
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        output_type: str = 'beta',  # 'beta', 'gaussian', 'dirichlet', 'quantile'
    ):
        super().__init__()

        # Shared backbone
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        # Output head based on type
        if output_type == 'beta':
            self.output_head = BetaOutputLayer(hidden_dim)
        elif output_type == 'gaussian':
            self.output_head = GaussianOutputLayer(hidden_dim, 1)
        elif output_type == 'dirichlet':
            self.output_head = DirichletOutputLayer(hidden_dim, 4)  # 4 regimes
        elif output_type == 'quantile':
            self.output_head = QuantileOutputLayer(hidden_dim)

    def forward(self, x: torch.Tensor) -> Dict:
        features = self.backbone(x)
        return {'direction': self.output_head(features)}

    def predict_with_confidence(self, x: torch.Tensor) -> Dict:
        """Get predictions with confidence scores."""
        self.eval()
        with torch.no_grad():
            output = self.forward(x)
            pred = output['direction']

            return {
                'direction': pred.direction,
                'probability': pred.mean,
                'confidence': pred.confidence,
                'alpha': pred.alpha,
                'beta': pred.beta,
                'concentration': pred.concentration,
            }
```

## 6. Position Sizing Based on Confidence

### 6.1 Confidence Thresholds

```python
CONFIDENCE_THRESHOLDS = {
    'very_high': 0.85,   # Full position (100%)
    'high': 0.70,        # Large position (75%)
    'moderate': 0.55,    # Small position (25%)
    'minimum': 0.50,     # No trade threshold
}
```

### 6.2 Position Sizing Logic

```python
def calculate_position_size(confidence: float, max_position: float = 1.0) -> float:
    """
    Calculate position size based on model confidence.

    Args:
        confidence: Model confidence score [0.5, 1.0]
        max_position: Maximum position size (1.0 = 100%)

    Returns:
        Position size as fraction of max_position
    """
    if confidence >= 0.85:
        return max_position * 1.0    # 100%
    elif confidence >= 0.70:
        return max_position * 0.50   # 50%
    elif confidence >= 0.55:
        return max_position * 0.25   # 25%
    else:
        return 0.0                    # NO TRADE
```

### 6.3 Position Sizing Table

| Confidence Range | Concentration (α+β) | Position Size | Rationale |
|------------------|---------------------|---------------|-----------|
| ≥ 0.85 | ≥ 15 | 100% | Very high confidence, full position |
| 0.70 - 0.85 | 8 - 15 | 50% | High confidence, reduced position |
| 0.55 - 0.70 | 4 - 8 | 25% | Moderate confidence, small position |
| < 0.55 | < 4 | 0% | Low confidence, NO TRADE |

## 7. Alternative Uncertainty Methods

While Beta output is recommended for direction prediction, the system also supports:

### 7.1 For Continuous Predictions (Price)

**Gaussian Output Layer**: Outputs mean (μ) and variance (σ²)
```python
# Use for price prediction with uncertainty bounds
# Output: GaussianPrediction(mean, variance)
# Confidence: 1 / (1 + sqrt(variance))
```

### 7.2 For Multi-Class (Market Regime)

**Dirichlet Output Layer**: Outputs concentration parameters for each class
```python
# Use for regime classification with uncertainty
# Output: DirichletPrediction(concentrations)
# Classes: trending_up, trending_down, ranging, volatile
```

### 7.3 For Risk Bounds

**Quantile Regression**: Outputs specific percentiles
```python
# Use for establishing risk bounds
# Output: QuantilePrediction(q10, q50, q90)
# Risk: Width of q10-q90 interval
```

### 7.4 Post-Hoc Methods (When Beta Not Feasible)

If you have an existing sigmoid model and can't retrain:

1. **Temperature Scaling**: Calibrates probability outputs
2. **MC Dropout**: Multiple forward passes with dropout enabled
3. **Ensemble Disagreement**: Variance across multiple models

These are implemented in:
- `src/models/confidence/calibration.py`
- `src/models/confidence/uncertainty.py`

## 8. Integration with Trading Pipeline

### 8.1 ConfidenceAwarePredictor

```python
from src.models.confidence import ConfidenceAwarePredictor

predictor = ConfidenceAwarePredictor(
    models={
        'short': short_term_model,
        'medium': medium_term_model,
        'long': long_term_model,
    },
    model_weights={'short': 0.5, 'medium': 0.3, 'long': 0.2},
    min_confidence=0.55,
)

prediction = predictor.predict(features, symbol="EURUSD")

print(f"Direction: {prediction.direction}")
print(f"Confidence: {prediction.confidence}")
print(f"Should Trade: {prediction.should_trade}")
print(f"Position Size: {prediction.position_size}")
```

### 8.2 Trading Decision Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    TRADING DECISION FLOW                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Input: Market Features (OHLCV + Indicators)                    │
│                          │                                       │
│                          ▼                                       │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │         Models with Beta Output Layers                   │    │
│  │  ┌───────────┐  ┌───────────┐  ┌───────────┐           │    │
│  │  │Short-Term │  │Medium-Term│  │Long-Term  │           │    │
│  │  │ Beta(α,β) │  │ Beta(α,β) │  │ Beta(α,β) │           │    │
│  │  └─────┬─────┘  └─────┬─────┘  └─────┬─────┘           │    │
│  └────────┼──────────────┼──────────────┼──────────────────┘    │
│           │              │              │                        │
│           ▼              ▼              ▼                        │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              Ensemble Combination                         │    │
│  │  - Weighted by model_weights                             │    │
│  │  - Adjusted by model confidence                          │    │
│  │  - Penalized for disagreement                            │    │
│  └─────────────────────────────────────────────────────────┘    │
│                          │                                       │
│                          ▼                                       │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              Final Prediction                             │    │
│  │  - direction: UP/DOWN/NEUTRAL                            │    │
│  │  - confidence: 0.5 - 1.0 (LEARNED)                       │    │
│  │  - should_trade: True/False                              │    │
│  │  - position_size: 0% - 100%                              │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## 9. Demonstrations and Testing

### 9.1 Run Comparison Demo

```bash
# See why Beta is better than Sigmoid
python -m src.models.confidence.comparison_demo
```

### 9.2 Run Full Examples

```bash
# See all confidence estimation methods
python -m src.models.confidence.examples
```

### 9.3 Expected Output

```
=======================================================================
   SIGMOID vs BETA: Why Learned Uncertainty is Better
=======================================================================

HIGH confidence UP (α=15, β=5)
  Mean: 0.75, Concentration: 20.0
  [--------------------########█####-------]
  Beta confidence:   0.857
  Sigmoid confidence: 0.750
  WARNING: DIFFERENCE: 0.107

LOW confidence UP (α=3, β=1)
  Mean: 0.75, Concentration: 4.0
  [------------------##########█##########-]
  Beta confidence:   0.589
  Sigmoid confidence: 0.750
  WARNING: DIFFERENCE: 0.161
```

## 10. Key Files Reference

| File | Purpose |
|------|---------|
| `src/models/confidence/__init__.py` | Module exports |
| `src/models/confidence/learned_uncertainty.py` | Beta/Gaussian/Dirichlet output layers |
| `src/models/confidence/calibration.py` | Temperature/Platt/Isotonic calibration |
| `src/models/confidence/uncertainty.py` | MC Dropout, Ensemble uncertainty |
| `src/models/confidence/integration.py` | ConfidenceAwarePredictor |
| `src/models/confidence/examples.py` | Demo functions |
| `src/models/confidence/comparison_demo.py` | Sigmoid vs Beta comparison |

## 11. Summary

**Why Beta Distribution for Direction Prediction?**

1. **Model LEARNS confidence** - Not derived from output value
2. **Can express "I don't know"** - Low concentration signals uncertainty
3. **Natural interpretation** - α = evidence for UP, β = evidence for DOWN
4. **Single forward pass** - Efficient for real-time trading
5. **Direct position sizing** - Use concentration for position size

**When to Use Each Method:**

| Output Type | Use Case | Key Advantage |
|-------------|----------|---------------|
| **Beta** | Direction (UP/DOWN) | Learned confidence - RECOMMENDED |
| **Gaussian** | Price prediction | Learned variance for bounds |
| **Dirichlet** | Market regime | Multi-class with uncertainty |
| **Quantile** | Risk management | Direct 10%/90% bounds |

---

*Document Version: 1.0*
*Last Updated: 2026-01-08*
*Author: AI Trader Development Team*
