"""
Learned Uncertainty: Models that directly predict their own uncertainty.

This is superior to deriving confidence from sigmoid outputs because:
1. The model LEARNS when it should be confident vs uncertain
2. Single forward pass (efficient)
3. Captures data-dependent (aleatoric) uncertainty
4. Can separate "I don't know" from "the data is noisy"

Methods implemented:
1. Gaussian Output - Predicts mean + variance (heteroscedastic)
2. Evidential Classification - Dirichlet distribution over classes
3. Quantile Output - Predicts multiple percentiles
4. Beta Output - Full distribution for binary classification

References:
- Kendall & Gal "What Uncertainties Do We Need?" (2017)
- Sensoy et al. "Evidential Deep Learning" (2018)
- Lakshminarayanan et al. "Deep Ensembles" (2017)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, List, NamedTuple
from dataclasses import dataclass
import math


# =============================================================================
# Output Structures
# =============================================================================

@dataclass
class GaussianPrediction:
    """Prediction with learned Gaussian uncertainty."""
    mean: torch.Tensor  # Predicted value
    variance: torch.Tensor  # Learned variance (uncertainty)

    @property
    def std(self) -> torch.Tensor:
        return torch.sqrt(self.variance)

    @property
    def confidence(self) -> torch.Tensor:
        """Higher variance = lower confidence. Scale to [0, 1]."""
        # Use inverse relationship: conf = 1 / (1 + std)
        return 1.0 / (1.0 + self.std)

    def to_interval(self, z: float = 1.96) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get confidence interval (default 95%)."""
        return self.mean - z * self.std, self.mean + z * self.std


@dataclass
class DirichletPrediction:
    """Prediction with Dirichlet-based uncertainty (for classification)."""
    alpha: torch.Tensor  # Dirichlet concentration parameters

    @property
    def probabilities(self) -> torch.Tensor:
        """Expected class probabilities."""
        return self.alpha / self.alpha.sum(dim=-1, keepdim=True)

    @property
    def total_evidence(self) -> torch.Tensor:
        """Total evidence (sum of alphas). Higher = more confident."""
        return self.alpha.sum(dim=-1)

    @property
    def uncertainty(self) -> torch.Tensor:
        """Epistemic uncertainty: K / total_evidence."""
        K = self.alpha.shape[-1]
        return K / self.total_evidence

    @property
    def confidence(self) -> torch.Tensor:
        """Confidence = 1 - uncertainty, scaled to [0.5, 1]."""
        # Map uncertainty [0, 1] to confidence [0.5, 1]
        return 1.0 - 0.5 * self.uncertainty.clamp(0, 1)


@dataclass
class BetaPrediction:
    """Prediction with Beta distribution (for binary classification)."""
    alpha: torch.Tensor  # Successes + 1
    beta: torch.Tensor   # Failures + 1

    @property
    def mean(self) -> torch.Tensor:
        """Expected probability of positive class."""
        return self.alpha / (self.alpha + self.beta)

    @property
    def variance(self) -> torch.Tensor:
        """Variance of the probability estimate."""
        total = self.alpha + self.beta
        return (self.alpha * self.beta) / (total ** 2 * (total + 1))

    @property
    def concentration(self) -> torch.Tensor:
        """Total concentration (alpha + beta). Higher = more confident."""
        return self.alpha + self.beta

    @property
    def confidence(self) -> torch.Tensor:
        """Confidence based on concentration and distance from 0.5."""
        # Two factors:
        # 1. How concentrated is the distribution (high α+β = confident)
        # 2. How far from 0.5 is the mean (edge = confident)

        concentration_factor = 1.0 - 2.0 / (self.concentration + 2.0)
        mean_factor = 2.0 * torch.abs(self.mean - 0.5)  # [0, 1]

        # Combine: need BOTH high concentration AND clear direction
        return 0.5 + 0.5 * concentration_factor * mean_factor

    @property
    def prediction(self) -> torch.Tensor:
        """Binary prediction (0 or 1)."""
        return (self.mean >= 0.5).long()


@dataclass
class QuantilePrediction:
    """Prediction with multiple quantiles for uncertainty bounds."""
    quantiles: Dict[float, torch.Tensor]  # {0.1: lower, 0.5: median, 0.9: upper}

    @property
    def median(self) -> torch.Tensor:
        return self.quantiles.get(0.5, list(self.quantiles.values())[len(self.quantiles)//2])

    @property
    def lower(self) -> torch.Tensor:
        """Lower bound (10th percentile by default)."""
        return self.quantiles.get(0.1, min(self.quantiles.values()))

    @property
    def upper(self) -> torch.Tensor:
        """Upper bound (90th percentile by default)."""
        return self.quantiles.get(0.9, max(self.quantiles.values()))

    @property
    def interval_width(self) -> torch.Tensor:
        """Width of prediction interval (proxy for uncertainty)."""
        return self.upper - self.lower

    @property
    def confidence(self) -> torch.Tensor:
        """Narrow interval = high confidence."""
        # Normalize by median to get relative width
        relative_width = self.interval_width / (torch.abs(self.median) + 1e-6)
        # Invert: narrow = confident
        return 1.0 / (1.0 + relative_width)


# =============================================================================
# Output Layers
# =============================================================================

class GaussianOutputLayer(nn.Module):
    """
    Output layer that predicts mean and variance.

    The model learns DATA-DEPENDENT uncertainty (aleatoric).
    When the input is noisy/ambiguous, variance is high.
    When the pattern is clear, variance is low.

    This is BETTER than sigmoid because:
    - Model explicitly learns "how sure am I about this input?"
    - Not deriving confidence from a single number
    """

    def __init__(self, input_dim: int, min_variance: float = 1e-6):
        super().__init__()
        self.mean_layer = nn.Linear(input_dim, 1)
        self.variance_layer = nn.Linear(input_dim, 1)
        self.min_variance = min_variance

    def forward(self, x: torch.Tensor) -> GaussianPrediction:
        mean = self.mean_layer(x)
        # Use softplus to ensure positive variance
        variance = F.softplus(self.variance_layer(x)) + self.min_variance
        return GaussianPrediction(mean=mean, variance=variance)


class BetaOutputLayer(nn.Module):
    """
    Output layer that predicts Beta distribution parameters.

    For BINARY CLASSIFICATION with uncertainty.

    Instead of: sigmoid → 0.75 → "75% confident it's UP"
    We get: Beta(α=15, β=5) → "I think UP, and I'm very sure"
            Beta(α=3, β=2) → "I think UP, but I'm not sure"
            Beta(α=1.1, β=1.1) → "I have no idea"

    The concentration (α + β) indicates HOW MUCH EVIDENCE the model has.
    Low concentration = "I don't know" (uncertainty)
    High concentration = "I'm confident"
    """

    def __init__(self, input_dim: int, min_concentration: float = 1.0):
        super().__init__()
        self.alpha_layer = nn.Linear(input_dim, 1)
        self.beta_layer = nn.Linear(input_dim, 1)
        self.min_concentration = min_concentration

    def forward(self, x: torch.Tensor) -> BetaPrediction:
        # Use softplus to ensure positive, add minimum
        alpha = F.softplus(self.alpha_layer(x)) + self.min_concentration
        beta = F.softplus(self.beta_layer(x)) + self.min_concentration
        return BetaPrediction(alpha=alpha.squeeze(-1), beta=beta.squeeze(-1))


class DirichletOutputLayer(nn.Module):
    """
    Output layer for Evidential Deep Learning (multi-class).

    Predicts Dirichlet distribution parameters (evidence for each class).

    Total evidence = sum of alphas = model's confidence
    Low evidence = "I don't know" (high uncertainty)
    High evidence for one class = "I'm confident it's this class"

    Key advantage: Separates WHAT the model predicts from HOW SURE it is.
    """

    def __init__(self, input_dim: int, num_classes: int = 2):
        super().__init__()
        self.evidence_layer = nn.Linear(input_dim, num_classes)
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> DirichletPrediction:
        # Evidence must be non-negative, use softplus or exp
        evidence = F.softplus(self.evidence_layer(x))
        # Alpha = evidence + 1 (Dirichlet prior)
        alpha = evidence + 1.0
        return DirichletPrediction(alpha=alpha)


class QuantileOutputLayer(nn.Module):
    """
    Output layer that predicts multiple quantiles.

    Predicts: 10th, 25th, 50th, 75th, 90th percentiles

    Advantages:
    - Direct uncertainty bounds (not derived)
    - No distributional assumptions
    - Actionable for risk management

    For trading:
    - 10th percentile: worst case scenario
    - 50th percentile: expected value
    - 90th percentile: best case scenario
    - Width of interval: uncertainty
    """

    def __init__(
        self,
        input_dim: int,
        quantiles: List[float] = [0.1, 0.25, 0.5, 0.75, 0.9]
    ):
        super().__init__()
        self.quantiles = quantiles
        self.output_layer = nn.Linear(input_dim, len(quantiles))

    def forward(self, x: torch.Tensor) -> QuantilePrediction:
        outputs = self.output_layer(x)

        # Enforce monotonicity (q10 < q25 < q50 < q75 < q90)
        # Use cumulative softplus to ensure ordering
        sorted_outputs = self._enforce_monotonicity(outputs)

        quantile_dict = {
            q: sorted_outputs[:, i:i+1]
            for i, q in enumerate(self.quantiles)
        }

        return QuantilePrediction(quantiles=quantile_dict)

    def _enforce_monotonicity(self, x: torch.Tensor) -> torch.Tensor:
        """Ensure quantiles are monotonically increasing."""
        # First output is the base
        # Each subsequent output adds a positive delta
        deltas = F.softplus(x)
        return torch.cumsum(deltas, dim=-1) - deltas[:, 0:1]  # Center around first


# =============================================================================
# Loss Functions
# =============================================================================

class GaussianNLLLoss(nn.Module):
    """
    Negative Log-Likelihood loss for Gaussian outputs.

    Trains the model to predict both mean AND variance.

    Loss = 0.5 * [log(variance) + (y - mean)² / variance]

    This encourages:
    - Accurate mean predictions
    - High variance when predictions are wrong
    - Low variance when predictions are accurate
    """

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(
        self,
        prediction: GaussianPrediction,
        target: torch.Tensor
    ) -> torch.Tensor:
        variance = prediction.variance + self.eps
        loss = 0.5 * (
            torch.log(variance) +
            (target - prediction.mean) ** 2 / variance
        )
        return loss.mean()


class BetaNLLLoss(nn.Module):
    """
    Loss for Beta distribution outputs.

    Uses the negative log-likelihood of the Beta distribution.

    For binary targets (0 or 1), this becomes:
    - If target=1: loss = -log(prob) where prob = α/(α+β)
    - If target=0: loss = -log(1-prob)

    Plus a regularization term that encourages appropriate concentration.
    """

    def __init__(self, reg_weight: float = 0.01):
        super().__init__()
        self.reg_weight = reg_weight

    def forward(
        self,
        prediction: BetaPrediction,
        target: torch.Tensor
    ) -> torch.Tensor:
        alpha = prediction.alpha
        beta = prediction.beta

        # For binary targets, use simplified loss
        # This is equivalent to Beta NLL for 0/1 targets
        prob = alpha / (alpha + beta)

        # Binary cross entropy (but with our predicted probability)
        bce = F.binary_cross_entropy(
            prob,
            target.float(),
            reduction='none'
        )

        # Regularization: penalize very low concentration (uncertain but wrong)
        # and very high concentration when wrong
        concentration = alpha + beta

        # KL divergence from uniform Beta(1,1)
        # Encourages confident predictions only when correct
        kl_reg = torch.lgamma(alpha + beta) - torch.lgamma(alpha) - torch.lgamma(beta)

        loss = bce + self.reg_weight * torch.abs(kl_reg)
        return loss.mean()


class EvidentialLoss(nn.Module):
    """
    Loss for Evidential Deep Learning (Dirichlet outputs).

    From Sensoy et al. "Evidential Deep Learning to Quantify Classification Uncertainty"

    Key idea:
    - Correct predictions should have HIGH evidence
    - Wrong predictions should have LOW evidence
    - Uncertain data should result in low total evidence

    Loss components:
    1. Cross-entropy-like term for classification accuracy
    2. KL divergence to uniform Dirichlet (regularization)
    """

    def __init__(self, kl_weight: float = 0.1, annealing_epochs: int = 10):
        super().__init__()
        self.kl_weight = kl_weight
        self.annealing_epochs = annealing_epochs
        self.current_epoch = 0

    def forward(
        self,
        prediction: DirichletPrediction,
        target: torch.Tensor
    ) -> torch.Tensor:
        alpha = prediction.alpha
        S = alpha.sum(dim=-1, keepdim=True)  # Total evidence

        # Convert target to one-hot if needed
        if target.dim() == 1:
            num_classes = alpha.shape[-1]
            target_onehot = F.one_hot(target.long(), num_classes).float()
        else:
            target_onehot = target

        # Type I loss: penalize wrong classifications
        # This is like cross-entropy but for Dirichlet
        p = alpha / S
        loss_ce = (target_onehot * (torch.digamma(S) - torch.digamma(alpha))).sum(dim=-1)

        # Type II loss: KL divergence to uniform Dirichlet
        # Removes misleading evidence for wrong classes
        alpha_tilde = target_onehot + (1 - target_onehot) * alpha
        S_tilde = alpha_tilde.sum(dim=-1, keepdim=True)

        kl = (
            torch.lgamma(S_tilde.squeeze(-1))
            - torch.lgamma(torch.tensor(alpha.shape[-1], dtype=alpha.dtype, device=alpha.device))
            - torch.lgamma(alpha_tilde).sum(dim=-1)
            + ((alpha_tilde - 1) * (torch.digamma(alpha_tilde) - torch.digamma(S_tilde))).sum(dim=-1)
        )

        # Annealing: gradually increase KL weight
        annealing_factor = min(1.0, self.current_epoch / max(1, self.annealing_epochs))

        loss = loss_ce + self.kl_weight * annealing_factor * kl
        return loss.mean()

    def set_epoch(self, epoch: int):
        """Call this at the start of each epoch for annealing."""
        self.current_epoch = epoch


class QuantileLoss(nn.Module):
    """
    Pinball loss for quantile regression.

    For each quantile q:
    - If y > pred: loss = q * (y - pred)      # Underestimate penalty
    - If y < pred: loss = (1-q) * (pred - y)  # Overestimate penalty

    This naturally trains the model to output the qth percentile.
    """

    def __init__(self, quantiles: List[float] = [0.1, 0.25, 0.5, 0.75, 0.9]):
        super().__init__()
        self.quantiles = quantiles

    def forward(
        self,
        prediction: QuantilePrediction,
        target: torch.Tensor
    ) -> torch.Tensor:
        total_loss = 0.0

        for q in self.quantiles:
            pred = prediction.quantiles[q]
            error = target - pred

            loss = torch.where(
                error >= 0,
                q * error,
                (q - 1) * error
            )
            total_loss += loss.mean()

        return total_loss / len(self.quantiles)


# =============================================================================
# Complete Trading Model with Learned Uncertainty
# =============================================================================

class TradingModelWithUncertainty(nn.Module):
    """
    Complete trading model that learns its own uncertainty.

    Architecture:
    1. Shared feature extractor (CNN/LSTM/Transformer)
    2. Multiple output heads:
       - Direction head (Beta distribution for binary up/down)
       - Magnitude head (Gaussian for expected return %)
       - Bounds head (Quantiles for risk management)

    This is BETTER than sigmoid because:
    - Model learns WHEN to be confident
    - Separate outputs for direction vs magnitude vs bounds
    - Natural integration with position sizing
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        output_type: str = 'beta'  # 'beta', 'dirichlet', 'gaussian', 'all'
    ):
        super().__init__()

        self.output_type = output_type

        # Shared feature extractor
        layers = []
        current_dim = input_dim
        for i in range(num_layers):
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ])
            current_dim = hidden_dim

        self.feature_extractor = nn.Sequential(*layers)

        # Output heads based on type
        if output_type in ['beta', 'all']:
            self.beta_head = BetaOutputLayer(hidden_dim)

        if output_type in ['dirichlet', 'all']:
            self.dirichlet_head = DirichletOutputLayer(hidden_dim, num_classes=2)

        if output_type in ['gaussian', 'all']:
            self.gaussian_head = GaussianOutputLayer(hidden_dim)

        if output_type == 'all':
            self.quantile_head = QuantileOutputLayer(hidden_dim)

    def forward(self, x: torch.Tensor) -> Dict[str, any]:
        """
        Forward pass returning predictions with uncertainty.

        Returns dict with:
        - 'direction': BetaPrediction or DirichletPrediction
        - 'magnitude': GaussianPrediction (if output_type='all')
        - 'bounds': QuantilePrediction (if output_type='all')
        """
        features = self.feature_extractor(x)

        outputs = {}

        if self.output_type == 'beta':
            outputs['direction'] = self.beta_head(features)
        elif self.output_type == 'dirichlet':
            outputs['direction'] = self.dirichlet_head(features)
        elif self.output_type == 'gaussian':
            outputs['magnitude'] = self.gaussian_head(features)
        elif self.output_type == 'all':
            outputs['direction'] = self.beta_head(features)
            outputs['magnitude'] = self.gaussian_head(features)
            outputs['bounds'] = self.quantile_head(features)

        return outputs

    def predict_with_confidence(
        self,
        x: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Make prediction with confidence score.

        Returns:
        - direction: 1 (up) or 0 (down)
        - probability: probability of up
        - confidence: how confident [0.5, 1.0]
        - uncertainty: inverse of confidence [0, 0.5]
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(x)

        if 'direction' in outputs:
            pred = outputs['direction']

            if isinstance(pred, BetaPrediction):
                return {
                    'direction': pred.prediction,
                    'probability': pred.mean,
                    'confidence': pred.confidence,
                    'uncertainty': 1.0 - pred.confidence,
                    'alpha': pred.alpha,
                    'beta': pred.beta,
                }
            elif isinstance(pred, DirichletPrediction):
                probs = pred.probabilities
                return {
                    'direction': (probs[:, 1] >= 0.5).long(),
                    'probability': probs[:, 1],
                    'confidence': pred.confidence,
                    'uncertainty': pred.uncertainty,
                    'evidence': pred.total_evidence,
                }

        return outputs


# =============================================================================
# Utility Functions
# =============================================================================

def compare_uncertainty_methods(
    x: torch.Tensor,
    y: torch.Tensor,
    input_dim: int,
    epochs: int = 100
) -> Dict[str, Dict]:
    """
    Compare different uncertainty methods on the same data.

    Returns metrics for each method.
    """
    methods = ['beta', 'dirichlet', 'gaussian']
    results = {}

    for method in methods:
        model = TradingModelWithUncertainty(
            input_dim=input_dim,
            hidden_dim=64,
            output_type=method
        )

        if method == 'beta':
            criterion = BetaNLLLoss()
        elif method == 'dirichlet':
            criterion = EvidentialLoss()
        else:
            criterion = GaussianNLLLoss()

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # Train
        model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = model(x)

            if method == 'gaussian':
                loss = criterion(outputs['magnitude'], y)
            else:
                loss = criterion(outputs['direction'], y)

            loss.backward()
            optimizer.step()

        # Evaluate
        model.eval()
        with torch.no_grad():
            preds = model.predict_with_confidence(x)

            results[method] = {
                'confidence_mean': preds['confidence'].mean().item(),
                'confidence_std': preds['confidence'].std().item(),
                'final_loss': loss.item(),
            }

    return results


def visualize_beta_distribution(alpha: float, beta: float) -> str:
    """
    ASCII visualization of Beta distribution.

    Useful for understanding model confidence.
    """
    from scipy import stats

    x = np.linspace(0, 1, 50)
    y = stats.beta.pdf(x, alpha, beta)
    y_normalized = y / y.max() * 10

    lines = []
    lines.append(f"Beta(α={alpha:.1f}, β={beta:.1f})")
    lines.append(f"Mean: {alpha/(alpha+beta):.2f}, Concentration: {alpha+beta:.1f}")
    lines.append("")

    for row in range(10, 0, -1):
        line = ""
        for col in range(50):
            if y_normalized[col] >= row:
                line += "█"
            else:
                line += " "
        lines.append(f"|{line}|")

    lines.append("+" + "-" * 50 + "+")
    lines.append(" 0" + " " * 23 + "0.5" + " " * 22 + "1")

    return "\n".join(lines)
