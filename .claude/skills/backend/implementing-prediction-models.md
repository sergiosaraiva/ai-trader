---
name: implementing-prediction-models
description: Implements new ML prediction models following the BaseModel abstract class pattern with registry registration. Use when creating CNN, LSTM, Transformer, or ensemble models for time series prediction. Python/PyTorch stack.
---

# Implementing Prediction Models

## Quick Reference

- Inherit from `BaseModel` in `src/models/base.py`
- Define `DEFAULT_CONFIG` class variable with model defaults
- Merge configs: `{**self.DEFAULT_CONFIG, **(config or {})}`
- Implement 4 abstract methods: `build()`, `train()`, `predict()`, `predict_batch()`
- Register with `ModelRegistry.register("name", ClassName)` at module end

## When to Use

- Creating a new time-horizon model (short/medium/long-term)
- Adding a new neural network architecture (CNN, LSTM, Transformer, N-BEATS)
- Implementing a specialized model for specific asset classes
- Building ensemble component models
- Extending existing model with new prediction heads

## When NOT to Use

- Simple feature calculations (use indicator pattern instead)
- One-off analysis scripts
- Non-ML statistical models (use dedicated module)

## Implementation Guide with Decision Tree

```
Is this a PyTorch neural network model?
├─ Yes → Inherit from TechnicalBaseModel (includes device handling)
│   └─ Does it need multiple prediction horizons?
│       ├─ Yes → Use List for prediction_horizon in config
│       └─ No → Use single int for prediction_horizon
└─ No → Inherit directly from BaseModel
    └─ Implement all 4 abstract methods manually
```

## Examples

**Example 1: Model Class Structure with DEFAULT_CONFIG**

```python
# From: src/models/technical/short_term.py:13-54
class ShortTermModel(TechnicalBaseModel):
    """
    Short-term prediction model for intraday trading.

    Architecture: CNN + Bi-LSTM + Multi-Head Attention

    Input: 168 hourly candles (7 days)
    Output: Price predictions for 1H, 4H, 12H, 24H ahead
    """

    DEFAULT_CONFIG = {
        "name": "short_term",
        "version": "1.0.0",
        "sequence_length": 168,
        "prediction_horizon": [1, 4, 12, 24],

        # CNN
        "cnn_filters": [64, 128, 256],
        "cnn_kernel_sizes": [3, 5, 7],
        "cnn_dropout": 0.2,

        # LSTM
        "lstm_hidden_size": 256,
        "lstm_num_layers": 2,
        "lstm_dropout": 0.3,
        "lstm_bidirectional": True,

        # Attention
        "attention_heads": 8,
        "attention_dim": 256,

        # Training
        "batch_size": 64,
        "learning_rate": 1e-4,
        "epochs": 100,
        "early_stopping_patience": 15,
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize short-term model."""
        merged_config = {**self.DEFAULT_CONFIG, **(config or {})}
        super().__init__(merged_config)
```

**Explanation**: DEFAULT_CONFIG defines sensible defaults. Constructor merges user config over defaults, allowing partial overrides.

**Example 2: Abstract Base Class Definition**

```python
# From: src/models/base.py:57-127
class BaseModel(ABC):
    """Abstract base class for all prediction models."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize model.

        Args:
            config: Model configuration dictionary
        """
        self.config = config or {}
        self.name = self.config.get("name", self.__class__.__name__)
        self.version = self.config.get("version", "1.0.0")
        self.model = None
        self.scalers: Dict[str, Any] = {}
        self.feature_names: List[str] = []
        self.is_trained = False

    @abstractmethod
    def build(self) -> None:
        """Build model architecture."""
        pass

    @abstractmethod
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Train the model."""
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> Prediction:
        """Make prediction."""
        pass

    @abstractmethod
    def predict_batch(self, X: np.ndarray) -> List[Prediction]:
        """Make batch predictions."""
        pass
```

**Explanation**: BaseModel defines the contract. All models must implement these 4 methods. `is_trained` flag tracks state.

**Example 3: Model Registry Pattern**

```python
# From: src/models/base.py:236-259
class ModelRegistry:
    """Registry for managing model classes."""

    _models: Dict[str, Type[BaseModel]] = {}

    @classmethod
    def register(cls, name: str, model_class: Type[BaseModel]) -> None:
        """Register a model class."""
        cls._models[name.lower()] = model_class

    @classmethod
    def create(cls, name: str, config: Optional[Dict[str, Any]] = None) -> BaseModel:
        """Create a model instance by name."""
        model_class = cls._models.get(name.lower())
        if model_class is None:
            available = ", ".join(cls._models.keys())
            raise ValueError(f"Unknown model: {name}. Available: {available}")
        return model_class(config)

# From: src/models/technical/short_term.py:319-320
# Register at module end
ModelRegistry.register("short_term", ShortTermModel)
```

**Explanation**: Registry enables factory pattern. Register at module end. Names are case-insensitive (lowercased).

**Example 4: Prediction Dataclass Return Type**

```python
# From: src/models/base.py:14-55
@dataclass
class Prediction:
    """Model prediction output."""

    timestamp: datetime
    symbol: str

    # Price predictions
    price_prediction: float
    price_predictions_multi: Dict[str, float] = field(default_factory=dict)

    # Direction
    direction: str = "neutral"  # bullish, bearish, neutral
    direction_probability: float = 0.5

    # Confidence
    confidence: float = 0.5
    prediction_lower: float = 0.0
    prediction_upper: float = 0.0

    # Model info
    model_name: str = ""
    model_version: str = ""
    prediction_horizon: int = 1

# Usage in predict_batch:
# From: src/models/technical/short_term.py:302-314
pred = Prediction(
    timestamp=datetime.now(),
    symbol="",
    price_prediction=float(price_preds[i, 0]),
    price_predictions_multi=price_multi,
    direction=direction,
    direction_probability=float(direction_probs[i, dir_idx]),
    confidence=float(confidence[i, 0]),
    model_name=self.name,
    model_version=self.version,
    prediction_horizon=horizons[0],
)
```

**Explanation**: Always return Prediction dataclass from predict methods. Multi-horizon predictions go in `price_predictions_multi` dict with string keys like "1h", "4h".

**Example 5: Training Loop with Early Stopping**

```python
# From: src/models/technical/short_term.py:213-268
# Training loop
history = {"train_loss": [], "val_loss": []}
best_val_loss = float("inf")
patience_counter = 0

for epoch in range(self.config["epochs"]):
    # Training
    self.model.train()
    train_loss = 0.0

    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(self.device)
        y_batch = y_batch.to(self.device)

        optimizer.zero_grad()
        outputs = self.model(X_batch)
        loss = criterion(outputs["price"][:, 0], y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        train_loss += loss.item()

    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
    else:
        patience_counter += 1

    if patience_counter >= self.config["early_stopping_patience"]:
        print(f"Early stopping at epoch {epoch + 1}")
        break

self.is_trained = True
return history
```

**Explanation**: Standard training loop pattern. Always set `self.is_trained = True` after training. Use gradient clipping for stability.

## Quality Checklist

- [ ] Class inherits from `BaseModel` or `TechnicalBaseModel`
- [ ] `DEFAULT_CONFIG` contains all hyperparameters with sensible defaults
- [ ] Constructor uses `{**self.DEFAULT_CONFIG, **(config or {})}` merge pattern
- [ ] All 4 abstract methods implemented: `build()`, `train()`, `predict()`, `predict_batch()`
- [ ] `predict()` and `predict_batch()` return `Prediction` dataclass
- [ ] `is_trained` flag set to `True` after training
- [ ] Model registered with `ModelRegistry.register()` at module end
- [ ] Type hints on all method signatures

## Common Mistakes

- **Forgetting to set is_trained**: Model state not tracked → Set `self.is_trained = True` at end of `train()` (see src/models/technical/short_term.py:267)
- **Not merging configs**: User overrides ignored → Use `{**self.DEFAULT_CONFIG, **(config or {})}` pattern
- **Returning raw arrays**: Downstream code expects Prediction → Always return Prediction dataclass
- **Case-sensitive registry**: Model not found → Registry lowercases names, use lowercase in create()

## Validation

- [ ] Pattern confirmed in `src/models/base.py:57-259`
- [ ] Implementation example in `src/models/technical/short_term.py:13-321`
- [ ] Registry pattern in `src/models/base.py:236-259`

## Related Skills

- [creating-technical-indicators](../feature-engineering/creating-technical-indicators.md) - For feature inputs to models
- [running-backtests](../trading-domain/running-backtests.md) - For evaluating model performance
- [creating-dataclasses](../quality-testing/creating-dataclasses.md) - For Prediction and other DTOs
