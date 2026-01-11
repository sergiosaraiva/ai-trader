# Multi-Timeframe Ensemble Implementation Guide

## 1. Executive Summary

This document specifies the implementation of a **3-timeframe ensemble model** that combines Short-term (1H), Medium-term (4H), and Long-term (Daily) predictions to generate higher-quality trading signals with reduced noise.

### Current State
- 1H XGBoost model: **59.4% win rate, 2.35 profit factor**
- 4H XGBoost model: **41.4% win rate, 1.40 profit factor**
- Daily model: **Not yet trained**
- Ensemble integration: **Not implemented**

### Target State
- 3 models trained at different timeframes
- Weighted ensemble combining all 3
- Expected improvement: **+2-5% win rate** through noise reduction

---

## 2. Architecture Overview

### 2.1 The Multi-Timeframe Ensemble Concept

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      3-TIMEFRAME ENSEMBLE                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   SHORT-TERM (1H)       MEDIUM-TERM (4H)        LONG-TERM (Daily)      │
│   ┌─────────────┐       ┌─────────────┐         ┌─────────────┐        │
│   │  XGBoost    │       │  XGBoost    │         │  XGBoost    │        │
│   │  ~115       │       │  ~113       │         │  ~110       │        │
│   │  features   │       │  features   │         │  features   │        │
│   └──────┬──────┘       └──────┬──────┘         └──────┬──────┘        │
│          │                     │                       │               │
│          ▼                     ▼                       ▼               │
│      Prediction            Prediction              Prediction          │
│      + Confidence          + Confidence            + Confidence        │
│          │                     │                       │               │
│          │    ┌────────────────┼───────────────────────┘               │
│          │    │                │                                       │
│          ▼    ▼                ▼                                       │
│   ┌─────────────────────────────────────────┐                          │
│   │         WEIGHTED ENSEMBLE               │                          │
│   │                                         │                          │
│   │   60% Short + 30% Medium + 10% Long     │                          │
│   │                                         │                          │
│   │   + Agreement Score (confidence boost)  │                          │
│   │   + Regime Adjustment (dynamic weights) │                          │
│   └─────────────────┬───────────────────────┘                          │
│                     │                                                   │
│                     ▼                                                   │
│           FINAL PREDICTION                                              │
│           (Direction + Confidence)                                      │
│                     │                                                   │
│                     ▼                                                   │
│           TRADING SIGNAL                                                │
│           (BUY/SELL/HOLD)                                               │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Why This Architecture Works

| Timeframe | Role | Weight | Captures |
|-----------|------|--------|----------|
| **1H (Short)** | Entry timing, immediate momentum | **60%** | Intraday patterns, quick reversals |
| **4H (Medium)** | Trend direction, swing context | **30%** | Multi-day moves, session dynamics |
| **Daily (Long)** | Major trend, regime detection | **10%** | Bull/bear markets, support/resistance |

### 2.3 Noise Reduction Cascade

```
Timeframe    Noise Level   Data Points/Year
───────────────────────────────────────────
1H           ~50%          ~6,500 bars
4H           ~40%          ~1,625 bars
Daily        ~30%          ~252 bars
```

Higher timeframes average out the noise of lower timeframes, providing cleaner trend signals.

---

## 3. Existing Code to Modify/Use

### 3.1 Key Files

| File | Purpose | Action |
|------|---------|--------|
| `src/models/multi_timeframe/improved_model.py` | Individual timeframe models | Use as-is, train Daily |
| `src/models/ensemble/combiner.py` | TechnicalEnsemble class | Integrate with improved models |
| `scripts/train_improved_models.py` | Training script | Extend for 3-TF training |
| `scripts/backtest_improved.py` | Backtesting | Extend for ensemble backtest |

### 3.2 TechnicalEnsemble (Already Exists)

Location: `src/models/ensemble/combiner.py`

```python
class TechnicalEnsemble(EnsembleModel):
    DEFAULT_CONFIG = {
        "default_weights": {
            "short_term": 0.3,   # Change to 0.6
            "medium_term": 0.4,  # Change to 0.3
            "long_term": 0.3,    # Change to 0.1
        },
        "use_dynamic_weights": True,
        "combination_method": "weighted_avg",
    }
```

This class already has:
- Weighted averaging of predictions
- Regime-based weight adjustment
- Performance-based weight adjustment
- Agreement score calculation

### 3.3 ImprovedModelConfig (Already Has Daily)

Location: `src/models/multi_timeframe/improved_model.py`

```python
@classmethod
def daily_model(cls) -> "ImprovedModelConfig":
    """Daily timeframe model for position trading."""
    return cls(
        name="D",
        base_timeframe="D",
        tp_pips=100.0,
        sl_pips=50.0,
        max_holding_bars=10,  # 2 weeks max
        n_estimators=100,
        max_depth=4,
    )
```

---

## 4. Implementation Steps

### Phase 1: Train the Daily Model

**Goal**: Train a Daily timeframe model using the existing framework.

```bash
# Modify train_improved_models.py to include Daily
python scripts/train_improved_models.py --timeframes 1H,4H,D
```

**Expected Output**:
- `models/improved_mtf/D_improved_model.pkl`
- Daily model with ~60-70% validation accuracy

### Phase 2: Create the Multi-Timeframe Ensemble

**Goal**: Create a new class that wraps 3 ImprovedTimeframeModels and combines their predictions.

Create: `src/models/multi_timeframe/mtf_ensemble.py`

```python
class MTFEnsemble:
    """Multi-Timeframe Ensemble combining 1H, 4H, and Daily models."""

    def __init__(
        self,
        model_dir: Path,
        weights: Dict[str, float] = None,
    ):
        self.weights = weights or {
            "1H": 0.6,   # Short-term: dominant
            "4H": 0.3,   # Medium-term: trend confirmation
            "D": 0.1,    # Long-term: regime context
        }

        # Load individual models
        self.models = {
            "1H": ImprovedTimeframeModel(ImprovedModelConfig.hourly_model()),
            "4H": ImprovedTimeframeModel(ImprovedModelConfig.four_hour_model()),
            "D": ImprovedTimeframeModel(ImprovedModelConfig.daily_model()),
        }

    def predict(self, df_5min: pd.DataFrame) -> Tuple[int, float]:
        """
        Make ensemble prediction.

        Returns:
            (direction, confidence) where direction is 0 (down) or 1 (up)
        """
        predictions = {}
        confidences = {}

        for tf, model in self.models.items():
            pred, conf, _, _ = model.predict(df_5min)
            predictions[tf] = pred
            confidences[tf] = conf

        # Weighted combination
        weighted_prob = sum(
            self.weights[tf] * (predictions[tf] * confidences[tf] +
                               (1 - predictions[tf]) * (1 - confidences[tf]))
            for tf in self.models
        )

        direction = 1 if weighted_prob > 0.5 else 0
        base_confidence = abs(weighted_prob - 0.5) * 2 + 0.5

        # Agreement bonus
        agreement = len(set(predictions.values())) == 1
        confidence = min(base_confidence + (0.05 if agreement else 0), 1.0)

        return direction, confidence
```

### Phase 3: Training Pipeline

Create: `scripts/train_mtf_ensemble.py`

```python
def main():
    # 1. Load data
    df_5min = load_data("data/forex/EURUSD_*.csv")

    # 2. Train all 3 models
    for tf in ["1H", "4H", "D"]:
        config = get_config(tf)
        model = ImprovedTimeframeModel(config)
        model.train(df_5min, ...)
        model.save(f"models/mtf_ensemble/{tf}_model.pkl")

    # 3. Validate ensemble
    ensemble = MTFEnsemble(model_dir="models/mtf_ensemble")
    validate_ensemble(ensemble, df_5min, test_ratio=0.2)
```

### Phase 4: Backtesting

Create: `scripts/backtest_mtf_ensemble.py`

The backtest should:
1. Load all 3 models
2. For each test bar:
   - Get prediction from each model
   - Combine with weights
   - Apply confidence threshold
   - Simulate trade with triple barrier

### Phase 5: (Optional) Learned Weights

Create a 4th "meta-model" that learns optimal weights:

```python
class MetaWeightLearner:
    """Learn optimal weights from individual model predictions."""

    def __init__(self):
        self.model = XGBClassifier(...)

    def train(self, predictions_df, labels):
        """
        Input features:
        - 1H prediction, 1H confidence
        - 4H prediction, 4H confidence
        - D prediction, D confidence
        - Agreement score
        - Market regime features

        Output: Optimal direction prediction
        """
        pass
```

---

## 5. Expected Results

### 5.1 Individual Model Targets

| Model | Val Accuracy | Win Rate | Profit Factor |
|-------|--------------|----------|---------------|
| 1H | 67%+ | 59%+ | 2.0+ |
| 4H | 65%+ | 55%+ | 1.5+ |
| Daily | 65%+ | 55%+ | 1.5+ |

### 5.2 Ensemble Targets

| Metric | Single Model (1H) | Ensemble Target |
|--------|-------------------|-----------------|
| Win Rate | 59.4% | **62-65%** |
| Profit Factor | 2.35 | **2.5-3.0** |
| High-Conf Win Rate | 65.5% | **70%+** |

### 5.3 When Ensemble Helps Most

| Scenario | Expected Benefit |
|----------|-----------------|
| All 3 agree | High conviction, full position |
| 2 of 3 agree | Normal position |
| All 3 disagree | Skip trade (avoid noise) |
| Trending market | Long-term weight increases |
| Ranging market | Short-term weight increases |

---

## 6. Testing Checklist

- [ ] Daily model trains successfully
- [ ] Daily model achieves >55% validation accuracy
- [ ] Ensemble combines predictions correctly
- [ ] Ensemble backtest runs without errors
- [ ] Ensemble win rate >= single model win rate
- [ ] Ensemble profit factor >= single model profit factor
- [ ] High-confidence trades have higher win rate
- [ ] Agreement score correlates with accuracy

---

## 7. File Structure After Implementation

```
models/
└── mtf_ensemble/
    ├── 1H_model.pkl
    ├── 4H_model.pkl
    ├── D_model.pkl
    ├── ensemble_config.json
    └── training_metadata.json

src/models/multi_timeframe/
├── improved_model.py          # Existing - individual models
├── mtf_ensemble.py            # NEW - ensemble class
└── ...

scripts/
├── train_mtf_ensemble.py      # NEW - training pipeline
├── backtest_mtf_ensemble.py   # NEW - ensemble backtest
└── ...
```

---

## 8. Key Configuration Parameters

```python
# Ensemble weights (can be tuned)
WEIGHTS = {
    "1H": 0.6,   # Short-term dominant
    "4H": 0.3,   # Medium-term confirmation
    "D": 0.1,    # Long-term context
}

# Regime-based adjustments
REGIME_ADJUSTMENTS = {
    "trending": {"1H": 0.5, "4H": 0.35, "D": 0.15},  # More weight to trends
    "ranging": {"1H": 0.7, "4H": 0.25, "D": 0.05},   # More weight to short-term
    "volatile": {"1H": 0.6, "4H": 0.35, "D": 0.05},  # Reduce long-term
}

# Trading thresholds
MIN_CONFIDENCE = 0.55
MIN_AGREEMENT = 0.5  # At least 2 of 3 must agree
```

---

## 9. References

- `docs/02-technical-analysis-model-design.md` - Original architecture design
- `src/models/ensemble/combiner.py` - Existing TechnicalEnsemble class
- `src/models/multi_timeframe/improved_model.py` - Individual model implementation
- `configs/profiles/trader.yaml` - Profile configuration for 1H/4H/1D

---

## 10. Notes for Implementation

1. **Start simple**: Use fixed weights first, add dynamic adjustment later
2. **Validate incrementally**: Test each model individually before ensemble
3. **Preserve rollback**: Keep individual model backtest scripts working
4. **Monitor agreement**: Track how often models agree/disagree
5. **Log everything**: Log individual predictions for analysis
