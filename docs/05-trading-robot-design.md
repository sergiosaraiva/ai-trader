# Trading Robot Design Specification

## Executive Summary

This document specifies a **world-class trading robot** that uses the trained ML models to generate trading signals, with configurable risk aversion, comprehensive loss protection, and production-ready safety mechanisms.

**Key Design Principles:**
1. **Confidence-First**: Only trade when model confidence exceeds thresholds
2. **Risk-Adaptive**: Configurable risk profiles from ultra-conservative to aggressive
3. **Self-Protecting**: Circuit breakers halt trading during adverse conditions
4. **Battle-Tested**: Simulation mode before production deployment

---

## 1. Architecture Overview

### 1.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           TRADING ROBOT ARCHITECTURE                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌────────────────┐                                                         │
│  │  Market Data   │─────┐                                                   │
│  │    Source      │     │                                                   │
│  └────────────────┘     │                                                   │
│                         ▼                                                   │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                      DATA PIPELINE                                   │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐  │   │
│  │  │ OHLCV Fetch  │─▶│  Timeframe   │─▶│  Technical Indicators    │  │   │
│  │  │              │  │  Conversion  │  │  (per profile config)    │  │   │
│  │  └──────────────┘  └──────────────┘  └──────────────────────────┘  │   │
│  └───────────────────────────────┬─────────────────────────────────────┘   │
│                                  │                                          │
│                                  ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                      PREDICTION ENGINE                               │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                 │   │
│  │  │ Short-Term  │  │ Medium-Term │  │ Long-Term   │                 │   │
│  │  │ Beta(α,β)   │  │ Beta(α,β)   │  │ Beta(α,β)   │                 │   │
│  │  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘                 │   │
│  │         │                │                │                         │   │
│  │         └────────────────┼────────────────┘                         │   │
│  │                          ▼                                          │   │
│  │  ┌─────────────────────────────────────────────────────────────┐   │   │
│  │  │              ENSEMBLE COMBINER                               │   │   │
│  │  │  - Weighted average with dynamic weights                     │   │   │
│  │  │  - Disagreement penalty                                      │   │   │
│  │  │  - Regime adjustment                                         │   │   │
│  │  └───────────────────────────────────────────────────────────────┘   │   │
│  └───────────────────────────────┬─────────────────────────────────────┘   │
│                                  │                                          │
│                                  ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                      DECISION ENGINE                                 │   │
│  │                                                                      │   │
│  │  ┌───────────────────────────────────────────────────────────────┐  │   │
│  │  │                 SIGNAL GENERATOR                               │  │   │
│  │  │  - Confidence threshold check                                  │  │   │
│  │  │  - Direction determination (BUY/SELL/HOLD)                     │  │   │
│  │  │  - Risk-adjusted position sizing                               │  │   │
│  │  └───────────────────────────────────────────────────────────────┘  │   │
│  │                          │                                          │   │
│  │                          ▼                                          │   │
│  │  ┌───────────────────────────────────────────────────────────────┐  │   │
│  │  │                 RISK MANAGER                                   │  │   │
│  │  │  - Position limits                                             │  │   │
│  │  │  - Drawdown protection                                         │  │   │
│  │  │  - Portfolio heat                                              │  │   │
│  │  │  - Exposure limits                                             │  │   │
│  │  └───────────────────────────────────────────────────────────────┘  │   │
│  │                          │                                          │   │
│  │                          ▼                                          │   │
│  │  ┌───────────────────────────────────────────────────────────────┐  │   │
│  │  │                 CIRCUIT BREAKERS                               │  │   │
│  │  │  - Consecutive loss protection                                 │  │   │
│  │  │  - Daily/weekly loss limits                                    │  │   │
│  │  │  - Model confidence degradation                                │  │   │
│  │  │  - Market instability detection                                │  │   │
│  │  └───────────────────────────────────────────────────────────────┘  │   │
│  └───────────────────────────────┬─────────────────────────────────────┘   │
│                                  │                                          │
│                                  ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                      EXECUTION ENGINE                                │   │
│  │  ┌────────────────────────────────────────────────────────────────┐ │   │
│  │  │              ORDER MANAGER                                      │ │   │
│  │  │  - Order creation & validation                                  │ │   │
│  │  │  - Stop-loss / Take-profit                                      │ │   │
│  │  │  - Order state machine                                          │ │   │
│  │  └────────────────────────────────────────────────────────────────┘ │   │
│  │                          │                                          │   │
│  │            ┌─────────────┴─────────────┐                           │   │
│  │            ▼                           ▼                           │   │
│  │  ┌─────────────────┐         ┌─────────────────┐                   │   │
│  │  │  SIMULATION     │         │   PRODUCTION    │                   │   │
│  │  │  (Paper Trade)  │         │   (Live Trade)  │                   │   │
│  │  └─────────────────┘         └─────────────────┘                   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 Component Responsibilities

| Component | Responsibility |
|-----------|---------------|
| **Data Pipeline** | Fetch, transform, and enrich market data |
| **Prediction Engine** | Run ML models, combine predictions, estimate confidence |
| **Signal Generator** | Convert predictions to BUY/SELL/HOLD signals |
| **Risk Manager** | Enforce position limits, drawdown protection |
| **Circuit Breakers** | Halt trading under adverse conditions |
| **Order Manager** | Create, validate, and track orders |
| **Execution Engine** | Route orders to broker (sim or live) |

---

## 2. Risk Aversion Configuration

### 2.1 Risk Profile Levels

The robot supports **five risk profiles** that control all risk-related parameters:

```python
@dataclass
class RiskProfile:
    """Defines risk tolerance for the trading robot."""

    # Profile identification
    name: str
    description: str

    # Confidence thresholds (from Beta output)
    min_confidence_to_trade: float      # Below this = HOLD
    full_position_confidence: float     # Above this = full size

    # Position sizing
    max_position_pct: float             # Max % of equity per position
    base_position_pct: float            # Starting position size
    kelly_fraction: float               # Fraction of Kelly criterion to use

    # Loss limits
    max_daily_loss_pct: float           # Daily loss limit
    max_weekly_loss_pct: float          # Weekly loss limit
    max_drawdown_pct: float             # Maximum drawdown before halt

    # Circuit breakers
    consecutive_loss_halt: int          # N losses in a row = halt
    cooldown_hours: int                 # Hours to wait after halt

    # Portfolio limits
    max_portfolio_heat: float           # Total risk exposure
    max_correlation_exposure: float     # Max in correlated assets
```

### 2.2 Predefined Risk Profiles

```yaml
# configs/risk_profiles.yaml

risk_profiles:
  # ULTRA-CONSERVATIVE: Only trade on extremely clear signals
  # Best for: Capital preservation, risk-averse investors
  ultra_conservative:
    name: "Ultra Conservative"
    description: "Maximum capital protection, minimal trading"

    # Very high confidence required
    min_confidence_to_trade: 0.85      # Only very confident predictions
    full_position_confidence: 0.95     # Need near-certainty for full size

    # Small positions
    max_position_pct: 0.01             # 1% max per position
    base_position_pct: 0.005           # 0.5% base size
    kelly_fraction: 0.125              # 1/8 Kelly

    # Tight loss limits
    max_daily_loss_pct: 0.005          # 0.5% daily limit
    max_weekly_loss_pct: 0.015         # 1.5% weekly limit
    max_drawdown_pct: 0.05             # 5% max drawdown

    # Very sensitive circuit breakers
    consecutive_loss_halt: 2           # 2 losses = halt
    cooldown_hours: 48                 # 2-day cooldown

    # Minimal exposure
    max_portfolio_heat: 0.03           # 3% total risk
    max_correlation_exposure: 0.02     # 2% in correlated assets

  # CONSERVATIVE: Careful trading with strong risk controls
  # Best for: Long-term traders, pension-style portfolios
  conservative:
    name: "Conservative"
    description: "Strong risk controls, selective trading"

    min_confidence_to_trade: 0.75
    full_position_confidence: 0.90

    max_position_pct: 0.02             # 2% max
    base_position_pct: 0.01            # 1% base
    kelly_fraction: 0.25               # 1/4 Kelly

    max_daily_loss_pct: 0.01           # 1% daily
    max_weekly_loss_pct: 0.03          # 3% weekly
    max_drawdown_pct: 0.10             # 10% max drawdown

    consecutive_loss_halt: 3
    cooldown_hours: 24

    max_portfolio_heat: 0.06           # 6% total risk
    max_correlation_exposure: 0.04

  # MODERATE: Balanced risk-reward approach
  # Best for: Active traders, swing trading
  moderate:
    name: "Moderate"
    description: "Balanced approach, industry-standard risk"

    min_confidence_to_trade: 0.65
    full_position_confidence: 0.85

    max_position_pct: 0.05             # 5% max
    base_position_pct: 0.02            # 2% base
    kelly_fraction: 0.50               # Half Kelly

    max_daily_loss_pct: 0.03           # 3% daily
    max_weekly_loss_pct: 0.07          # 7% weekly
    max_drawdown_pct: 0.15             # 15% max drawdown

    consecutive_loss_halt: 5
    cooldown_hours: 12

    max_portfolio_heat: 0.10           # 10% total risk
    max_correlation_exposure: 0.06

  # AGGRESSIVE: Higher risk for higher potential returns
  # Best for: Experienced traders, high-risk tolerance
  aggressive:
    name: "Aggressive"
    description: "Higher risk tolerance, more frequent trading"

    min_confidence_to_trade: 0.55      # Trade on moderate confidence
    full_position_confidence: 0.80

    max_position_pct: 0.10             # 10% max
    base_position_pct: 0.05            # 5% base
    kelly_fraction: 0.75               # 3/4 Kelly

    max_daily_loss_pct: 0.05           # 5% daily
    max_weekly_loss_pct: 0.10          # 10% weekly
    max_drawdown_pct: 0.25             # 25% max drawdown

    consecutive_loss_halt: 7
    cooldown_hours: 6

    max_portfolio_heat: 0.15           # 15% total risk
    max_correlation_exposure: 0.10

  # ULTRA-AGGRESSIVE: Maximum trading activity (NOT RECOMMENDED)
  # Best for: Testing, research, experienced day traders only
  ultra_aggressive:
    name: "Ultra Aggressive"
    description: "Maximum trading activity - USE WITH CAUTION"

    min_confidence_to_trade: 0.52      # Trade on slight edge
    full_position_confidence: 0.75

    max_position_pct: 0.15             # 15% max
    base_position_pct: 0.08            # 8% base
    kelly_fraction: 1.0                # Full Kelly (risky!)

    max_daily_loss_pct: 0.08           # 8% daily
    max_weekly_loss_pct: 0.15          # 15% weekly
    max_drawdown_pct: 0.35             # 35% max drawdown

    consecutive_loss_halt: 10
    cooldown_hours: 4

    max_portfolio_heat: 0.25           # 25% total risk
    max_correlation_exposure: 0.15
```

### 2.3 Risk Profile Selection Matrix

| Factor | Ultra-Cons | Conservative | Moderate | Aggressive | Ultra-Agg |
|--------|------------|--------------|----------|------------|-----------|
| **Min Confidence** | 85% | 75% | 65% | 55% | 52% |
| **Max Position** | 1% | 2% | 5% | 10% | 15% |
| **Daily Loss Limit** | 0.5% | 1% | 3% | 5% | 8% |
| **Max Drawdown** | 5% | 10% | 15% | 25% | 35% |
| **Loss Streak Halt** | 2 | 3 | 5 | 7 | 10 |
| **Kelly Fraction** | 1/8 | 1/4 | 1/2 | 3/4 | Full |
| **Trade Frequency** | Very Low | Low | Medium | High | Very High |
| **Expected Return** | Low | Low-Med | Medium | High | Highest |
| **Risk Level** | Minimal | Low | Medium | High | Extreme |

---

## 3. Circuit Breakers and Loss Protection

### 3.1 Circuit Breaker System Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        CIRCUIT BREAKER SYSTEM                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                    TRADING STATE MACHINE                               │ │
│  │                                                                        │ │
│  │    ┌─────────┐                                   ┌─────────┐          │ │
│  │    │  ACTIVE │◄─────── (cooldown expires) ──────│  HALTED │          │ │
│  │    │         │                                   │         │          │ │
│  │    │ Trading │──────── (breaker triggers) ──────▶│  Paused │          │ │
│  │    │ Enabled │                                   │         │          │ │
│  │    └────┬────┘                                   └────┬────┘          │ │
│  │         │                                             │               │ │
│  │         │ (reduced activity)                          │ (manual only) │ │
│  │         ▼                                             ▼               │ │
│  │    ┌─────────┐                                   ┌─────────┐          │ │
│  │    │ REDUCED │◄─────── (partial recovery) ──────│ RECOVER │          │ │
│  │    │         │                                   │         │          │ │
│  │    │ Limited │──────── (continues poor) ────────▶│  Testing│          │ │
│  │    │ Trading │                                   │ Mode    │          │ │
│  │    └─────────┘                                   └─────────┘          │ │
│  │                                                                        │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                    CIRCUIT BREAKERS                                    │ │
│  │                                                                        │ │
│  │  ┌─────────────────────────┐  ┌─────────────────────────┐            │ │
│  │  │  CONSECUTIVE LOSS       │  │  DAILY LOSS LIMIT       │            │ │
│  │  │                         │  │                         │            │ │
│  │  │  Tracks: Loss streak    │  │  Tracks: Daily P&L      │            │ │
│  │  │  Trigger: N losses      │  │  Trigger: % of equity   │            │ │
│  │  │  Action: HALT           │  │  Action: HALT           │            │ │
│  │  └─────────────────────────┘  └─────────────────────────┘            │ │
│  │                                                                        │ │
│  │  ┌─────────────────────────┐  ┌─────────────────────────┐            │ │
│  │  │  DRAWDOWN PROTECTION    │  │  MODEL DEGRADATION      │            │ │
│  │  │                         │  │                         │            │ │
│  │  │  Tracks: Peak equity    │  │  Tracks: Rolling acc.   │            │ │
│  │  │  Trigger: DD% exceeded  │  │  Trigger: Acc. < min    │            │ │
│  │  │  Action: HALT or REDUCE │  │  Action: REDUCE/HALT    │            │ │
│  │  └─────────────────────────┘  └─────────────────────────┘            │ │
│  │                                                                        │ │
│  │  ┌─────────────────────────┐  ┌─────────────────────────┐            │ │
│  │  │  MARKET INSTABILITY     │  │  CONFIDENCE COLLAPSE    │            │ │
│  │  │                         │  │                         │            │ │
│  │  │  Tracks: Volatility,    │  │  Tracks: Avg confidence │            │ │
│  │  │          ensemble agree │  │  Trigger: Below thresh  │            │ │
│  │  │  Trigger: High vol/low  │  │  Action: HOLD only      │            │ │
│  │  │           agreement     │  │                         │            │ │
│  │  │  Action: REDUCE/HALT    │  │                         │            │ │
│  │  └─────────────────────────┘  └─────────────────────────┘            │ │
│  │                                                                        │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Circuit Breaker Implementations

#### 3.2.1 Consecutive Loss Circuit Breaker

**Purpose**: Detect when the model is consistently wrong and halt before catastrophic losses.

```python
@dataclass
class ConsecutiveLossBreaker:
    """
    Halts trading after N consecutive losses.

    RATIONALE:
    If the model is correct ~55% of the time, the probability of
    N consecutive losses is (0.45)^N:

    | N Losses | Probability | Should Happen Every |
    |----------|-------------|---------------------|
    | 3        | 9.1%        | 11 trades           |
    | 4        | 4.1%        | 25 trades           |
    | 5        | 1.8%        | 55 trades           |
    | 6        | 0.8%        | 120 trades          |
    | 7        | 0.4%        | 270 trades          |

    For a CONSERVATIVE profile (3 loss halt), we expect false triggers
    about 1 in 11 trades - acceptable for capital protection.

    For MODERATE profile (5 loss halt), false triggers 1 in 55 trades.
    """

    max_consecutive_losses: int
    consecutive_losses: int = 0
    trade_history: List[TradeResult] = field(default_factory=list)

    def record_trade(self, trade: TradeResult) -> CircuitBreakerAction:
        """Record trade and return action."""
        self.trade_history.append(trade)

        if trade.is_loss:
            self.consecutive_losses += 1

            if self.consecutive_losses >= self.max_consecutive_losses:
                return CircuitBreakerAction(
                    action=TradingState.HALTED,
                    reason=f"Consecutive loss limit reached: {self.consecutive_losses}",
                    severity=Severity.HIGH,
                    recovery_requirement=RecoveryRequirement(
                        cooldown_hours=self._get_cooldown_hours(),
                        reduced_size_on_resume=0.5,
                        wins_to_restore=3,
                    )
                )
        else:
            self.consecutive_losses = 0

        return CircuitBreakerAction(action=TradingState.ACTIVE)

    def _get_cooldown_hours(self) -> int:
        """Longer cooldown for longer losing streaks."""
        base_cooldown = 12
        return base_cooldown * (self.consecutive_losses - self.max_consecutive_losses + 1)
```

#### 3.2.2 Daily/Weekly Loss Limit Breaker

```python
@dataclass
class PeriodLossBreaker:
    """
    Halts trading when period loss limit is exceeded.

    Prevents "revenge trading" - the tendency to increase risk
    after losses to try to recover quickly.
    """

    daily_loss_limit_pct: float
    weekly_loss_limit_pct: float

    daily_pnl: float = 0.0
    weekly_pnl: float = 0.0
    start_of_day_equity: float = 0.0
    start_of_week_equity: float = 0.0

    def update(self, current_equity: float) -> CircuitBreakerAction:
        """Check if period limits are breached."""

        daily_loss_pct = (self.start_of_day_equity - current_equity) / self.start_of_day_equity
        weekly_loss_pct = (self.start_of_week_equity - current_equity) / self.start_of_week_equity

        if daily_loss_pct >= self.daily_loss_limit_pct:
            return CircuitBreakerAction(
                action=TradingState.HALTED,
                reason=f"Daily loss limit reached: {daily_loss_pct:.2%}",
                severity=Severity.HIGH,
                recovery_requirement=RecoveryRequirement(
                    cooldown_hours=24,  # Until next trading day
                    reduced_size_on_resume=0.75,
                )
            )

        if weekly_loss_pct >= self.weekly_loss_limit_pct:
            return CircuitBreakerAction(
                action=TradingState.HALTED,
                reason=f"Weekly loss limit reached: {weekly_loss_pct:.2%}",
                severity=Severity.CRITICAL,
                recovery_requirement=RecoveryRequirement(
                    cooldown_hours=168,  # Until next week
                    reduced_size_on_resume=0.50,
                )
            )

        # Warning levels (reduce position size)
        if daily_loss_pct >= self.daily_loss_limit_pct * 0.7:
            return CircuitBreakerAction(
                action=TradingState.REDUCED,
                reason=f"Approaching daily loss limit: {daily_loss_pct:.2%}",
                severity=Severity.MEDIUM,
                size_multiplier=0.5,
            )

        return CircuitBreakerAction(action=TradingState.ACTIVE)
```

#### 3.2.3 Drawdown Protection Breaker

```python
@dataclass
class DrawdownBreaker:
    """
    Progressive protection as drawdown increases.

    STRATEGY:
    - 50% of limit: Reduce position sizes
    - 75% of limit: Only very high confidence trades
    - 100% of limit: Full halt
    """

    max_drawdown_pct: float
    peak_equity: float = 0.0

    def update(self, current_equity: float) -> CircuitBreakerAction:
        """Check drawdown and return appropriate action."""

        # Update peak
        self.peak_equity = max(self.peak_equity, current_equity)

        # Calculate current drawdown
        drawdown_pct = (self.peak_equity - current_equity) / self.peak_equity
        drawdown_ratio = drawdown_pct / self.max_drawdown_pct

        if drawdown_ratio >= 1.0:
            return CircuitBreakerAction(
                action=TradingState.HALTED,
                reason=f"Maximum drawdown reached: {drawdown_pct:.2%}",
                severity=Severity.CRITICAL,
                recovery_requirement=RecoveryRequirement(
                    cooldown_hours=72,
                    reduced_size_on_resume=0.25,
                    wins_to_restore=5,
                )
            )

        if drawdown_ratio >= 0.75:
            return CircuitBreakerAction(
                action=TradingState.REDUCED,
                reason=f"Drawdown at {drawdown_pct:.2%} (75% of limit)",
                severity=Severity.HIGH,
                size_multiplier=0.25,
                min_confidence_override=0.85,  # Only very confident trades
            )

        if drawdown_ratio >= 0.50:
            return CircuitBreakerAction(
                action=TradingState.REDUCED,
                reason=f"Drawdown at {drawdown_pct:.2%} (50% of limit)",
                severity=Severity.MEDIUM,
                size_multiplier=0.50,
            )

        return CircuitBreakerAction(action=TradingState.ACTIVE)
```

#### 3.2.4 Model Degradation Breaker

**KEY INSIGHT**: This breaker detects when the market has changed and the model is no longer effective.

```python
@dataclass
class ModelDegradationBreaker:
    """
    Detects when model performance has degraded.

    DISTINGUISHES BETWEEN:
    1. Normal variance: Model is still valid, just unlucky
    2. Model degradation: Market has changed, model needs recalibration
    3. Market unpredictability: No model would work in current conditions

    DETECTION SIGNALS:
    - Rolling accuracy drops below threshold
    - Ensemble disagreement increases
    - Confidence is high but accuracy is low (miscalibration)
    - Feature distribution shift
    """

    min_rolling_accuracy: float = 0.50  # Below this = degraded
    rolling_window: int = 20  # Number of trades to consider
    max_ensemble_disagreement: float = 0.30  # Std dev of predictions

    recent_trades: List[TradeResult] = field(default_factory=list)
    recent_confidences: List[float] = field(default_factory=list)
    recent_ensemble_disagreements: List[float] = field(default_factory=list)

    def record_trade(
        self,
        trade: TradeResult,
        confidence: float,
        ensemble_disagreement: float
    ) -> CircuitBreakerAction:
        """Analyze trade and model metrics."""

        self.recent_trades.append(trade)
        self.recent_confidences.append(confidence)
        self.recent_ensemble_disagreements.append(ensemble_disagreement)

        # Keep only rolling window
        if len(self.recent_trades) > self.rolling_window:
            self.recent_trades.pop(0)
            self.recent_confidences.pop(0)
            self.recent_ensemble_disagreements.pop(0)

        if len(self.recent_trades) < self.rolling_window // 2:
            return CircuitBreakerAction(action=TradingState.ACTIVE)

        # Calculate metrics
        rolling_accuracy = sum(1 for t in self.recent_trades if t.is_win) / len(self.recent_trades)
        avg_confidence = sum(self.recent_confidences) / len(self.recent_confidences)
        avg_disagreement = sum(self.recent_ensemble_disagreements) / len(self.recent_ensemble_disagreements)

        # Check for miscalibration (high confidence but low accuracy)
        calibration_gap = avg_confidence - rolling_accuracy

        # CRITICAL: High confidence + low accuracy = broken model
        if calibration_gap > 0.20 and rolling_accuracy < 0.45:
            return CircuitBreakerAction(
                action=TradingState.HALTED,
                reason=f"Model miscalibrated: {avg_confidence:.1%} conf, {rolling_accuracy:.1%} acc",
                severity=Severity.CRITICAL,
                recovery_requirement=RecoveryRequirement(
                    cooldown_hours=168,  # 1 week - needs investigation
                    requires_recalibration=True,
                )
            )

        # High ensemble disagreement = uncertain market
        if avg_disagreement > self.max_ensemble_disagreement:
            return CircuitBreakerAction(
                action=TradingState.REDUCED,
                reason=f"High model disagreement: {avg_disagreement:.1%}",
                severity=Severity.MEDIUM,
                size_multiplier=0.5,
                min_confidence_override=0.80,
            )

        # Low accuracy but not miscalibrated = tough market
        if rolling_accuracy < self.min_rolling_accuracy:
            return CircuitBreakerAction(
                action=TradingState.REDUCED,
                reason=f"Rolling accuracy low: {rolling_accuracy:.1%}",
                severity=Severity.MEDIUM,
                size_multiplier=0.25,
            )

        return CircuitBreakerAction(action=TradingState.ACTIVE)
```

#### 3.2.5 Market Instability Breaker

```python
@dataclass
class MarketInstabilityBreaker:
    """
    Detects unstable market conditions where trading is risky.

    TRIGGERS:
    - Extreme volatility (ATR >> historical norm)
    - Low ensemble agreement (models confused)
    - Rapid regime changes
    - News/event periods (if configured)
    """

    volatility_threshold_multiplier: float = 2.5  # X times normal ATR
    min_ensemble_agreement: float = 0.60  # 60% of models must agree

    historical_volatility: float = 0.0
    volatility_window: int = 50
    recent_volatilities: List[float] = field(default_factory=list)

    def update(
        self,
        current_atr: float,
        ensemble_agreement: float,
        is_news_period: bool = False,
    ) -> CircuitBreakerAction:
        """Check market stability."""

        # Update volatility history
        self.recent_volatilities.append(current_atr)
        if len(self.recent_volatilities) > self.volatility_window:
            self.recent_volatilities.pop(0)

        if len(self.recent_volatilities) >= self.volatility_window // 2:
            self.historical_volatility = sum(self.recent_volatilities) / len(self.recent_volatilities)

        # News blackout
        if is_news_period:
            return CircuitBreakerAction(
                action=TradingState.HALTED,
                reason="News blackout period",
                severity=Severity.LOW,
                recovery_requirement=RecoveryRequirement(
                    cooldown_hours=1,
                )
            )

        # Extreme volatility
        if self.historical_volatility > 0:
            vol_ratio = current_atr / self.historical_volatility
            if vol_ratio > self.volatility_threshold_multiplier:
                return CircuitBreakerAction(
                    action=TradingState.HALTED,
                    reason=f"Extreme volatility: {vol_ratio:.1f}x normal",
                    severity=Severity.HIGH,
                    recovery_requirement=RecoveryRequirement(
                        cooldown_hours=4,
                    )
                )

        # Low ensemble agreement
        if ensemble_agreement < self.min_ensemble_agreement:
            return CircuitBreakerAction(
                action=TradingState.REDUCED,
                reason=f"Low model agreement: {ensemble_agreement:.1%}",
                severity=Severity.MEDIUM,
                size_multiplier=0.50,
            )

        return CircuitBreakerAction(action=TradingState.ACTIVE)
```

### 3.3 Recovery Protocol

After a circuit breaker triggers, the robot follows a **graduated recovery** process:

```python
@dataclass
class RecoveryProtocol:
    """
    Manages recovery after circuit breaker activation.

    RECOVERY PHASES:
    1. COOLDOWN: No trading, monitor only
    2. TESTING: Reduced position sizes, paper trades
    3. GRADUATED: Increasing position sizes with wins
    4. RESTORED: Full trading restored
    """

    class Phase(Enum):
        COOLDOWN = "cooldown"
        TESTING = "testing"
        GRADUATED = "graduated"
        RESTORED = "restored"

    current_phase: Phase = Phase.COOLDOWN
    cooldown_end: datetime = None

    # Testing phase
    testing_trades: int = 0
    testing_wins: int = 0
    testing_required: int = 5

    # Graduated phase
    graduated_wins_needed: int = 3
    graduated_consecutive_wins: int = 0
    current_size_multiplier: float = 0.25

    def update(self, trade_result: Optional[TradeResult] = None) -> RecoveryState:
        """Update recovery state after trade or time passage."""

        if self.current_phase == Phase.COOLDOWN:
            if datetime.now() >= self.cooldown_end:
                self.current_phase = Phase.TESTING
                return RecoveryState(
                    phase=Phase.TESTING,
                    can_trade=True,
                    size_multiplier=0.10,  # 10% size during testing
                    message="Entering testing phase with minimal position sizes"
                )
            return RecoveryState(
                phase=Phase.COOLDOWN,
                can_trade=False,
                time_remaining=self.cooldown_end - datetime.now(),
            )

        if self.current_phase == Phase.TESTING:
            if trade_result:
                self.testing_trades += 1
                if trade_result.is_win:
                    self.testing_wins += 1

            if self.testing_trades >= self.testing_required:
                win_rate = self.testing_wins / self.testing_trades
                if win_rate >= 0.5:  # At least 50% win rate
                    self.current_phase = Phase.GRADUATED
                    return RecoveryState(
                        phase=Phase.GRADUATED,
                        can_trade=True,
                        size_multiplier=0.25,
                        message=f"Testing passed ({win_rate:.0%}), entering graduated recovery"
                    )
                else:
                    # Back to cooldown
                    self._reset_to_cooldown(hours=24)
                    return RecoveryState(
                        phase=Phase.COOLDOWN,
                        can_trade=False,
                        message=f"Testing failed ({win_rate:.0%}), returning to cooldown"
                    )

            return RecoveryState(
                phase=Phase.TESTING,
                can_trade=True,
                size_multiplier=0.10,
                progress=f"{self.testing_trades}/{self.testing_required} test trades"
            )

        if self.current_phase == Phase.GRADUATED:
            if trade_result:
                if trade_result.is_win:
                    self.graduated_consecutive_wins += 1
                    # Increase size multiplier
                    self.current_size_multiplier = min(
                        1.0,
                        self.current_size_multiplier + 0.25
                    )
                else:
                    self.graduated_consecutive_wins = 0
                    # Decrease size multiplier
                    self.current_size_multiplier = max(
                        0.25,
                        self.current_size_multiplier - 0.25
                    )

            if self.graduated_consecutive_wins >= self.graduated_wins_needed:
                self.current_phase = Phase.RESTORED
                return RecoveryState(
                    phase=Phase.RESTORED,
                    can_trade=True,
                    size_multiplier=1.0,
                    message="Full trading restored!"
                )

            return RecoveryState(
                phase=Phase.GRADUATED,
                can_trade=True,
                size_multiplier=self.current_size_multiplier,
                progress=f"{self.graduated_consecutive_wins}/{self.graduated_wins_needed} consecutive wins"
            )

        return RecoveryState(
            phase=Phase.RESTORED,
            can_trade=True,
            size_multiplier=1.0
        )
```

---

## 4. Signal Generation and Decision Logic

### 4.1 Signal Generation Flow

```python
@dataclass
class TradingSignal:
    """Output from signal generator."""

    action: Action  # BUY, SELL, HOLD
    confidence: float  # 0.0 - 1.0
    direction_probability: float  # Probability of predicted direction

    # Position sizing
    position_size_pct: float  # % of equity for this trade

    # Risk management
    stop_loss_pct: float  # Stop loss distance
    take_profit_pct: float  # Take profit distance

    # Model details
    short_term_signal: float
    medium_term_signal: float
    long_term_signal: float
    ensemble_agreement: float

    # Metadata
    timestamp: datetime
    symbol: str
    timeframe: str


class SignalGenerator:
    """
    Converts model predictions to trading signals.

    DECISION LOGIC:
    1. Get predictions from all models
    2. Combine with ensemble weights
    3. Check confidence threshold (risk profile)
    4. Generate BUY/SELL/HOLD signal
    5. Calculate position size based on confidence
    """

    def __init__(
        self,
        prediction_engine: PredictionEngine,
        risk_profile: RiskProfile,
        circuit_breaker_manager: CircuitBreakerManager,
    ):
        self.prediction_engine = prediction_engine
        self.risk_profile = risk_profile
        self.circuit_breakers = circuit_breaker_manager

    def generate_signal(
        self,
        features: Dict[str, torch.Tensor],
        symbol: str,
        current_position: Optional[Position] = None,
    ) -> TradingSignal:
        """Generate trading signal from features."""

        # Step 1: Get predictions
        prediction = self.prediction_engine.predict(features)

        # Step 2: Check circuit breakers
        breaker_state = self.circuit_breakers.check_all(
            ensemble_agreement=prediction.ensemble_agreement,
            current_volatility=features.get('atr', 0),
        )

        if breaker_state.action == TradingState.HALTED:
            return TradingSignal(
                action=Action.HOLD,
                confidence=0.0,
                direction_probability=prediction.direction_probability,
                position_size_pct=0.0,
                reason=f"Circuit breaker: {breaker_state.reason}",
                ...
            )

        # Step 3: Determine action based on confidence
        action, position_size = self._determine_action(
            prediction=prediction,
            breaker_state=breaker_state,
            current_position=current_position,
        )

        # Step 4: Calculate stop loss and take profit
        stop_loss, take_profit = self._calculate_exits(
            prediction=prediction,
            action=action,
        )

        return TradingSignal(
            action=action,
            confidence=prediction.confidence,
            direction_probability=prediction.direction_probability,
            position_size_pct=position_size,
            stop_loss_pct=stop_loss,
            take_profit_pct=take_profit,
            short_term_signal=prediction.short_term_signal,
            medium_term_signal=prediction.medium_term_signal,
            long_term_signal=prediction.long_term_signal,
            ensemble_agreement=prediction.ensemble_agreement,
            timestamp=datetime.now(),
            symbol=symbol,
        )

    def _determine_action(
        self,
        prediction: EnsemblePrediction,
        breaker_state: CircuitBreakerState,
        current_position: Optional[Position],
    ) -> Tuple[Action, float]:
        """Determine action and position size."""

        # Get effective minimum confidence (may be overridden by breaker)
        min_confidence = max(
            self.risk_profile.min_confidence_to_trade,
            breaker_state.min_confidence_override or 0,
        )

        # Below threshold = HOLD
        if prediction.confidence < min_confidence:
            return Action.HOLD, 0.0

        # Calculate direction
        if prediction.direction_probability > 0.5:
            base_action = Action.BUY
        else:
            base_action = Action.SELL

        # If we already have a position in opposite direction
        if current_position:
            if current_position.side == Side.LONG and base_action == Action.SELL:
                base_action = Action.CLOSE_AND_REVERSE
            elif current_position.side == Side.SHORT and base_action == Action.BUY:
                base_action = Action.CLOSE_AND_REVERSE

        # Calculate position size
        position_size = self._calculate_position_size(
            prediction=prediction,
            breaker_state=breaker_state,
        )

        return base_action, position_size

    def _calculate_position_size(
        self,
        prediction: EnsemblePrediction,
        breaker_state: CircuitBreakerState,
    ) -> float:
        """
        Calculate position size based on confidence and Kelly criterion.

        FORMULA:
        size = base_size * confidence_factor * kelly_factor * breaker_multiplier

        Where:
        - base_size: From risk profile
        - confidence_factor: Linear scaling with confidence
        - kelly_factor: Based on edge and Kelly fraction
        - breaker_multiplier: From circuit breaker state
        """

        # Base size from risk profile
        base_size = self.risk_profile.base_position_pct

        # Confidence factor: scale linearly from min to full confidence
        conf_range = self.risk_profile.full_position_confidence - self.risk_profile.min_confidence_to_trade
        conf_above_min = prediction.confidence - self.risk_profile.min_confidence_to_trade
        confidence_factor = min(1.0, conf_above_min / conf_range) if conf_range > 0 else 1.0

        # Kelly factor (simplified)
        # Full Kelly = (p * b - q) / b where p = win prob, b = win/loss ratio, q = 1-p
        estimated_edge = (prediction.confidence - 0.5) * 2  # Scale to [0, 1]
        kelly_size = estimated_edge * self.risk_profile.kelly_fraction
        kelly_factor = max(0.1, min(1.0, kelly_size))

        # Ensemble agreement factor
        agreement_factor = prediction.ensemble_agreement ** 0.5

        # Circuit breaker multiplier
        breaker_multiplier = breaker_state.size_multiplier if breaker_state.size_multiplier else 1.0

        # Calculate final size
        position_size = base_size * confidence_factor * kelly_factor * agreement_factor * breaker_multiplier

        # Cap at maximum
        position_size = min(position_size, self.risk_profile.max_position_pct)

        return position_size
```

### 4.2 Position Sizing Visualization

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        POSITION SIZING LOGIC                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Model Confidence                                                            │
│  0.50        0.65        0.75        0.85        0.95        1.00           │
│    │           │           │           │           │           │            │
│    ▼           ▼           ▼           ▼           ▼           ▼            │
│  ┌─────────┬───────────┬───────────┬───────────┬───────────┬─────────┐     │
│  │  HOLD   │   25%     │    50%    │    75%    │   100%    │  100%   │     │
│  │  (0%)   │  (small)  │  (medium) │  (large)  │  (full)   │ (full)  │     │
│  └─────────┴───────────┴───────────┴───────────┴───────────┴─────────┘     │
│                                                                              │
│  Position Size = Base * Confidence * Kelly * Agreement * BreakerMultiplier  │
│                                                                              │
│  EXAMPLE (Moderate Profile):                                                 │
│  ───────────────────────────────────────────────────────────────────────     │
│  Base Position:    2%                                                        │
│  Model Confidence: 0.75                                                      │
│  Kelly Factor:     0.5                                                       │
│  Ensemble Agree:   0.80 (√0.80 = 0.89)                                      │
│  Breaker State:    ACTIVE (1.0)                                             │
│                                                                              │
│  Final Size = 2% × 0.75 × 0.5 × 0.89 × 1.0 = 0.67%                         │
│                                                                              │
│  With REDUCED breaker state (0.5):                                          │
│  Final Size = 2% × 0.75 × 0.5 × 0.89 × 0.5 = 0.33%                         │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 5. Simulation vs Production Mode

### 5.1 Mode Architecture

```python
from abc import ABC, abstractmethod


class ExecutionMode(ABC):
    """Abstract base class for execution modes."""

    @abstractmethod
    def submit_order(self, order: Order) -> OrderResult:
        """Submit order for execution."""
        pass

    @abstractmethod
    def get_positions(self) -> List[Position]:
        """Get current positions."""
        pass

    @abstractmethod
    def get_account_info(self) -> AccountInfo:
        """Get account information."""
        pass


class SimulationMode(ExecutionMode):
    """
    Paper trading simulation mode.

    FEATURES:
    - Uses real market data
    - Simulates fills with configurable slippage
    - Tracks virtual P&L
    - No real money at risk

    USE FOR:
    - Strategy validation
    - Model testing
    - Risk parameter tuning
    """

    def __init__(
        self,
        initial_capital: float,
        slippage_model: SlippageModel,
        latency_model: LatencyModel,
        commission_model: CommissionModel,
    ):
        self.capital = initial_capital
        self.slippage = slippage_model
        self.latency = latency_model
        self.commissions = commission_model

        self.positions: Dict[str, Position] = {}
        self.order_history: List[Order] = []
        self.trade_history: List[Trade] = []

    def submit_order(self, order: Order) -> OrderResult:
        """Simulate order execution."""

        # Simulate latency
        execution_delay = self.latency.simulate_delay()

        # Get fill price with slippage
        fill_price = self.slippage.apply(
            order_price=order.price,
            side=order.side,
            size=order.quantity,
        )

        # Calculate commission
        commission = self.commissions.calculate(
            quantity=order.quantity,
            price=fill_price,
        )

        # Create trade
        trade = Trade(
            symbol=order.symbol,
            side=order.side,
            quantity=order.quantity,
            price=fill_price,
            commission=commission,
            timestamp=datetime.now() + timedelta(milliseconds=execution_delay),
        )

        # Update position
        self._update_position(trade)

        return OrderResult(
            status=OrderStatus.FILLED,
            fill_price=fill_price,
            fill_quantity=order.quantity,
            commission=commission,
            latency_ms=execution_delay,
        )


class ProductionMode(ExecutionMode):
    """
    Live trading production mode.

    ADDITIONAL SAFETY:
    - Real broker integration
    - Order validation
    - Position reconciliation
    - Kill switch integration
    """

    def __init__(
        self,
        broker: BrokerAdapter,
        kill_switch: KillSwitch,
        order_validator: OrderValidator,
    ):
        self.broker = broker
        self.kill_switch = kill_switch
        self.validator = order_validator

    def submit_order(self, order: Order) -> OrderResult:
        """Submit real order to broker."""

        # Check kill switch
        if not self.kill_switch.is_active:
            raise TradingHaltedError("Kill switch engaged")

        # Validate order
        validation = self.validator.validate(order)
        if not validation.is_valid:
            raise OrderValidationError(validation.errors)

        # Submit to broker
        result = self.broker.submit_order(order)

        # Verify fill
        if result.status == OrderStatus.FILLED:
            self._verify_position_sync()

        return result
```

### 5.2 Slippage and Latency Models

```python
@dataclass
class SlippageModel:
    """
    Models realistic slippage for simulation.

    COMPONENTS:
    1. Spread crossing: Always pay half the spread
    2. Market impact: Large orders move price
    3. Random component: Market microstructure noise
    """

    base_spread_pct: float = 0.0002  # 2 pips for forex
    market_impact_factor: float = 0.1  # Impact per 1% of ADV
    random_std: float = 0.0001  # Random noise

    def apply(
        self,
        order_price: float,
        side: Side,
        size: float,
        avg_daily_volume: float = None,
    ) -> float:
        """Apply slippage to order price."""

        # Base spread (always pay half)
        spread_slippage = order_price * self.base_spread_pct / 2

        # Market impact (for large orders)
        if avg_daily_volume and avg_daily_volume > 0:
            participation_rate = size / avg_daily_volume
            impact_slippage = order_price * participation_rate * self.market_impact_factor
        else:
            impact_slippage = 0

        # Random component
        random_slippage = np.random.normal(0, order_price * self.random_std)

        # Apply direction
        total_slippage = spread_slippage + impact_slippage + abs(random_slippage)

        if side == Side.BUY:
            return order_price + total_slippage  # Buy higher
        else:
            return order_price - total_slippage  # Sell lower


@dataclass
class LatencyModel:
    """Models realistic execution latency."""

    base_latency_ms: float = 50  # Base network latency
    processing_latency_ms: float = 10  # Order processing
    jitter_std_ms: float = 20  # Random variation

    def simulate_delay(self) -> float:
        """Return simulated latency in milliseconds."""
        base = self.base_latency_ms + self.processing_latency_ms
        jitter = abs(np.random.normal(0, self.jitter_std_ms))
        return base + jitter
```

### 5.3 Simulation to Production Checklist

```python
SIMULATION_VALIDATION_REQUIREMENTS = {
    'duration': {
        'minimum_days': 30,
        'minimum_trades': 100,
        'must_include': ['trending', 'ranging', 'volatile'],
    },

    'performance': {
        'min_sharpe_ratio': 1.0,
        'min_win_rate': 0.50,
        'max_drawdown': 'within_profile_limit * 1.5',
        'profit_factor': 1.3,
    },

    'risk_metrics': {
        'circuit_breakers_tested': True,
        'recovery_protocol_tested': True,
        'edge_cases_covered': [
            'consecutive_losses',
            'drawdown_scenarios',
            'high_volatility',
            'low_liquidity',
        ],
    },

    'infrastructure': {
        'monitoring_active': True,
        'alerting_configured': True,
        'kill_switch_verified': True,
        'logging_complete': True,
    },
}


PRODUCTION_TRANSITION_PHASES = [
    {
        'phase': 'Micro-Live',
        'duration': '2 weeks',
        'position_size_multiplier': 0.10,  # 10% of normal
        'description': 'Verify real execution matches simulation',
        'success_criteria': [
            'Slippage within 20% of model',
            'Latency within acceptable range',
            'No technical errors',
        ],
    },
    {
        'phase': 'Small-Live',
        'duration': '2 weeks',
        'position_size_multiplier': 0.25,
        'description': 'Increase exposure, validate metrics',
        'success_criteria': [
            'Performance matches simulation +/- 20%',
            'No unexpected behavior',
            'All risk controls functioning',
        ],
    },
    {
        'phase': 'Medium-Live',
        'duration': '2 weeks',
        'position_size_multiplier': 0.50,
        'description': 'Half position sizes',
        'success_criteria': [
            'Consistent performance',
            'Risk metrics within bounds',
            'Operational stability',
        ],
    },
    {
        'phase': 'Full-Production',
        'duration': 'Ongoing',
        'position_size_multiplier': 1.00,
        'description': 'Full position sizes per risk profile',
        'success_criteria': [
            'All validation requirements met',
            'Management approval obtained',
        ],
    },
]
```

---

## 6. Implementation Structure

### 6.1 Module Structure

```
src/trading/
├── __init__.py
├── robot/
│   ├── __init__.py
│   ├── core.py                    # TradingRobot main class
│   ├── state.py                   # Trading state machine
│   └── config.py                  # Robot configuration
├── signals/
│   ├── __init__.py
│   ├── generator.py               # SignalGenerator
│   └── actions.py                 # Action enums and types
├── risk/
│   ├── __init__.py
│   ├── profiles.py                # RiskProfile definitions
│   ├── manager.py                 # RiskManager
│   ├── position_sizing.py         # Position sizing logic
│   └── limits.py                  # Limit checking
├── circuit_breakers/
│   ├── __init__.py
│   ├── base.py                    # CircuitBreaker base class
│   ├── consecutive_loss.py        # ConsecutiveLossBreaker
│   ├── period_loss.py             # DailyLossBreaker, WeeklyLossBreaker
│   ├── drawdown.py                # DrawdownBreaker
│   ├── model_degradation.py       # ModelDegradationBreaker
│   ├── market_instability.py      # MarketInstabilityBreaker
│   ├── manager.py                 # CircuitBreakerManager
│   └── recovery.py                # RecoveryProtocol
├── execution/
│   ├── __init__.py
│   ├── base.py                    # ExecutionMode base class
│   ├── simulation.py              # SimulationMode
│   ├── production.py              # ProductionMode
│   ├── slippage.py                # SlippageModel
│   └── orders.py                  # Order types and management
├── brokers/
│   ├── __init__.py
│   ├── base.py                    # BrokerAdapter base class
│   ├── alpaca.py                  # Alpaca integration
│   ├── mt5.py                     # MetaTrader 5 integration
│   └── interactive_brokers.py     # IB integration
├── monitoring/
│   ├── __init__.py
│   ├── metrics.py                 # Trading metrics
│   ├── logging.py                 # Trade logging
│   └── alerts.py                  # Alert system
└── safety/
    ├── __init__.py
    ├── kill_switch.py             # Kill switch implementation
    ├── validation.py              # Order validation
    └── reconciliation.py          # Position reconciliation
```

### 6.2 Core Robot Class

```python
class TradingRobot:
    """
    Main trading robot orchestrator.

    RESPONSIBILITIES:
    - Coordinate all components
    - Run trading loop
    - Handle state transitions
    - Manage lifecycle
    """

    def __init__(
        self,
        config: RobotConfig,
        risk_profile: RiskProfile,
        execution_mode: ExecutionMode,
        prediction_engine: PredictionEngine,
    ):
        self.config = config
        self.risk_profile = risk_profile
        self.execution_mode = execution_mode
        self.prediction_engine = prediction_engine

        # Initialize components
        self.signal_generator = SignalGenerator(
            prediction_engine=prediction_engine,
            risk_profile=risk_profile,
        )

        self.risk_manager = RiskManager(risk_profile=risk_profile)

        self.circuit_breakers = CircuitBreakerManager(
            risk_profile=risk_profile,
        )

        self.state_machine = TradingStateMachine()

        self.logger = TradingLogger()
        self.metrics = TradingMetrics()

    async def run(self):
        """Main trading loop."""

        self.logger.info("Trading robot starting...")

        while self.state_machine.is_running:
            try:
                await self._trading_cycle()
            except Exception as e:
                self.logger.error(f"Trading cycle error: {e}")
                await self._handle_error(e)

            await asyncio.sleep(self.config.cycle_interval_seconds)

    async def _trading_cycle(self):
        """Single trading cycle."""

        # Check if trading is allowed
        if not self.state_machine.can_trade:
            self.logger.debug(f"Trading paused: {self.state_machine.pause_reason}")
            return

        # Fetch latest market data
        market_data = await self._fetch_market_data()

        # Generate features
        features = self._prepare_features(market_data)

        # Generate signal
        signal = self.signal_generator.generate_signal(
            features=features,
            symbol=self.config.symbol,
            current_position=self.execution_mode.get_position(self.config.symbol),
        )

        # Log signal
        self.logger.log_signal(signal)

        # Execute if actionable
        if signal.action != Action.HOLD:
            # Risk check
            risk_check = self.risk_manager.check(signal)
            if not risk_check.approved:
                self.logger.info(f"Trade rejected: {risk_check.reason}")
                return

            # Execute trade
            result = await self._execute_trade(signal)

            # Update metrics and circuit breakers
            self._update_after_trade(result)

    async def _execute_trade(self, signal: TradingSignal) -> TradeResult:
        """Execute trading signal."""

        # Create order
        order = self._create_order(signal)

        # Submit order
        result = self.execution_mode.submit_order(order)

        # Log execution
        self.logger.log_execution(order, result)

        return result

    def _update_after_trade(self, result: TradeResult):
        """Update state after trade completion."""

        # Update circuit breakers
        breaker_action = self.circuit_breakers.record_trade(result)

        if breaker_action.action != TradingState.ACTIVE:
            self.state_machine.transition(breaker_action.action)
            self.logger.warning(f"Circuit breaker triggered: {breaker_action.reason}")

        # Update metrics
        self.metrics.record_trade(result)
```

---

## 7. Configuration Files

### 7.1 Robot Configuration

```yaml
# configs/trading_robot.yaml

robot:
  name: "AI-Trader Robot"
  version: "1.0.0"

  # Basic settings
  symbol: "EURUSD"
  timeframe_profile: "trader"  # scalper, trader, investor

  # Execution
  mode: "simulation"  # simulation, production
  cycle_interval_seconds: 60  # How often to check for signals

  # Risk profile
  risk_profile: "moderate"  # ultra_conservative, conservative, moderate, aggressive, ultra_aggressive

  # Models
  model_paths:
    short_term: "models/short_term_v1.pt"
    medium_term: "models/medium_term_v1.pt"
    long_term: "models/long_term_v1.pt"

  ensemble_weights:
    short_term: 0.5
    medium_term: 0.3
    long_term: 0.2

  # Safety
  kill_switch:
    enabled: true
    max_daily_trades: 50
    max_position_value: 100000
    emergency_close_on_disconnect: true

  # Monitoring
  monitoring:
    log_level: "INFO"
    metrics_interval_seconds: 60
    alert_channels:
      - type: "email"
        address: "alerts@example.com"
      - type: "telegram"
        chat_id: "123456"

# Simulation-specific settings
simulation:
  initial_capital: 100000

  slippage:
    base_spread_pct: 0.0002
    market_impact_factor: 0.1

  latency:
    base_ms: 50
    jitter_std_ms: 20

  commissions:
    type: "per_trade"
    amount: 0.0

# Production-specific settings
production:
  broker: "alpaca"

  alpaca:
    api_key_env: "ALPACA_API_KEY"
    secret_key_env: "ALPACA_SECRET_KEY"
    paper: true  # Use paper trading endpoint

  position_reconciliation:
    enabled: true
    interval_seconds: 300
```

---

## 8. Summary and Recommendations

### 8.1 Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Beta distribution for confidence** | Model learns when to be confident, enabling intelligent position sizing |
| **Five risk profiles** | Accommodates different user risk tolerances without code changes |
| **Multiple circuit breakers** | Defense in depth - no single point of failure |
| **Graduated recovery** | Prevents immediate return to full trading after problems |
| **Simulation-first approach** | Validates strategy before risking real money |

### 8.2 Best Practices Summary

1. **Never trade below confidence threshold** - HOLD is always an option
2. **Position size scales with confidence** - More certain = larger position
3. **Circuit breakers protect capital** - Automatic halt on adverse conditions
4. **Recovery is graduated** - Earn back full trading through wins
5. **Simulation validates production** - Test everything in paper mode first

### 8.3 Expected Improvements Over Basic Approach

| Metric | Basic Approach | Our Design | Improvement |
|--------|---------------|------------|-------------|
| **Capital Protection** | None (fixed size) | Circuit breakers | Prevents blowup |
| **Position Sizing** | Fixed | Confidence-based | 2-3x better risk-adjusted returns |
| **Loss Recovery** | None | Graduated protocol | Faster recovery, fewer repeat losses |
| **Adaptability** | Static | Dynamic risk profiles | Adapts to market conditions |
| **Production Safety** | Limited | Kill switches + validation | Enterprise-grade |

---

*Document Version: 1.0*
*Last Updated: 2026-01-08*
*Author: AI Trader Development Team*
