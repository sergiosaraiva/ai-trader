# AI-Trader Implementation Plan

## Executive Summary

This document provides a **step-by-step implementation plan** for building the complete AI-Trader system, from data pipeline through trained models to a production-ready trading robot. The plan is designed to be executed in phases, with each phase building on the previous one.

**Total Phases**: 8
**Estimated Components**: ~50 files
**Testing Coverage Target**: 80%+

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        AI-TRADER COMPLETE SYSTEM                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  PHASE 1-2: DATA PIPELINE                                                        │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐       │
│  │ Data Source │───▶│  Timeframe  │───▶│  Technical  │───▶│  Feature    │       │
│  │ (MT5/Alpaca)│    │  Transform  │    │  Indicators │    │  Store      │       │
│  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘       │
│                                                                                  │
│  PHASE 3-4: MODEL TRAINING                                                       │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                          │
│  │ Short-Term  │    │ Medium-Term │    │ Long-Term   │                          │
│  │ Model       │    │ Model       │    │ Model       │                          │
│  │ (Beta out)  │    │ (Beta out)  │    │ (Beta out)  │                          │
│  └──────┬──────┘    └──────┬──────┘    └──────┬──────┘                          │
│         └──────────────────┼──────────────────┘                                 │
│                            ▼                                                     │
│  PHASE 5: ENSEMBLE        ┌─────────────┐                                       │
│                           │  Ensemble   │                                       │
│                           │  Combiner   │                                       │
│                           └──────┬──────┘                                       │
│                                  │                                              │
│  PHASE 6-7: TRADING ROBOT        ▼                                              │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐       │
│  │   Signal    │───▶│    Risk     │───▶│  Circuit    │───▶│  Execution  │       │
│  │  Generator  │    │   Manager   │    │  Breakers   │    │   Engine    │       │
│  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘       │
│                                                                                  │
│  PHASE 8: PRODUCTION                                                             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                          │
│  │ Simulation  │───▶│   Paper     │───▶│ Production  │                          │
│  │  Testing    │    │  Trading    │    │    Live     │                          │
│  └─────────────┘    └─────────────┘    └─────────────┘                          │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## Phase 1: Data Pipeline Foundation

### Objective
Build robust data ingestion and storage system that fetches OHLCV data from multiple sources and stores it efficiently.

### Prerequisites
- Python 3.11+ environment with venv
- Sample data in `data/sample/`

### Tasks

#### 1.1 Data Source Abstraction
**File**: `src/data/sources/base.py`

```python
# Interface for all data sources
class DataSource(ABC):
    @abstractmethod
    async def fetch_ohlcv(symbol, timeframe, start, end) -> pd.DataFrame
    @abstractmethod
    async def get_latest(symbol, timeframe, n_bars) -> pd.DataFrame
    @abstractmethod
    def get_supported_symbols() -> List[str]
```

**Acceptance Criteria**:
- [ ] Abstract base class with clear interface
- [ ] Async support for non-blocking I/O
- [ ] Error handling with retries
- [ ] Rate limiting support

#### 1.2 Data Source Implementations
**Files**:
- `src/data/sources/csv_source.py` - Local CSV files (for development)
- `src/data/sources/alpaca_source.py` - Alpaca Markets API
- `src/data/sources/mt5_source.py` - MetaTrader 5

**Acceptance Criteria**:
- [ ] CSV source works with sample data
- [ ] Alpaca source handles authentication
- [ ] MT5 source connects to terminal
- [ ] All sources return consistent DataFrame format

#### 1.3 Data Storage Layer
**Files**:
- `src/data/storage/base.py` - Storage interface
- `src/data/storage/parquet_store.py` - Parquet file storage
- `src/data/storage/sqlite_store.py` - SQLite for metadata

**Acceptance Criteria**:
- [ ] Efficient storage with compression
- [ ] Fast retrieval by date range
- [ ] Incremental updates (append new data)
- [ ] Data integrity validation

#### 1.4 Data Pipeline Orchestrator
**File**: `src/data/pipeline.py`

```python
class DataPipeline:
    def __init__(self, source: DataSource, storage: DataStorage)
    async def fetch_and_store(symbol, timeframe, start, end)
    async def update_latest(symbol, timeframe)  # Incremental update
    def get_data(symbol, timeframe, start, end) -> pd.DataFrame
```

**Acceptance Criteria**:
- [ ] Coordinates source and storage
- [ ] Handles gaps in data
- [ ] Logging and monitoring
- [ ] Configuration via YAML

### Tests for Phase 1
- `tests/data/test_sources.py`
- `tests/data/test_storage.py`
- `tests/data/test_pipeline.py`

### Deliverables
- [ ] Working data pipeline with CSV source
- [ ] Data stored in Parquet format
- [ ] 100% test coverage for data module
- [ ] Documentation in docstrings

---

## Phase 2: Feature Engineering Pipeline

### Objective
Build comprehensive technical indicator calculation with timeframe transformation.

### Prerequisites
- Phase 1 complete
- pandas-ta library installed

### Tasks

#### 2.1 Timeframe Transformation
**File**: `src/data/processors/timeframe_transformer.py`

```python
class TimeframeTransformer:
    def transform(df: pd.DataFrame, source_tf: str, target_tf: str) -> pd.DataFrame
    def aggregate_ohlcv(df, rule) -> pd.DataFrame  # Proper OHLCV aggregation
```

**Acceptance Criteria**:
- [ ] Correct OHLCV aggregation (Open=first, High=max, Low=min, Close=last, Volume=sum)
- [ ] Support for all standard timeframes (1m to 1M)
- [ ] Handle gaps and missing data
- [ ] Timezone-aware processing

#### 2.2 Technical Indicator Calculator
**File**: `src/features/technical/calculator.py`

```python
class TechnicalIndicatorCalculator:
    def __init__(self, config: IndicatorConfig)
    def calculate_all(df: pd.DataFrame) -> pd.DataFrame
    def calculate_for_model(df, model_type: str) -> pd.DataFrame  # short/medium/long
```

**Acceptance Criteria**:
- [ ] All indicators from `docs/03-technical-indicators-configuration.md`
- [ ] Configurable per model type
- [ ] NaN handling strategy
- [ ] Performance optimized (vectorized operations)

#### 2.3 Feature Store
**File**: `src/features/store.py`

```python
class FeatureStore:
    def compute_and_store(symbol, timeframe, start, end)
    def get_features(symbol, timeframe, start, end) -> pd.DataFrame
    def get_feature_names(model_type: str) -> List[str]
```

**Acceptance Criteria**:
- [ ] Caches computed features
- [ ] Invalidation on source data change
- [ ] Memory-efficient retrieval
- [ ] Feature versioning

#### 2.4 Data Loader for Training
**File**: `src/data/loaders/training_loader.py`

```python
class TrainingDataLoader:
    def __init__(self, feature_store, config)
    def create_sequences(df, seq_length, horizon) -> Tuple[np.array, np.array]
    def get_train_val_test_split(symbol, timeframe) -> Tuple[DataLoader, ...]
    def create_labels(df, method='direction') -> np.array  # direction, returns, etc.
```

**Acceptance Criteria**:
- [ ] Proper time-series splitting (no leakage!)
- [ ] Sequence creation for LSTM/Transformer
- [ ] Label generation (direction, returns)
- [ ] PyTorch DataLoader integration
- [ ] Normalization with saved scalers

### Tests for Phase 2
- `tests/features/test_timeframe_transformer.py`
- `tests/features/test_indicator_calculator.py`
- `tests/features/test_feature_store.py`
- `tests/data/test_training_loader.py`

### Deliverables
- [ ] Complete feature pipeline
- [ ] All indicators working
- [ ] Training data loaders ready
- [ ] Feature documentation

---

## Phase 3: Model Architecture Implementation

### Objective
Implement the three time-horizon models with Beta distribution outputs for learned confidence.

### Prerequisites
- Phase 2 complete
- PyTorch installed

### Tasks

#### 3.1 Base Model with Beta Output
**File**: `src/models/base.py` (update existing)

```python
class BaseModelWithUncertainty(nn.Module):
    """Base class for all models with Beta output."""

    def __init__(self, config: ModelConfig)

    @abstractmethod
    def forward(self, x) -> Dict[str, BetaPrediction]

    def predict_with_confidence(self, x) -> Dict:
        """Returns direction, probability, confidence, alpha, beta"""
```

**Acceptance Criteria**:
- [ ] Beta output layer integration
- [ ] Standard interface for all models
- [ ] Checkpoint saving/loading
- [ ] Device management (CPU/GPU)

#### 3.2 Short-Term Model (CNN-LSTM-Attention)
**File**: `src/models/technical/short_term.py`

```python
class ShortTermModel(BaseModelWithUncertainty):
    """
    Architecture:
    - Multi-scale CNN for pattern detection
    - Bi-LSTM for temporal dependencies
    - Multi-head attention for focus
    - Beta output for direction + confidence
    """
```

**Architecture Details**:
```
Input: (batch, seq_len, features)
  ↓
CNN Block (filters: 64, 128, 256; kernels: 3, 5, 7)
  ↓
Bi-LSTM (hidden: 256, layers: 2, dropout: 0.3)
  ↓
Multi-Head Attention (heads: 8, dim: 256)
  ↓
Beta Output Layer → BetaPrediction(α, β)
```

**Acceptance Criteria**:
- [ ] Architecture matches specification
- [ ] Forward pass works with sample data
- [ ] Beta output produces valid distributions
- [ ] Attention weights extractable for interpretability

#### 3.3 Medium-Term Model (TFT-style)
**File**: `src/models/technical/medium_term.py`

```python
class MediumTermModel(BaseModelWithUncertainty):
    """
    Temporal Fusion Transformer (simplified):
    - Variable selection network
    - LSTM encoder
    - Multi-head attention
    - Beta output
    """
```

**Acceptance Criteria**:
- [ ] Variable selection working
- [ ] Temporal attention
- [ ] Interpretable feature importance
- [ ] Beta output integration

#### 3.4 Long-Term Model (N-BEATS + Transformer)
**File**: `src/models/technical/long_term.py`

```python
class LongTermModel(BaseModelWithUncertainty):
    """
    N-BEATS backbone with Transformer enhancement:
    - N-BEATS stacks for trend/seasonality
    - Transformer for long-range dependencies
    - Beta output
    - Optional regime classification (Dirichlet)
    """
```

**Acceptance Criteria**:
- [ ] N-BEATS blocks implemented
- [ ] Transformer integration
- [ ] Regime output (optional)
- [ ] Beta output for direction

#### 3.5 Loss Functions
**File**: `src/models/losses.py`

```python
class TradingLoss(nn.Module):
    """Combined loss for trading models."""

    def __init__(self, direction_weight=1.0, calibration_weight=0.1):
        self.beta_nll = BetaNLLLoss()
        self.calibration = CalibrationLoss()

    def forward(self, predictions, targets):
        direction_loss = self.beta_nll(predictions['direction'], targets['direction'])
        # Optional: price loss, regime loss
        return combined_loss
```

**Acceptance Criteria**:
- [ ] Beta NLL loss working
- [ ] Optional calibration regularization
- [ ] Multi-task loss weighting
- [ ] Gradient logging for debugging

### Tests for Phase 3
- `tests/models/test_short_term.py`
- `tests/models/test_medium_term.py`
- `tests/models/test_long_term.py`
- `tests/models/test_losses.py`

### Deliverables
- [ ] Three model architectures implemented
- [ ] All models output BetaPrediction
- [ ] Loss functions ready
- [ ] Model unit tests passing

---

## Phase 4: Training Pipeline

### Objective
Build complete training infrastructure with experiment tracking, early stopping, and model checkpointing.

### Prerequisites
- Phase 3 complete
- MLflow installed

### Tasks

#### 4.1 Training Configuration
**File**: `src/training/config.py`

```python
@dataclass
class TrainingConfig:
    # Model
    model_type: str  # 'short_term', 'medium_term', 'long_term'
    model_config: ModelConfig

    # Training
    batch_size: int = 64
    epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5

    # Early stopping
    patience: int = 15
    min_delta: float = 1e-4

    # Checkpointing
    checkpoint_dir: str = 'checkpoints'
    save_top_k: int = 3
```

#### 4.2 Trainer Class
**File**: `src/training/trainer.py`

```python
class ModelTrainer:
    def __init__(self, model, config: TrainingConfig, device: str = 'auto')

    def train(self, train_loader, val_loader) -> TrainingResult:
        """
        Training loop with:
        - Learning rate scheduling (OneCycleLR)
        - Gradient clipping
        - Early stopping
        - MLflow logging
        - Checkpoint saving
        """

    def evaluate(self, test_loader) -> EvaluationResult:
        """Comprehensive evaluation with trading metrics."""

    def load_checkpoint(self, path: str):
        """Load model from checkpoint."""
```

**Acceptance Criteria**:
- [ ] Complete training loop
- [ ] Early stopping working
- [ ] Checkpoints saved correctly
- [ ] MLflow experiment tracking
- [ ] GPU support

#### 4.3 Experiment Manager
**File**: `src/training/experiment.py`

```python
class ExperimentManager:
    def __init__(self, mlflow_tracking_uri: str)

    def run_experiment(self, config: TrainingConfig, data_config: DataConfig):
        """
        Full experiment:
        1. Load and prepare data
        2. Train model
        3. Evaluate on test set
        4. Log all metrics and artifacts
        5. Save model
        """

    def hyperparameter_search(self, search_space: Dict, n_trials: int):
        """Optuna-based hyperparameter optimization."""
```

**Acceptance Criteria**:
- [ ] End-to-end experiment execution
- [ ] All metrics logged to MLflow
- [ ] Hyperparameter search working
- [ ] Reproducible experiments (seeding)

#### 4.4 Training Script
**File**: `scripts/train_model.py`

```bash
# Usage examples:
python scripts/train_model.py --model short_term --profile trader --symbol EURUSD
python scripts/train_model.py --model all --profile trader --symbol EURUSD
python scripts/train_model.py --config configs/training/experiment_1.yaml
```

**Acceptance Criteria**:
- [ ] CLI interface with argparse
- [ ] Config file support
- [ ] Progress logging
- [ ] Error handling

#### 4.5 Model Evaluation
**File**: `src/training/evaluation.py`

```python
class ModelEvaluator:
    def evaluate_direction_accuracy(predictions, actuals) -> float
    def evaluate_confidence_calibration(predictions, actuals) -> CalibrationMetrics
    def evaluate_trading_metrics(predictions, actuals, prices) -> TradingMetrics
    def generate_evaluation_report(results) -> str
```

**Metrics to Calculate**:
- Direction accuracy
- Precision, Recall, F1
- Calibration (ECE, MCE)
- Brier score
- Sharpe ratio (simulated)
- Maximum drawdown (simulated)

### Tests for Phase 4
- `tests/training/test_trainer.py`
- `tests/training/test_experiment.py`
- `tests/training/test_evaluation.py`

### Deliverables
- [ ] Complete training pipeline
- [ ] MLflow integration working
- [ ] Training script ready
- [ ] At least one trained model per type

---

## Phase 5: Ensemble System

### Objective
Combine the three models into a unified ensemble with dynamic weighting and confidence aggregation.

### Prerequisites
- Phase 4 complete
- At least one trained model per type

### Tasks

#### 5.1 Ensemble Combiner
**File**: `src/models/ensemble/combiner.py`

```python
class TechnicalEnsemble:
    def __init__(
        self,
        short_model: ShortTermModel,
        medium_model: MediumTermModel,
        long_model: LongTermModel,
        weights: Dict[str, float] = None,
    )

    def predict(self, features: Dict[str, torch.Tensor]) -> EnsemblePrediction:
        """
        Combine predictions from all models:
        1. Get individual predictions (direction, confidence)
        2. Weight by model weights and individual confidence
        3. Penalize for disagreement
        4. Return combined prediction with agreement metric
        """

    def update_weights(self, performance_metrics: Dict):
        """Dynamic weight adjustment based on recent performance."""
```

**Acceptance Criteria**:
- [ ] Proper weighted combination
- [ ] Disagreement penalty implemented
- [ ] Agreement metric calculated
- [ ] Confidence aggregation correct

#### 5.2 Dynamic Weight Calculator
**File**: `src/models/ensemble/weights.py`

```python
class DynamicWeightCalculator:
    def __init__(self, base_weights, lookback_trades=50)

    def calculate_weights(
        self,
        recent_performance: Dict[str, List[TradeResult]],
        market_regime: str,
        volatility_level: float,
    ) -> Dict[str, float]:
        """
        Adjust weights based on:
        - Recent model performance
        - Market regime (trending vs ranging)
        - Volatility conditions
        """
```

**Weight Adjustment Rules**:
- Trending market: Favor medium/long term (0.3, 0.35, 0.35)
- Ranging market: Favor short term (0.5, 0.3, 0.2)
- High volatility: Reduce all weights, increase confidence threshold

#### 5.3 Ensemble Predictor Integration
**File**: `src/models/ensemble/predictor.py`

```python
class EnsemblePredictor:
    """Production-ready ensemble prediction."""

    def __init__(self, ensemble: TechnicalEnsemble, feature_store: FeatureStore)

    def predict(self, symbol: str) -> EnsemblePrediction:
        """
        Full prediction pipeline:
        1. Get latest data
        2. Compute features for each timeframe
        3. Run ensemble prediction
        4. Return with all metadata
        """
```

### Tests for Phase 5
- `tests/ensemble/test_combiner.py`
- `tests/ensemble/test_weights.py`
- `tests/ensemble/test_predictor.py`

### Deliverables
- [ ] Working ensemble system
- [ ] Dynamic weights implemented
- [ ] Integration with feature store
- [ ] End-to-end prediction working

---

## Phase 6: Trading Robot Core

### Objective
Implement the trading robot with signal generation, risk management, and circuit breakers.

### Prerequisites
- Phase 5 complete
- Risk profiles and circuit breakers from earlier design

### Tasks

#### 6.1 Signal Generator Enhancement
**File**: `src/trading/signals/generator.py` (enhance existing)

Add integration with:
- EnsemblePredictor
- Real-time feature computation
- Stop-loss/take-profit calculation

**Acceptance Criteria**:
- [ ] Generates signals from live predictions
- [ ] Position sizing based on confidence
- [ ] Risk parameters from profile
- [ ] All signal metadata populated

#### 6.2 Order Management
**File**: `src/trading/orders/manager.py`

```python
class OrderManager:
    def __init__(self, execution_mode: ExecutionMode)

    def create_order(self, signal: TradingSignal, account: Account) -> Order
    def submit_order(self, order: Order) -> OrderResult
    def cancel_order(self, order_id: str) -> bool
    def get_open_orders(self) -> List[Order]

    # Bracket orders
    def create_bracket_order(
        self,
        signal: TradingSignal,
        account: Account,
    ) -> BracketOrder:
        """Creates entry + stop-loss + take-profit."""
```

**Acceptance Criteria**:
- [ ] Order creation from signals
- [ ] Bracket order support
- [ ] Order state tracking
- [ ] Partial fill handling

#### 6.3 Position Manager
**File**: `src/trading/positions/manager.py`

```python
class PositionManager:
    def __init__(self, execution_mode: ExecutionMode)

    def get_position(self, symbol: str) -> Optional[Position]
    def get_all_positions(self) -> List[Position]
    def update_positions(self) -> None  # Sync with broker
    def calculate_pnl(self) -> Dict[str, float]
    def calculate_exposure(self) -> float
```

#### 6.4 Account Manager
**File**: `src/trading/account/manager.py`

```python
class AccountManager:
    def __init__(self, execution_mode: ExecutionMode)

    def get_balance(self) -> float
    def get_equity(self) -> float
    def get_margin_available(self) -> float
    def get_daily_pnl(self) -> float
    def reset_daily_counters(self) -> None
```

#### 6.5 Trading Robot Core
**File**: `src/trading/robot/core.py`

```python
class TradingRobot:
    def __init__(
        self,
        config: RobotConfig,
        ensemble_predictor: EnsemblePredictor,
        risk_profile: RiskProfile,
        execution_mode: ExecutionMode,
    )

    async def start(self):
        """Start the trading robot."""

    async def stop(self):
        """Gracefully stop the robot."""

    async def _trading_cycle(self):
        """
        Single trading cycle:
        1. Check if trading allowed (circuit breakers)
        2. Get latest market data
        3. Generate prediction
        4. Generate signal
        5. Execute if actionable
        6. Update state
        """

    def get_status(self) -> RobotStatus:
        """Get current robot status."""
```

**Acceptance Criteria**:
- [ ] Complete trading cycle
- [ ] Async operation
- [ ] Graceful shutdown
- [ ] State persistence
- [ ] Comprehensive logging

### Tests for Phase 6
- `tests/trading/test_order_manager.py`
- `tests/trading/test_position_manager.py`
- `tests/trading/test_robot_core.py`

### Deliverables
- [ ] Complete trading robot
- [ ] Order management working
- [ ] Position tracking working
- [ ] All tests passing

---

## Phase 7: Simulation Mode

### Objective
Build complete simulation environment for paper trading with realistic market modeling.

### Prerequisites
- Phase 6 complete

### Tasks

#### 7.1 Simulation Execution Mode
**File**: `src/trading/execution/simulation.py`

```python
class SimulationMode(ExecutionMode):
    def __init__(
        self,
        initial_capital: float,
        slippage_model: SlippageModel,
        latency_model: LatencyModel,
        commission_model: CommissionModel,
    )

    def submit_order(self, order: Order) -> OrderResult:
        """Simulate order execution with realistic fills."""

    def get_market_price(self, symbol: str) -> float:
        """Get current simulated market price."""
```

#### 7.2 Market Simulator
**File**: `src/simulation/market_simulator.py`

```python
class MarketSimulator:
    def __init__(self, data_source: DataSource)

    def start(self, start_time: datetime):
        """Start simulation from specific time."""

    def advance(self, bars: int = 1):
        """Advance simulation by N bars."""

    def get_current_bar(self, symbol: str) -> Dict:
        """Get current OHLCV bar."""

    def is_market_open(self) -> bool:
        """Check if market is open."""
```

#### 7.3 Backtester
**File**: `src/simulation/backtester.py`

```python
class Backtester:
    def __init__(
        self,
        robot: TradingRobot,
        market_simulator: MarketSimulator,
    )

    def run(
        self,
        start_date: datetime,
        end_date: datetime,
        symbols: List[str],
    ) -> BacktestResult:
        """
        Run full backtest:
        1. Initialize simulation
        2. For each bar:
           - Update market data
           - Run robot cycle
           - Record trades and equity
        3. Calculate metrics
        4. Generate report
        """

    def generate_report(self, result: BacktestResult) -> str:
        """Generate comprehensive backtest report."""
```

#### 7.4 Performance Metrics
**File**: `src/simulation/metrics.py`

```python
class PerformanceMetrics:
    @staticmethod
    def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0) -> float

    @staticmethod
    def calculate_sortino_ratio(returns: pd.Series, risk_free_rate: float = 0) -> float

    @staticmethod
    def calculate_max_drawdown(equity_curve: pd.Series) -> Tuple[float, datetime, datetime]

    @staticmethod
    def calculate_calmar_ratio(returns: pd.Series, max_dd: float) -> float

    @staticmethod
    def calculate_win_rate(trades: List[Trade]) -> float

    @staticmethod
    def calculate_profit_factor(trades: List[Trade]) -> float

    @staticmethod
    def generate_summary(trades, equity_curve) -> Dict:
        """Complete performance summary."""
```

#### 7.5 Backtest Script
**File**: `scripts/run_backtest.py`

```bash
# Usage:
python scripts/run_backtest.py \
    --symbol EURUSD \
    --start 2023-01-01 \
    --end 2024-01-01 \
    --profile trader \
    --risk-profile moderate \
    --output results/backtest_001
```

### Tests for Phase 7
- `tests/simulation/test_market_simulator.py`
- `tests/simulation/test_backtester.py`
- `tests/simulation/test_metrics.py`

### Deliverables
- [ ] Complete simulation mode
- [ ] Backtester working
- [ ] Performance metrics calculated
- [ ] Backtest reports generated
- [ ] At least one full backtest run

---

## Phase 8: Production Mode

### Objective
Implement production trading with broker integration, monitoring, and safety mechanisms.

### Prerequisites
- Phase 7 complete
- Successful backtest results
- Broker API credentials

### Tasks

#### 8.1 Broker Adapters
**Files**:
- `src/trading/brokers/alpaca.py`
- `src/trading/brokers/mt5.py`

```python
class AlpacaBroker(BrokerAdapter):
    def __init__(self, api_key: str, secret_key: str, paper: bool = True)

    def submit_order(self, order: Order) -> OrderResult
    def cancel_order(self, order_id: str) -> bool
    def get_positions(self) -> List[Position]
    def get_account(self) -> AccountInfo
    def stream_quotes(self, symbols: List[str]) -> AsyncIterator[Quote]
```

**Acceptance Criteria**:
- [ ] Authentication working
- [ ] Order submission working
- [ ] Position sync working
- [ ] Real-time quotes (optional)

#### 8.2 Production Execution Mode
**File**: `src/trading/execution/production.py`

```python
class ProductionMode(ExecutionMode):
    def __init__(
        self,
        broker: BrokerAdapter,
        kill_switch: KillSwitch,
        order_validator: OrderValidator,
    )

    def submit_order(self, order: Order) -> OrderResult:
        """
        Production order submission:
        1. Check kill switch
        2. Validate order
        3. Submit to broker
        4. Verify execution
        5. Reconcile position
        """
```

#### 8.3 Kill Switch
**File**: `src/trading/safety/kill_switch.py`

```python
class KillSwitch:
    def __init__(self, config: KillSwitchConfig)

    @property
    def is_active(self) -> bool:
        """Check if trading is allowed."""

    def trigger(self, reason: str):
        """Emergency halt."""

    def reset(self, authorization: str):
        """Reset with authorization."""

    # Automatic triggers
    def check_daily_trades(self, count: int) -> bool
    def check_position_value(self, value: float) -> bool
    def check_connectivity(self) -> bool
```

#### 8.4 Monitoring System
**File**: `src/trading/monitoring/monitor.py`

```python
class TradingMonitor:
    def __init__(self, config: MonitoringConfig)

    def start(self):
        """Start monitoring."""

    def record_metric(self, name: str, value: float):
        """Record custom metric."""

    def send_alert(self, level: str, message: str):
        """Send alert via configured channels."""

    def get_dashboard_data(self) -> Dict:
        """Get data for monitoring dashboard."""
```

#### 8.5 Production Deployment Script
**File**: `scripts/deploy_robot.py`

```bash
# Start in paper trading mode
python scripts/deploy_robot.py \
    --mode paper \
    --broker alpaca \
    --symbol EURUSD \
    --risk-profile conservative

# Start in production mode (requires confirmation)
python scripts/deploy_robot.py \
    --mode production \
    --broker alpaca \
    --symbol EURUSD \
    --risk-profile moderate \
    --confirm
```

### Tests for Phase 8
- `tests/brokers/test_alpaca.py`
- `tests/trading/test_production_mode.py`
- `tests/trading/test_kill_switch.py`

### Deliverables
- [ ] Broker integration working
- [ ] Kill switch implemented
- [ ] Monitoring system active
- [ ] Paper trading validated
- [ ] Production deployment script ready

---

## Implementation Checklist

### Phase 1: Data Pipeline Foundation
- [ ] `src/data/sources/base.py`
- [ ] `src/data/sources/csv_source.py`
- [ ] `src/data/sources/alpaca_source.py`
- [ ] `src/data/storage/parquet_store.py`
- [ ] `src/data/pipeline.py`
- [ ] Tests passing
- [ ] Documentation complete

### Phase 2: Feature Engineering Pipeline
- [ ] `src/data/processors/timeframe_transformer.py`
- [ ] `src/features/technical/calculator.py`
- [ ] `src/features/store.py`
- [ ] `src/data/loaders/training_loader.py`
- [ ] Tests passing
- [ ] Documentation complete

### Phase 3: Model Architecture Implementation
- [ ] `src/models/base.py` (updated)
- [ ] `src/models/technical/short_term.py`
- [ ] `src/models/technical/medium_term.py`
- [ ] `src/models/technical/long_term.py`
- [ ] `src/models/losses.py`
- [ ] Tests passing
- [ ] Documentation complete

### Phase 4: Training Pipeline
- [ ] `src/training/config.py`
- [ ] `src/training/trainer.py`
- [ ] `src/training/experiment.py`
- [ ] `src/training/evaluation.py`
- [ ] `scripts/train_model.py`
- [ ] Tests passing
- [ ] At least one trained model

### Phase 5: Ensemble System
- [ ] `src/models/ensemble/combiner.py`
- [ ] `src/models/ensemble/weights.py`
- [ ] `src/models/ensemble/predictor.py`
- [ ] Tests passing
- [ ] End-to-end prediction working

### Phase 6: Trading Robot Core
- [ ] `src/trading/signals/generator.py` (enhanced)
- [ ] `src/trading/orders/manager.py`
- [ ] `src/trading/positions/manager.py`
- [ ] `src/trading/account/manager.py`
- [ ] `src/trading/robot/core.py`
- [ ] Tests passing
- [ ] Robot cycle working

### Phase 7: Simulation Mode
- [ ] `src/trading/execution/simulation.py`
- [ ] `src/simulation/market_simulator.py`
- [ ] `src/simulation/backtester.py`
- [ ] `src/simulation/metrics.py`
- [ ] `scripts/run_backtest.py`
- [ ] Tests passing
- [ ] Successful backtest completed

### Phase 8: Production Mode
- [ ] `src/trading/brokers/alpaca.py`
- [ ] `src/trading/execution/production.py`
- [ ] `src/trading/safety/kill_switch.py`
- [ ] `src/trading/monitoring/monitor.py`
- [ ] `scripts/deploy_robot.py`
- [ ] Tests passing
- [ ] Paper trading validated

---

## Agent Roles for Implementation

### Requirements Analyst Agent
**Responsibilities**:
- Clarify requirements for each phase
- Validate acceptance criteria are met
- Ensure no scope creep
- Review documentation completeness

### Code Engineer Agent
**Responsibilities**:
- Implement code according to specifications
- Follow project coding conventions
- Write clean, maintainable code
- Integrate with existing codebase

### Quality Guardian Agent
**Responsibilities**:
- Review code for quality and security
- Ensure type hints and docstrings
- Check for code smells and anti-patterns
- Validate error handling

### Test Automator Agent
**Responsibilities**:
- Write comprehensive unit tests
- Create integration tests
- Ensure test coverage targets met
- Set up CI/CD test automation

---

## Success Criteria

### Minimum Viable Product (MVP)
- [ ] Data pipeline fetches and stores EURUSD data
- [ ] Three models trained with >52% directional accuracy
- [ ] Ensemble produces predictions with confidence
- [ ] Backtest shows positive Sharpe ratio (>1.0)
- [ ] Robot runs in simulation mode without errors

### Production Ready
- [ ] All 8 phases complete
- [ ] 80%+ test coverage
- [ ] Paper trading stable for 30+ days
- [ ] Performance metrics meet targets
- [ ] Monitoring and alerts working
- [ ] Documentation complete

---

*Document Version: 1.0*
*Created: 2026-01-08*
*Author: AI Trader Development Team*
