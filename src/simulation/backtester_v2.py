"""
Enhanced Backtester with Trading Robot Integration.

Provides comprehensive backtesting capabilities with full integration
to the Phase 6 Trading Robot components.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable, Tuple
import logging
import json
import asyncio

import pandas as pd
import numpy as np

from .market_simulator import MarketSimulator, MarketSnapshot, MarketBar, MarketSession
from .metrics import PerformanceMetrics
from ..trading.execution.simulation import (
    SimulationExecutionEngine,
    SimulationConfig,
    SlippageModel,
    LatencyModel,
    CommissionModel,
    FixedSlippageModel,
    FixedLatencyModel,
    PercentageCommissionModel,
    FillEvent,
)
from ..trading.signals.generator import SignalGenerator, EnsemblePrediction
from ..trading.signals.actions import Action, TradingSignal
from ..trading.orders.manager import (
    OrderManager,
    ExecutionMode,
    Order,
    OrderType,
    OrderSide,
    OrderStatus,
)
from ..trading.positions.manager import PositionManager, Position
from ..trading.account.manager import AccountManager
from ..trading.risk.profiles import RiskProfile, load_risk_profile
from ..trading.circuit_breakers.manager import CircuitBreakerManager
from ..trading.circuit_breakers.base import TradeResult, TradingState

logger = logging.getLogger(__name__)


class BacktestStatus(Enum):
    """Backtest status enumeration."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class BacktestConfig:
    """Configuration for backtest execution."""
    # Basic settings
    name: str = "backtest"
    symbol: str = "EURUSD"
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None

    # Capital and risk
    initial_capital: float = 100000.0
    risk_profile_name: str = "moderate"
    leverage: float = 10.0

    # Execution simulation
    slippage_pct: float = 0.0001
    commission_pct: float = 0.0001
    latency_ms: float = 50.0

    # Trading parameters
    min_confidence: float = 0.65
    warmup_bars: int = 100

    # Output
    output_dir: Optional[str] = None
    save_trades: bool = True
    save_equity_curve: bool = True
    verbose: bool = True


@dataclass
class Trade:
    """Represents a completed round-trip trade."""
    trade_id: str
    symbol: str
    side: str
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    quantity: float
    pnl: float
    pnl_pct: float
    commission: float
    slippage: float
    exit_reason: str
    holding_period: timedelta = field(default_factory=timedelta)
    max_favorable: float = 0.0
    max_adverse: float = 0.0

    def __post_init__(self):
        """Calculate derived fields."""
        self.holding_period = self.exit_time - self.entry_time

    @property
    def is_winner(self) -> bool:
        """Check if trade was profitable."""
        return self.pnl > 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (with native Python types for JSON serialization)."""
        return {
            "trade_id": self.trade_id,
            "symbol": self.symbol,
            "side": self.side,
            "entry_time": self.entry_time.isoformat(),
            "exit_time": self.exit_time.isoformat(),
            "entry_price": float(self.entry_price),
            "exit_price": float(self.exit_price),
            "quantity": float(self.quantity),
            "pnl": float(self.pnl),
            "pnl_pct": float(self.pnl_pct),
            "commission": float(self.commission),
            "slippage": float(self.slippage),
            "exit_reason": self.exit_reason,
            "holding_period_hours": self.holding_period.total_seconds() / 3600,
            "max_favorable": float(self.max_favorable),
            "max_adverse": float(self.max_adverse),
            "is_winner": bool(self.is_winner),
        }


@dataclass
class BacktestResult:
    """Complete backtest results."""
    # Basic info
    config: BacktestConfig
    status: BacktestStatus
    start_time: datetime
    end_time: datetime
    duration_seconds: float

    # Equity metrics
    initial_capital: float
    final_equity: float
    peak_equity: float
    low_equity: float

    # Return metrics
    total_return: float
    total_return_pct: float
    annualized_return: float
    cagr: float

    # Risk metrics
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    max_drawdown_duration: timedelta
    volatility: float
    downside_volatility: float

    # Trade metrics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    profit_factor: float
    average_win: float
    average_loss: float
    largest_win: float
    largest_loss: float
    average_trade: float
    expectancy: float

    # Time metrics
    average_holding_period: timedelta
    max_holding_period: timedelta
    min_holding_period: timedelta

    # Additional metrics
    trades_per_month: float
    recovery_factor: float
    risk_reward_ratio: float

    # Data series
    equity_curve: pd.Series = field(default_factory=pd.Series)
    drawdown_series: pd.Series = field(default_factory=pd.Series)
    returns_series: pd.Series = field(default_factory=pd.Series)
    trades: List[Trade] = field(default_factory=list)
    signals: List[Dict[str, Any]] = field(default_factory=list)

    # Execution stats
    total_slippage: float = 0.0
    total_commission: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (excluding large data)."""
        return {
            "config": {
                "name": self.config.name,
                "symbol": self.config.symbol,
                "initial_capital": self.config.initial_capital,
                "risk_profile": self.config.risk_profile_name,
            },
            "status": self.status.value,
            "period": {
                "start": self.start_time.isoformat(),
                "end": self.end_time.isoformat(),
                "duration_seconds": self.duration_seconds,
            },
            "equity": {
                "initial": self.initial_capital,
                "final": self.final_equity,
                "peak": self.peak_equity,
                "low": self.low_equity,
            },
            "returns": {
                "total_return": self.total_return,
                "total_return_pct": self.total_return_pct,
                "annualized_return": self.annualized_return,
                "cagr": self.cagr,
            },
            "risk": {
                "sharpe_ratio": self.sharpe_ratio,
                "sortino_ratio": self.sortino_ratio,
                "calmar_ratio": self.calmar_ratio,
                "max_drawdown": self.max_drawdown,
                "max_drawdown_duration_days": self.max_drawdown_duration.days,
                "volatility": self.volatility,
            },
            "trades": {
                "total": self.total_trades,
                "winning": self.winning_trades,
                "losing": self.losing_trades,
                "win_rate": self.win_rate,
                "profit_factor": self.profit_factor,
                "average_win": self.average_win,
                "average_loss": self.average_loss,
                "largest_win": self.largest_win,
                "largest_loss": self.largest_loss,
                "expectancy": self.expectancy,
            },
            "execution": {
                "total_slippage": self.total_slippage,
                "total_commission": self.total_commission,
            },
        }

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    def print_summary(self) -> None:
        """Print human-readable summary."""
        print("\n" + "=" * 60)
        print(f"BACKTEST RESULTS: {self.config.name}")
        print("=" * 60)
        print(f"\nPeriod: {self.start_time.date()} to {self.end_time.date()}")
        print(f"Symbol: {self.config.symbol}")
        print(f"Initial Capital: ${self.initial_capital:,.2f}")
        print(f"Final Equity: ${self.final_equity:,.2f}")

        print("\n--- RETURNS ---")
        print(f"Total Return: ${self.total_return:,.2f} ({self.total_return_pct:.2%})")
        print(f"Annualized Return: {self.annualized_return:.2%}")
        print(f"CAGR: {self.cagr:.2%}")

        print("\n--- RISK METRICS ---")
        print(f"Sharpe Ratio: {self.sharpe_ratio:.2f}")
        print(f"Sortino Ratio: {self.sortino_ratio:.2f}")
        print(f"Calmar Ratio: {self.calmar_ratio:.2f}")
        print(f"Max Drawdown: {self.max_drawdown:.2%}")
        print(f"Volatility: {self.volatility:.2%}")

        print("\n--- TRADE STATISTICS ---")
        print(f"Total Trades: {self.total_trades}")
        print(f"Win Rate: {self.win_rate:.2%}")
        print(f"Profit Factor: {self.profit_factor:.2f}")
        print(f"Average Win: ${self.average_win:,.2f}")
        print(f"Average Loss: ${self.average_loss:,.2f}")
        print(f"Expectancy: ${self.expectancy:,.2f}")

        print("\n--- EXECUTION COSTS ---")
        print(f"Total Slippage: ${self.total_slippage:,.2f}")
        print(f"Total Commission: ${self.total_commission:,.2f}")
        print("=" * 60 + "\n")


class EnhancedBacktester:
    """
    Enhanced backtester with Trading Robot integration.

    Features:
    - Full integration with Phase 6 trading components
    - Realistic execution simulation
    - Comprehensive performance metrics
    - Signal and trade logging
    - Walk-forward validation support
    """

    def __init__(
        self,
        config: Optional[BacktestConfig] = None,
        predictor: Optional[Any] = None,
        feature_provider: Optional[Callable[[str, int], Dict[str, Any]]] = None,
    ):
        """
        Initialize backtester.

        Args:
            config: Backtest configuration
            predictor: Ensemble predictor for generating signals
            feature_provider: Function to get features for prediction
        """
        self.config = config or BacktestConfig()
        self.predictor = predictor
        self.feature_provider = feature_provider

        # Initialize components
        self.risk_profile = load_risk_profile(self.config.risk_profile_name)

        # Market simulator
        self.market_simulator = MarketSimulator()

        # Execution engine
        self.execution_engine = SimulationExecutionEngine(
            SimulationConfig(
                initial_capital=self.config.initial_capital,
                slippage_model=FixedSlippageModel(self.config.slippage_pct),
                latency_model=FixedLatencyModel(self.config.latency_ms),
                commission_model=PercentageCommissionModel(self.config.commission_pct),
            )
        )

        # Trading components
        self.signal_generator = SignalGenerator(risk_profile=self.risk_profile)
        self.account_manager = AccountManager(
            initial_balance=self.config.initial_capital,
            leverage=self.config.leverage,
        )
        self.position_manager = PositionManager()
        self.circuit_breaker_manager = CircuitBreakerManager(
            risk_profile=self.risk_profile,
            initial_equity=self.config.initial_capital,
        )

        # Metrics calculator
        self.metrics = PerformanceMetrics()

        # State
        self._status = BacktestStatus.PENDING
        self._trades: List[Trade] = []
        self._signals: List[Dict[str, Any]] = []
        self._equity_history: List[Dict[str, Any]] = []
        self._current_trade_id = 0

        # Position tracking for PnL
        self._open_positions: Dict[str, Dict[str, Any]] = {}

    def load_data(
        self,
        symbol: str,
        data: pd.DataFrame,
        features: Optional[pd.DataFrame] = None,
    ) -> None:
        """
        Load historical data for backtesting.

        Args:
            symbol: Trading symbol
            data: OHLCV DataFrame
            features: Pre-computed features
        """
        self.market_simulator.load_data(symbol, data, features)

    def run(self) -> BacktestResult:
        """
        Run the backtest.

        Returns:
            BacktestResult with all metrics
        """
        run_start = datetime.now()
        self._status = BacktestStatus.RUNNING
        self._reset_state()

        try:
            # Start market simulation
            self.market_simulator.start(
                start_time=self.config.start_date,
                end_time=self.config.end_date,
            )

            bar_count = 0
            total_bars = self.market_simulator._end_index - self.market_simulator._start_index + 1

            if self.config.verbose:
                logger.info(f"Starting backtest: {total_bars} bars to process")

            # Main backtest loop
            for snapshot in self.market_simulator.iter_bars():
                bar_count += 1

                # Skip warmup period
                if bar_count < self.config.warmup_bars:
                    continue

                # Process bar
                self._process_bar(snapshot)

                # Progress logging
                if self.config.verbose and bar_count % 100 == 0:
                    progress = bar_count / total_bars * 100
                    equity = self.account_manager.equity
                    logger.info(f"Progress: {progress:.1f}% | Equity: ${equity:,.2f}")

            # Close any remaining positions
            self._close_all_positions(self.market_simulator.get_current_snapshot())

            self._status = BacktestStatus.COMPLETED

        except Exception as e:
            logger.error(f"Backtest failed: {e}", exc_info=True)
            self._status = BacktestStatus.FAILED
            raise

        run_end = datetime.now()
        duration = (run_end - run_start).total_seconds()

        # Calculate results
        result = self._calculate_results(duration)

        # Save results if configured
        if self.config.output_dir:
            self._save_results(result)

        if self.config.verbose:
            result.print_summary()

        return result

    def _reset_state(self) -> None:
        """Reset backtest state."""
        self._trades = []
        self._signals = []
        self._equity_history = []
        self._current_trade_id = 0
        self._open_positions = {}

        self.account_manager.reset()
        self.position_manager.reset()
        self.circuit_breaker_manager.force_resume()
        self.execution_engine.reset()

    def _process_bar(self, snapshot: MarketSnapshot) -> None:
        """
        Process a single bar.

        Args:
            snapshot: Current market snapshot
        """
        symbol = self.config.symbol
        bar = snapshot.get_bar(symbol)

        if bar is None:
            return

        current_price = bar.close
        current_time = bar.timestamp

        # Update execution engine with market data
        fills = self.execution_engine.update_market_data(
            symbol=symbol,
            price=current_price,
            timestamp=current_time,
            market_data={
                "high": bar.high,
                "low": bar.low,
                "volume": bar.volume,
                "atr": bar.atr,
            }
        )

        # Process any fills
        for fill in fills:
            self._process_fill(fill, bar)

        # Update position prices and PnL
        self._update_positions(bar)

        # Check circuit breakers
        breaker_state = self.circuit_breaker_manager.check_all(
            current_equity=self.account_manager.equity,
        )

        # Generate prediction and signal
        signal = self._generate_signal(bar, breaker_state)

        if signal:
            self._signals.append({
                "timestamp": current_time.isoformat(),
                "action": signal.action.value,
                "confidence": signal.confidence,
                "reason": signal.reason,
            })

            # Execute signal
            if signal.action in [Action.BUY, Action.SELL]:
                self._execute_signal(signal, bar)
            elif signal.action in [Action.CLOSE_LONG, Action.CLOSE_SHORT]:
                self._close_position(symbol, bar, signal.reason)

        # Record equity
        self._record_equity(current_time)

    def _generate_signal(
        self,
        bar: MarketBar,
        breaker_state: Any,
    ) -> Optional[TradingSignal]:
        """Generate trading signal for current bar."""
        symbol = bar.symbol

        # Get prediction
        prediction = None
        if self.predictor and self.feature_provider:
            try:
                features = self.feature_provider(symbol, self.market_simulator.current_index)
                if features:
                    pred_output = self.predictor.predict(features=features, symbol=symbol)
                    prediction = EnsemblePrediction.from_predictor_output(pred_output)
            except Exception as e:
                logger.debug(f"Prediction error: {e}")
                prediction = None

        # If no predictor, use mock prediction based on price action
        if prediction is None:
            prediction = self._create_mock_prediction(bar)

        if prediction is None:
            return None

        # Get current position
        position = self.position_manager.get_position(symbol)
        current_position = None
        if position:
            from ..trading.signals.generator import Position as SignalPosition
            current_position = SignalPosition(
                symbol=position.symbol,
                side=position.side.value,
                quantity=position.quantity,
                entry_price=position.average_entry_price,
                current_price=position.current_price,
                unrealized_pnl=position.unrealized_pnl,
            )

        # Generate signal
        signal = self.signal_generator.generate_signal(
            prediction=prediction,
            symbol=symbol,
            current_price=bar.close,
            breaker_state=breaker_state,
            current_position=current_position,
            atr=bar.atr,
        )

        return signal

    def _create_mock_prediction(self, bar: MarketBar) -> Optional[EnsemblePrediction]:
        """Create mock prediction when no predictor available."""
        # Simple momentum-based mock prediction
        historical = self.market_simulator.get_historical_data(bar.symbol, n_bars=20)
        if len(historical) < 20:
            return None

        returns = historical["close"].pct_change().dropna()
        momentum = returns.mean()

        if momentum > 0.001:
            direction_prob = 0.65
        elif momentum < -0.001:
            direction_prob = 0.35
        else:
            direction_prob = 0.50

        confidence = 0.70  # Fixed confidence for mock

        return EnsemblePrediction(
            direction_probability=direction_prob,
            confidence=confidence,
        )

    def _execute_signal(self, signal: TradingSignal, bar: MarketBar) -> None:
        """Execute a trading signal."""
        symbol = bar.symbol

        # Calculate position size
        position_value = self.account_manager.equity * signal.position_size_pct
        quantity = position_value / bar.close

        # Create order
        side = OrderSide.BUY if signal.action == Action.BUY else OrderSide.SELL
        order = Order(
            order_id=f"order_{self._current_trade_id}",
            symbol=symbol,
            side=side,
            order_type=OrderType.MARKET,
            quantity=quantity,
            metadata={
                "signal_confidence": signal.confidence,
                "stop_loss": signal.stop_loss_price,
                "take_profit": signal.take_profit_price,
            }
        )

        # Submit order
        self.execution_engine.submit_order(order)
        self._current_trade_id += 1

        # Store entry info for position tracking
        self._open_positions[symbol] = {
            "entry_time": bar.timestamp,
            "entry_price": bar.close,
            "side": side.value,
            "quantity": quantity,
            "stop_loss": signal.stop_loss_price,
            "take_profit": signal.take_profit_price,
            "max_favorable": 0.0,
            "max_adverse": 0.0,
        }

    def _process_fill(self, fill: FillEvent, bar: MarketBar) -> None:
        """Process an order fill."""
        # Update position manager
        order = Order(
            order_id=fill.order_id,
            symbol=fill.symbol,
            side=fill.side,
            order_type=fill.order_type,
            quantity=fill.fill_quantity,
            status=OrderStatus.FILLED,
            filled_quantity=fill.fill_quantity,
            average_fill_price=fill.fill_price,
            commission=fill.commission,
        )

        self.position_manager.process_fill(order)

        # Update account margin
        exposure = self.position_manager.calculate_exposure()
        self.account_manager.update_margin(
            exposure["gross_exposure"] * self.account_manager.margin_requirement
        )

    def _update_positions(self, bar: MarketBar) -> None:
        """Update positions with current prices."""
        symbol = bar.symbol
        current_price = bar.close

        # Update position prices directly
        position = self.position_manager.get_position(symbol)
        if position is not None:
            position.update_price(current_price)

        # Calculate unrealized PnL
        pnl = self.position_manager.calculate_total_pnl()
        self.account_manager.update_unrealized_pnl(pnl["unrealized_pnl"])

        # Track max favorable/adverse excursion
        if symbol in self._open_positions:
            pos_info = self._open_positions[symbol]
            entry_price = pos_info["entry_price"]

            if pos_info["side"] == "buy":
                favorable = (bar.high - entry_price) / entry_price
                adverse = (entry_price - bar.low) / entry_price
            else:
                favorable = (entry_price - bar.low) / entry_price
                adverse = (bar.high - entry_price) / entry_price

            pos_info["max_favorable"] = max(pos_info["max_favorable"], favorable)
            pos_info["max_adverse"] = max(pos_info["max_adverse"], adverse)

            # Check stop loss and take profit
            self._check_exit_conditions(symbol, bar)

    def _check_exit_conditions(self, symbol: str, bar: MarketBar) -> None:
        """Check stop loss and take profit conditions."""
        if symbol not in self._open_positions:
            return

        pos_info = self._open_positions[symbol]
        current_price = bar.close

        should_exit = False
        exit_reason = ""

        if pos_info["side"] == "buy":
            if pos_info.get("stop_loss") and bar.low <= pos_info["stop_loss"]:
                should_exit = True
                exit_reason = "stop_loss"
            elif pos_info.get("take_profit") and bar.high >= pos_info["take_profit"]:
                should_exit = True
                exit_reason = "take_profit"
        else:
            if pos_info.get("stop_loss") and bar.high >= pos_info["stop_loss"]:
                should_exit = True
                exit_reason = "stop_loss"
            elif pos_info.get("take_profit") and bar.low <= pos_info["take_profit"]:
                should_exit = True
                exit_reason = "take_profit"

        if should_exit:
            self._close_position(symbol, bar, exit_reason)

    def _close_position(self, symbol: str, bar: MarketBar, reason: str) -> None:
        """Close position and record trade."""
        position = self.position_manager.get_position(symbol)
        if position is None:
            return

        # Create closing order
        side = OrderSide.SELL if position.side.value == "long" else OrderSide.BUY
        order = Order(
            order_id=f"close_{self._current_trade_id}",
            symbol=symbol,
            side=side,
            order_type=OrderType.MARKET,
            quantity=position.quantity,
        )

        self.execution_engine.submit_order(order)

        # Record trade
        if symbol in self._open_positions:
            pos_info = self._open_positions[symbol]
            pnl = position.unrealized_pnl
            pnl_pct = position.pnl_percentage / 100

            trade = Trade(
                trade_id=f"trade_{len(self._trades)}",
                symbol=symbol,
                side=pos_info["side"],
                entry_time=pos_info["entry_time"],
                exit_time=bar.timestamp,
                entry_price=pos_info["entry_price"],
                exit_price=bar.close,
                quantity=position.quantity,
                pnl=pnl,
                pnl_pct=pnl_pct,
                commission=position.total_commission,
                slippage=0.0,  # Calculated separately
                exit_reason=reason,
                max_favorable=pos_info["max_favorable"],
                max_adverse=pos_info["max_adverse"],
            )
            self._trades.append(trade)

            # Update account
            self.account_manager.record_trade_result(
                realized_pnl=pnl,
                commission=position.total_commission,
            )

            # Update circuit breakers
            trade_result = TradeResult(
                symbol=symbol,
                side=pos_info["side"],
                entry_price=pos_info["entry_price"],
                exit_price=bar.close,
                quantity=position.quantity,
                pnl=pnl,
                pnl_pct=pnl_pct,
                entry_time=pos_info["entry_time"],
                exit_time=bar.timestamp,
            )
            self.circuit_breaker_manager.record_trade(trade_result)

            del self._open_positions[symbol]

        # Close in position manager
        self.position_manager.close_position(symbol, bar.close)

    def _close_all_positions(self, snapshot: MarketSnapshot) -> None:
        """Close all open positions at end of backtest."""
        for symbol in list(self._open_positions.keys()):
            bar = snapshot.get_bar(symbol)
            if bar:
                self._close_position(symbol, bar, "backtest_end")

    def _record_equity(self, timestamp: datetime) -> None:
        """Record current equity."""
        self._equity_history.append({
            "timestamp": timestamp,
            "equity": self.account_manager.equity,
            "balance": self.account_manager.balance,
            "unrealized_pnl": self.account_manager._unrealized_pnl,
        })

    def _calculate_results(self, duration: float) -> BacktestResult:
        """Calculate comprehensive backtest results."""
        # Build equity series
        equity_df = pd.DataFrame(self._equity_history)
        if equity_df.empty:
            return self._empty_result(duration)

        equity_df.set_index("timestamp", inplace=True)
        equity_series = equity_df["equity"]

        # Calculate metrics
        initial = self.config.initial_capital
        final = equity_series.iloc[-1] if len(equity_series) > 0 else initial
        peak = equity_series.max()
        low = equity_series.min()

        total_return = final - initial
        total_return_pct = total_return / initial

        # Time calculations
        if len(equity_series) >= 2:
            days = (equity_series.index[-1] - equity_series.index[0]).days
            years = days / 365.25 if days > 0 else 1
        else:
            years = 1

        annualized_return = (1 + total_return_pct) ** (1 / years) - 1 if years > 0 else 0
        cagr = annualized_return

        # Returns series
        returns = equity_series.pct_change().dropna()

        # Risk metrics
        sharpe = self.metrics.sharpe_ratio(returns) if len(returns) > 1 else 0
        sortino = self.metrics.sortino_ratio(returns) if len(returns) > 1 else 0
        max_dd, dd_series = self.metrics.max_drawdown(equity_series)
        calmar = annualized_return / max_dd if max_dd > 0 else 0
        volatility = returns.std() * np.sqrt(252) if len(returns) > 1 else 0
        downside_vol = returns[returns < 0].std() * np.sqrt(252) if len(returns[returns < 0]) > 1 else 0

        # Max drawdown duration
        dd_duration = self._calculate_max_dd_duration(dd_series)

        # Trade statistics
        trade_stats = self._calculate_trade_stats()

        # Execution stats
        total_slippage = sum(abs(t.slippage) for t in self._trades)
        total_commission = sum(t.commission for t in self._trades)

        # Holding periods
        if self._trades:
            holding_periods = [t.holding_period for t in self._trades]
            avg_holding = sum(holding_periods, timedelta()) / len(holding_periods)
            max_holding = max(holding_periods)
            min_holding = min(holding_periods)
        else:
            avg_holding = max_holding = min_holding = timedelta()

        # Trades per month
        trades_per_month = len(self._trades) / (years * 12) if years > 0 else 0

        # Recovery factor
        recovery_factor = total_return / (max_dd * initial) if max_dd > 0 else 0

        return BacktestResult(
            config=self.config,
            status=self._status,
            start_time=equity_series.index[0] if len(equity_series) > 0 else datetime.now(),
            end_time=equity_series.index[-1] if len(equity_series) > 0 else datetime.now(),
            duration_seconds=duration,
            initial_capital=initial,
            final_equity=final,
            peak_equity=peak,
            low_equity=low,
            total_return=total_return,
            total_return_pct=total_return_pct,
            annualized_return=annualized_return,
            cagr=cagr,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            max_drawdown=max_dd,
            max_drawdown_duration=dd_duration,
            volatility=volatility,
            downside_volatility=downside_vol,
            **trade_stats,
            average_holding_period=avg_holding,
            max_holding_period=max_holding,
            min_holding_period=min_holding,
            trades_per_month=trades_per_month,
            recovery_factor=recovery_factor,
            risk_reward_ratio=trade_stats["average_win"] / abs(trade_stats["average_loss"]) if trade_stats["average_loss"] != 0 else 0,
            equity_curve=equity_series,
            drawdown_series=dd_series,
            returns_series=returns,
            trades=self._trades,
            signals=self._signals,
            total_slippage=total_slippage,
            total_commission=total_commission,
        )

    def _calculate_trade_stats(self) -> Dict[str, Any]:
        """Calculate trade statistics."""
        if not self._trades:
            return {
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "average_win": 0.0,
                "average_loss": 0.0,
                "largest_win": 0.0,
                "largest_loss": 0.0,
                "average_trade": 0.0,
                "expectancy": 0.0,
            }

        wins = [t for t in self._trades if t.pnl > 0]
        losses = [t for t in self._trades if t.pnl < 0]

        total_trades = len(self._trades)
        winning_trades = len(wins)
        losing_trades = len(losses)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0

        gross_profit = sum(t.pnl for t in wins)
        gross_loss = abs(sum(t.pnl for t in losses))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        average_win = np.mean([t.pnl for t in wins]) if wins else 0
        average_loss = np.mean([t.pnl for t in losses]) if losses else 0
        largest_win = max([t.pnl for t in wins]) if wins else 0
        largest_loss = min([t.pnl for t in losses]) if losses else 0
        average_trade = np.mean([t.pnl for t in self._trades])
        expectancy = (win_rate * average_win) + ((1 - win_rate) * average_loss)

        return {
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "average_win": average_win,
            "average_loss": average_loss,
            "largest_win": largest_win,
            "largest_loss": largest_loss,
            "average_trade": average_trade,
            "expectancy": expectancy,
        }

    def _calculate_max_dd_duration(self, dd_series: pd.Series) -> timedelta:
        """Calculate maximum drawdown duration."""
        if len(dd_series) < 2:
            return timedelta()

        # Find drawdown periods
        in_drawdown = dd_series < 0
        dd_starts = in_drawdown & ~in_drawdown.shift(1, fill_value=False)
        dd_ends = ~in_drawdown & in_drawdown.shift(1, fill_value=False)

        start_times = dd_series.index[dd_starts]
        end_times = dd_series.index[dd_ends]

        if len(start_times) == 0:
            return timedelta()

        # Calculate durations
        max_duration = timedelta()
        for i, start in enumerate(start_times):
            if i < len(end_times):
                duration = end_times[i] - start
                max_duration = max(max_duration, duration)

        return max_duration

    def _empty_result(self, duration: float) -> BacktestResult:
        """Return empty result when no trades."""
        return BacktestResult(
            config=self.config,
            status=self._status,
            start_time=datetime.now(),
            end_time=datetime.now(),
            duration_seconds=duration,
            initial_capital=self.config.initial_capital,
            final_equity=self.config.initial_capital,
            peak_equity=self.config.initial_capital,
            low_equity=self.config.initial_capital,
            total_return=0,
            total_return_pct=0,
            annualized_return=0,
            cagr=0,
            sharpe_ratio=0,
            sortino_ratio=0,
            calmar_ratio=0,
            max_drawdown=0,
            max_drawdown_duration=timedelta(),
            volatility=0,
            downside_volatility=0,
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            win_rate=0,
            profit_factor=0,
            average_win=0,
            average_loss=0,
            largest_win=0,
            largest_loss=0,
            average_trade=0,
            expectancy=0,
            average_holding_period=timedelta(),
            max_holding_period=timedelta(),
            min_holding_period=timedelta(),
            trades_per_month=0,
            recovery_factor=0,
            risk_reward_ratio=0,
        )

    def _save_results(self, result: BacktestResult) -> None:
        """Save results to output directory."""
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save summary
        summary_path = output_dir / "summary.json"
        with open(summary_path, "w") as f:
            f.write(result.to_json())

        # Save equity curve
        if self.config.save_equity_curve and len(result.equity_curve) > 0:
            equity_path = output_dir / "equity_curve.csv"
            result.equity_curve.to_csv(equity_path)

        # Save trades
        if self.config.save_trades and result.trades:
            trades_path = output_dir / "trades.json"
            with open(trades_path, "w") as f:
                json.dump([t.to_dict() for t in result.trades], f, indent=2)

        logger.info(f"Results saved to {output_dir}")

    def walk_forward(
        self,
        data: pd.DataFrame,
        features: pd.DataFrame,
        n_splits: int = 5,
        train_ratio: float = 0.7,
        retrain_callback: Optional[Callable] = None,
    ) -> List[BacktestResult]:
        """
        Perform walk-forward optimization.

        Args:
            data: Full OHLCV data
            features: Full feature data
            n_splits: Number of splits
            train_ratio: Ratio of training to total in each split
            retrain_callback: Function to retrain model on each split

        Returns:
            List of backtest results for each split
        """
        results = []
        total_size = len(data)
        split_size = total_size // n_splits

        for i in range(n_splits):
            # Calculate split boundaries
            split_end = (i + 1) * split_size
            train_end = int((i * split_size) + (split_size * train_ratio))
            test_start = train_end
            test_end = split_end

            # Extract test data
            test_data = data.iloc[test_start:test_end]
            test_features = features.iloc[test_start:test_end] if features is not None else None

            # Retrain model if callback provided
            if retrain_callback:
                train_data = data.iloc[i * split_size:train_end]
                train_features = features.iloc[i * split_size:train_end] if features is not None else None
                retrain_callback(train_data, train_features)

            # Run backtest on test period
            self.load_data(self.config.symbol, test_data, test_features)
            result = self.run()
            results.append(result)

            logger.info(f"Walk-forward split {i+1}/{n_splits}: "
                       f"Return={result.total_return_pct:.2%}, "
                       f"Sharpe={result.sharpe_ratio:.2f}")

        return results
