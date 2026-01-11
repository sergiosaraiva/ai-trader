"""
Simulation Execution Engine.

Provides realistic order execution simulation with configurable models for:
- Slippage (fixed, volume-based, volatility-based)
- Latency (fixed, random)
- Commission (fixed, percentage, tiered)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
import random
import logging
import math

from ..orders.manager import Order, OrderType, OrderSide, OrderStatus

logger = logging.getLogger(__name__)


# =============================================================================
# Slippage Models
# =============================================================================

class SlippageModel(ABC):
    """Abstract base class for slippage models."""

    @abstractmethod
    def calculate_slippage(
        self,
        order: Order,
        market_price: float,
        market_data: Optional[Dict[str, Any]] = None,
    ) -> float:
        """
        Calculate slippage for an order.

        Args:
            order: The order being executed
            market_price: Current market price
            market_data: Optional additional market data (spread, volume, etc.)

        Returns:
            Slippage amount (positive = worse fill, negative = better fill)
        """
        pass


class FixedSlippageModel(SlippageModel):
    """
    Fixed slippage model.

    Applies a constant slippage percentage to all orders.
    """

    def __init__(self, slippage_pct: float = 0.0001):
        """
        Initialize fixed slippage model.

        Args:
            slippage_pct: Slippage as a percentage (0.0001 = 0.01% = 1 pip)
        """
        self.slippage_pct = slippage_pct

    def calculate_slippage(
        self,
        order: Order,
        market_price: float,
        market_data: Optional[Dict[str, Any]] = None,
    ) -> float:
        """Calculate fixed slippage."""
        slippage = market_price * self.slippage_pct

        # Always adverse to trader (buy higher, sell lower)
        if order.side == OrderSide.BUY:
            return slippage
        else:
            return -slippage


class VolumeBasedSlippageModel(SlippageModel):
    """
    Volume-based slippage model.

    Larger orders relative to market volume experience more slippage.
    """

    def __init__(
        self,
        base_slippage_pct: float = 0.0001,
        volume_impact_factor: float = 0.1,
        max_slippage_pct: float = 0.005,
    ):
        """
        Initialize volume-based slippage model.

        Args:
            base_slippage_pct: Minimum slippage percentage
            volume_impact_factor: How much order size affects slippage
            max_slippage_pct: Maximum slippage cap
        """
        self.base_slippage_pct = base_slippage_pct
        self.volume_impact_factor = volume_impact_factor
        self.max_slippage_pct = max_slippage_pct

    def calculate_slippage(
        self,
        order: Order,
        market_price: float,
        market_data: Optional[Dict[str, Any]] = None,
    ) -> float:
        """Calculate volume-based slippage."""
        market_data = market_data or {}
        avg_volume = market_data.get("avg_volume", 1000000)

        # Order size relative to market volume
        order_value = order.quantity * market_price
        volume_ratio = order_value / avg_volume if avg_volume > 0 else 0

        # Calculate slippage percentage
        slippage_pct = self.base_slippage_pct + (
            self.volume_impact_factor * math.sqrt(volume_ratio)
        )
        slippage_pct = min(slippage_pct, self.max_slippage_pct)

        slippage = market_price * slippage_pct

        if order.side == OrderSide.BUY:
            return slippage
        else:
            return -slippage


class VolatilitySlippageModel(SlippageModel):
    """
    Volatility-based slippage model.

    Higher volatility periods experience more slippage.
    """

    def __init__(
        self,
        base_slippage_pct: float = 0.0001,
        volatility_multiplier: float = 2.0,
        normal_atr_pct: float = 0.005,
    ):
        """
        Initialize volatility-based slippage model.

        Args:
            base_slippage_pct: Minimum slippage percentage
            volatility_multiplier: How much ATR affects slippage
            normal_atr_pct: Normal ATR percentage for baseline
        """
        self.base_slippage_pct = base_slippage_pct
        self.volatility_multiplier = volatility_multiplier
        self.normal_atr_pct = normal_atr_pct

    def calculate_slippage(
        self,
        order: Order,
        market_price: float,
        market_data: Optional[Dict[str, Any]] = None,
    ) -> float:
        """Calculate volatility-based slippage."""
        market_data = market_data or {}
        atr = market_data.get("atr", market_price * self.normal_atr_pct)

        # ATR as percentage of price
        atr_pct = atr / market_price if market_price > 0 else 0

        # Volatility ratio compared to normal
        volatility_ratio = atr_pct / self.normal_atr_pct if self.normal_atr_pct > 0 else 1

        # Calculate slippage
        slippage_pct = self.base_slippage_pct * (
            1 + self.volatility_multiplier * (volatility_ratio - 1)
        )
        slippage_pct = max(self.base_slippage_pct, slippage_pct)

        slippage = market_price * slippage_pct

        if order.side == OrderSide.BUY:
            return slippage
        else:
            return -slippage


# =============================================================================
# Latency Models
# =============================================================================

class LatencyModel(ABC):
    """Abstract base class for latency models."""

    @abstractmethod
    def get_latency(self) -> timedelta:
        """
        Get simulated latency for order execution.

        Returns:
            Latency as timedelta
        """
        pass


class FixedLatencyModel(LatencyModel):
    """Fixed latency model with constant delay."""

    def __init__(self, latency_ms: float = 50.0):
        """
        Initialize fixed latency model.

        Args:
            latency_ms: Latency in milliseconds
        """
        self.latency_ms = latency_ms

    def get_latency(self) -> timedelta:
        """Get fixed latency."""
        return timedelta(milliseconds=self.latency_ms)


class RandomLatencyModel(LatencyModel):
    """Random latency model with variable delay."""

    def __init__(
        self,
        min_latency_ms: float = 10.0,
        max_latency_ms: float = 100.0,
        spike_probability: float = 0.05,
        spike_multiplier: float = 5.0,
    ):
        """
        Initialize random latency model.

        Args:
            min_latency_ms: Minimum latency
            max_latency_ms: Maximum normal latency
            spike_probability: Probability of latency spike
            spike_multiplier: Multiplier for spike latency
        """
        self.min_latency_ms = min_latency_ms
        self.max_latency_ms = max_latency_ms
        self.spike_probability = spike_probability
        self.spike_multiplier = spike_multiplier

    def get_latency(self) -> timedelta:
        """Get random latency with occasional spikes."""
        base_latency = random.uniform(self.min_latency_ms, self.max_latency_ms)

        # Occasionally simulate latency spike
        if random.random() < self.spike_probability:
            base_latency *= self.spike_multiplier

        return timedelta(milliseconds=base_latency)


# =============================================================================
# Commission Models
# =============================================================================

class CommissionModel(ABC):
    """Abstract base class for commission models."""

    @abstractmethod
    def calculate_commission(
        self,
        order: Order,
        fill_price: float,
        fill_quantity: float,
    ) -> float:
        """
        Calculate commission for a fill.

        Args:
            order: The order being filled
            fill_price: Price at which fill occurred
            fill_quantity: Quantity filled

        Returns:
            Commission amount
        """
        pass


class FixedCommissionModel(CommissionModel):
    """Fixed commission per trade."""

    def __init__(self, commission_per_trade: float = 1.0):
        """
        Initialize fixed commission model.

        Args:
            commission_per_trade: Fixed commission per trade
        """
        self.commission_per_trade = commission_per_trade

    def calculate_commission(
        self,
        order: Order,
        fill_price: float,
        fill_quantity: float,
    ) -> float:
        """Calculate fixed commission."""
        return self.commission_per_trade


class PercentageCommissionModel(CommissionModel):
    """Percentage-based commission."""

    def __init__(
        self,
        commission_pct: float = 0.0001,
        min_commission: float = 0.0,
        max_commission: float = float("inf"),
    ):
        """
        Initialize percentage commission model.

        Args:
            commission_pct: Commission as percentage of trade value
            min_commission: Minimum commission
            max_commission: Maximum commission cap
        """
        self.commission_pct = commission_pct
        self.min_commission = min_commission
        self.max_commission = max_commission

    def calculate_commission(
        self,
        order: Order,
        fill_price: float,
        fill_quantity: float,
    ) -> float:
        """Calculate percentage-based commission."""
        trade_value = fill_price * fill_quantity
        commission = trade_value * self.commission_pct

        return max(self.min_commission, min(commission, self.max_commission))


class TieredCommissionModel(CommissionModel):
    """Tiered commission based on trade value."""

    def __init__(
        self,
        tiers: Optional[List[Tuple[float, float]]] = None,
    ):
        """
        Initialize tiered commission model.

        Args:
            tiers: List of (value_threshold, commission_pct) tuples
                   Example: [(0, 0.001), (10000, 0.0005), (100000, 0.0001)]
        """
        self.tiers = tiers or [
            (0, 0.001),      # 0.1% for values < 10k
            (10000, 0.0005), # 0.05% for values 10k-100k
            (100000, 0.0001), # 0.01% for values > 100k
        ]
        # Sort tiers by threshold descending
        self.tiers = sorted(self.tiers, key=lambda x: x[0], reverse=True)

    def calculate_commission(
        self,
        order: Order,
        fill_price: float,
        fill_quantity: float,
    ) -> float:
        """Calculate tiered commission."""
        trade_value = fill_price * fill_quantity

        # Find applicable tier
        for threshold, rate in self.tiers:
            if trade_value >= threshold:
                return trade_value * rate

        # Default to first tier (highest threshold)
        return trade_value * self.tiers[-1][1]


# =============================================================================
# Fill Simulator
# =============================================================================

@dataclass
class FillEvent:
    """Represents a fill event."""
    order_id: str
    symbol: str
    side: OrderSide
    fill_price: float
    fill_quantity: float
    commission: float
    timestamp: datetime
    latency: timedelta
    slippage: float
    order_type: OrderType

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "order_id": self.order_id,
            "symbol": self.symbol,
            "side": self.side.value,
            "fill_price": self.fill_price,
            "fill_quantity": self.fill_quantity,
            "commission": self.commission,
            "timestamp": self.timestamp.isoformat(),
            "latency_ms": self.latency.total_seconds() * 1000,
            "slippage": self.slippage,
            "order_type": self.order_type.value,
        }


class FillSimulator:
    """
    Simulates order fills with realistic behavior.

    Handles different order types and applies slippage, latency, and commission.
    """

    def __init__(
        self,
        slippage_model: Optional[SlippageModel] = None,
        latency_model: Optional[LatencyModel] = None,
        commission_model: Optional[CommissionModel] = None,
        partial_fill_probability: float = 0.0,
    ):
        """
        Initialize fill simulator.

        Args:
            slippage_model: Model for calculating slippage
            latency_model: Model for simulating latency
            commission_model: Model for calculating commission
            partial_fill_probability: Probability of partial fill (0-1)
        """
        self.slippage_model = slippage_model or FixedSlippageModel()
        self.latency_model = latency_model or FixedLatencyModel()
        self.commission_model = commission_model or PercentageCommissionModel()
        self.partial_fill_probability = partial_fill_probability

    def simulate_fill(
        self,
        order: Order,
        current_price: float,
        current_time: datetime,
        market_data: Optional[Dict[str, Any]] = None,
    ) -> Optional[FillEvent]:
        """
        Simulate order fill.

        Args:
            order: Order to fill
            current_price: Current market price
            current_time: Current simulation time
            market_data: Additional market data (high, low, volume, atr)

        Returns:
            FillEvent if filled, None otherwise
        """
        market_data = market_data or {}

        # Check if order should fill based on type
        should_fill, base_price = self._check_fill_conditions(
            order, current_price, market_data
        )

        if not should_fill:
            return None

        # Calculate slippage
        slippage = self.slippage_model.calculate_slippage(
            order, base_price, market_data
        )
        fill_price = base_price + slippage

        # Determine fill quantity (possibly partial)
        if random.random() < self.partial_fill_probability:
            fill_quantity = order.remaining_quantity * random.uniform(0.3, 0.9)
        else:
            fill_quantity = order.remaining_quantity

        # Calculate commission
        commission = self.commission_model.calculate_commission(
            order, fill_price, fill_quantity
        )

        # Get latency
        latency = self.latency_model.get_latency()
        fill_time = current_time + latency

        return FillEvent(
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side,
            fill_price=fill_price,
            fill_quantity=fill_quantity,
            commission=commission,
            timestamp=fill_time,
            latency=latency,
            slippage=slippage,
            order_type=order.order_type,
        )

    def _check_fill_conditions(
        self,
        order: Order,
        current_price: float,
        market_data: Dict[str, Any],
    ) -> Tuple[bool, float]:
        """
        Check if order should fill and determine base fill price.

        Args:
            order: Order to check
            current_price: Current market price
            market_data: Market data with high/low

        Returns:
            Tuple of (should_fill, base_price)
        """
        high = market_data.get("high", current_price * 1.01)
        low = market_data.get("low", current_price * 0.99)

        if order.order_type == OrderType.MARKET:
            # Market orders always fill at current price
            return True, current_price

        elif order.order_type == OrderType.LIMIT:
            if order.side == OrderSide.BUY:
                # Buy limit fills if price goes at or below limit
                if low <= order.limit_price:
                    return True, min(order.limit_price, current_price)
            else:
                # Sell limit fills if price goes at or above limit
                if high >= order.limit_price:
                    return True, max(order.limit_price, current_price)

        elif order.order_type == OrderType.STOP:
            if order.side == OrderSide.BUY:
                # Buy stop triggers if price goes at or above stop
                if high >= order.stop_price:
                    return True, max(order.stop_price, current_price)
            else:
                # Sell stop triggers if price goes at or below stop
                if low <= order.stop_price:
                    return True, min(order.stop_price, current_price)

        elif order.order_type == OrderType.STOP_LIMIT:
            if order.side == OrderSide.BUY:
                # Buy stop-limit: triggers at stop, fills at limit
                if high >= order.stop_price:
                    if low <= order.limit_price:
                        return True, min(order.limit_price, current_price)
            else:
                # Sell stop-limit: triggers at stop, fills at limit
                if low <= order.stop_price:
                    if high >= order.limit_price:
                        return True, max(order.limit_price, current_price)

        return False, 0.0


# =============================================================================
# Simulation Execution Engine
# =============================================================================

@dataclass
class SimulationConfig:
    """Configuration for simulation execution."""
    initial_capital: float = 100000.0
    slippage_model: Optional[SlippageModel] = None
    latency_model: Optional[LatencyModel] = None
    commission_model: Optional[CommissionModel] = None
    partial_fill_probability: float = 0.0
    use_bid_ask_spread: bool = True
    default_spread_pips: float = 1.0

    def __post_init__(self):
        """Set default models if not provided."""
        if self.slippage_model is None:
            self.slippage_model = FixedSlippageModel(slippage_pct=0.0001)
        if self.latency_model is None:
            self.latency_model = FixedLatencyModel(latency_ms=50.0)
        if self.commission_model is None:
            self.commission_model = PercentageCommissionModel(commission_pct=0.0001)


class SimulationExecutionEngine:
    """
    Simulation execution engine for backtesting and paper trading.

    Features:
    - Realistic order fill simulation
    - Configurable slippage, latency, commission models
    - Support for all order types
    - Bid/ask spread simulation
    - Partial fill support
    """

    def __init__(self, config: Optional[SimulationConfig] = None):
        """
        Initialize simulation execution engine.

        Args:
            config: Simulation configuration
        """
        self.config = config or SimulationConfig()

        self.fill_simulator = FillSimulator(
            slippage_model=self.config.slippage_model,
            latency_model=self.config.latency_model,
            commission_model=self.config.commission_model,
            partial_fill_probability=self.config.partial_fill_probability,
        )

        # Order tracking
        self._pending_orders: Dict[str, Order] = {}
        self._fill_history: List[FillEvent] = []

        # Current market state
        self._current_prices: Dict[str, float] = {}
        self._current_time: datetime = datetime.now()

    def update_market_data(
        self,
        symbol: str,
        price: float,
        timestamp: datetime,
        market_data: Optional[Dict[str, Any]] = None,
    ) -> List[FillEvent]:
        """
        Update market data and check for order fills.

        Args:
            symbol: Trading symbol
            price: Current price
            timestamp: Current time
            market_data: Additional data (high, low, volume, atr)

        Returns:
            List of fill events
        """
        self._current_prices[symbol] = price
        self._current_time = timestamp

        fills = []

        # Check pending orders for this symbol
        orders_to_check = [
            order for order in self._pending_orders.values()
            if order.symbol == symbol and order.is_open
        ]

        for order in orders_to_check:
            fill = self.fill_simulator.simulate_fill(
                order=order,
                current_price=price,
                current_time=timestamp,
                market_data=market_data,
            )

            if fill:
                self._process_fill(order, fill)
                fills.append(fill)

        return fills

    def submit_order(self, order: Order) -> bool:
        """
        Submit order to simulation.

        Args:
            order: Order to submit

        Returns:
            True if submitted successfully
        """
        if order.order_id in self._pending_orders:
            logger.warning(f"Order {order.order_id} already submitted")
            return False

        order.status = OrderStatus.SUBMITTED
        order.submitted_at = self._current_time

        # Market orders fill immediately at next price update
        if order.order_type == OrderType.MARKET:
            order.status = OrderStatus.ACCEPTED

        self._pending_orders[order.order_id] = order
        logger.debug(f"Order {order.order_id} submitted: {order.side.value} {order.quantity} {order.symbol}")

        return True

    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an open order.

        Args:
            order_id: ID of order to cancel

        Returns:
            True if cancelled successfully
        """
        if order_id not in self._pending_orders:
            return False

        order = self._pending_orders[order_id]
        if not order.is_open:
            return False

        order.status = OrderStatus.CANCELLED
        order.cancelled_at = self._current_time

        del self._pending_orders[order_id]
        logger.debug(f"Order {order_id} cancelled")

        return True

    def _process_fill(self, order: Order, fill: FillEvent) -> None:
        """Process a fill event."""
        order.filled_quantity += fill.fill_quantity
        order.commission += fill.commission

        # Update average fill price
        if order.filled_quantity > 0:
            prev_value = order.average_fill_price * (order.filled_quantity - fill.fill_quantity)
            fill_value = fill.fill_price * fill.fill_quantity
            order.average_fill_price = (prev_value + fill_value) / order.filled_quantity

        # Update status
        if order.filled_quantity >= order.quantity:
            order.status = OrderStatus.FILLED
            order.filled_at = fill.timestamp
            del self._pending_orders[order.order_id]
        else:
            order.status = OrderStatus.PARTIALLY_FILLED

        self._fill_history.append(fill)
        logger.debug(
            f"Order {order.order_id} filled: {fill.fill_quantity} @ {fill.fill_price:.5f} "
            f"(slippage: {fill.slippage:.5f}, commission: {fill.commission:.2f})"
        )

    def get_pending_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """Get pending orders, optionally filtered by symbol."""
        if symbol:
            return [o for o in self._pending_orders.values() if o.symbol == symbol]
        return list(self._pending_orders.values())

    def get_fill_history(self, symbol: Optional[str] = None) -> List[FillEvent]:
        """Get fill history, optionally filtered by symbol."""
        if symbol:
            return [f for f in self._fill_history if f.symbol == symbol]
        return list(self._fill_history)

    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price for symbol."""
        return self._current_prices.get(symbol)

    def reset(self) -> None:
        """Reset simulation state."""
        self._pending_orders.clear()
        self._fill_history.clear()
        self._current_prices.clear()
        self._current_time = datetime.now()

    def get_stats(self) -> Dict[str, Any]:
        """Get execution statistics."""
        total_fills = len(self._fill_history)
        total_slippage = sum(abs(f.slippage) for f in self._fill_history)
        total_commission = sum(f.commission for f in self._fill_history)
        avg_latency_ms = (
            sum(f.latency.total_seconds() * 1000 for f in self._fill_history) / total_fills
            if total_fills > 0 else 0
        )

        return {
            "total_fills": total_fills,
            "total_slippage": total_slippage,
            "total_commission": total_commission,
            "average_latency_ms": avg_latency_ms,
            "pending_orders": len(self._pending_orders),
            "symbols_tracked": list(self._current_prices.keys()),
        }
