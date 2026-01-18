"""
Execution module for order execution handling.

Provides execution modes for simulation, paper trading, and production.
"""

from .simulation import (
    SlippageModel,
    FixedSlippageModel,
    VolumeBasedSlippageModel,
    VolatilitySlippageModel,
    LatencyModel,
    FixedLatencyModel,
    RandomLatencyModel,
    CommissionModel,
    FixedCommissionModel,
    PercentageCommissionModel,
    TieredCommissionModel,
    SimulationExecutionEngine,
    FillSimulator,
)

from .production import (
    ProductionExecutionEngine,
    ProductionConfig,
    OrderValidator,
    OrderValidationConfig,
    OrderValidationResult,
    ValidationError,
    ReconciliationError,
    BrokerExecutor,
)

__all__ = [
    # Slippage models
    "SlippageModel",
    "FixedSlippageModel",
    "VolumeBasedSlippageModel",
    "VolatilitySlippageModel",
    # Latency models
    "LatencyModel",
    "FixedLatencyModel",
    "RandomLatencyModel",
    # Commission models
    "CommissionModel",
    "FixedCommissionModel",
    "PercentageCommissionModel",
    "TieredCommissionModel",
    # Simulation execution
    "SimulationExecutionEngine",
    "FillSimulator",
    # Production execution
    "ProductionExecutionEngine",
    "ProductionConfig",
    "OrderValidator",
    "OrderValidationConfig",
    "OrderValidationResult",
    "ValidationError",
    "ReconciliationError",
    "BrokerExecutor",
]
