"""Verify Phase 4 implementation imports correctly.

This script just checks that all modules can be imported without errors.
"""

import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

print("Verifying Phase 4 implementation...")
print("="*60)

try:
    print("\n[1/4] Importing agent models...")
    from src.agent.models import CycleResult, PredictionData, SignalData
    print("✓ agent.models imports successfully")

    print("\n[2/4] Importing trading cycle...")
    from src.agent.trading_cycle import TradingCycle
    print("✓ agent.trading_cycle imports successfully")

    print("\n[3/4] Importing agent runner...")
    from src.agent.runner import AgentRunner, AgentStatus
    print("✓ agent.runner imports successfully")

    print("\n[4/4] Importing agent package...")
    from src.agent import (
        AgentConfig,
        AgentRunner,
        AgentStatus,
        TradingCycle,
        CycleResult,
        PredictionData,
        SignalData,
    )
    print("✓ agent package imports successfully")

    print("\n" + "="*60)
    print("Phase 4 Import Verification: PASSED")
    print("="*60)

    # Display key classes
    print("\nAvailable classes:")
    print(f"  - AgentConfig: {AgentConfig}")
    print(f"  - AgentRunner: {AgentRunner}")
    print(f"  - TradingCycle: {TradingCycle}")
    print(f"  - CycleResult: {CycleResult}")
    print(f"  - PredictionData: {PredictionData}")
    print(f"  - SignalData: {SignalData}")

    sys.exit(0)

except Exception as e:
    print(f"\n✗ Import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
