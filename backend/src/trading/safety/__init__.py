"""
Safety Module.

Provides safety mechanisms for production trading.
"""

from .kill_switch import (
    KillSwitch,
    KillSwitchConfig,
    KillSwitchState,
    KillSwitchTrigger,
    TriggerType,
)

__all__ = [
    "KillSwitch",
    "KillSwitchConfig",
    "KillSwitchState",
    "KillSwitchTrigger",
    "TriggerType",
]
