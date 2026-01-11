"""
Risk Management Module.

Provides risk profiles and position sizing for the trading robot.
"""

from .profiles import (
    RiskProfile,
    RiskLevel,
    RISK_PROFILES,
    get_risk_profile,
    load_risk_profile,
    create_custom_profile,
    compare_profiles,
)

__all__ = [
    'RiskProfile',
    'RiskLevel',
    'RISK_PROFILES',
    'get_risk_profile',
    'load_risk_profile',
    'create_custom_profile',
    'compare_profiles',
]
