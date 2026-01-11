"""
Trading Signals Module.

Provides signal generation from ML model predictions.
"""

from .actions import Action, TradingSignal, SignalStrength, get_signal_strength
from .generator import SignalGenerator, EnsemblePrediction, Position

__all__ = [
    'Action',
    'TradingSignal',
    'SignalStrength',
    'get_signal_strength',
    'SignalGenerator',
    'EnsemblePrediction',
    'Position',
]
