"""Feature selection module for MTF Ensemble.

This module provides recursive feature elimination with cross-validation (RFECV)
for selecting optimal features per timeframe model.
"""

from .rfecv_config import RFECVConfig
from .rfecv_selector import RFECVSelector
from .manager import FeatureSelectionManager

__all__ = [
    "RFECVConfig",
    "RFECVSelector",
    "FeatureSelectionManager",
]
