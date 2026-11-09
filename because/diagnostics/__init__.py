"""
Diagnostics module for because-py: WAIC and model comparison tools.
"""

from .waic import (
    calculate_waic,
    calculate_waic_internal,
    calculate_waic_ensemble,
    accumulate_waic_from_log_lik,
)

__all__ = [
    "calculate_waic",
    "calculate_waic_internal",
    "calculate_waic_ensemble",
    "accumulate_waic_from_log_lik",
]
