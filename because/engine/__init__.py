"""
Engine modules for because-py: JAX environment setup, MCMC runners, and ensemble samplers.
"""

from .env import _detect_available_cores, _init_jax, get_jax_primitives
from .runner import run_mcmc
from .ensemble import run_ensemble_chains, wrap_ensemble_model

__all__ = [
    "_detect_available_cores",
    "_init_jax",
    "get_jax_primitives",
    "run_mcmc",
    "run_ensemble_chains",
    "wrap_ensemble_model",
]
