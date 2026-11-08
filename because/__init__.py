"""
because-py: A Python backend for the because multiscale causal engine.
"""

__version__ = "0.2.11"

from .api import fit, get_dsep_equations, summary, dsep_summary

__all__ = ["__version__", "fit", "get_dsep_equations", "summary", "dsep_summary"]
