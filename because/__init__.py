"""
because-py: A Python backend for the because multiscale causal engine.
"""

from .api import fit, get_dsep_equations, summary

__all__ = ["fit", "get_dsep_equations", "summary"]
