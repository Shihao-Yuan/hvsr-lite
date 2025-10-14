"""
hvsr-lite: Fast, reproducible HVSR processing for dense nodal arrays

A Python package for Horizontal-to-Vertical Spectral Ratio (HVSR) analysis
designed for large nodal arrays, with batch-oriented workflows, SESAME-inspired
defaults, and minimal dependencies.

Author: Shihao Yuan (syuan@mines.edu)

DISCLAIMER:
This is a development build. The code may contain errors or unstable functionality. 
Contributions and feedback are welcome.
"""

__version__ = "0.1.0"
__author__ = "Shihao Yuan"
__email__ = "syuan@mines.edu"

# Flat API 
from .core import compute_hvsr, HVSRResult, compute_hvsr_batch, compute_hvsr_array, compute_hvsr_parallel  # noqa: F401

# Notebook utilities
from .utils import stream_to_dict  # noqa: F401

# Back-compat soft-exports (kept only if present)
BatchProcessor = None  # type: ignore
NodalArray = None  # type: ignore

__all__ = [
    "compute_hvsr",
    "HVSRResult",
    "compute_hvsr_batch",
    "compute_hvsr_array", 
    "compute_hvsr_parallel",
    "stream_to_dict",
]
