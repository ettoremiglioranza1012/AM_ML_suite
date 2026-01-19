"""
AM - Additive Manufacturing Topology Optimization Package

This package provides tools for topology optimization in additive manufacturing,
structured in three main submodules:

- **core**: Shared problem definitions (geometry, loads)
- **numerical**: Python-based FEM solver (SIMP method) - Ground Truth
- **ai**: Deep Learning models for fast inference (future)
"""

__version__ = "0.2.0"

from .core.geometry import VoxelDomain, create_bracket_domain
from .core.loads import LoadCase, create_brk_a_01_static_case_1
from .numerical.fem import MaterialProperties
from .numerical.topopt import SIMPOptimizer, SIMPParams, TOResult

__all__ = [
    # Core
    "VoxelDomain",
    "create_bracket_domain",
    "LoadCase",
    "create_brk_a_01_static_case_1",
    # Numerical
    "MaterialProperties",
    "SIMPOptimizer",
    "SIMPParams",
    "TOResult",
]
