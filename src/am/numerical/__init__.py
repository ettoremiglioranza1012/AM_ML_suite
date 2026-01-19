"""
AM Numerical - Python FEM Solver (Ground Truth)

This module contains the original Python-based topology optimization solver
using the SIMP (Solid Isotropic Material with Penalization) method.

Used as:
- Reference implementation for validation
- Ground truth for AI model training
- Prototyping and debugging
"""

from .fem import (
    MaterialProperties,
    get_element_stiffness_matrix,
    get_element_dof_indices,
    assemble_global_stiffness,
    solve_fem,
    compute_element_compliance,
)
from .topopt import (
    SIMPParams,
    TOResult,
    SIMPOptimizer,
    threshold_density,
)

__all__ = [
    # FEM
    "MaterialProperties",
    "get_element_stiffness_matrix",
    "get_element_dof_indices",
    "assemble_global_stiffness",
    "solve_fem",
    "compute_element_compliance",
    # TopOpt
    "SIMPParams",
    "TOResult",
    "SIMPOptimizer",
    "threshold_density",
]
