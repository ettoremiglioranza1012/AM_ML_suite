"""
AM Core - Shared Problem Definitions

Contains geometry and load case definitions used by:
- Numerical solver (Python FEM)
- AI inference module
- C++ HPC engine (via serialization)
"""

from .geometry import VoxelDomain, create_bracket_domain
from .loads import (
    DOF,
    BoundaryCondition,
    PointLoad,
    LoadCase,
    create_brk_a_01_static_case_1,
)

__all__ = [
    "VoxelDomain",
    "create_bracket_domain",
    "DOF",
    "BoundaryCondition",
    "PointLoad",
    "LoadCase",
    "create_brk_a_01_static_case_1",
]
