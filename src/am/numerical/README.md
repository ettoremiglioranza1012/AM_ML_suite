# AM - Additive Manufacturing Topology Optimization

## 🎯 Project Objective

End-to-end prototype of **Topology Optimization** for aeronautical components in Metal AM.  
First milestone: `parametric inputs → TO → 3D density field` for the pilot case BRK-A-01.

---

## 🔧 Technical Stack (v1.0)

| Component | Choice | Rationale |
|-----------|--------|-----------|
| **Language** | Python 3.11+ | Mature scientific ecosystem (NumPy, SciPy) |
| **Representation** | 3D Voxel | Regular grid, 1 mm step |
| **TO Solver** | Custom SIMP 3D | Simplified in-house implementation, extensible |
| **Linear Algebra** | SciPy sparse | Sparse matrices for FEM on large grids |
| **Visualization** | PyVista / Matplotlib | 3D density field rendering |

### Core Dependencies
```
numpy>=1.24
scipy>=1.11
pyvista>=0.42
```

---

## 📋 TECHNICAL SPECIFICATION: PILOT CASE - AERO-BRACKET V1

### 1. Part Family and Process

| Parameter | Value |
|-----------|-------|
| **Object** | Actuator support bracket (Pylon/Engine Bracket) |
| **Material** | Titanium Alloy **Ti6Al4V** |
| **Process** | Metal AM - **L-PBF** (Laser Powder Bed Fusion) |

#### Ti6Al4V Material Properties
| Property | Value | Unit |
|----------|-------|------|
| Young's Modulus (E) | 113.8 | GPa |
| Poisson's Ratio (ν) | 0.342 | - |
| Density (ρ) | 4430 | kg/m³ |
| Yield Strength (σ_y) | 880 | MPa |

---

### 2. Problem Domain (Geometry and Mesh)

#### Bounding Box
- **Dimensions:** 120 × 60 × 80 mm

#### Fixed Interfaces (Non-Design Space)

| Interface | Description |
|-----------|-------------|
| **Base** | 4 through-holes (Ø 8mm) with 5mm thick flange (pylon mounting constraint) |
| **Upper Eyelet** | 1 hole (Ø 12mm) for actuator pin, 4mm solid material offset around radius |

#### Computational Resolution
- **Voxel Grid:** 120 × 60 × 80 mm with 1 mm step
- **Total Elements:** 576,000 voxels

---

### 3. Loads and Objectives (Physics)

#### Boundary Conditions (Load Cases)

| Case | Description | Value |
|------|-------------|-------|
| **Static 1** | Vertical load (Z) on eyelet | 15,000 N |
| **Static 2** | Inclined load at 30° (X-Z component) | 10,000 N |

#### Optimization Objectives

| Parameter | Value |
|-----------|-------|
| **Primary Objective** | Compliance minimization (maximum stiffness) |
| **Mass Constraint** | Volume fraction ≤ 0.25 (remove 75% of material) |
| **Safety Factor** | SF ≥ 1.5 relative to σ_y = 880 MPa |

---

### 4. Manufacturing Constraints (DfAM)

| Constraint | Value | Notes |
|------------|-------|-------|
| **Overhang Angle** | ≥ 45° | Relative to XY build plane |
| **Minimum Member Size** | 2 mm | Structural stability + laser fusion quality |
| **Symmetry** | Y-Z Plane | Longitudinal symmetry constraint |

---

### 5. Validation Dataset

| ID | Description | Status |
|----|-------------|--------|
| **BRK-A-01** | Purely vertical load (Baseline) | 🎯 Target v1 |
| **BRK-A-02** | Eyelet position variation (+10mm in X) | Planned |
| **BRK-A-03** | Combined Tension + Torsion load | Planned |

---

## 💻 Data Structures (Developer Reference)

### Domain Tensor (3D)
```python
# 3D Matrix: (120, 60, 80) with 1mm step
# Values:
#   1  = Fixed (Non-Design Space)
#   0  = Optimizable (Design Space)
#  -1  = Void (outside domain)

domain: np.ndarray  # shape: (nx, ny, nz), dtype: int8
```

### Boundary Conditions Schema
```python
@dataclass
class BoundaryCondition:
    """Boundary condition for FEM."""
    node_coords: np.ndarray   # (N, 3) constrained node coordinates
    dof_mask: np.ndarray      # (N, 3) bool - which DOFs are locked
    
@dataclass  
class LoadCase:
    """Load case."""
    name: str
    node_coords: np.ndarray   # (M, 3) application points
    force_vectors: np.ndarray # (M, 3) force vectors [Fx, Fy, Fz] in N
```

### Optimization Parameters
```python
@dataclass
class SIMPParams:
    """SIMP Topology Optimization parameters."""
    volume_fraction: float = 0.25    # Target volume
    penalty: float = 3.0             # SIMP penalization (p)
    filter_radius: float = 2.0       # Density filter radius [mm]
    move_limit: float = 0.2          # Update limit per iteration
    max_iterations: int = 100
    convergence_tol: float = 0.01
```

### Output: Density Field
```python
@dataclass
class TOResult:
    """Topology Optimization result."""
    density: np.ndarray       # (nx, ny, nz), values in [0, 1]
    compliance: float         # Final compliance
    volume_fraction: float    # Effective volume fraction
    iterations: int           # Completed iterations
    converged: bool
```

---

## 🧮 SIMP 3D Algorithm (Simplified)

### Mathematical Formulation

**Objective:**
$$\min_{\rho} \quad C(\rho) = \mathbf{U}^T \mathbf{K}(\rho) \mathbf{U}$$

**Subject to:**
$$\frac{V(\rho)}{V_0} \leq f \quad \text{(volume constraint)}$$
$$0 < \rho_{min} \leq \rho_e \leq 1 \quad \text{(density bounds)}$$

**SIMP Interpolation:**
$$E_e(\rho_e) = E_{min} + \rho_e^p (E_0 - E_{min})$$

where:
- $\rho_e$ = element density
- $p$ = penalty (typically 3)
- $E_0$ = Young's modulus of solid material

### Sensitivity Analysis
$$\frac{\partial C}{\partial \rho_e} = -p \rho_e^{p-1} \mathbf{u}_e^T \mathbf{k}_0 \mathbf{u}_e$$

---

## 📁 Repository Structure

```
AM/
├── README.md                    # This file
├── pyproject.toml               # Project configuration
├── main.py                      # Entry point
│
├── data/
│   └── brk_a_01/                # Pilot case data
│       ├── .gitkeep             # Placeholder
│       ├── density_field.npy    # Output: density field (generated)
│       └── metadata.json        # Run metadata (generated)
│
├── src/
│   ├── __init__.py              # Package init
│   ├── geometry.py              # Voxel grid + design/non-design marking
│   ├── loads.py                 # Load cases and boundary conditions
│   ├── fem.py                   # K matrix assembly, FEM solver
│   └── topopt.py                # SIMP loop: density update, compliance
│
└── notebooks/
    └── 01_brk_a_01_topopt.ipynb # Interactive pilot case test
```

### Module Descriptions

| Module | Responsibility |
|--------|----------------|
| `geometry.py` | Generates 3D voxel grid, marks design (0) vs non-design (1) vs void (-1) zones. Defines base, eyelet, bounding box. |
| `loads.py` | Defines LoadCase with boundary conditions and forces. Currently only Static Case 1 (15 kN vertical). |
| `fem.py` | Stiffness matrix K assembly for isotropic voxel grid (8-node hexahedron). Sparse solver. |
| `topopt.py` | Complete SIMP loop: density update with OC, sensitivity filter, projection to 25% volume fraction. |

---

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install numpy scipy matplotlib

# 2. Test modules
python -c "from src.geometry import create_bracket_domain; print(create_bracket_domain())"

# 3. Run interactive notebook
jupyter notebook notebooks/01_brk_a_01_topopt.ipynb
```

### Code Example

```python
from src.geometry import create_bracket_domain
from src.loads import create_brk_a_01_static_case_1
from src.fem import MaterialProperties
from src.topopt import SIMPOptimizer, SIMPParams

# 1. Define domain
domain = create_bracket_domain(
    size_mm=(120, 60, 80),
    resolution_mm=5.0  # Use 1.0 for production
)

# 2. Load case
load_case = create_brk_a_01_static_case_1(domain.shape)

# 3. Configure and run TO
optimizer = SIMPOptimizer(
    domain=domain,
    load_cases=[load_case],
    material=MaterialProperties(),
    params=SIMPParams(volume_fraction=0.25)
)
result = optimizer.run()

# 4. Result: 3D density field
print(f"Density shape: {result.density.shape}")
print(f"Final compliance: {result.final_compliance:.4e}")
```

---

## 📊 Milestone v1.0

- [ ] Voxel domain definition with NDS
- [ ] 3D stiffness matrix assembly (linear hexahedron)
- [ ] Sparse FEM solver (SciPy)
- [ ] Basic SIMP loop
- [ ] Density filter
- [ ] Density field visualization
- [ ] BRK-A-01 test case

---

## 📚 References

1. Bendsøe, M.P., Sigmund, O. (2003). *Topology Optimization: Theory, Methods and Applications*
2. Andreassen, E. et al. (2011). *Efficient topology optimization in MATLAB using 88 lines of code*
3. Liu, K., Tovar, A. (2014). *An efficient 3D topology optimization code written in Matlab*

---

*Last updated: January 17, 2026*
