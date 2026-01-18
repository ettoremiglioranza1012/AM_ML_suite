"""
fem.py - Assemblaggio matrice di rigidezza per FEM 3D.

Questo modulo gestisce:
- Matrice di rigidezza elemento (esaedro 8 nodi)
- Assemblaggio matrice globale K (sparse)
- Soluzione sistema lineare K*u = F
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve, cg
from dataclasses import dataclass
from typing import Tuple, Optional
import warnings


@dataclass
class MaterialProperties:
    """
    Proprietà materiale per FEM elastico lineare.
    
    Attributes:
        E: Modulo di Young [Pa]
        nu: Coefficiente di Poisson [-]
        rho: Densità [kg/m³]
    """
    E: float = 113.8e9      # Ti6Al4V: 113.8 GPa
    nu: float = 0.342       # Ti6Al4V
    rho: float = 4430.0     # Ti6Al4V: 4430 kg/m³
    
    @property
    def lame_lambda(self) -> float:
        """Primo parametro di Lamé."""
        return self.E * self.nu / ((1 + self.nu) * (1 - 2 * self.nu))
    
    @property
    def lame_mu(self) -> float:
        """Secondo parametro di Lamé (modulo di taglio)."""
        return self.E / (2 * (1 + self.nu))


def get_element_stiffness_matrix(
    element_size: float,
    material: MaterialProperties
) -> np.ndarray:
    """
    Calcola la matrice di rigidezza per un elemento esaedrico a 8 nodi.
    
    Usa integrazione analitica per elemento cubico isotropo.
    Riferimento: Cook et al., "Concepts and Applications of FEA"
    
    Args:
        element_size: Dimensione lato elemento (cubo) [m]
        material: Proprietà del materiale
        
    Returns:
        Matrice 24x24 (8 nodi x 3 DOF)
    """
    E = material.E
    nu = material.nu
    a = element_size / 2  # Semi-lato
    
    # Matrice costitutiva elastica 3D (Voigt notation)
    C = E / ((1 + nu) * (1 - 2 * nu)) * np.array([
        [1 - nu, nu, nu, 0, 0, 0],
        [nu, 1 - nu, nu, 0, 0, 0],
        [nu, nu, 1 - nu, 0, 0, 0],
        [0, 0, 0, (1 - 2 * nu) / 2, 0, 0],
        [0, 0, 0, 0, (1 - 2 * nu) / 2, 0],
        [0, 0, 0, 0, 0, (1 - 2 * nu) / 2]
    ])
    
    # Coordinate nodali locali (elemento riferimento [-1, 1]³)
    nodes_local = np.array([
        [-1, -1, -1],
        [+1, -1, -1],
        [+1, +1, -1],
        [-1, +1, -1],
        [-1, -1, +1],
        [+1, -1, +1],
        [+1, +1, +1],
        [-1, +1, +1]
    ], dtype=float)
    
    # Punti di Gauss (2x2x2)
    gp = 1.0 / np.sqrt(3)
    gauss_points = np.array([
        [-gp, -gp, -gp],
        [+gp, -gp, -gp],
        [+gp, +gp, -gp],
        [-gp, +gp, -gp],
        [-gp, -gp, +gp],
        [+gp, -gp, +gp],
        [+gp, +gp, +gp],
        [-gp, +gp, +gp]
    ])
    
    Ke = np.zeros((24, 24))
    
    for gp_coord in gauss_points:
        xi, eta, zeta = gp_coord
        
        # Derivate funzioni di forma rispetto a xi, eta, zeta
        dN_dxi = np.zeros((8, 3))
        for i, (xi_i, eta_i, zeta_i) in enumerate(nodes_local):
            dN_dxi[i, 0] = 0.125 * xi_i * (1 + eta_i * eta) * (1 + zeta_i * zeta)
            dN_dxi[i, 1] = 0.125 * eta_i * (1 + xi_i * xi) * (1 + zeta_i * zeta)
            dN_dxi[i, 2] = 0.125 * zeta_i * (1 + xi_i * xi) * (1 + eta_i * eta)
        
        # Jacobiano (per cubo regolare è costante)
        J = a * np.eye(3)
        detJ = a ** 3
        invJ = np.linalg.inv(J)
        
        # Derivate rispetto a x, y, z
        dN_dx = dN_dxi @ invJ.T
        
        # Matrice B (strain-displacement)
        B = np.zeros((6, 24))
        for i in range(8):
            B[0, 3*i] = dN_dx[i, 0]      # epsilon_xx
            B[1, 3*i+1] = dN_dx[i, 1]    # epsilon_yy
            B[2, 3*i+2] = dN_dx[i, 2]    # epsilon_zz
            B[3, 3*i] = dN_dx[i, 1]      # gamma_xy
            B[3, 3*i+1] = dN_dx[i, 0]
            B[4, 3*i+1] = dN_dx[i, 2]    # gamma_yz
            B[4, 3*i+2] = dN_dx[i, 1]
            B[5, 3*i] = dN_dx[i, 2]      # gamma_xz
            B[5, 3*i+2] = dN_dx[i, 0]
        
        # Contributo al punto di Gauss
        Ke += B.T @ C @ B * detJ
    
    return Ke


def get_element_dof_indices(
    element_ijk: Tuple[int, int, int],
    grid_shape: Tuple[int, int, int]
) -> np.ndarray:
    """
    Calcola gli indici DOF globali per un elemento.
    
    Numerazione nodi: (nx+1) x (ny+1) x (nz+1)
    Ogni nodo ha 3 DOF: ux, uy, uz
    
    Args:
        element_ijk: Indici (i, j, k) dell'elemento nella griglia
        grid_shape: Dimensioni griglia elementi (nx, ny, nz)
        
    Returns:
        Array di 24 indici DOF
    """
    i, j, k = element_ijk
    nx, ny, nz = grid_shape
    
    # Nodi dell'elemento (8 vertici)
    # Ordine: seguendo convenzione esaedro standard
    node_offsets = [
        (0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
        (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)
    ]
    
    dof_indices = []
    for di, dj, dk in node_offsets:
        # Indice nodo globale
        node_idx = (i + di) * (ny + 1) * (nz + 1) + (j + dj) * (nz + 1) + (k + dk)
        # 3 DOF per nodo
        dof_indices.extend([3 * node_idx, 3 * node_idx + 1, 3 * node_idx + 2])
    
    return np.array(dof_indices, dtype=np.int64)


def assemble_global_stiffness(
    domain_grid: np.ndarray,
    density: np.ndarray,
    element_size: float,
    material: MaterialProperties,
    penalty: float = 3.0,
    E_min: float = 1e-9
) -> sparse.csr_matrix:
    """
    Assembla la matrice di rigidezza globale (sparse).
    
    Usa interpolazione SIMP: E(rho) = E_min + rho^p * (E0 - E_min)
    
    Args:
        domain_grid: Griglia 3D con marking (-1=void, 0=design, 1=fixed)
        density: Campo di densità (0 a 1) per elementi design
        element_size: Dimensione elemento [m]
        material: Proprietà materiale
        penalty: Esponente SIMP (p)
        E_min: Modulo minimo per evitare singolarità
        
    Returns:
        Matrice K globale in formato CSR
    """
    nx, ny, nz = domain_grid.shape
    n_nodes = (nx + 1) * (ny + 1) * (nz + 1)
    n_dofs = 3 * n_nodes
    
    # Matrice elemento base (materiale pieno)
    Ke0 = get_element_stiffness_matrix(element_size, material)
    
    # Liste per costruzione COO
    rows = []
    cols = []
    data = []
    
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                cell_type = domain_grid[i, j, k]
                
                if cell_type == -1:
                    # Void: skip
                    continue
                
                # Calcola modulo effettivo
                if cell_type == 1:
                    # Fixed: materiale pieno
                    E_eff = material.E
                else:
                    # Design: interpola con SIMP
                    rho = density[i, j, k]
                    E_eff = E_min + (rho ** penalty) * (material.E - E_min)
                
                # Scala matrice elemento
                scale = E_eff / material.E
                Ke = scale * Ke0
                
                # Indici DOF
                dof_idx = get_element_dof_indices((i, j, k), (nx, ny, nz))
                
                # Aggiungi contributi
                for ii in range(24):
                    for jj in range(24):
                        rows.append(dof_idx[ii])
                        cols.append(dof_idx[jj])
                        data.append(Ke[ii, jj])
    
    # Costruisci matrice sparse
    K = sparse.coo_matrix((data, (rows, cols)), shape=(n_dofs, n_dofs))
    K = K.tocsr()
    K = (K + K.T) / 2  # Assicura simmetria
    
    return K


def solve_fem(
    K: sparse.csr_matrix,
    F: np.ndarray,
    constrained_dofs: np.ndarray,
    method: str = "direct"
) -> np.ndarray:
    """
    Risolve il sistema FEM K*u = F con vincoli.
    
    Args:
        K: Matrice rigidezza globale
        F: Vettore forze
        constrained_dofs: Indici DOF vincolati
        method: 'direct' (spsolve) o 'iterative' (CG)
        
    Returns:
        Vettore spostamenti u
    """
    n_dofs = K.shape[0]
    
    # DOF liberi
    all_dofs = np.arange(n_dofs)
    free_dofs = np.setdiff1d(all_dofs, constrained_dofs)
    
    # Estrai sottosistema
    K_ff = K[free_dofs, :][:, free_dofs]
    F_f = F[free_dofs]
    
    # Risolvi
    if method == "direct":
        u_f = spsolve(K_ff, F_f)
    elif method == "iterative":
        u_f, info = cg(K_ff, F_f, tol=1e-8, maxiter=5000)
        if info != 0:
            warnings.warn(f"CG non converge: info={info}")
    else:
        raise ValueError(f"Metodo sconosciuto: {method}")
    
    # Ricostruisci vettore completo
    u = np.zeros(n_dofs)
    u[free_dofs] = u_f
    
    return u


def compute_element_compliance(
    u: np.ndarray,
    domain_grid: np.ndarray,
    density: np.ndarray,
    element_size: float,
    material: MaterialProperties,
    penalty: float = 3.0,
    E_min: float = 1e-9
) -> Tuple[float, np.ndarray]:
    """
    Calcola compliance globale e contributi per elemento.
    
    Compliance: C = u^T * K * u = sum_e (rho_e^p * u_e^T * K0_e * u_e)
    
    Args:
        u: Vettore spostamenti
        domain_grid: Griglia dominio
        density: Campo densità
        element_size: Dimensione elemento
        material: Proprietà materiale
        penalty: Esponente SIMP
        E_min: Modulo minimo
        
    Returns:
        (compliance_totale, compliance_per_elemento)
    """
    nx, ny, nz = domain_grid.shape
    Ke0 = get_element_stiffness_matrix(element_size, material)
    
    element_compliance = np.zeros((nx, ny, nz))
    
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                if domain_grid[i, j, k] == -1:
                    continue
                
                # Estrai spostamenti elemento
                dof_idx = get_element_dof_indices((i, j, k), (nx, ny, nz))
                u_e = u[dof_idx]
                
                # Compliance elemento (con materiale pieno)
                ce = u_e @ Ke0 @ u_e
                
                # Scala per densità
                if domain_grid[i, j, k] == 0:
                    rho = density[i, j, k]
                    E_ratio = E_min / material.E + (rho ** penalty) * (1 - E_min / material.E)
                else:
                    E_ratio = 1.0
                
                element_compliance[i, j, k] = E_ratio * ce
    
    total_compliance = np.sum(element_compliance)
    
    return total_compliance, element_compliance


if __name__ == "__main__":
    # Test: verifica matrice elemento
    material = MaterialProperties()
    Ke = get_element_stiffness_matrix(0.001, material)  # 1mm
    
    print(f"Element stiffness matrix shape: {Ke.shape}")
    print(f"Symmetry check: {np.allclose(Ke, Ke.T)}")
    print(f"Max value: {Ke.max():.2e}")
    print(f"Condition number estimate: {np.linalg.cond(Ke):.2e}")
