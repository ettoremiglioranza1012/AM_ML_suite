"""
topopt.py - Loop SIMP per Topology Optimization.

Questo modulo gestisce:
- Algoritmo SIMP (Solid Isotropic Material with Penalization)
- Update densità con Optimality Criteria (OC)
- Filtro densità per controllo mesh-dependency
- Proiezione su vincolo di volume
"""

import numpy as np
from scipy import ndimage
from scipy import sparse
from dataclasses import dataclass, field
from typing import Tuple, Optional, List, Callable
import time

from .geometry import VoxelDomain
from .loads import LoadCase
from .fem import (
    MaterialProperties,
    assemble_global_stiffness,
    solve_fem,
    compute_element_compliance,
    get_element_stiffness_matrix,
    get_element_dof_indices
)


@dataclass
class SIMPParams:
    """
    Parametri per SIMP Topology Optimization.
    
    Attributes:
        volume_fraction: Frazione di volume target (es. 0.25 = 25%)
        penalty: Esponente SIMP (p), tipicamente 3
        filter_radius: Raggio filtro densità in elementi
        move_limit: Limite variazione densità per iterazione
        rho_min: Densità minima (evita singolarità)
        max_iterations: Numero massimo iterazioni
        convergence_tol: Tolleranza convergenza su variazione densità
        E_min: Modulo minimo per elementi vuoti
    """
    volume_fraction: float = 0.25
    penalty: float = 3.0
    filter_radius: float = 2.0
    move_limit: float = 0.2
    rho_min: float = 0.01  # Increased from 0.001 for numerical stability
    max_iterations: int = 100
    convergence_tol: float = 0.01
    E_min: float = 1e-9


@dataclass
class TOResult:
    """
    Risultato Topology Optimization.
    
    Attributes:
        density: Campo di densità finale (nx, ny, nz)
        compliance_history: Storia compliance per iterazione
        volume_history: Storia volume per iterazione
        iterations: Numero iterazioni eseguite
        converged: Se l'algoritmo è converso
        elapsed_time: Tempo di calcolo [s]
    """
    density: np.ndarray
    compliance_history: List[float] = field(default_factory=list)
    volume_history: List[float] = field(default_factory=list)
    iterations: int = 0
    converged: bool = False
    elapsed_time: float = 0.0
    
    @property
    def final_compliance(self) -> float:
        return self.compliance_history[-1] if self.compliance_history else 0.0
    
    @property
    def final_volume(self) -> float:
        return self.volume_history[-1] if self.volume_history else 0.0


class SIMPOptimizer:
    """
    Ottimizzatore topologico basato su metodo SIMP.
    
    Implementa:
    - Filtro densità (sensitivity filter)
    - Optimality Criteria (OC) update
    - Multi-load case support
    """
    
    def __init__(
        self,
        domain: VoxelDomain,
        load_cases: List[LoadCase],
        material: MaterialProperties = None,
        params: SIMPParams = None,
        verbose: bool = True
    ):
        """
        Inizializza l'ottimizzatore.
        
        Args:
            domain: Dominio voxel con marking design/non-design
            load_cases: Lista di casi di carico
            material: Proprietà materiale (default Ti6Al4V)
            params: Parametri SIMP (default valori standard)
            verbose: Stampa info durante ottimizzazione
        """
        self.domain = domain
        self.load_cases = load_cases
        self.material = material or MaterialProperties()
        self.params = params or SIMPParams()
        self.verbose = verbose
        
        # Dimensioni
        self.nx, self.ny, self.nz = domain.shape
        self.n_elements = domain.n_elements
        self.element_size = domain.resolution_mm / 1000  # mm -> m
        
        # Inizializza densità uniforme al volume target
        self.density = np.ones((self.nx, self.ny, self.nz)) * self.params.volume_fraction
        
        # Maschera design space
        self.design_mask = (domain.grid == 0)
        
        # Pre-calcola filtro
        self._build_filter()
        
        # Pre-calcola matrice elemento
        self.Ke0 = get_element_stiffness_matrix(self.element_size, self.material)
    
    def _build_filter(self) -> None:
        """Costruisce il kernel del filtro densità."""
        r = self.params.filter_radius
        
        # Crea kernel sferico
        size = int(2 * np.ceil(r) + 1)
        center = size // 2
        
        kernel = np.zeros((size, size, size))
        for i in range(size):
            for j in range(size):
                for k in range(size):
                    dist = np.sqrt((i - center)**2 + (j - center)**2 + (k - center)**2)
                    if dist <= r:
                        kernel[i, j, k] = r - dist
        
        self.filter_kernel = kernel / kernel.sum()
    
    def _apply_filter(self, x: np.ndarray) -> np.ndarray:
        """Applica filtro densità (convolution)."""
        x_filtered = ndimage.convolve(x, self.filter_kernel, mode='reflect')
        return x_filtered
    
    def _compute_sensitivities(
        self,
        density: np.ndarray,
        u: np.ndarray
    ) -> np.ndarray:
        """
        Calcola sensibilità della compliance rispetto alla densità.
        
        dC/drho_e = -p * rho_e^(p-1) * u_e^T * K0 * u_e
        
        Args:
            density: Campo densità corrente
            u: Spostamenti dalla soluzione FEM
            
        Returns:
            Array sensibilità (nx, ny, nz)
        """
        p = self.params.penalty
        dc = np.zeros((self.nx, self.ny, self.nz))
        
        for i in range(self.nx):
            for j in range(self.ny):
                for k in range(self.nz):
                    if not self.design_mask[i, j, k]:
                        continue
                    
                    # Estrai spostamenti elemento
                    dof_idx = get_element_dof_indices(
                        (i, j, k), 
                        (self.nx, self.ny, self.nz)
                    )
                    u_e = u[dof_idx]
                    
                    # Compliance elemento
                    ce = u_e @ self.Ke0 @ u_e
                    
                    # Sensibilità
                    rho = density[i, j, k]
                    dc[i, j, k] = -p * (rho ** (p - 1)) * ce
        
        return dc
    
    def _oc_update(
        self,
        density: np.ndarray,
        dc: np.ndarray,
        dv: np.ndarray
    ) -> np.ndarray:
        """
        Aggiorna densità con Optimality Criteria.
        
        Args:
            density: Densità corrente
            dc: Sensibilità compliance
            dv: Sensibilità volume (= 1 per ogni elemento)
            
        Returns:
            Nuova densità
        """
        move = self.params.move_limit
        rho_min = self.params.rho_min
        vf = self.params.volume_fraction
        
        # Bisection per trovare Lagrangiano
        l1, l2 = 1e-9, 1e9  # Avoid division by zero
        
        # Volume design space
        n_design = np.sum(self.design_mask)
        
        for _ in range(100):  # Max iterations for bisection
            if l1 + l2 < 1e-12:  # Safety check
                break
            if (l2 - l1) / max((l1 + l2), 1e-10) <= 1e-3:
                break
                
            lmid = 0.5 * (l1 + l2)
            
            # Update formula OC
            Be = np.sqrt(-dc / (dv * lmid + 1e-10))
            
            # Limiti
            rho_new = np.maximum(
                rho_min,
                np.maximum(
                    density - move,
                    np.minimum(
                        1.0,
                        np.minimum(
                            density + move,
                            density * Be
                        )
                    )
                )
            )
            
            # Applica solo al design space
            rho_new = np.where(self.design_mask, rho_new, density)
            
            # Check volume
            current_vf = np.sum(rho_new[self.design_mask]) / n_design
            
            if current_vf > vf:
                l1 = lmid
            else:
                l2 = lmid
        
        return rho_new
    
    def run(self, callback: Optional[Callable] = None) -> TOResult:
        """
        Esegue l'ottimizzazione topologica.
        
        Args:
            callback: Funzione chiamata ad ogni iterazione 
                     callback(iter, density, compliance)
                     
        Returns:
            TOResult con risultati ottimizzazione
        """
        start_time = time.time()
        
        compliance_history = []
        volume_history = []
        
        density = self.density.copy()
        
        # Sensitività volume (costante = 1)
        dv = np.ones_like(density)
        
        if self.verbose:
            print("=" * 60)
            print("SIMP Topology Optimization")
            print("=" * 60)
            print(f"Grid size: {self.nx} x {self.ny} x {self.nz}")
            print(f"Design elements: {np.sum(self.design_mask):,}")
            print(f"Target volume fraction: {self.params.volume_fraction:.2%}")
            print(f"Filter radius: {self.params.filter_radius}")
            print(f"Penalty: {self.params.penalty}")
            print("-" * 60)
        
        converged = False
        
        for iteration in range(self.params.max_iterations):
            # Filtra densità
            density_filtered = self._apply_filter(density)
            density_filtered = np.clip(density_filtered, self.params.rho_min, 1.0)
            
            # Assembla K e risolvi per ogni load case
            total_compliance = 0.0
            dc_total = np.zeros_like(density)
            
            for lc in self.load_cases:
                # Assembla matrice rigidezza
                K = assemble_global_stiffness(
                    self.domain.grid,
                    density_filtered,
                    self.element_size,
                    self.material,
                    self.params.penalty,
                    self.params.E_min
                )
                
                # Vettore forze
                n_nodes = (self.nx + 1) * (self.ny + 1) * (self.nz + 1)
                n_dofs = 3 * n_nodes
                F = lc.get_force_vector(n_dofs)
                
                # Vincoli
                constrained_dofs = lc.get_constrained_dofs()
                
                # Risolvi
                u = solve_fem(K, F, constrained_dofs, method="direct")
                
                # Compliance
                c, _ = compute_element_compliance(
                    u, self.domain.grid, density_filtered,
                    self.element_size, self.material,
                    self.params.penalty, self.params.E_min
                )
                total_compliance += c
                
                # Sensibilità
                dc = self._compute_sensitivities(density_filtered, u)
                dc_total += dc
            
            # Filtra sensibilità
            dc_filtered = self._apply_filter(dc_total * density) / np.maximum(density, 1e-3)
            
            # Update densità
            density_old = density.copy()
            density = self._oc_update(density, dc_filtered, dv)
            
            # Calcola volume corrente
            current_vf = np.mean(density[self.design_mask])
            
            # Salva storia
            compliance_history.append(total_compliance)
            volume_history.append(current_vf)
            
            # Convergenza
            change = np.max(np.abs(density - density_old))
            
            if self.verbose:
                print(f"Iter {iteration+1:3d} | "
                      f"Compliance: {total_compliance:.4e} | "
                      f"Volume: {current_vf:.4f} | "
                      f"Change: {change:.4f}")
            
            if callback:
                callback(iteration, density, total_compliance)
            
            if change < self.params.convergence_tol:
                converged = True
                if self.verbose:
                    print("-" * 60)
                    print(f"Converged at iteration {iteration + 1}")
                break
        
        elapsed_time = time.time() - start_time
        
        if self.verbose:
            print("=" * 60)
            print(f"Optimization complete in {elapsed_time:.1f}s")
            print(f"Final compliance: {compliance_history[-1]:.4e}")
            print(f"Final volume fraction: {volume_history[-1]:.4f}")
            print("=" * 60)
        
        return TOResult(
            density=density,
            compliance_history=compliance_history,
            volume_history=volume_history,
            iterations=iteration + 1,
            converged=converged,
            elapsed_time=elapsed_time
        )


def threshold_density(
    density: np.ndarray,
    threshold: float = 0.5
) -> np.ndarray:
    """
    Applica threshold al campo di densità.
    
    Args:
        density: Campo densità continuo [0, 1]
        threshold: Valore soglia
        
    Returns:
        Campo binario (0 o 1)
    """
    return (density >= threshold).astype(np.float64)


if __name__ == "__main__":
    # Test base (senza dati reali)
    print("topopt.py module loaded successfully")
    print(f"SIMPParams defaults: {SIMPParams()}")
