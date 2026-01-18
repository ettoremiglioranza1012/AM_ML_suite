"""
loads.py - Definizione load cases e boundary conditions.

Questo modulo gestisce:
- Definizione dei casi di carico (load cases)
- Condizioni al contorno (vincoli di spostamento)
- Applicazione forze su nodi specifici
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from enum import Enum


class DOF(Enum):
    """Gradi di libertà per nodo 3D."""
    UX = 0  # Spostamento X
    UY = 1  # Spostamento Y
    UZ = 2  # Spostamento Z


@dataclass
class BoundaryCondition:
    """
    Condizione al contorno (vincolo di spostamento).
    
    Attributes:
        node_indices: Indici dei nodi vincolati (nel sistema globale)
        constrained_dofs: Lista di DOF vincolati per ogni nodo
        prescribed_values: Valori prescritti (default 0 = vincolo fisso)
    """
    node_indices: np.ndarray
    constrained_dofs: List[DOF] = field(default_factory=lambda: [DOF.UX, DOF.UY, DOF.UZ])
    prescribed_values: float = 0.0
    
    @property
    def n_nodes(self) -> int:
        return len(self.node_indices)
    
    def get_dof_indices(self, dofs_per_node: int = 3) -> np.ndarray:
        """
        Restituisce gli indici DOF globali vincolati.
        
        Args:
            dofs_per_node: Numero di DOF per nodo (default 3 per 3D)
            
        Returns:
            Array di indici DOF da vincolare
        """
        dof_indices = []
        for node_idx in self.node_indices:
            for dof in self.constrained_dofs:
                dof_indices.append(node_idx * dofs_per_node + dof.value)
        return np.array(dof_indices, dtype=np.int64)


@dataclass
class PointLoad:
    """
    Carico concentrato su un nodo.
    
    Attributes:
        node_index: Indice del nodo di applicazione
        force_vector: Vettore forza [Fx, Fy, Fz] in Newton
    """
    node_index: int
    force_vector: np.ndarray  # [Fx, Fy, Fz] in N
    
    def __post_init__(self):
        self.force_vector = np.asarray(self.force_vector, dtype=np.float64)
        assert self.force_vector.shape == (3,), "Force vector must be (3,)"


@dataclass
class DistributedLoad:
    """
    Carico distribuito su più nodi.
    
    Attributes:
        node_indices: Indici dei nodi di applicazione
        total_force: Forza totale [Fx, Fy, Fz] in Newton
        distribution: 'uniform' o 'proportional'
    """
    node_indices: np.ndarray
    total_force: np.ndarray  # [Fx, Fy, Fz] in N
    distribution: str = "uniform"
    
    def __post_init__(self):
        self.node_indices = np.asarray(self.node_indices, dtype=np.int64)
        self.total_force = np.asarray(self.total_force, dtype=np.float64)
    
    def get_nodal_forces(self) -> np.ndarray:
        """
        Calcola le forze nodali distribuite.
        
        Returns:
            Array (N, 3) di forze per ogni nodo
        """
        n_nodes = len(self.node_indices)
        if self.distribution == "uniform":
            return np.tile(self.total_force / n_nodes, (n_nodes, 1))
        else:
            raise NotImplementedError(f"Distribution '{self.distribution}' not implemented")


@dataclass
class LoadCase:
    """
    Caso di carico completo per analisi FEM.
    
    Attributes:
        name: Nome identificativo del caso
        description: Descrizione del caso di carico
        boundary_conditions: Lista di condizioni al contorno
        point_loads: Lista di carichi concentrati
        distributed_loads: Lista di carichi distribuiti
    """
    name: str
    description: str = ""
    boundary_conditions: List[BoundaryCondition] = field(default_factory=list)
    point_loads: List[PointLoad] = field(default_factory=list)
    distributed_loads: List[DistributedLoad] = field(default_factory=list)
    
    def get_force_vector(self, n_dofs: int) -> np.ndarray:
        """
        Assembla il vettore delle forze globale.
        
        Args:
            n_dofs: Numero totale di DOF nel sistema
            
        Returns:
            Vettore forze (n_dofs,)
        """
        F = np.zeros(n_dofs, dtype=np.float64)
        
        # Carichi concentrati
        for load in self.point_loads:
            for i, dof in enumerate([DOF.UX, DOF.UY, DOF.UZ]):
                F[load.node_index * 3 + dof.value] += load.force_vector[i]
        
        # Carichi distribuiti
        for dist_load in self.distributed_loads:
            nodal_forces = dist_load.get_nodal_forces()
            for node_idx, force in zip(dist_load.node_indices, nodal_forces):
                for i, dof in enumerate([DOF.UX, DOF.UY, DOF.UZ]):
                    F[node_idx * 3 + dof.value] += force[i]
        
        return F
    
    def get_constrained_dofs(self) -> np.ndarray:
        """Restituisce tutti i DOF vincolati."""
        all_dofs = []
        for bc in self.boundary_conditions:
            all_dofs.extend(bc.get_dof_indices())
        return np.unique(np.array(all_dofs, dtype=np.int64))


# =============================================================================
# CASI DI CARICO PREDEFINITI PER BRK-A-01
# =============================================================================

def create_brk_a_01_static_case_1(
    domain_shape: Tuple[int, int, int],
    resolution_mm: float = 1.0
) -> LoadCase:
    """
    Crea il Caso Statico 1 per BRK-A-01.
    
    Specifiche:
    - Vincolo: Base completamente fissa (Z=0)
    - Carico: 15,000 N verticale (Z negativo) sull'occhiello
    
    Args:
        domain_shape: Dimensioni griglia (nx, ny, nz)
        resolution_mm: Risoluzione in mm
        
    Returns:
        LoadCase configurato
    """
    nx, ny, nz = domain_shape
    
    # === BOUNDARY CONDITIONS ===
    # Nodi alla base (Z=0) sono vincolati in tutte le direzioni
    base_nodes = []
    for i in range(nx + 1):  # +1 perché i nodi sono ai vertici
        for j in range(ny + 1):
            node_idx = i * (ny + 1) * (nz + 1) + j * (nz + 1) + 0
            base_nodes.append(node_idx)
    
    bc_base = BoundaryCondition(
        node_indices=np.array(base_nodes, dtype=np.int64),
        constrained_dofs=[DOF.UX, DOF.UY, DOF.UZ]
    )
    
    # === LOADS ===
    # Carico sull'occhiello: 15,000 N in direzione -Z
    # L'occhiello è al centro superiore del dominio
    eyelet_z = nz - 15  # 15mm dal top
    eyelet_x = nx // 2
    eyelet_y = ny // 2
    
    # Nodi attorno al foro dell'occhiello
    eyelet_nodes = []
    eyelet_radius = 10  # mm (raggio esterno dell'occhiello)
    
    for i in range(max(0, eyelet_x - eyelet_radius), min(nx + 1, eyelet_x + eyelet_radius + 1)):
        for j in range(max(0, eyelet_y - eyelet_radius), min(ny + 1, eyelet_y + eyelet_radius + 1)):
            dist = np.sqrt((i - eyelet_x)**2 + (j - eyelet_y)**2)
            if 6 <= dist <= eyelet_radius:  # Anello tra raggio 6 e 10
                node_idx = i * (ny + 1) * (nz + 1) + j * (nz + 1) + eyelet_z
                eyelet_nodes.append(node_idx)
    
    load_eyelet = DistributedLoad(
        node_indices=np.array(eyelet_nodes, dtype=np.int64),
        total_force=np.array([0.0, 0.0, -15000.0]),  # 15 kN verso il basso
        distribution="uniform"
    )
    
    return LoadCase(
        name="BRK-A-01_Static_1",
        description="Carico verticale 15kN sull'occhiello, base vincolata",
        boundary_conditions=[bc_base],
        distributed_loads=[load_eyelet]
    )


def create_brk_a_01_static_case_2(
    domain_shape: Tuple[int, int, int],
    resolution_mm: float = 1.0
) -> LoadCase:
    """
    Crea il Caso Statico 2 per BRK-A-01.
    
    Specifiche:
    - Vincolo: Base completamente fissa (Z=0)
    - Carico: 10,000 N inclinato a 30° (componente X-Z)
    
    Args:
        domain_shape: Dimensioni griglia (nx, ny, nz)
        resolution_mm: Risoluzione in mm
        
    Returns:
        LoadCase configurato
    """
    # Riutilizza la logica del caso 1 per i vincoli
    case_1 = create_brk_a_01_static_case_1(domain_shape, resolution_mm)
    
    # Modifica il carico: 10 kN a 30°
    angle_rad = np.radians(30)
    Fx = 10000.0 * np.sin(angle_rad)  # Componente X
    Fz = -10000.0 * np.cos(angle_rad)  # Componente Z (negativa, verso il basso)
    
    # Aggiorna il carico distribuito
    if case_1.distributed_loads:
        case_1.distributed_loads[0].total_force = np.array([Fx, 0.0, Fz])
    
    case_1.name = "BRK-A-01_Static_2"
    case_1.description = "Carico inclinato 30° (10kN) sull'occhiello, base vincolata"
    
    return case_1


# Dizionario dei casi di carico disponibili
LOAD_CASES = {
    "static_1": create_brk_a_01_static_case_1,
    "static_2": create_brk_a_01_static_case_2,
}


if __name__ == "__main__":
    # Test: crea e stampa info sul caso di carico
    domain_shape = (120, 60, 80)
    case = create_brk_a_01_static_case_1(domain_shape)
    
    print(f"Load Case: {case.name}")
    print(f"Description: {case.description}")
    print(f"Boundary conditions: {len(case.boundary_conditions)}")
    print(f"Point loads: {len(case.point_loads)}")
    print(f"Distributed loads: {len(case.distributed_loads)}")
    
    if case.distributed_loads:
        dl = case.distributed_loads[0]
        print(f"  - Nodes: {len(dl.node_indices)}")
        print(f"  - Total force: {dl.total_force} N")
