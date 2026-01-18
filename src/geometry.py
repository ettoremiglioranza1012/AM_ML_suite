"""
geometry.py - Generazione griglia voxel e marking design/non-design space.

Questo modulo gestisce:
- Creazione della griglia voxel 3D con risoluzione 1mm
- Marking delle zone: design space vs non-design space
- Definizione interfacce fisse (base con 4 fori, occhiello superiore)
- Bounding box del dominio di ottimizzazione
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class VoxelDomain:
    """
    Dominio voxel 3D per topology optimization.
    
    Attributes:
        grid: Array 3D con marking delle zone
              1  = Fixed (Non-Design Space)
              0  = Optimizable (Design Space)
             -1  = Void (fuori dominio)
        resolution_mm: Risoluzione della griglia in mm
        origin: Origine del sistema di coordinate (mm)
    """
    grid: np.ndarray
    resolution_mm: float
    origin: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    
    @property
    def shape(self) -> Tuple[int, int, int]:
        """Dimensioni della griglia (nx, ny, nz)."""
        return self.grid.shape
    
    @property
    def size_mm(self) -> Tuple[float, float, float]:
        """Dimensioni fisiche in mm."""
        return tuple(s * self.resolution_mm for s in self.shape)
    
    @property
    def n_elements(self) -> int:
        """Numero totale di elementi."""
        return int(np.prod(self.shape))
    
    @property
    def n_design(self) -> int:
        """Numero di elementi nel design space."""
        return int(np.sum(self.grid == 0))
    
    @property
    def n_fixed(self) -> int:
        """Numero di elementi fissi (non-design)."""
        return int(np.sum(self.grid == 1))
    
    def get_design_indices(self) -> np.ndarray:
        """Restituisce gli indici degli elementi ottimizzabili."""
        return np.argwhere(self.grid == 0)
    
    def get_fixed_indices(self) -> np.ndarray:
        """Restituisce gli indici degli elementi fissi."""
        return np.argwhere(self.grid == 1)
    
    def voxel_to_coords(self, indices: np.ndarray) -> np.ndarray:
        """
        Converte indici voxel in coordinate fisiche (centro del voxel).
        
        Args:
            indices: Array (N, 3) di indici voxel
            
        Returns:
            Array (N, 3) di coordinate in mm
        """
        return (indices + 0.5) * self.resolution_mm + np.array(self.origin)


def create_bracket_domain(
    size_mm: Tuple[int, int, int] = (120, 60, 80),
    resolution_mm: float = 1.0,
    base_thickness_mm: float = 5.0,
    hole_diameter_mm: float = 8.0,
    eyelet_diameter_mm: float = 12.0,
    eyelet_offset_mm: float = 4.0,
) -> VoxelDomain:
    """
    Crea il dominio voxel per la staffa aeronautica BRK-A-01.
    
    Geometria:
    - Base: flangia con 4 fori passanti (Ø 8mm), spessore 5mm
    - Occhiello: foro superiore (Ø 12mm) con offset 4mm di materiale
    - Volume centrale: design space ottimizzabile
    
    Args:
        size_mm: Dimensioni bounding box (X, Y, Z) in mm
        resolution_mm: Risoluzione griglia in mm
        base_thickness_mm: Spessore della flangia base
        hole_diameter_mm: Diametro fori base
        eyelet_diameter_mm: Diametro foro occhiello
        eyelet_offset_mm: Offset materiale attorno all'occhiello
        
    Returns:
        VoxelDomain con grid marcata
    """
    # Calcola dimensioni griglia
    nx = int(size_mm[0] / resolution_mm)
    ny = int(size_mm[1] / resolution_mm)
    nz = int(size_mm[2] / resolution_mm)
    
    # Inizializza tutto come design space (0)
    grid = np.zeros((nx, ny, nz), dtype=np.int8)
    
    # === MARCA NON-DESIGN SPACE ===
    
    # 1. Base (flangia) - primi 5mm in Z sono fissi
    base_z = max(1, int(base_thickness_mm / resolution_mm))
    grid[:, :, :base_z] = 1
    
    # 2. Fori nella base (4 fori agli angoli) - marcati come void
    hole_radius = hole_diameter_mm / 2 / resolution_mm
    # Posizioni fori (offset dal bordo in elementi)
    hole_offset_x_mm = 15  # mm dal bordo
    hole_offset_y_mm = 15  # mm dal bordo
    hole_offset_x = int(hole_offset_x_mm / resolution_mm)
    hole_offset_y = int(hole_offset_y_mm / resolution_mm)
    
    hole_positions = [
        (hole_offset_x, hole_offset_y),
        (nx - hole_offset_x, hole_offset_y),
        (hole_offset_x, ny - hole_offset_y),
        (nx - hole_offset_x, ny - hole_offset_y),
    ]
    
    for hx, hy in hole_positions:
        _mark_cylinder_void(grid, hx, hy, 0, hole_radius, base_z)
    
    # 3. Occhiello superiore - foro con materiale attorno
    # All dimensions scaled by resolution
    eyelet_z_offset_mm = 15  # mm from top
    eyelet_height_mm = 20    # mm height
    
    eyelet_z_offset = int(eyelet_z_offset_mm / resolution_mm)
    eyelet_height = int(eyelet_height_mm / resolution_mm)
    eyelet_height = max(1, eyelet_height)  # At least 1 element
    
    eyelet_z_center = nz - eyelet_z_offset
    eyelet_x_center = nx // 2  # centrato in X
    eyelet_y_center = ny // 2  # centrato in Y
    
    # Ensure z_start is within bounds
    z_start = max(0, eyelet_z_center - eyelet_height // 2)
    z_end = min(nz, z_start + eyelet_height)
    actual_height = z_end - z_start
    
    if actual_height > 0:
        # Materiale attorno all'occhiello (fixed)
        outer_radius = (eyelet_diameter_mm / 2 + eyelet_offset_mm) / resolution_mm
        _mark_cylinder_fixed(
            grid, 
            eyelet_x_center, 
            eyelet_y_center, 
            z_start,
            outer_radius,
            actual_height
        )
        
        # Foro centrale dell'occhiello (void)
        inner_radius = eyelet_diameter_mm / 2 / resolution_mm
        _mark_cylinder_void(
            grid,
            eyelet_x_center,
            eyelet_y_center, 
            z_start,
            inner_radius,
            actual_height
        )
    
    # 4. PILASTRO DI CONNESSIONE: Garantisce continuità strutturale base-occhiello
    # Questo evita che l'occhiello sia una "bolla" isolata durante l'ottimizzazione
    # Il pilastro è marcato come Fixed (1) per garantire il percorso di carico
    pillar_radius = (eyelet_diameter_mm / 2 + eyelet_offset_mm) / resolution_mm  # stesso raggio esterno occhiello
    pillar_z_start = base_z  # parte dalla cima della base
    pillar_z_end = z_start   # arriva alla base dell'occhiello
    
    if pillar_z_end > pillar_z_start:
        _mark_cylinder_fixed(
            grid,
            eyelet_x_center,
            eyelet_y_center,
            pillar_z_start,
            pillar_radius,
            pillar_z_end - pillar_z_start
        )
    
    return VoxelDomain(grid=grid, resolution_mm=resolution_mm)


def _mark_cylinder_void(
    grid: np.ndarray,
    cx: float, cy: float, z_start: int,
    radius: float, height: int
) -> None:
    """Marca un cilindro come void (-1) nella griglia."""
    nx, ny, nz = grid.shape
    for i in range(max(0, int(cx - radius - 1)), min(nx, int(cx + radius + 2))):
        for j in range(max(0, int(cy - radius - 1)), min(ny, int(cy + radius + 2))):
            if (i - cx)**2 + (j - cy)**2 <= radius**2:
                for k in range(z_start, min(nz, z_start + height)):
                    grid[i, j, k] = -1


def _mark_cylinder_fixed(
    grid: np.ndarray,
    cx: float, cy: float, z_start: int,
    radius: float, height: int
) -> None:
    """Marca un cilindro come fixed (1) nella griglia."""
    nx, ny, nz = grid.shape
    for i in range(max(0, int(cx - radius - 1)), min(nx, int(cx + radius + 2))):
        for j in range(max(0, int(cy - radius - 1)), min(ny, int(cy + radius + 2))):
            if (i - cx)**2 + (j - cy)**2 <= radius**2:
                for k in range(z_start, min(nz, z_start + height)):
                    if grid[i, j, k] != -1:  # Non sovrascrivere void
                        grid[i, j, k] = 1


def visualize_domain_slice(domain: VoxelDomain, z_level: int) -> None:
    """
    Visualizza una slice del dominio a un dato livello Z.
    
    Args:
        domain: VoxelDomain da visualizzare
        z_level: Indice Z della slice
    """
    import matplotlib.pyplot as plt
    
    slice_data = domain.grid[:, :, z_level].T
    
    fig, ax = plt.subplots(figsize=(10, 5))
    cmap = plt.cm.colors.ListedColormap(['white', 'lightgray', 'darkblue'])
    bounds = [-1.5, -0.5, 0.5, 1.5]
    norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)
    
    im = ax.imshow(slice_data, cmap=cmap, norm=norm, origin='lower')
    ax.set_xlabel('X [voxel]')
    ax.set_ylabel('Y [voxel]')
    ax.set_title(f'Domain slice at Z={z_level} mm')
    
    cbar = plt.colorbar(im, ax=ax, ticks=[-1, 0, 1])
    cbar.ax.set_yticklabels(['Void', 'Design', 'Fixed'])
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Test: crea dominio e stampa statistiche
    domain = create_bracket_domain()
    print(f"Domain shape: {domain.shape}")
    print(f"Domain size: {domain.size_mm} mm")
    print(f"Total elements: {domain.n_elements:,}")
    print(f"Design space elements: {domain.n_design:,}")
    print(f"Fixed elements: {domain.n_fixed:,}")
    print(f"Void elements: {domain.n_elements - domain.n_design - domain.n_fixed:,}")
