"""
AM - Additive Manufacturing Topology Optimization
Entry point for BRK-A-01 pilot case.
"""

import json
import numpy as np
from pathlib import Path

from src.geometry import create_bracket_domain
from src.loads import create_brk_a_01_static_case_1
from src.fem import MaterialProperties
from src.topopt import SIMPOptimizer, SIMPParams


def main():
    """Run topology optimization for BRK-A-01 pilot case."""
    print("=" * 60)
    print("AM - Topology Optimization for BRK-A-01")
    print("=" * 60)
    
    # 1. Create domain (use 5mm for fast test, 1mm for production)
    resolution = 3.0  # mm
    domain = create_bracket_domain(
        size_mm=(120, 60, 80),
        resolution_mm=resolution
    )
    print(f"\nDomain: {domain.shape}, {domain.n_design:,} design elements")
    
    # 2. Load case
    load_case = create_brk_a_01_static_case_1(domain.shape, resolution)
    print(f"Load case: {load_case.name}")
    
    # 3. Material (Ti6Al4V defaults)
    material = MaterialProperties()
    print(f"Material: Ti6Al4V (E={material.E/1e9:.1f} GPa)")
    
    # 4. Optimization parameters
    params = SIMPParams(
        volume_fraction=0.25,
        max_iterations=50
    )
    
    optimizer = SIMPOptimizer(
        domain=domain,
        load_cases=[load_case],
        material=material,
        params=params,
        verbose=True
    )
    
    # 5. Run optimization
    print("\n" + "-" * 60)
    result = optimizer.run()
    
    # 6. Save results
    output_dir = Path("data/brk_a_01")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save density field
    np.save(output_dir / "density_field.npy", result.density)
    
    # Save metadata
    metadata = {
        "case": "BRK-A-01",
        "resolution_mm": resolution,
        "domain_shape": list(domain.shape),
        "volume_fraction_target": params.volume_fraction,
        "volume_fraction_final": float(result.final_volume),
        "compliance_final": float(result.final_compliance),
        "iterations": result.iterations,
        "converged": result.converged,
        "elapsed_time_s": result.elapsed_time
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nResults saved to {output_dir}/")
    print(f"  - density_field.npy")
    print(f"  - metadata.json")
    
    return result


if __name__ == "__main__":
    main()
