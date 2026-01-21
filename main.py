"""
AM - Additive Manufacturing Topology Optimization
Unified Entry Point

Supports multiple execution modes:
  --mode numerical  : Python SIMP solver (Ground Truth)
  --mode ai         : Neural network inference (requires trained model)
"""

import argparse
import json
import sys
import uuid
from pathlib import Path

import numpy as np

# Package imports (new structure)
from src.am.core.geometry import create_bracket_domain
from src.am.core.loads import create_brk_a_01_static_case_1
from src.am.numerical.fem import MaterialProperties
from src.am.numerical.topopt import SIMPOptimizer, SIMPParams


def run_numerical(args: argparse.Namespace) -> dict:
    """
    Run topology optimization using Python SIMP solver.
    
    This is the original prototype, now serving as:
    - Ground Truth for validation
    - Reference implementation
    """
    # Generate unique run ID
    run_id = str(uuid.uuid4())[:8]
    
    print("=" * 60)
    print("AM - Topology Optimization (Numerical Solver)")
    print(f"Run ID: {run_id}")
    print("=" * 60)
    
    # 1. Create domain
    resolution = args.resolution
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
        volume_fraction=args.volume_fraction,
        max_iterations=args.max_iter
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
    output_dir = Path("data") / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save density field
    np.save(output_dir / "density_field.npy", result.density)
    
    # Save metadata
    metadata = {
        "case": "BRK-A-01",
        "mode": "numerical",
        "run_id": run_id,
        "provenance": "python_prototype",
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
    
    return metadata


def run_ai(args: argparse.Namespace) -> dict:
    """
    Run topology optimization using AI inference.
    
    Requires a pre-trained model checkpoint.
    """
    # Generate unique run ID
    run_id = str(uuid.uuid4())[:8]
    
    print("=" * 60)
    print("AM - Topology Optimization (AI Inference)")
    print(f"Run ID: {run_id}")
    print("=" * 60)
    
    # Import AI module (lazy to avoid PyTorch dependency when not needed)
    try:
        from src.am.ai.inference import AIOptimizer
    except ImportError as e:
        print(f"\n❌ AI module not available: {e}")
        print("   Install PyTorch: pip install torch")
        sys.exit(1)
    
    if args.model_path is None:
        print("\n❌ AI mode requires --model-path to a trained checkpoint")
        print("   Example: python main.py --mode ai --model-path models/topopt_unet.pt")
        sys.exit(1)
    
    # Create problem definition
    resolution = args.resolution
    domain = create_bracket_domain(
        size_mm=(120, 60, 80),
        resolution_mm=resolution
    )
    load_case = create_brk_a_01_static_case_1(domain.shape, resolution)
    
    print(f"\nDomain: {domain.shape}")
    print(f"Model: {args.model_path}")
    
    try:
        optimizer = AIOptimizer(model_path=args.model_path)
        density = optimizer.predict(domain, [load_case])
        
        # Save results
        output_dir = Path("data") / run_id
        output_dir.mkdir(parents=True, exist_ok=True)
        np.save(output_dir / "density_field.npy", density)
        
        metadata = {
            "case": "BRK-A-01",
            "mode": "ai",
            "run_id": run_id,
            "provenance": "python_prototype",
            "model_path": str(args.model_path),
            "resolution_mm": resolution,
            "domain_shape": list(domain.shape),
        }
        with open(output_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\nResults saved to {output_dir}/")
        return metadata
        
    except NotImplementedError as e:
        print(f"\n⚠️  {e}")
        print("\nAI model not yet trained. Use --mode numerical for now.")
        sys.exit(1)


def main():
    """Main entry point with mode selection."""
    parser = argparse.ArgumentParser(
        description="AM Topology Optimization - Unified Entry Point",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            Examples:
            # Run numerical solver (default, Python SIMP)
            python main.py --mode numerical --resolution 2.0 --max-iter 50

            # Run AI inference (requires trained model)
            python main.py --mode ai --model-path models/topopt_unet.pt
            
            # For C++ HPC solver, use the standalone executable:
            ./cpp_engine/build/am_topopt --resolution 2.0 --iterations 100
                    """
    )
    
    # Mode selection
    parser.add_argument(
        "--mode", "-m",
        choices=["numerical", "ai"],
        default="numerical",
        help="Execution mode: 'numerical' (Python SIMP) or 'ai' (neural network)"
    )
    
    # Common parameters
    parser.add_argument(
        "--resolution", "-r",
        type=float,
        default=1.0,
        help="Voxel resolution in mm (default: 1.0)"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="data/brk_a_01",
        help="Output directory for results"
    )
    
    # Numerical-specific parameters
    parser.add_argument(
        "--volume-fraction", "-vf",
        type=float,
        default=0.25,
        help="Target volume fraction (default: 0.25)"
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=50,
        help="Maximum iterations for numerical solver (default: 50)"
    )
    
    # AI-specific parameters
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to trained model checkpoint (required for --mode ai)"
    )
    
    args = parser.parse_args()
    
    # Dispatch to appropriate runner
    if args.mode == "numerical":
        run_numerical(args)
    elif args.mode == "ai":
        run_ai(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
