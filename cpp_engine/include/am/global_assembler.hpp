/**
 * @file global_assembler.hpp
 * @brief Global stiffness matrix assembly with OpenMP parallelization
 * 
 * This module implements thread-safe global FEM matrix assembly using
 * the Thread-Local Accumulation pattern to avoid race conditions.
 * 
 * Key features:
 * - Per-thread triplet vectors (no locks in hot loop)
 * - SIMP density interpolation
 * - Efficient memory pre-allocation
 * - Eigen SparseMatrix output (CSR format)
 * 
 * @author AM Project
 * @date January 2026
 */

#ifndef AM_GLOBAL_ASSEMBLER_HPP
#define AM_GLOBAL_ASSEMBLER_HPP

#include <Eigen/Sparse>
#include <vector>
#include <array>
#include <cmath>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "am/element_kernel.hpp"

namespace am {
namespace fem {

// =============================================================================
// Type Definitions
// =============================================================================

/// Sparse matrix type (CSR format for efficient linear algebra)
using SparseMatrix = Eigen::SparseMatrix<Scalar, Eigen::RowMajor>;

/// Triplet for sparse matrix construction
using Triplet = Eigen::Triplet<Scalar>;


// =============================================================================
// SIMP Parameters
// =============================================================================

/**
 * @brief Parameters for SIMP (Solid Isotropic Material with Penalization)
 */
struct SIMPParams {
    Scalar penalty = 3.0;       ///< SIMP penalty exponent (p)
    Scalar E_min = 1e-6;        ///< Minimum Young's modulus ratio (1e-6 for CG stability)
    Scalar rho_min = 0.001;     ///< Minimum density threshold
    
    SIMPParams() = default;
    SIMPParams(Scalar p, Scalar e_min, Scalar r_min = 0.001)
        : penalty(p), E_min(e_min), rho_min(r_min) {}
};


// =============================================================================
// Global Assembler
// =============================================================================

/**
 * @brief Thread-safe global stiffness matrix assembler
 * 
 * Assembles the global stiffness matrix K from element matrices using
 * OpenMP parallelization with thread-local storage to avoid race conditions.
 * 
 * Algorithm:
 * 1. Each thread fills its own local triplet vector
 * 2. Element matrices are computed and scaled by SIMP density
 * 3. Triplets are merged at the end (single critical section)
 * 4. Eigen builds sparse matrix and sums duplicate entries
 * 
 * Performance:
 * - No locks in hot loop (linear scaling with threads)
 * - Memory pre-allocated to avoid reallocations
 * - Expected: <1s for 500K elements with 10 threads
 * 
 * @example
 * ```cpp
 * GlobalAssembler assembler;
 * auto K = assembler.assemble(
 *     120, 60, 80,           // Grid dimensions
 *     0.001,                  // Element size (1mm in meters)
 *     kernel,                 // ElementKernel
 *     density,                // Density field (flat vector)
 *     domain_mask,            // Domain mask (-1=void, 0=design, 1=fixed)
 *     simp_params             // SIMP parameters
 * );
 * ```
 */
class GlobalAssembler {
public:
    /**
     * @brief Assemble global stiffness matrix
     * 
     * @param nx Number of elements in X direction
     * @param ny Number of elements in Y direction
     * @param nz Number of elements in Z direction
     * @param element_size Element edge length [m]
     * @param kernel ElementKernel for computing Ke
     * @param density Density field (nx*ny*nz), values in [0,1]
     * @param domain_mask Domain mask: -1=void, 0=design, 1=fixed (optional)
     * @param simp SIMP parameters
     * @return Sparse stiffness matrix K
     */
    SparseMatrix assemble(
        int nx, int ny, int nz,
        Scalar element_size,
        const ElementKernel& kernel,
        const std::vector<Scalar>& density,
        const std::vector<int8_t>& domain_mask,
        const SIMPParams& simp = SIMPParams()
    ) const;
    
    /**
     * @brief Simplified assemble without domain mask (all elements active)
     */
    SparseMatrix assemble(
        int nx, int ny, int nz,
        Scalar element_size,
        const ElementKernel& kernel,
        const std::vector<Scalar>& density,
        const SIMPParams& simp = SIMPParams()
    ) const;
    
    /**
     * @brief Get number of DOFs for given grid dimensions
     */
    static int64_t numDofs(int nx, int ny, int nz) {
        return 3 * static_cast<int64_t>(nx + 1) * (ny + 1) * (nz + 1);
    }
    
    /**
     * @brief Get number of elements for given grid dimensions
     */
    static int64_t numElements(int nx, int ny, int nz) {
        return static_cast<int64_t>(nx) * ny * nz;
    }

private:
    /**
     * @brief Compute element DOF indices for given (i,j,k)
     * 
     * Maps local element nodes to global DOF indices.
     * Node ordering follows standard Hex8 convention.
     */
    static std::array<int64_t, 24> getDofIndices(
        int i, int j, int k,
        int nx, int ny, int nz
    );
    
    /**
     * @brief Flatten (i,j,k) to linear element index
     */
    static int64_t flatIndex(int i, int j, int k, int nx, int ny, int nz) {
        return static_cast<int64_t>(i) * ny * nz + j * nz + k;
    }
    
    /**
     * @brief Compute SIMP-interpolated Young's modulus ratio
     * 
     * E_eff/E0 = E_min + rho^p * (1 - E_min)
     */
    static Scalar simpInterpolation(Scalar rho, const SIMPParams& simp) {
        return simp.E_min + std::pow(rho, simp.penalty) * (1.0 - simp.E_min);
    }
};


// =============================================================================
// Assembly Statistics
// =============================================================================

/**
 * @brief Statistics from assembly process
 */
struct AssemblyStats {
    int64_t n_elements;           ///< Total elements in grid
    int64_t n_active_elements;    ///< Elements actually assembled
    int64_t n_dofs;               ///< Total degrees of freedom
    int64_t n_triplets;           ///< Number of triplets generated
    int64_t nnz;                  ///< Non-zeros in final matrix
    double assembly_time_s;       ///< Time for assembly [s]
    double matrix_build_time_s;   ///< Time for setFromTriplets [s]
    int n_threads;                ///< Number of OpenMP threads used
};


}  // namespace fem
}  // namespace am

#endif  // AM_GLOBAL_ASSEMBLER_HPP
