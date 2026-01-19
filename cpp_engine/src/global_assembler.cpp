/**
 * @file global_assembler.cpp
 * @brief Implementation of parallel global stiffness matrix assembly
 * 
 * Uses OpenMP with Thread-Local Accumulation pattern:
 * - Each thread has its own vector of triplets
 * - No locks in the hot loop
 * - Single merge operation at the end
 */

#include "am/global_assembler.hpp"
#include <iostream>
#include <chrono>

namespace am {
namespace fem {

// =============================================================================
// DOF Index Computation
// =============================================================================

std::array<int64_t, 24> GlobalAssembler::getDofIndices(
    int i, int j, int k,
    int nx, int ny, int nz
) {
    std::array<int64_t, 24> dof_indices;
    
    // Node offsets for 8-node hexahedron (standard FEM ordering)
    // Must match Python: src/am/numerical/fem.py::get_element_dof_indices
    constexpr int NODE_OFFSETS[8][3] = {
        {0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0},
        {0, 0, 1}, {1, 0, 1}, {1, 1, 1}, {0, 1, 1}
    };
    
    const int64_t ny1 = ny + 1;
    const int64_t nz1 = nz + 1;
    
    for (int n = 0; n < 8; ++n) {
        const int64_t ni = i + NODE_OFFSETS[n][0];
        const int64_t nj = j + NODE_OFFSETS[n][1];
        const int64_t nk = k + NODE_OFFSETS[n][2];
        
        // Global node index (i varies fastest, then j, then k)
        // Must match Python: node_idx = (i+di)*(ny+1)*(nz+1) + (j+dj)*(nz+1) + (k+dk)
        const int64_t node_idx = ni * ny1 * nz1 + nj * nz1 + nk;
        
        // 3 DOFs per node (ux, uy, uz)
        dof_indices[3 * n]     = 3 * node_idx;
        dof_indices[3 * n + 1] = 3 * node_idx + 1;
        dof_indices[3 * n + 2] = 3 * node_idx + 2;
    }
    
    return dof_indices;
}


// =============================================================================
// Assembly Implementation
// =============================================================================

SparseMatrix GlobalAssembler::assemble(
    int nx, int ny, int nz,
    Scalar element_size,
    const ElementKernel& kernel,
    const std::vector<Scalar>& density,
    const std::vector<int8_t>& domain_mask,
    const SIMPParams& simp
) const {
    const int64_t n_elements = numElements(nx, ny, nz);
    const int64_t n_dofs = numDofs(nx, ny, nz);
    
    // Validate input sizes
    if (static_cast<int64_t>(density.size()) != n_elements) {
        throw std::invalid_argument(
            "Density vector size mismatch: expected " + std::to_string(n_elements) +
            ", got " + std::to_string(density.size())
        );
    }
    
    bool use_mask = !domain_mask.empty();
    if (use_mask && static_cast<int64_t>(domain_mask.size()) != n_elements) {
        throw std::invalid_argument("Domain mask size mismatch");
    }
    
    // Get base element stiffness matrix (computed once, read by all threads)
    Matrix24 Ke0;
    kernel.computeStiffness(element_size, Ke0);
    
    // Estimate number of triplets per element (24*24 = 576)
    // Active elements contribute all 576, so reserve generously
    const int64_t triplets_per_element = 576;
    
#ifdef _OPENMP
    const int n_threads = omp_get_max_threads();
#else
    const int n_threads = 1;
#endif
    
    // Thread-local triplet vectors
    std::vector<std::vector<Triplet>> thread_triplets(n_threads);
    
    // Reserve memory for each thread (estimate ~equal distribution)
    const int64_t elements_per_thread = (n_elements + n_threads - 1) / n_threads;
    for (auto& triplets : thread_triplets) {
        triplets.reserve(elements_per_thread * triplets_per_element);
    }
    
    // Counters for active elements (atomic for thread safety)
    std::atomic<int64_t> active_count{0};
    
    // =========================================================================
    // PARALLEL REGION: Each thread fills its local triplet vector
    // =========================================================================
    
    #pragma omp parallel
    {
#ifdef _OPENMP
        const int tid = omp_get_thread_num();
#else
        const int tid = 0;
#endif
        std::vector<Triplet>& local_triplets = thread_triplets[tid];
        
        // Scaled element matrix (thread-local)
        Matrix24 Ke_scaled;
        
        // Iterate over elements (static scheduling for regular workload)
        #pragma omp for schedule(static)
        for (int i = 0; i < nx; ++i) {
            for (int j = 0; j < ny; ++j) {
                for (int k = 0; k < nz; ++k) {
                    const int64_t elem_idx = flatIndex(i, j, k, nx, ny, nz);
                    
                    // Check domain mask if provided
                    if (use_mask) {
                        int8_t mask_val = domain_mask[elem_idx];
                        if (mask_val == -1) {
                            // Void element: skip entirely
                            continue;
                        }
                    }
                    
                    // Get density
                    Scalar rho = density[elem_idx];
                    
                    // Skip very low density elements (optional optimization)
                    if (rho < simp.rho_min) {
                        continue;
                    }
                    
                    // Compute SIMP scaling factor
                    Scalar scale;
                    if (use_mask && domain_mask[elem_idx] == 1) {
                        // Fixed element (non-design): full material
                        scale = 1.0;
                    } else {
                        // Design element: SIMP interpolation
                        scale = simpInterpolation(rho, simp);
                    }
                    
                    // Scale element matrix
                    Ke_scaled = scale * Ke0;
                    
                    // Get global DOF indices for this element
                    auto dof_indices = getDofIndices(i, j, k, nx, ny, nz);
                    
                    // Add contributions to local triplet list
                    for (int ii = 0; ii < 24; ++ii) {
                        const int64_t row = dof_indices[ii];
                        for (int jj = 0; jj < 24; ++jj) {
                            const int64_t col = dof_indices[jj];
                            const Scalar val = Ke_scaled(ii, jj);
                            
                            // Only add non-negligible values
                            if (std::abs(val) > 1e-20) {
                                local_triplets.emplace_back(row, col, val);
                            }
                        }
                    }
                    
                    // Increment active element counter
                    active_count.fetch_add(1, std::memory_order_relaxed);
                }
            }
        }
    }
    // =========================================================================
    // END PARALLEL REGION
    // =========================================================================
    
    // Merge all thread-local triplets into global list
    std::vector<Triplet> all_triplets;
    
    // Calculate total size first
    size_t total_triplets = 0;
    for (const auto& triplets : thread_triplets) {
        total_triplets += triplets.size();
    }
    all_triplets.reserve(total_triplets);
    
    // Merge (sequential, but fast - just memory copies)
    for (auto& triplets : thread_triplets) {
        all_triplets.insert(
            all_triplets.end(),
            std::make_move_iterator(triplets.begin()),
            std::make_move_iterator(triplets.end())
        );
        // Free thread-local memory
        triplets.clear();
        triplets.shrink_to_fit();
    }
    
    // Build sparse matrix from triplets
    // Eigen automatically sums duplicate entries
    SparseMatrix K(n_dofs, n_dofs);
    K.setFromTriplets(all_triplets.begin(), all_triplets.end());
    
    // Ensure symmetry (K should already be symmetric, but this guarantees it)
    // K = (K + SparseMatrix(K.transpose())) * 0.5;
    
    return K;
}


// Simplified version without domain mask
SparseMatrix GlobalAssembler::assemble(
    int nx, int ny, int nz,
    Scalar element_size,
    const ElementKernel& kernel,
    const std::vector<Scalar>& density,
    const SIMPParams& simp
) const {
    // Create empty mask (will be ignored)
    std::vector<int8_t> empty_mask;
    return assemble(nx, ny, nz, element_size, kernel, density, empty_mask, simp);
}


}  // namespace fem
}  // namespace am
