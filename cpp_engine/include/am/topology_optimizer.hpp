/**
 * @file topology_optimizer.hpp
 * @brief SIMP-based Topology Optimization Loop
 * 
 * Orchestrates the optimization loop using:
 * - ElementKernel for Ke computation
 * - GlobalAssembler for parallel K assembly
 * - LinearSolver for K*u=F solution
 * 
 * Implements:
 * - Sensitivity computation (parallel)
 * - Density filter (parallel)
 * - Optimality Criteria (OC) update with bisection
 */

#pragma once

#include <vector>
#include <array>
#include <functional>
#include <chrono>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <numeric>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include "am/element_kernel.hpp"
#include "am/global_assembler.hpp"
#include "am/linear_solver.hpp"

namespace am {
namespace topopt {

using Scalar = double;
using Vector = Eigen::VectorXd;
using Matrix24 = Eigen::Matrix<Scalar, 24, 24>;


/**
 * @brief Configuration for SIMP optimization
 */
struct SIMPConfig {
    Scalar volume_fraction = 0.25;      ///< Target volume fraction (e.g., 0.25 = 25%)
    Scalar penalty = 3.0;               ///< SIMP penalty exponent (p)
    Scalar filter_radius = 2.0;         ///< Filter radius in elements
    Scalar move_limit = 0.2;            ///< Maximum density change per iteration
    Scalar rho_min = 0.01;              ///< Minimum density (avoids singularity)
    Scalar E_min = 1e-9;                ///< Minimum modulus for void elements
    int max_iterations = 100;           ///< Maximum optimization iterations
    Scalar convergence_tol = 0.01;      ///< Convergence tolerance (max density change)
    bool verbose = true;                ///< Print progress
};


/**
 * @brief Result of a single optimization iteration
 */
struct IterationResult {
    double compliance = 0.0;            ///< Total compliance
    double volume_fraction = 0.0;       ///< Current volume fraction
    double max_density_change = 0.0;    ///< Maximum density change
    int solver_iterations = 0;          ///< CG iterations for solve
    double assembly_time_s = 0.0;       ///< Assembly time
    double solve_time_s = 0.0;          ///< Solve time
    double sensitivity_time_s = 0.0;    ///< Sensitivity time
    double update_time_s = 0.0;         ///< Density update time
};


/**
 * @brief Result of full optimization run
 */
struct OptimizationResult {
    std::vector<Scalar> final_density;          ///< Final density field
    std::vector<double> compliance_history;     ///< Compliance per iteration
    std::vector<double> volume_history;         ///< Volume fraction per iteration
    int iterations = 0;                         ///< Iterations performed
    bool converged = false;                     ///< Convergence flag
    double elapsed_time_s = 0.0;                ///< Total time
};


/**
 * @brief Callback function type for iteration monitoring
 * 
 * @param iteration Current iteration number
 * @param density Current density field
 * @param result Iteration result (compliance, etc.)
 * @return true to continue, false to stop early
 */
using IterationCallback = std::function<bool(
    int iteration,
    const std::vector<Scalar>& density,
    const IterationResult& result
)>;


/**
 * @brief Precomputed filter weights for efficient filtering
 */
struct FilterWeight {
    int64_t neighbor_idx;   ///< Linear index of neighbor element
    Scalar weight;          ///< Filter weight (r_min - distance)
};


/**
 * @brief SIMP Topology Optimizer
 * 
 * Orchestrates the full optimization loop with parallel sensitivity
 * computation and density filtering.
 */
class TopologyOptimizer {
public:
    /**
     * @brief Construct optimizer for given domain
     * 
     * @param nx Number of elements in X direction
     * @param ny Number of elements in Y direction
     * @param nz Number of elements in Z direction
     * @param element_size Element edge length [m]
     * @param material Material properties
     * @param config SIMP configuration
     */
    TopologyOptimizer(
        int nx, int ny, int nz,
        Scalar element_size,
        const fem::MaterialProperties& material,
        const SIMPConfig& config = SIMPConfig()
    );
    
    /**
     * @brief Run a single optimization iteration
     * 
     * Performs: assembly -> solve -> sensitivity -> filter -> OC update
     * 
     * @param density Current density field (modified in-place)
     * @param forces Force vector (constant across iterations)
     * @param fixed_dofs Fixed DOF indices for BCs
     * @return IterationResult with compliance and change metrics
     */
    IterationResult runIteration(
        std::vector<Scalar>& density,
        const Vector& forces,
        const std::vector<int64_t>& fixed_dofs
    );
    
    /**
     * @brief Run full optimization loop
     * 
     * @param initial_density Starting density field
     * @param forces Force vector
     * @param fixed_dofs Fixed DOF indices
     * @param callback Optional callback for monitoring (can be nullptr)
     * @return OptimizationResult with final density and history
     */
    OptimizationResult run(
        std::vector<Scalar> initial_density,
        const Vector& forces,
        const std::vector<int64_t>& fixed_dofs,
        IterationCallback callback = nullptr
    );
    
    // Accessors
    int nx() const { return nx_; }
    int ny() const { return ny_; }
    int nz() const { return nz_; }
    int64_t numElements() const { return n_elements_; }
    int64_t numDofs() const { return n_dofs_; }
    
    const SIMPConfig& config() const { return config_; }
    
private:
    // Domain
    int nx_, ny_, nz_;
    int64_t n_elements_;
    int64_t n_dofs_;
    Scalar element_size_;
    
    // Configuration
    SIMPConfig config_;
    
    // FEM components
    fem::ElementKernel kernel_;
    fem::GlobalAssembler assembler_;
    fem::LinearSolver solver_;
    
    // Precomputed data
    Matrix24 Ke0_;  ///< Base element stiffness (rho=1)
    std::vector<std::vector<FilterWeight>> filter_weights_;  ///< Per-element filter neighbors
    
    // =========================================================================
    // Private Methods
    // =========================================================================
    
    /**
     * @brief Build filter weight structure for all elements
     * 
     * Precomputes neighbor indices and weights for O(1) lookup during filtering.
     */
    void buildFilter();
    
    /**
     * @brief Convert 3D index to linear element index
     */
    int64_t toLinearIndex(int ix, int iy, int iz) const {
        return static_cast<int64_t>(iz) * ny_ * nx_ + 
               static_cast<int64_t>(iy) * nx_ + 
               static_cast<int64_t>(ix);
    }
    
    /**
     * @brief Convert linear index to 3D indices
     */
    void toGridIndex(int64_t idx, int& ix, int& iy, int& iz) const {
        iz = static_cast<int>(idx / (nx_ * ny_));
        int rem = static_cast<int>(idx % (nx_ * ny_));
        iy = rem / nx_;
        ix = rem % nx_;
    }
    
    /**
     * @brief Compute sensitivities for all elements (parallel)
     * 
     * @param u Global displacement vector
     * @param density Current density field
     * @return Sensitivity array (one per element)
     */
    std::vector<Scalar> computeSensitivities(
        const Vector& u,
        const std::vector<Scalar>& density
    ) const;
    
    /**
     * @brief Apply density filter to sensitivities (parallel)
     * 
     * @param dc Raw sensitivities
     * @param density Current density field
     * @return Filtered sensitivities
     */
    std::vector<Scalar> filterSensitivities(
        const std::vector<Scalar>& dc,
        const std::vector<Scalar>& density
    ) const;
    
    /**
     * @brief Update densities using Optimality Criteria
     * 
     * @param density Current density (modified in-place)
     * @param dc_filtered Filtered sensitivities
     * @return Maximum density change
     */
    Scalar updateDensitiesOC(
        std::vector<Scalar>& density,
        const std::vector<Scalar>& dc_filtered
    ) const;
    
    /**
     * @brief Compute total compliance from displacements
     */
    Scalar computeCompliance(
        const Vector& u,
        const Vector& forces
    ) const {
        return u.dot(forces);
    }
};


}  // namespace topopt
}  // namespace am
