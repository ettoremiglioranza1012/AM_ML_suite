/**
 * @file linear_solver.hpp
 * @brief Iterative linear solver for FEM systems with boundary conditions
 * 
 * This module provides:
 * - Penalty method for applying Dirichlet boundary conditions
 * - Preconditioned Conjugate Gradient (PCG) solver
 * - Optimized for large sparse symmetric positive-definite systems
 * 
 * Key features:
 * - No matrix reallocation when applying BCs
 * - Exploits symmetry (Lower|Upper)
 * - Diagonal preconditioning for fast convergence
 * 
 * @author AM Project
 * @date January 2026
 */

#ifndef AM_LINEAR_SOLVER_HPP
#define AM_LINEAR_SOLVER_HPP

#include <Eigen/Sparse>
#include <Eigen/IterativeLinearSolvers>
#include <vector>
#include <iostream>
#include <chrono>

namespace am {
namespace fem {

// =============================================================================
// Type Definitions
// =============================================================================

using Scalar = double;
using SparseMatrix = Eigen::SparseMatrix<Scalar, Eigen::RowMajor>;
using Vector = Eigen::VectorXd;


// =============================================================================
// Solver Statistics
// =============================================================================

/**
 * @brief Statistics from linear solve
 */
struct SolveStats {
    int iterations = 0;           ///< Number of CG iterations
    Scalar residual = 0.0;        ///< Final residual norm
    Scalar solve_time_s = 0.0;    ///< Solve time [s]
    Scalar bc_time_s = 0.0;       ///< BC application time [s]
    bool converged = false;       ///< Whether solver converged
    Scalar max_displacement = 0.0;
    Scalar min_displacement = 0.0;
};


// =============================================================================
// Solver Configuration
// =============================================================================

/**
 * @brief Configuration for linear solver
 */
struct SolverConfig {
    Scalar tolerance = 1e-8;         ///< Convergence tolerance
    int max_iterations = 10000;       ///< Maximum CG iterations
    Scalar penalty_factor = 1e10;     ///< Multiplier for penalty method
    bool verbose = false;             ///< Print solver progress
    
    SolverConfig() = default;
    SolverConfig(Scalar tol, int max_iter, bool verb = false)
        : tolerance(tol), max_iterations(max_iter), verbose(verb) {}
};


// =============================================================================
// Linear Solver
// =============================================================================

/**
 * @brief Iterative linear solver for symmetric positive-definite FEM systems
 * 
 * Uses Eigen's Conjugate Gradient solver with diagonal preconditioning.
 * Boundary conditions are applied using the penalty method, which:
 * - Preserves matrix sparsity pattern
 * - Avoids matrix reallocation
 * - Is numerically stable for well-conditioned systems
 * 
 * @example
 * ```cpp
 * LinearSolver solver;
 * 
 * // Define fixed DOFs (e.g., base nodes)
 * std::vector<int64_t> fixed_dofs = {0, 1, 2, 3, 4, 5, ...};
 * 
 * // Apply boundary conditions
 * solver.applyBoundaryConditions(K, F, fixed_dofs);
 * 
 * // Solve
 * auto [u, stats] = solver.solve(K, F);
 * std::cout << "Converged in " << stats.iterations << " iterations" << std::endl;
 * ```
 */
class LinearSolver {
public:
    /**
     * @brief Construct solver with default configuration
     */
    LinearSolver() : config_() {}
    
    /**
     * @brief Construct solver with custom configuration
     */
    explicit LinearSolver(const SolverConfig& config) : config_(config) {}
    
    /**
     * @brief Apply Dirichlet boundary conditions using penalty method
     * 
     * For each fixed DOF i:
     *   K(i,i) += penalty * max_diagonal
     *   F(i) = prescribed_value * penalty * max_diagonal
     * 
     * This forces u[i] → prescribed_value without modifying sparsity.
     * 
     * @param K Stiffness matrix (modified in-place)
     * @param F Force vector (modified in-place)
     * @param fixed_dofs Indices of DOFs to constrain
     * @param prescribed_values Values to prescribe (default: all zeros)
     */
    void applyBoundaryConditions(
        SparseMatrix& K,
        Vector& F,
        const std::vector<int64_t>& fixed_dofs,
        const std::vector<Scalar>& prescribed_values = {}
    ) const;
    
    /**
     * @brief Overload for int indices (common case)
     */
    void applyBoundaryConditions(
        SparseMatrix& K,
        Vector& F,
        const std::vector<int>& fixed_dofs,
        Scalar prescribed_value = 0.0
    ) const;
    
    /**
     * @brief Solve the linear system K * u = F
     * 
     * Uses Preconditioned Conjugate Gradient with diagonal preconditioning.
     * Matrix K must be symmetric positive-definite.
     * 
     * @param K Stiffness matrix (should have BCs already applied)
     * @param F Force vector (should have BCs already applied)
     * @return Pair of (displacement vector, solve statistics)
     */
    std::pair<Vector, SolveStats> solve(
        const SparseMatrix& K,
        const Vector& F
    ) const;
    
    /**
     * @brief Convenience method: apply BCs and solve in one call
     */
    std::pair<Vector, SolveStats> solveWithBCs(
        SparseMatrix& K,
        Vector& F,
        const std::vector<int64_t>& fixed_dofs,
        const std::vector<Scalar>& prescribed_values = {}
    ) const;
    
    /**
     * @brief Get/set solver configuration
     */
    SolverConfig& config() { return config_; }
    const SolverConfig& config() const { return config_; }

private:
    SolverConfig config_;
    
    /**
     * @brief Find maximum diagonal element in sparse matrix
     */
    static Scalar findMaxDiagonal(const SparseMatrix& K);
};


}  // namespace fem
}  // namespace am

#endif  // AM_LINEAR_SOLVER_HPP
