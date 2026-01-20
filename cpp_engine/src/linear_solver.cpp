/**
 * @file linear_solver.cpp
 * @brief Implementation of iterative linear solver with penalty BCs
 */

#include "am/linear_solver.hpp"
#include <algorithm>
#include <cmath>

namespace am {
namespace fem {

// =============================================================================
// Helper Functions
// =============================================================================

Scalar LinearSolver::findMaxDiagonal(const SparseMatrix& K) {
    Scalar max_diag = 0.0;
    
    for (int i = 0; i < K.outerSize(); ++i) {
        Scalar diag_val = K.coeff(i, i);
        if (std::abs(diag_val) > max_diag) {
            max_diag = std::abs(diag_val);
        }
    }
    
    return max_diag;
}


// =============================================================================
// Boundary Conditions
// =============================================================================

void LinearSolver::applyBoundaryConditions(
    SparseMatrix& K,
    Vector& F,
    const std::vector<int64_t>& fixed_dofs,
    const std::vector<Scalar>& prescribed_values
) const {
    if (fixed_dofs.empty()) {
        return;
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    
    // Find maximum diagonal for scaling penalty
    Scalar max_diag = findMaxDiagonal(K);
    Scalar penalty = max_diag * config_.penalty_factor;
    
    if (config_.verbose) {
        std::cout << "Applying BCs: " << fixed_dofs.size() << " fixed DOFs" << std::endl;
        std::cout << "  Max diagonal: " << max_diag << std::endl;
        std::cout << "  Penalty factor: " << penalty << std::endl;
    }
    
    // Check if we have prescribed values
    bool has_values = !prescribed_values.empty();
    if (has_values && prescribed_values.size() != fixed_dofs.size()) {
        throw std::invalid_argument(
            "prescribed_values size must match fixed_dofs size"
        );
    }
    
    // Apply penalty method
    for (size_t idx = 0; idx < fixed_dofs.size(); ++idx) {
        int64_t i = fixed_dofs[idx];
        
        if (i < 0 || i >= K.rows()) {
            throw std::out_of_range(
                "Fixed DOF index out of range: " + std::to_string(i)
            );
        }
        
        // Get prescribed value (default 0)
        Scalar u_prescribed = has_values ? prescribed_values[idx] : 0.0;
        
        // Apply penalty to diagonal
        K.coeffRef(i, i) += penalty;
        
        // Apply penalty to RHS
        F(i) = u_prescribed * penalty;
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    if (config_.verbose) {
        std::cout << "  BC application time: " << duration.count() / 1000.0 << " ms" << std::endl;
    }
}


void LinearSolver::applyBoundaryConditions(
    SparseMatrix& K,
    Vector& F,
    const std::vector<int>& fixed_dofs,
    Scalar prescribed_value
) const {
    // Convert to int64_t
    std::vector<int64_t> dofs_64(fixed_dofs.begin(), fixed_dofs.end());
    
    // Create uniform prescribed values
    std::vector<Scalar> values;
    if (prescribed_value != 0.0) {
        values.assign(fixed_dofs.size(), prescribed_value);
    }
    
    applyBoundaryConditions(K, F, dofs_64, values);
}


// =============================================================================
// Solve
// =============================================================================

std::pair<Vector, SolveStats> LinearSolver::solve(
    const SparseMatrix& K,
    const Vector& F
) const {
    SolveStats stats;
    
    if (config_.verbose) {
        std::cout << "Solving linear system..." << std::endl;
        std::cout << "  Matrix size: " << K.rows() << " x " << K.cols() << std::endl;
        std::cout << "  Non-zeros: " << K.nonZeros() << std::endl;
        std::cout << "  Tolerance: " << config_.tolerance << std::endl;
        std::cout << "  Max iterations: " << config_.max_iterations << std::endl;
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    
    // Configure Conjugate Gradient solver with Incomplete Cholesky preconditioner
    // IC is much more effective than Diagonal (Jacobi) for ill-conditioned FEM matrices
    // Use Lower|Upper to exploit symmetry
    Eigen::ConjugateGradient<
        SparseMatrix,
        Eigen::Lower | Eigen::Upper,
        Eigen::IncompleteCholesky<Scalar>
    > cg;
    
    cg.setTolerance(config_.tolerance);
    cg.setMaxIterations(config_.max_iterations);
    
    // Compute factorization (for CG, this just analyzes pattern)
    cg.compute(K);
    
    if (cg.info() != Eigen::Success) {
        std::cerr << "ERROR: CG decomposition failed!" << std::endl;
        stats.converged = false;
        return {Vector::Zero(F.size()), stats};
    }
    
    // Solve
    Vector u = cg.solve(F);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    // Collect statistics
    stats.iterations = static_cast<int>(cg.iterations());
    stats.residual = cg.error();
    stats.solve_time_s = duration.count() / 1000.0;
    stats.converged = (cg.info() == Eigen::Success);
    stats.max_displacement = u.maxCoeff();
    stats.min_displacement = u.minCoeff();
    
    if (config_.verbose) {
        std::cout << "  Iterations: " << stats.iterations << std::endl;
        std::cout << "  Residual: " << stats.residual << std::endl;
        std::cout << "  Solve time: " << stats.solve_time_s << " s" << std::endl;
        std::cout << "  Converged: " << (stats.converged ? "YES" : "NO") << std::endl;
    }
    
    if (!stats.converged) {
        std::cerr << "WARNING: CG solver did not converge!" << std::endl;
        std::cerr << "  Iterations: " << stats.iterations << std::endl;
        std::cerr << "  Residual: " << stats.residual << std::endl;
        std::cerr << "  Consider increasing max_iterations or checking matrix conditioning." << std::endl;
    }
    
    return {u, stats};
}


std::pair<Vector, SolveStats> LinearSolver::solveWithBCs(
    SparseMatrix& K,
    Vector& F,
    const std::vector<int64_t>& fixed_dofs,
    const std::vector<Scalar>& prescribed_values
) const {
    auto bc_start = std::chrono::high_resolution_clock::now();
    
    applyBoundaryConditions(K, F, fixed_dofs, prescribed_values);
    
    auto bc_end = std::chrono::high_resolution_clock::now();
    auto bc_duration = std::chrono::duration_cast<std::chrono::microseconds>(bc_end - bc_start);
    
    auto [u, stats] = solve(K, F);
    stats.bc_time_s = bc_duration.count() / 1e6;
    
    return {u, stats};
}


}  // namespace fem
}  // namespace am
