/**
 * @file topology_optimizer.cpp
 * @brief Implementation of SIMP Topology Optimizer
 */

#include "am/topology_optimizer.hpp"
#include <iomanip>

namespace am {
namespace topopt {

// =============================================================================
// Constructor
// =============================================================================

TopologyOptimizer::TopologyOptimizer(
    int nx, int ny, int nz,
    Scalar element_size,
    const fem::MaterialProperties& material,
    const SIMPConfig& config
) : nx_(nx), ny_(ny), nz_(nz),
    element_size_(element_size),
    config_(config),
    kernel_(material),
    assembler_(),
    solver_()
{
    n_elements_ = static_cast<int64_t>(nx_) * ny_ * nz_;
    n_dofs_ = fem::GlobalAssembler::numDofs(nx_, ny_, nz_);
    
    // Precompute base element stiffness (rho = 1)
    kernel_.computeStiffness(element_size_, Ke0_);
    
    // Configure solver
    fem::SolverConfig solver_config;
    solver_config.tolerance = 1e-6;
    solver_config.max_iterations = 5000;
    solver_config.penalty_factor = 1e8;
    solver_config.verbose = false;
    solver_ = fem::LinearSolver(solver_config);
    
    // Build filter structure
    buildFilter();
    
    if (config_.verbose) {
        std::cout << "TopologyOptimizer initialized:" << std::endl;
        std::cout << "  Grid: " << nx_ << " x " << ny_ << " x " << nz_ << std::endl;
        std::cout << "  Elements: " << n_elements_ << std::endl;
        std::cout << "  DOFs: " << n_dofs_ << std::endl;
        std::cout << "  Filter radius: " << config_.filter_radius << std::endl;
        std::cout << "  Volume fraction: " << config_.volume_fraction << std::endl;
    }
}


// =============================================================================
// Filter Construction
// =============================================================================

void TopologyOptimizer::buildFilter() {
    Scalar r = config_.filter_radius;
    int r_int = static_cast<int>(std::ceil(r));
    
    filter_weights_.resize(n_elements_);
    
    auto start = std::chrono::high_resolution_clock::now();
    
    #pragma omp parallel for schedule(dynamic)
    for (int64_t e = 0; e < n_elements_; ++e) {
        int ix, iy, iz;
        toGridIndex(e, ix, iy, iz);
        
        std::vector<FilterWeight>& neighbors = filter_weights_[e];
        
        // Search within bounding box
        for (int dz = -r_int; dz <= r_int; ++dz) {
            int nz = iz + dz;
            if (nz < 0 || nz >= nz_) continue;
            
            for (int dy = -r_int; dy <= r_int; ++dy) {
                int ny = iy + dy;
                if (ny < 0 || ny >= ny_) continue;
                
                for (int dx = -r_int; dx <= r_int; ++dx) {
                    int nx = ix + dx;
                    if (nx < 0 || nx >= nx_) continue;
                    
                    // Compute distance
                    Scalar dist = std::sqrt(
                        static_cast<Scalar>(dx*dx + dy*dy + dz*dz)
                    );
                    
                    if (dist <= r) {
                        FilterWeight fw;
                        fw.neighbor_idx = toLinearIndex(nx, ny, nz);
                        fw.weight = std::max(Scalar(0), r - dist);
                        neighbors.push_back(fw);
                    }
                }
            }
        }
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    if (config_.verbose) {
        // Sample average neighbors
        size_t total_neighbors = 0;
        for (const auto& n : filter_weights_) {
            total_neighbors += n.size();
        }
        double avg_neighbors = static_cast<double>(total_neighbors) / n_elements_;
        
        std::cout << "  Filter build time: " << duration.count() << " ms" << std::endl;
        std::cout << "  Average neighbors: " << std::fixed << std::setprecision(1) 
                  << avg_neighbors << std::endl;
    }
}


// =============================================================================
// Sensitivity Computation
// =============================================================================

std::vector<Scalar> TopologyOptimizer::computeSensitivities(
    const Vector& u,
    const std::vector<Scalar>& density
) const {
    std::vector<Scalar> dc(n_elements_);
    Scalar p = config_.penalty;
    
    #pragma omp parallel
    {
        // Thread-local storage for element displacement
        Eigen::Matrix<Scalar, 24, 1> u_e;
        
        #pragma omp for schedule(static)
        for (int64_t e = 0; e < n_elements_; ++e) {
            int ix, iy, iz;
            toGridIndex(e, ix, iy, iz);
            
            // Get DOF indices for this element
            auto dof_indices = fem::getElementDofIndices(ix, iy, iz, nx_, ny_, nz_);
            
            // Extract element displacement
            for (int i = 0; i < 24; ++i) {
                u_e(i) = u(dof_indices[i]);
            }
            
            // Compliance energy: c_e = u_e^T * Ke0 * u_e
            Scalar ce = u_e.transpose() * Ke0_ * u_e;
            
            // Sensitivity: dc/drho = -p * rho^(p-1) * c_e
            Scalar rho = density[e];
            dc[e] = -p * std::pow(rho, p - 1) * ce;
        }
    }
    
    return dc;
}


// =============================================================================
// Sensitivity Filter
// =============================================================================

std::vector<Scalar> TopologyOptimizer::filterSensitivities(
    const std::vector<Scalar>& dc,
    const std::vector<Scalar>& density
) const {
    std::vector<Scalar> dc_filtered(n_elements_);
    
    #pragma omp parallel for schedule(static)
    for (int64_t e = 0; e < n_elements_; ++e) {
        const std::vector<FilterWeight>& neighbors = filter_weights_[e];
        
        Scalar numerator = 0.0;
        Scalar denominator = 0.0;
        
        for (const auto& fw : neighbors) {
            Scalar rho_n = density[fw.neighbor_idx];
            Scalar dc_n = dc[fw.neighbor_idx];
            
            numerator += fw.weight * rho_n * dc_n;
            denominator += fw.weight * rho_n;
        }
        
        // Avoid division by zero
        if (denominator > 1e-15) {
            dc_filtered[e] = numerator / denominator;
        } else {
            dc_filtered[e] = dc[e];
        }
    }
    
    return dc_filtered;
}


// =============================================================================
// Optimality Criteria Update
// =============================================================================

Scalar TopologyOptimizer::updateDensitiesOC(
    std::vector<Scalar>& density,
    const std::vector<Scalar>& dc_filtered
) const {
    Scalar move = config_.move_limit;
    Scalar rho_min = config_.rho_min;
    Scalar vf_target = config_.volume_fraction;
    
    // Bisection bounds for Lagrange multiplier
    Scalar l1 = 1e-20;
    Scalar l2 = 1e20;
    
    std::vector<Scalar> rho_new(n_elements_);
    Scalar max_change = 0.0;
    
    // Use precomputed design mask (set in run() from initial densities)
    // This prevents misclassifying design elements that reach rho=1.0
    int64_t n_design = n_design_elements_;
    if (n_design == 0) {
        n_design = n_elements_;  // Fallback
    }
    
    // Bisection loop
    const int max_bisect = 100;
    for (int iter = 0; iter < max_bisect; ++iter) {
        if ((l2 - l1) / (l1 + l2) < 1e-4) break;
        
        Scalar lmid = 0.5 * (l1 + l2);
        
        // Update all elements
        Scalar vol_sum = 0.0;
        
        #pragma omp parallel for reduction(+:vol_sum) schedule(static)
        for (int64_t e = 0; e < n_elements_; ++e) {
            Scalar rho = density[e];
            Scalar dc = dc_filtered[e];
            
            // Skip non-design elements using precomputed mask
            if (!design_mask_[e]) {
                rho_new[e] = rho;  // Keep unchanged
                continue;
            }
            
            // Volume sensitivity is 1 for uniform mesh
            Scalar dv = 1.0;
            
            // OC update formula: rho_new = rho * sqrt(-dc / (lmid * dv))
            // Note: dc is negative for minimization, so -dc is positive
            Scalar Be = std::sqrt(-dc / (lmid * dv + 1e-20));
            
            // Apply move limits and bounds
            Scalar rho_candidate = rho * Be;
            rho_candidate = std::max(rho_min, std::max(rho - move, 
                            std::min(1.0, std::min(rho + move, rho_candidate))));
            
            rho_new[e] = rho_candidate;
            vol_sum += rho_candidate;
        }
        
        // Check volume constraint only on design space
        Scalar current_vf = vol_sum / n_design;
        
        if (current_vf > vf_target) {
            l1 = lmid;
        } else {
            l2 = lmid;
        }
    }
    
    // Compute max change and update density
    #pragma omp parallel for reduction(max:max_change) schedule(static)
    for (int64_t e = 0; e < n_elements_; ++e) {
        Scalar change = std::abs(rho_new[e] - density[e]);
        if (change > max_change) {
            max_change = change;
        }
    }
    
    // Copy to output
    density = std::move(rho_new);
    
    return max_change;
}


// =============================================================================
// Single Iteration
// =============================================================================

IterationResult TopologyOptimizer::runIteration(
    std::vector<Scalar>& density,
    const Vector& forces,
    const std::vector<int64_t>& fixed_dofs
) {
    IterationResult result;
    auto total_start = std::chrono::high_resolution_clock::now();
    
    // -------------------------------------------------------------------------
    // 1. Assemble global stiffness matrix
    // -------------------------------------------------------------------------
    auto t0 = std::chrono::high_resolution_clock::now();
    
    fem::SIMPParams simp(config_.penalty, config_.E_min, config_.rho_min);
    
    auto K = assembler_.assemble(
        nx_, ny_, nz_,
        element_size_,
        kernel_,
        density,
        simp
    );
    
    auto t1 = std::chrono::high_resolution_clock::now();
    result.assembly_time_s = std::chrono::duration<double>(t1 - t0).count();
    
    // -------------------------------------------------------------------------
    // 2. Solve K * u = F with BCs
    // -------------------------------------------------------------------------
    Vector F = forces;  // Copy because solver modifies it
    
    auto [u, stats] = solver_.solveWithBCs(K, F, fixed_dofs);
    
    auto t2 = std::chrono::high_resolution_clock::now();
    result.solve_time_s = std::chrono::duration<double>(t2 - t1).count();
    result.solver_iterations = stats.iterations;
    
    if (!stats.converged) {
        std::cerr << "WARNING: Solver did not converge at iteration!" << std::endl;
    }
    
    // -------------------------------------------------------------------------
    // 3. Compute compliance
    // -------------------------------------------------------------------------
    result.compliance = computeCompliance(u, forces);
    
    // -------------------------------------------------------------------------
    // 4. Compute sensitivities (parallel)
    // -------------------------------------------------------------------------
    auto dc = computeSensitivities(u, density);
    
    auto t3 = std::chrono::high_resolution_clock::now();
    result.sensitivity_time_s = std::chrono::duration<double>(t3 - t2).count();
    
    // -------------------------------------------------------------------------
    // 5. Filter sensitivities (parallel)
    // -------------------------------------------------------------------------
    auto dc_filtered = filterSensitivities(dc, density);
    
    // -------------------------------------------------------------------------
    // 6. Update densities with OC
    // -------------------------------------------------------------------------
    result.max_density_change = updateDensitiesOC(density, dc_filtered);
    
    auto t4 = std::chrono::high_resolution_clock::now();
    result.update_time_s = std::chrono::duration<double>(t4 - t3).count();
    
    // -------------------------------------------------------------------------
    // 7. Compute current volume fraction (design space only)
    // -------------------------------------------------------------------------
    Scalar vol_sum = 0.0;
    int64_t design_count = 0;
    
    #pragma omp parallel for reduction(+:vol_sum, design_count)
    for (int64_t e = 0; e < n_elements_; ++e) {
        // Use design_mask if available, otherwise fallback to density-based check
        bool is_design = design_mask_.empty() ? 
            (density[e] > config_.rho_min + 1e-6 && density[e] < 0.99) :
            design_mask_[e];
        
        if (is_design) {
            vol_sum += density[e];
            ++design_count;
        }
    }
    
    // Volume fraction is average density in design space
    result.volume_fraction = (design_count > 0) ? vol_sum / design_count : vol_sum / n_elements_;
    
    return result;
}


// =============================================================================
// Full Optimization Run
// =============================================================================

OptimizationResult TopologyOptimizer::run(
    std::vector<Scalar> initial_density,
    const Vector& forces,
    const std::vector<int64_t>& fixed_dofs,
    IterationCallback callback
) {
    OptimizationResult result;
    auto start = std::chrono::high_resolution_clock::now();
    
    std::vector<Scalar> density = std::move(initial_density);
    
    // Ensure proper initialization
    if (density.size() != static_cast<size_t>(n_elements_)) {
        density.assign(n_elements_, config_.volume_fraction);
    }
    
    // ==========================================================================
    // Initialize design mask from initial densities
    // ==========================================================================
    // Elements are classified based on their INITIAL density:
    // - Fixed (non-design): rho >= 0.99 (always solid)
    // - Void (non-design): rho <= rho_min (always empty)
    // - Design space: everything else (optimizable)
    //
    // This mask is computed ONCE and remains constant during optimization,
    // preventing design elements from being misclassified as Fixed when they
    // reach rho=1.0 during the optimization process.
    // ==========================================================================
    design_mask_.resize(n_elements_);
    n_design_elements_ = 0;
    
    for (int64_t e = 0; e < n_elements_; ++e) {
        Scalar rho = density[e];
        // Design space: not fixed (rho < 0.99) and not void (rho > rho_min)
        bool is_design = (rho > config_.rho_min + 1e-6) && (rho < 0.99);
        design_mask_[e] = is_design;
        if (is_design) ++n_design_elements_;
    }
    
    if (config_.verbose) {
        int64_t n_fixed = 0, n_void = 0;
        for (int64_t e = 0; e < n_elements_; ++e) {
            if (density[e] >= 0.99) ++n_fixed;
            else if (density[e] <= config_.rho_min + 1e-6) ++n_void;
        }
        std::cout << "Design space: " << n_design_elements_ << " elements "
                  << "(Fixed: " << n_fixed << ", Void: " << n_void << ")" << std::endl;
    }
    
    if (config_.verbose) {
        std::cout << "\n========================================" << std::endl;
        std::cout << "SIMP Topology Optimization" << std::endl;
        std::cout << "========================================" << std::endl;
        std::cout << std::setw(6) << "Iter" 
                  << std::setw(14) << "Compliance" 
                  << std::setw(10) << "Vol" 
                  << std::setw(12) << "Change"
                  << std::setw(10) << "CG Iter"
                  << std::setw(10) << "Time"
                  << std::endl;
        std::cout << std::string(62, '-') << std::endl;
    }
    
    result.converged = false;
    
    for (int iter = 0; iter < config_.max_iterations; ++iter) {
        auto iter_start = std::chrono::high_resolution_clock::now();
        
        auto iter_result = runIteration(density, forces, fixed_dofs);
        
        auto iter_end = std::chrono::high_resolution_clock::now();
        double iter_time = std::chrono::duration<double>(iter_end - iter_start).count();
        
        result.compliance_history.push_back(iter_result.compliance);
        result.volume_history.push_back(iter_result.volume_fraction);
        result.iterations = iter + 1;
        
        if (config_.verbose) {
            std::cout << std::setw(6) << iter
                      << std::setw(14) << std::scientific << std::setprecision(4) 
                      << iter_result.compliance
                      << std::setw(10) << std::fixed << std::setprecision(3) 
                      << iter_result.volume_fraction
                      << std::setw(12) << std::scientific << std::setprecision(2) 
                      << iter_result.max_density_change
                      << std::setw(10) << iter_result.solver_iterations
                      << std::setw(10) << std::fixed << std::setprecision(2) 
                      << iter_time << "s"
                      << std::endl;
        }
        
        // Check callback
        if (callback) {
            bool should_continue = callback(iter, density, iter_result);
            if (!should_continue) {
                if (config_.verbose) {
                    std::cout << "Stopped by callback at iteration " << iter << std::endl;
                }
                break;
            }
        }
        
        // Check convergence
        if (iter_result.max_density_change < config_.convergence_tol) {
            result.converged = true;
            if (config_.verbose) {
                std::cout << "Converged at iteration " << iter 
                          << " (change = " << iter_result.max_density_change << ")" << std::endl;
            }
            break;
        }
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    result.elapsed_time_s = std::chrono::duration<double>(end - start).count();
    result.final_density = std::move(density);
    
    if (config_.verbose) {
        std::cout << "========================================" << std::endl;
        std::cout << "Optimization " << (result.converged ? "CONVERGED" : "completed") << std::endl;
        std::cout << "  Iterations: " << result.iterations << std::endl;
        std::cout << "  Final compliance: " << result.compliance_history.back() << std::endl;
        std::cout << "  Final volume: " << result.volume_history.back() << std::endl;
        std::cout << "  Total time: " << std::fixed << std::setprecision(2) 
                  << result.elapsed_time_s << " s" << std::endl;
        std::cout << "========================================" << std::endl;
    }
    
    return result;
}


}  // namespace topopt
}  // namespace am
