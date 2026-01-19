/**
 * @file main.cpp
 * @brief Test and validation for element kernel and global assembler
 * 
 * This executable validates the C++ FEM components:
 * 1. Element kernel: symmetry, positive semi-definiteness
 * 2. Global assembler: parallel assembly performance
 * 
 * Usage:
 *   ./am_element_test [element_size_mm]
 * 
 * Default element size: 1.0 mm
 */

#include <iostream>
#include <iomanip>
#include <chrono>
#include <cmath>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "am/element_kernel.hpp"
#include "am/global_assembler.hpp"
#include "am/linear_solver.hpp"
#include "am/topology_optimizer.hpp"

using namespace am::fem;


/**
 * @brief Validate stiffness matrix properties
 */
struct ValidationResult {
    bool symmetric = false;
    bool positive_semidefinite = false;
    double symmetry_error = 0.0;
    double min_eigenvalue = 0.0;
    double max_value = 0.0;
    double condition_estimate = 0.0;
};


ValidationResult validateStiffnessMatrix(const Matrix24& Ke) {
    ValidationResult result;
    
    // Check symmetry: ||K - K^T|| / ||K||
    Matrix24 diff = Ke - Ke.transpose();
    result.symmetry_error = diff.norm() / Ke.norm();
    result.symmetric = (result.symmetry_error < 1e-12);
    
    // Max value
    result.max_value = Ke.maxCoeff();
    
    // Eigenvalue analysis for positive semi-definiteness
    Eigen::SelfAdjointEigenSolver<Matrix24> solver(Ke);
    auto eigenvalues = solver.eigenvalues();
    result.min_eigenvalue = eigenvalues.minCoeff();
    double max_eigenvalue = eigenvalues.maxCoeff();
    
    // Allow small negative eigenvalues due to numerical precision
    result.positive_semidefinite = (result.min_eigenvalue > -1e-6 * max_eigenvalue);
    
    // Condition number estimate
    if (std::abs(result.min_eigenvalue) > 1e-15) {
        result.condition_estimate = max_eigenvalue / std::abs(result.min_eigenvalue);
    } else {
        result.condition_estimate = std::numeric_limits<double>::infinity();
    }
    
    return result;
}


/**
 * @brief Benchmark kernel performance (serial)
 */
void benchmarkKernel(const ElementKernel& kernel, Scalar element_size, int iterations) {
    Matrix24 Ke;
    
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < iterations; ++i) {
        kernel.computeStiffness(element_size, Ke);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    double us_per_call = static_cast<double>(duration.count()) / iterations;
    double calls_per_second = 1e6 / us_per_call;
    
    std::cout << "\n=== Performance Benchmark (Serial) ===" << std::endl;
    std::cout << "Iterations: " << iterations << std::endl;
    std::cout << "Time per call: " << std::fixed << std::setprecision(3) 
              << us_per_call << " µs" << std::endl;
    std::cout << "Throughput: " << std::scientific << std::setprecision(2)
              << calls_per_second << " calls/sec" << std::endl;
    
    // Estimate for full mesh
    int elements_1mm = 120 * 60 * 80;  // BRK-A-01 at 1mm resolution
    double full_mesh_time_s = elements_1mm * us_per_call / 1e6;
    std::cout << "Estimated time for " << elements_1mm << " elements (serial): "
              << std::fixed << std::setprecision(2) << full_mesh_time_s << " s" << std::endl;
}


/**
 * @brief Benchmark kernel performance with OpenMP parallelism
 */
void benchmarkKernelParallel(const ElementKernel& kernel, Scalar element_size, int n_elements) {
#ifdef _OPENMP
    int n_threads = omp_get_max_threads();
    std::cout << "\n=== Performance Benchmark (OpenMP) ===" << std::endl;
    std::cout << "Elements: " << n_elements << std::endl;
    std::cout << "Threads: " << n_threads << std::endl;
    
    // Allocate per-thread Ke matrices to avoid false sharing
    std::vector<Matrix24> Ke_per_thread(n_threads);
    
    auto start = std::chrono::high_resolution_clock::now();
    
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        Matrix24& Ke = Ke_per_thread[tid];
        
        #pragma omp for schedule(static)
        for (int i = 0; i < n_elements; ++i) {
            kernel.computeStiffness(element_size, Ke);
        }
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    double time_s = duration.count() / 1000.0;
    double elements_per_second = n_elements / time_s;
    
    std::cout << "Total time: " << std::fixed << std::setprecision(3) << time_s << " s" << std::endl;
    std::cout << "Throughput: " << std::scientific << std::setprecision(2)
              << elements_per_second << " elements/sec" << std::endl;
    std::cout << "Speedup vs serial estimate: ~" << std::fixed << std::setprecision(1)
              << n_threads << "x (ideal)" << std::endl;
#else
    std::cout << "\n=== OpenMP not available ===" << std::endl;
    std::cout << "Compile with OpenMP to enable parallel benchmark." << std::endl;
    (void)kernel; (void)element_size; (void)n_elements;
#endif
}


int main(int argc, char* argv[]) {
    std::cout << "========================================" << std::endl;
    std::cout << "AM C++ Engine - Element Kernel Test" << std::endl;
    std::cout << "========================================" << std::endl;
    
    // Parse element size from command line (default 1mm = 0.001m)
    Scalar element_size_mm = 1.0;
    if (argc > 1) {
        element_size_mm = std::stod(argv[1]);
    }
    Scalar element_size = element_size_mm / 1000.0;  // Convert to meters
    
    std::cout << "\nElement size: " << element_size_mm << " mm" << std::endl;
    
    // Create kernel with Ti6Al4V properties
    MaterialProperties material;
    std::cout << "Material: Ti6Al4V" << std::endl;
    std::cout << "  E  = " << material.E / 1e9 << " GPa" << std::endl;
    std::cout << "  ν  = " << material.nu << std::endl;
    std::cout << "  ρ  = " << material.rho << " kg/m³" << std::endl;
    
    ElementKernel kernel(material);
    
    // Show constitutive matrix
    std::cout << "\n=== Constitutive Matrix (6×6) ===" << std::endl;
    std::cout << std::scientific << std::setprecision(4);
    std::cout << kernel.constitutiveMatrix() << std::endl;
    
    // Compute stiffness matrix
    Matrix24 Ke;
    kernel.computeStiffness(element_size, Ke);
    
    // Validate
    std::cout << "\n=== Stiffness Matrix Validation ===" << std::endl;
    auto validation = validateStiffnessMatrix(Ke);
    
    std::cout << "Symmetric: " << (validation.symmetric ? "✓ YES" : "✗ NO") 
              << " (error = " << std::scientific << validation.symmetry_error << ")" << std::endl;
    std::cout << "Positive semi-definite: " 
              << (validation.positive_semidefinite ? "✓ YES" : "✗ NO")
              << " (min eigenvalue = " << validation.min_eigenvalue << ")" << std::endl;
    std::cout << "Max value: " << validation.max_value << std::endl;
    std::cout << "Condition number estimate: " << validation.condition_estimate << std::endl;
    
    // Print corner of Ke for manual inspection
    std::cout << "\n=== Stiffness Matrix (first 6×6 block) ===" << std::endl;
    std::cout << Ke.topLeftCorner<6, 6>() << std::endl;
    
    // Test DOF indices
    std::cout << "\n=== DOF Indices Test ===" << std::endl;
    auto dofs = getElementDofIndices(0, 0, 0, 10, 10, 10);
    std::cout << "Element (0,0,0) DOF indices: [";
    for (int i = 0; i < 24; ++i) {
        std::cout << dofs[i];
        if (i < 23) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    
    // Benchmark serial
    benchmarkKernel(kernel, element_size, 100000);
    
    // Benchmark parallel (simulating full BRK-A-01 mesh)
    int elements_1mm = 120 * 60 * 80;  // 576,000 elements
    benchmarkKernelParallel(kernel, element_size, elements_1mm);
    
    // =========================================================================
    // Global Assembly Benchmark
    // =========================================================================
    std::cout << "\n========================================" << std::endl;
    std::cout << "Global Assembly Benchmark" << std::endl;
    std::cout << "========================================" << std::endl;
    
    // Test grid dimensions (smaller for quick test)
    int test_nx = 30, test_ny = 15, test_nz = 20;  // 9,000 elements
    int64_t test_n_elements = static_cast<int64_t>(test_nx) * test_ny * test_nz;
    int64_t test_n_dofs = GlobalAssembler::numDofs(test_nx, test_ny, test_nz);
    
    std::cout << "Test grid: " << test_nx << " x " << test_ny << " x " << test_nz << std::endl;
    std::cout << "Elements: " << test_n_elements << std::endl;
    std::cout << "DOFs: " << test_n_dofs << std::endl;
    
    // Create uniform density field (all 1.0 for test)
    std::vector<Scalar> test_density(test_n_elements, 1.0);
    
    // SIMP parameters
    SIMPParams simp(3.0, 1e-9, 0.001);
    
    // Assembler
    GlobalAssembler assembler;
    
    // Warm-up run
    auto K_warmup = assembler.assemble(
        test_nx, test_ny, test_nz,
        element_size,
        kernel,
        test_density,
        simp
    );
    
    // Timed run
    auto assembly_start = std::chrono::high_resolution_clock::now();
    
    auto K = assembler.assemble(
        test_nx, test_ny, test_nz,
        element_size,
        kernel,
        test_density,
        simp
    );
    
    auto assembly_end = std::chrono::high_resolution_clock::now();
    auto assembly_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        assembly_end - assembly_start
    );
    
    double assembly_time_s = assembly_duration.count() / 1000.0;
    
    std::cout << "\n=== Assembly Results ===" << std::endl;
    std::cout << "Assembly time: " << std::fixed << std::setprecision(3) 
              << assembly_time_s << " s" << std::endl;
    std::cout << "Matrix size: " << K.rows() << " x " << K.cols() << std::endl;
    std::cout << "Non-zeros: " << K.nonZeros() << std::endl;
    std::cout << "Sparsity: " << std::fixed << std::setprecision(2)
              << 100.0 * (1.0 - double(K.nonZeros()) / (double(K.rows()) * K.cols()))
              << "%" << std::endl;
    
    // Validate symmetry
    bool matrix_symmetric = true;
    Scalar max_sym_error = 0.0;
    Scalar max_val = 0.0;
    for (int k_outer = 0; k_outer < K.outerSize() && k_outer < 100; ++k_outer) {
        for (SparseMatrix::InnerIterator it(K, k_outer); it; ++it) {
            Scalar val = it.value();
            if (std::abs(val) > max_val) max_val = std::abs(val);
            Scalar val_T = K.coeff(it.col(), it.row());
            Scalar err = std::abs(val - val_T);
            if (err > max_sym_error) max_sym_error = err;
        }
    }
    // Relative symmetry check (error relative to max value)
    Scalar rel_sym_error = max_sym_error / (max_val + 1e-15);
    matrix_symmetric = (rel_sym_error < 1e-12);
    std::cout << "Symmetric: " << (matrix_symmetric ? "✓ YES" : "✗ NO")
              << " (rel error = " << std::scientific << rel_sym_error << ")" << std::endl;
    
    // Estimate for full BRK-A-01 mesh (576k elements)
    double full_mesh_estimate_s = assembly_time_s * (576000.0 / test_n_elements);
    std::cout << "\nEstimated time for 576K elements: " << std::fixed << std::setprecision(2)
              << full_mesh_estimate_s << " s" << std::endl;
    
    // =========================================================================
    // Linear Solver Benchmark
    // =========================================================================
    std::cout << "\n========================================" << std::endl;
    std::cout << "Linear Solver Benchmark" << std::endl;
    std::cout << "========================================" << std::endl;
    
    // Create force vector (distributed load on top surface)
    Vector F = Vector::Zero(K.rows());
    
    // Apply load on last DOF (Z-direction of a top corner node)
    // This is DOF index (K.rows() - 1) which is the Z-component of the last node
    F(K.rows() - 1) = -1000.0;  // -1 kN downward
    
    // Collect fixed DOFs: fix bottom surface (z=0)
    std::vector<int64_t> fixed_dofs;
    int nz_nodes = test_nz + 1;  // Number of nodes in z-direction
    int ny_nodes = test_ny + 1;
    int nx_nodes = test_nx + 1;
    
    // Fix all nodes on z=0 plane (bottom surface)
    for (int iy = 0; iy <= test_ny; ++iy) {
        for (int ix = 0; ix <= test_nx; ++ix) {
            // Node index at (ix, iy, iz=0)
            int64_t node_idx = iy * nx_nodes + ix;  // z=0 plane
            // 3 DOFs per node: x, y, z
            fixed_dofs.push_back(3 * node_idx + 0);  // x
            fixed_dofs.push_back(3 * node_idx + 1);  // y
            fixed_dofs.push_back(3 * node_idx + 2);  // z
        }
    }
    
    std::cout << "Fixed DOFs (bottom surface): " << fixed_dofs.size() << std::endl;
    std::cout << "Applied load: -1000 N at top corner" << std::endl;
    
    // Configure solver
    SolverConfig solver_config;
    solver_config.tolerance = 1e-6;
    solver_config.max_iterations = 5000;
    solver_config.verbose = true;
    solver_config.penalty_factor = 1e8;
    
    LinearSolver solver(solver_config);
    
    // Solve (note: K is modified in-place by BCs)
    auto [u, stats] = solver.solveWithBCs(K, F, fixed_dofs);
    
    bool solver_ok = stats.converged;
    
    std::cout << "\n=== Solution Statistics ===" << std::endl;
    std::cout << "Converged: " << (stats.converged ? "✓ YES" : "✗ NO") << std::endl;
    std::cout << "Iterations: " << stats.iterations << std::endl;
    std::cout << "Residual: " << std::scientific << stats.residual << std::endl;
    std::cout << "Solve time: " << std::fixed << std::setprecision(3) << stats.solve_time_s << " s" << std::endl;
    std::cout << "Max displacement: " << std::scientific << stats.max_displacement << " m" << std::endl;
    std::cout << "Min displacement: " << stats.min_displacement << " m" << std::endl;
    
    // Verify fixed DOFs are zero
    Scalar max_fixed_disp = 0.0;
    for (int64_t i : fixed_dofs) {
        max_fixed_disp = std::max(max_fixed_disp, std::abs(u(i)));
    }
    bool bc_satisfied = (max_fixed_disp < 1e-10);
    std::cout << "BC satisfaction: " << (bc_satisfied ? "✓ YES" : "✗ NO")
              << " (max fixed DOF displacement = " << max_fixed_disp << ")" << std::endl;
    
    // =========================================================================
    // Topology Optimization Test
    // =========================================================================
    std::cout << "\n========================================" << std::endl;
    std::cout << "Topology Optimization Test (SIMP)" << std::endl;
    std::cout << "========================================" << std::endl;
    
    // Smaller grid for quick TO test
    int to_nx = 20, to_ny = 10, to_nz = 10;  // 2000 elements
    
    // Configure SIMP
    am::topopt::SIMPConfig simp_config;
    simp_config.volume_fraction = 0.4;      // 40% material
    simp_config.penalty = 3.0;
    simp_config.filter_radius = 1.5;
    simp_config.move_limit = 0.2;
    simp_config.max_iterations = 20;        // Quick test
    simp_config.convergence_tol = 0.01;
    simp_config.verbose = true;
    
    // Create optimizer
    am::topopt::TopologyOptimizer optimizer(
        to_nx, to_ny, to_nz,
        element_size,
        material,
        simp_config
    );
    
    // Setup BCs: fix left face (x=0), load on right face (x=nx)
    std::vector<int64_t> to_fixed_dofs;
    int to_nx_nodes = to_nx + 1;
    int to_ny_nodes = to_ny + 1;
    int to_nz_nodes = to_nz + 1;
    
    // Fix x=0 face (all DOFs)
    for (int iz = 0; iz <= to_nz; ++iz) {
        for (int iy = 0; iy <= to_ny; ++iy) {
            int64_t node_idx = static_cast<int64_t>(iz) * to_ny_nodes * to_nx_nodes +
                               static_cast<int64_t>(iy) * to_nx_nodes + 0;  // ix=0
            to_fixed_dofs.push_back(3 * node_idx + 0);
            to_fixed_dofs.push_back(3 * node_idx + 1);
            to_fixed_dofs.push_back(3 * node_idx + 2);
        }
    }
    
    // Apply distributed load on right face center (x=nx)
    int64_t to_ndofs = optimizer.numDofs();
    Eigen::VectorXd to_forces = Eigen::VectorXd::Zero(to_ndofs);
    
    // Load on center nodes of right face
    int center_iy = to_ny / 2;
    int center_iz = to_nz / 2;
    for (int diy = -1; diy <= 1; ++diy) {
        for (int diz = -1; diz <= 1; ++diz) {
            int iy = center_iy + diy;
            int iz = center_iz + diz;
            if (iy < 0 || iy > to_ny || iz < 0 || iz > to_nz) continue;
            
            int64_t node_idx = static_cast<int64_t>(iz) * to_ny_nodes * to_nx_nodes +
                               static_cast<int64_t>(iy) * to_nx_nodes + to_nx;  // ix=to_nx
            to_forces(3 * node_idx + 1) = -100.0;  // Y-direction load (bending)
        }
    }
    
    std::cout << "Fixed DOFs: " << to_fixed_dofs.size() << std::endl;
    std::cout << "Total load: " << to_forces.sum() << " N" << std::endl;
    
    // Initial density (uniform at volume fraction)
    std::vector<Scalar> initial_density(optimizer.numElements(), simp_config.volume_fraction);
    
    // Run optimization
    auto to_result = optimizer.run(initial_density, to_forces, to_fixed_dofs);
    
    // Validate: compliance should decrease monotonically (mostly)
    bool compliance_decreasing = true;
    int increase_count = 0;
    for (size_t i = 1; i < to_result.compliance_history.size(); ++i) {
        if (to_result.compliance_history[i] > to_result.compliance_history[i-1] * 1.01) {
            increase_count++;
        }
    }
    // Allow up to 2 small increases due to OC oscillations
    compliance_decreasing = (increase_count <= 2);
    
    std::cout << "\n=== Optimization Validation ===" << std::endl;
    std::cout << "Compliance trend: " << (compliance_decreasing ? "✓ Generally decreasing" : "✗ Issues detected") << std::endl;
    std::cout << "Initial compliance: " << to_result.compliance_history.front() << std::endl;
    std::cout << "Final compliance: " << to_result.compliance_history.back() << std::endl;
    std::cout << "Reduction: " << std::fixed << std::setprecision(1)
              << 100.0 * (1.0 - to_result.compliance_history.back() / to_result.compliance_history.front())
              << "%" << std::endl;
    
    bool to_ok = compliance_decreasing && to_result.iterations > 0;
    
    // Final status
    std::cout << "\n========================================" << std::endl;
    if (validation.symmetric && validation.positive_semidefinite && matrix_symmetric && solver_ok && bc_satisfied && to_ok) {
        std::cout << "✓ All validations PASSED" << std::endl;
        return 0;
    } else {
        std::cout << "✗ Some validations FAILED" << std::endl;
        return 1;
    }
}
