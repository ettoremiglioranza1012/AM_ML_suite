/**
 * @file element_kernel.hpp
 * @brief Element stiffness matrix computation for 8-node hexahedral elements
 * 
 * This module provides thread-safe, SIMD-optimized computation of element
 * stiffness matrices for 3D FEM topology optimization.
 * 
 * Design Principles:
 * - Pure functions / const methods for OpenMP thread safety
 * - Fixed-size Eigen matrices for stack allocation (no heap contention)
 * - Pre-computed constitutive matrix for efficiency
 * - SIMD vectorization via Eigen
 * 
 * @author AM Project
 * @date January 2026
 */

#ifndef AM_ELEMENT_KERNEL_HPP
#define AM_ELEMENT_KERNEL_HPP

#include <Eigen/Dense>
#include <cmath>
#include <array>

namespace am {
namespace fem {

// =============================================================================
// Type Definitions
// =============================================================================

/// Scalar type for all computations (double for accuracy, float for speed)
using Scalar = double;

/// 24x24 element stiffness matrix (8 nodes × 3 DOF)
using Matrix24 = Eigen::Matrix<Scalar, 24, 24>;

/// 6x6 constitutive matrix (Voigt notation)
using Matrix6 = Eigen::Matrix<Scalar, 6, 6>;

/// 6x24 strain-displacement matrix
using Matrix6x24 = Eigen::Matrix<Scalar, 6, 24>;

/// 8x3 shape function derivatives
using Matrix8x3 = Eigen::Matrix<Scalar, 8, 3>;

/// 3x3 matrix (Jacobian, etc.)
using Matrix3 = Eigen::Matrix<Scalar, 3, 3>;

/// 3D vector
using Vector3 = Eigen::Matrix<Scalar, 3, 1>;


// =============================================================================
// Material Properties
// =============================================================================

/**
 * @brief Material properties for isotropic linear elastic material
 */
struct MaterialProperties {
    Scalar E;    ///< Young's modulus [Pa]
    Scalar nu;   ///< Poisson's ratio [-]
    Scalar rho;  ///< Density [kg/m³] (for future dynamic analysis)
    
    /// Default: Ti6Al4V properties
    static constexpr Scalar DEFAULT_E = 113.8e9;
    static constexpr Scalar DEFAULT_NU = 0.342;
    static constexpr Scalar DEFAULT_RHO = 4430.0;
    
    MaterialProperties(
        Scalar E_ = DEFAULT_E,
        Scalar nu_ = DEFAULT_NU,
        Scalar rho_ = DEFAULT_RHO
    ) : E(E_), nu(nu_), rho(rho_) {}
    
    /// Lame's first parameter
    Scalar lambda() const {
        return E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
    }
    
    /// Lame's second parameter (shear modulus)
    Scalar mu() const {
        return E / (2.0 * (1.0 + nu));
    }
};


// =============================================================================
// Element Kernel
// =============================================================================

/**
 * @brief Thread-safe element stiffness matrix kernel for 8-node hexahedron
 * 
 * This class computes the 24×24 stiffness matrix for a cubic hexahedral
 * element using 2×2×2 Gauss quadrature.
 * 
 * Thread Safety:
 * - All computation methods are const
 * - No mutable state or global variables
 * - Multiple threads can call computeStiffness() simultaneously
 * 
 * Performance:
 * - Constitutive matrix is pre-computed in constructor
 * - All matrices are fixed-size (stack allocated)
 * - Gauss points are compile-time constants
 * - SIMD vectorization via Eigen
 * 
 * @example
 * ```cpp
 * am::fem::ElementKernel kernel(113.8e9, 0.342);
 * am::fem::Matrix24 Ke;
 * 
 * #pragma omp parallel for
 * for (int i = 0; i < n_elements; ++i) {
 *     kernel.computeStiffness(element_size, Ke);  // Thread-safe
 *     // ... assemble into global K ...
 * }
 * ```
 */
class ElementKernel {
public:
    /**
     * @brief Construct kernel with material properties
     * @param E Young's modulus [Pa]
     * @param nu Poisson's ratio [-]
     */
    ElementKernel(Scalar E, Scalar nu);
    
    /**
     * @brief Construct kernel from MaterialProperties struct
     */
    explicit ElementKernel(const MaterialProperties& mat);
    
    /**
     * @brief Compute element stiffness matrix (THREAD-SAFE)
     * 
     * This is the core computational kernel. It's marked const to ensure
     * thread safety - multiple threads can call this simultaneously.
     * 
     * @param element_size Edge length of cubic element [m]
     * @param[out] Ke Output stiffness matrix (24×24)
     */
    void computeStiffness(Scalar element_size, Matrix24& Ke) const;
    
    /**
     * @brief Get pre-computed constitutive matrix
     * @return 6×6 constitutive matrix in Voigt notation
     */
    const Matrix6& constitutiveMatrix() const { return C_; }
    
    /**
     * @brief Get material properties
     */
    const MaterialProperties& material() const { return material_; }

private:
    MaterialProperties material_;
    Matrix6 C_;  ///< Pre-computed constitutive matrix
    
    /// Pre-compute constitutive matrix (called once in constructor)
    void precomputeConstitutiveMatrix();
    
    /**
     * @brief Compute strain-displacement matrix B at a Gauss point
     * 
     * @param xi Natural coordinate ξ ∈ [-1, 1]
     * @param eta Natural coordinate η ∈ [-1, 1]
     * @param zeta Natural coordinate ζ ∈ [-1, 1]
     * @param inv_a Inverse of half-element-size (1/a)
     * @param[out] B Strain-displacement matrix (6×24)
     */
    void computeBMatrix(
        Scalar xi, Scalar eta, Scalar zeta,
        Scalar inv_a,
        Matrix6x24& B
    ) const;
};


// =============================================================================
// Utility Functions
// =============================================================================

/**
 * @brief Compute element DOF indices in global system
 * 
 * Given element indices (i, j, k) in a structured grid, returns the
 * 24 global DOF indices for the element's 8 nodes.
 * 
 * @param i Element index in X direction
 * @param j Element index in Y direction
 * @param k Element index in Z direction
 * @param nx Number of elements in X
 * @param ny Number of elements in Y
 * @param nz Number of elements in Z
 * @return Array of 24 DOF indices
 */
std::array<int64_t, 24> getElementDofIndices(
    int i, int j, int k,
    int nx, int ny, int nz
);


}  // namespace fem
}  // namespace am

#endif  // AM_ELEMENT_KERNEL_HPP
