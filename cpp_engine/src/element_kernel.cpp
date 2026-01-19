/**
 * @file element_kernel.cpp
 * @brief Implementation of element stiffness matrix computation
 * 
 * Ported from Python: src/am/numerical/fem.py::get_element_stiffness_matrix()
 * 
 * Key optimizations:
 * - Pre-computed constitutive matrix
 * - Fixed-size Eigen matrices (stack allocation, SIMD)
 * - No dynamic memory allocation in hot path
 * - Const methods for thread safety
 */

#include "am/element_kernel.hpp"

namespace am {
namespace fem {

// =============================================================================
// Constants
// =============================================================================

namespace {

/// Gauss point coordinate (1/√3)
constexpr Scalar GP = 0.5773502691896257;  // 1.0 / std::sqrt(3.0)

/// 2×2×2 Gauss quadrature points (pre-computed at compile time)
constexpr Scalar GAUSS_POINTS[8][3] = {
    {-GP, -GP, -GP},
    {+GP, -GP, -GP},
    {+GP, +GP, -GP},
    {-GP, +GP, -GP},
    {-GP, -GP, +GP},
    {+GP, -GP, +GP},
    {+GP, +GP, +GP},
    {-GP, +GP, +GP}
};

/// Local node coordinates in reference element [-1, 1]³
constexpr Scalar NODE_COORDS[8][3] = {
    {-1, -1, -1},
    {+1, -1, -1},
    {+1, +1, -1},
    {-1, +1, -1},
    {-1, -1, +1},
    {+1, -1, +1},
    {+1, +1, +1},
    {-1, +1, +1}
};

}  // anonymous namespace


// =============================================================================
// ElementKernel Implementation
// =============================================================================

ElementKernel::ElementKernel(Scalar E, Scalar nu)
    : material_(E, nu, MaterialProperties::DEFAULT_RHO)
{
    precomputeConstitutiveMatrix();
}

ElementKernel::ElementKernel(const MaterialProperties& mat)
    : material_(mat)
{
    precomputeConstitutiveMatrix();
}

void ElementKernel::precomputeConstitutiveMatrix() {
    const Scalar E = material_.E;
    const Scalar nu = material_.nu;
    
    // 3D isotropic elasticity matrix (Voigt notation)
    // C = E / ((1+ν)(1-2ν)) * [...]
    const Scalar factor = E / ((1.0 + nu) * (1.0 - 2.0 * nu));
    const Scalar diag = 1.0 - nu;
    const Scalar off_diag = nu;
    const Scalar shear = (1.0 - 2.0 * nu) / 2.0;
    
    C_.setZero();
    
    // Normal stress terms
    C_(0, 0) = diag * factor;
    C_(1, 1) = diag * factor;
    C_(2, 2) = diag * factor;
    
    // Coupling terms
    C_(0, 1) = off_diag * factor;
    C_(0, 2) = off_diag * factor;
    C_(1, 0) = off_diag * factor;
    C_(1, 2) = off_diag * factor;
    C_(2, 0) = off_diag * factor;
    C_(2, 1) = off_diag * factor;
    
    // Shear terms
    C_(3, 3) = shear * factor;
    C_(4, 4) = shear * factor;
    C_(5, 5) = shear * factor;
}

void ElementKernel::computeBMatrix(
    Scalar xi, Scalar eta, Scalar zeta,
    Scalar inv_a,
    Matrix6x24& B
) const {
    B.setZero();
    
    // Shape function derivatives w.r.t. natural coordinates
    // Then transformed to physical coordinates via Jacobian
    // For a regular cube: dN/dx = (1/a) * dN/dξ
    
    for (int n = 0; n < 8; ++n) {
        const Scalar xi_n = NODE_COORDS[n][0];
        const Scalar eta_n = NODE_COORDS[n][1];
        const Scalar zeta_n = NODE_COORDS[n][2];
        
        // dN/dξ, dN/dη, dN/dζ (shape function derivatives in natural coords)
        const Scalar dN_dxi = 0.125 * xi_n * (1.0 + eta_n * eta) * (1.0 + zeta_n * zeta);
        const Scalar dN_deta = 0.125 * eta_n * (1.0 + xi_n * xi) * (1.0 + zeta_n * zeta);
        const Scalar dN_dzeta = 0.125 * zeta_n * (1.0 + xi_n * xi) * (1.0 + eta_n * eta);
        
        // For regular cube: J = a*I, so dN/dx = (1/a) * dN/dξ
        const Scalar dN_dx = inv_a * dN_dxi;
        const Scalar dN_dy = inv_a * dN_deta;
        const Scalar dN_dz = inv_a * dN_dzeta;
        
        // Fill B matrix columns for node n
        const int col = 3 * n;
        
        // ε_xx = du/dx
        B(0, col) = dN_dx;
        
        // ε_yy = dv/dy
        B(1, col + 1) = dN_dy;
        
        // ε_zz = dw/dz
        B(2, col + 2) = dN_dz;
        
        // γ_xy = du/dy + dv/dx
        B(3, col) = dN_dy;
        B(3, col + 1) = dN_dx;
        
        // γ_yz = dv/dz + dw/dy
        B(4, col + 1) = dN_dz;
        B(4, col + 2) = dN_dy;
        
        // γ_xz = du/dz + dw/dx
        B(5, col) = dN_dz;
        B(5, col + 2) = dN_dx;
    }
}

void ElementKernel::computeStiffness(Scalar element_size, Matrix24& Ke) const {
    // Reset output matrix
    Ke.setZero();
    
    const Scalar a = element_size / 2.0;  // Half-edge length
    const Scalar inv_a = 1.0 / a;
    const Scalar detJ = a * a * a;  // Jacobian determinant for regular cube
    
    // Temporary B matrix (stack allocated, no heap)
    Matrix6x24 B;
    
    // 2×2×2 Gauss quadrature
    for (int gp = 0; gp < 8; ++gp) {
        const Scalar xi = GAUSS_POINTS[gp][0];
        const Scalar eta = GAUSS_POINTS[gp][1];
        const Scalar zeta = GAUSS_POINTS[gp][2];
        
        // Compute B matrix at this Gauss point
        computeBMatrix(xi, eta, zeta, inv_a, B);
        
        // Ke += B^T * C * B * detJ
        // Eigen handles SIMD vectorization automatically
        Ke.noalias() += B.transpose() * C_ * B * detJ;
    }
}


// =============================================================================
// Utility Functions
// =============================================================================

std::array<int64_t, 24> getElementDofIndices(
    int i, int j, int k,
    int nx, int ny, int nz
) {
    std::array<int64_t, 24> dof_indices;
    
    // Node offsets for 8-node hexahedron
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
        
        // Global node index
        const int64_t node_idx = ni * ny1 * nz1 + nj * nz1 + nk;
        
        // 3 DOFs per node
        dof_indices[3 * n] = 3 * node_idx;
        dof_indices[3 * n + 1] = 3 * node_idx + 1;
        dof_indices[3 * n + 2] = 3 * node_idx + 2;
    }
    
    return dof_indices;
}


}  // namespace fem
}  // namespace am
