/**
 * @file am_topopt.cpp
 * @brief Standalone C++ topology optimization for BRK-A-01 bracket
 * 
 * This is a self-contained executable that:
 * 1. Creates the BRK-A-01 bracket geometry
 * 2. Applies boundary conditions and loads
 * 3. Runs SIMP topology optimization
 * 4. Saves results to NumPy-compatible format
 * 
 * Usage:
 *   ./am_topopt [options]
 * 
 * Options:
 *   --resolution <mm>     Grid resolution (default: 2.0)
 *   --volume <fraction>   Target volume fraction (default: 0.25)
 *   --iterations <n>      Max iterations (default: 100)
 *   --output <dir>        Output directory (default: ./output)
 *   --verbose             Enable verbose output
 */

#include <iostream>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <cmath>
#include <vector>
#include <string>
#include <sstream>
#include <filesystem>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "am/topology_optimizer.hpp"

using namespace am::topopt;
using namespace am::fem;

namespace fs = std::filesystem;

// =============================================================================
// Configuration
// =============================================================================

struct Config {
    // Geometry (BRK-A-01)
    double size_x_mm = 120.0;
    double size_y_mm = 60.0;
    double size_z_mm = 80.0;
    double resolution_mm = 2.0;
    
    // Bracket features
    double base_thickness_mm = 5.0;
    double hole_diameter_mm = 8.0;
    double hole_offset_mm = 15.0;
    double eyelet_diameter_mm = 12.0;
    double eyelet_offset_mm = 4.0;
    double eyelet_z_offset_mm = 15.0;
    double eyelet_height_mm = 20.0;
    
    // Load
    double load_force_N = 15000.0;
    
    // Optimization
    double volume_fraction = 0.25;
    int max_iterations = 100;
    double tolerance = 0.01;
    
    // SIMP parameters
    double filter_radius = 3.5;
    double E_min = 1e-4;
    double penalty = 3.0;
    double move_limit = 0.1;
    
    // Output
    std::string output_dir = "./output";
    bool verbose = true;
};

// =============================================================================
// Domain Grid
// =============================================================================

/**
 * Element types in the domain grid
 */
enum class ElementType : int8_t {
    VOID = -1,      // Hole (excluded from optimization)
    DESIGN = 0,     // Optimizable region
    FIXED = 1       // Non-design (always solid)
};

/**
 * Create BRK-A-01 bracket domain
 */
class BracketDomain {
public:
    int nx, ny, nz;
    std::vector<int8_t> grid;
    double resolution_mm;
    
    BracketDomain(const Config& cfg) 
        : resolution_mm(cfg.resolution_mm)
    {
        nx = static_cast<int>(cfg.size_x_mm / cfg.resolution_mm);
        ny = static_cast<int>(cfg.size_y_mm / cfg.resolution_mm);
        nz = static_cast<int>(cfg.size_z_mm / cfg.resolution_mm);
        
        grid.resize(nx * ny * nz, static_cast<int8_t>(ElementType::DESIGN));
        
        // 1. Base (flangia) - first few mm in Z are fixed
        int base_z = std::max(1, static_cast<int>(cfg.base_thickness_mm / cfg.resolution_mm));
        markSlabFixed(0, base_z);
        
        // 2. Holes in base (4 corners)
        double hole_radius = cfg.hole_diameter_mm / 2.0 / cfg.resolution_mm;
        int hole_offset_x = static_cast<int>(cfg.hole_offset_mm / cfg.resolution_mm);
        int hole_offset_y = static_cast<int>(cfg.hole_offset_mm / cfg.resolution_mm);
        
        std::vector<std::pair<int, int>> hole_positions = {
            {hole_offset_x, hole_offset_y},
            {nx - hole_offset_x, hole_offset_y},
            {hole_offset_x, ny - hole_offset_y},
            {nx - hole_offset_x, ny - hole_offset_y}
        };
        
        for (auto& [hx, hy] : hole_positions) {
            markCylinderVoid(hx, hy, 0, hole_radius, base_z);
        }
        
        // 3. Eyelet (top hole with material around)
        int eyelet_z_offset = static_cast<int>(cfg.eyelet_z_offset_mm / cfg.resolution_mm);
        int eyelet_height = std::max(1, static_cast<int>(cfg.eyelet_height_mm / cfg.resolution_mm));
        int eyelet_z_center = nz - eyelet_z_offset;
        int eyelet_x = nx / 2;
        int eyelet_y = ny / 2;
        
        int z_start = std::max(0, eyelet_z_center - eyelet_height / 2);
        int z_end = std::min(nz, z_start + eyelet_height);
        int actual_height = z_end - z_start;
        
        if (actual_height > 0) {
            // Outer ring (fixed material)
            double outer_radius = (cfg.eyelet_diameter_mm / 2.0 + cfg.eyelet_offset_mm) / cfg.resolution_mm;
            markCylinderFixed(eyelet_x, eyelet_y, z_start, outer_radius, actual_height);
            
            // Inner hole (void)
            double inner_radius = cfg.eyelet_diameter_mm / 2.0 / cfg.resolution_mm;
            markCylinderVoid(eyelet_x, eyelet_y, z_start, inner_radius, actual_height);
        }
        
        // 4. Connection pillar (ensures structural continuity)
        double pillar_radius = (cfg.eyelet_diameter_mm / 2.0 + cfg.eyelet_offset_mm) / cfg.resolution_mm;
        int pillar_z_start = base_z;
        int pillar_z_end = z_start;
        
        if (pillar_z_end > pillar_z_start) {
            markCylinderFixed(eyelet_x, eyelet_y, pillar_z_start, pillar_radius, pillar_z_end - pillar_z_start);
        }
        
        // Count elements
        countElements();
    }
    
    int64_t index(int i, int j, int k) const {
        return static_cast<int64_t>(k) * ny * nx + 
               static_cast<int64_t>(j) * nx + 
               static_cast<int64_t>(i);
    }
    
    int n_design = 0;
    int n_fixed = 0;
    int n_void = 0;
    
private:
    void markSlabFixed(int z_start, int z_end) {
        for (int i = 0; i < nx; ++i) {
            for (int j = 0; j < ny; ++j) {
                for (int k = z_start; k < z_end && k < nz; ++k) {
                    grid[index(i, j, k)] = static_cast<int8_t>(ElementType::FIXED);
                }
            }
        }
    }
    
    void markCylinderVoid(double cx, double cy, int z_start, double radius, int height) {
        int i_min = std::max(0, static_cast<int>(cx - radius - 1));
        int i_max = std::min(nx, static_cast<int>(cx + radius + 2));
        int j_min = std::max(0, static_cast<int>(cy - radius - 1));
        int j_max = std::min(ny, static_cast<int>(cy + radius + 2));
        
        for (int i = i_min; i < i_max; ++i) {
            for (int j = j_min; j < j_max; ++j) {
                double di = i - cx;
                double dj = j - cy;
                if (di*di + dj*dj <= radius*radius) {
                    for (int k = z_start; k < std::min(nz, z_start + height); ++k) {
                        grid[index(i, j, k)] = static_cast<int8_t>(ElementType::VOID);
                    }
                }
            }
        }
    }
    
    void markCylinderFixed(double cx, double cy, int z_start, double radius, int height) {
        int i_min = std::max(0, static_cast<int>(cx - radius - 1));
        int i_max = std::min(nx, static_cast<int>(cx + radius + 2));
        int j_min = std::max(0, static_cast<int>(cy - radius - 1));
        int j_max = std::min(ny, static_cast<int>(cy + radius + 2));
        
        for (int i = i_min; i < i_max; ++i) {
            for (int j = j_min; j < j_max; ++j) {
                double di = i - cx;
                double dj = j - cy;
                if (di*di + dj*dj <= radius*radius) {
                    for (int k = z_start; k < std::min(nz, z_start + height); ++k) {
                        auto& elem = grid[index(i, j, k)];
                        if (elem != static_cast<int8_t>(ElementType::VOID)) {
                            elem = static_cast<int8_t>(ElementType::FIXED);
                        }
                    }
                }
            }
        }
    }
    
    void countElements() {
        n_design = 0;
        n_fixed = 0;
        n_void = 0;
        for (auto& e : grid) {
            if (e == static_cast<int8_t>(ElementType::DESIGN)) ++n_design;
            else if (e == static_cast<int8_t>(ElementType::FIXED)) ++n_fixed;
            else ++n_void;
        }
    }
};

// =============================================================================
// Load Case Setup
// =============================================================================

/**
 * Create force vector for BRK-A-01 load case
 */
Vector createForceVector(const BracketDomain& domain, const Config& cfg) {
    int n_nodes = (domain.nx + 1) * (domain.ny + 1) * (domain.nz + 1);
    int n_dofs = 3 * n_nodes;
    Vector F = Vector::Zero(n_dofs);
    
    // Find eyelet load nodes (ring around the hole at eyelet center)
    int eyelet_z_offset = static_cast<int>(cfg.eyelet_z_offset_mm / cfg.resolution_mm);
    int eyelet_height = std::max(1, static_cast<int>(cfg.eyelet_height_mm / cfg.resolution_mm));
    int eyelet_z_center = domain.nz - eyelet_z_offset;
    
    int z_start = std::max(0, eyelet_z_center - eyelet_height / 2);
    int z_end = std::min(domain.nz, z_start + eyelet_height);
    int eyelet_z = (z_start + z_end) / 2;
    eyelet_z = std::max(1, std::min(domain.nz - 1, eyelet_z));
    
    int eyelet_x = domain.nx / 2;
    int eyelet_y = domain.ny / 2;
    
    // Safety margin for load application
    double safety_margin = cfg.resolution_mm * 1.25;
    double inner_radius_mm = (cfg.eyelet_diameter_mm / 2.0) + safety_margin;
    double outer_radius_mm = (cfg.eyelet_diameter_mm / 2.0) + cfg.eyelet_offset_mm;
    
    if (inner_radius_mm >= outer_radius_mm) {
        inner_radius_mm = (cfg.eyelet_diameter_mm / 2.0) + (cfg.eyelet_offset_mm * 0.5);
    }
    
    double inner_radius = inner_radius_mm / cfg.resolution_mm;
    double outer_radius = outer_radius_mm / cfg.resolution_mm;
    
    // Collect nodes in ring
    std::vector<int64_t> load_nodes;
    int search = static_cast<int>(outer_radius) + 2;
    
    for (int i = std::max(0, eyelet_x - search); i <= std::min(domain.nx, eyelet_x + search); ++i) {
        for (int j = std::max(0, eyelet_y - search); j <= std::min(domain.ny, eyelet_y + search); ++j) {
            double dx = i - eyelet_x;
            double dy = j - eyelet_y;
            double dist = std::sqrt(dx*dx + dy*dy);
            
            if (dist > inner_radius && dist <= outer_radius) {
                int64_t node_idx = i * (domain.ny + 1) * (domain.nz + 1) + 
                                   j * (domain.nz + 1) + eyelet_z;
                load_nodes.push_back(node_idx);
            }
        }
    }
    
    // Fallback to center if no ring nodes found
    if (load_nodes.empty()) {
        int64_t center = eyelet_x * (domain.ny + 1) * (domain.nz + 1) + 
                         eyelet_y * (domain.nz + 1) + eyelet_z;
        load_nodes.push_back(center);
    }
    
    // Distribute force uniformly
    double force_per_node = cfg.load_force_N / load_nodes.size();
    for (auto node_idx : load_nodes) {
        // Z component (DOF index 2 for each node)
        F(node_idx * 3 + 2) = -force_per_node;  // -Z direction
    }
    
    std::cout << "Load: " << cfg.load_force_N << " N distributed over " 
              << load_nodes.size() << " nodes" << std::endl;
    
    return F;
}

/**
 * Create fixed DOF list for base constraint
 */
std::vector<int64_t> createFixedDofs(const BracketDomain& domain) {
    std::vector<int64_t> fixed_dofs;
    
    // All nodes at Z=0 are fixed
    for (int i = 0; i <= domain.nx; ++i) {
        for (int j = 0; j <= domain.ny; ++j) {
            int64_t node_idx = i * (domain.ny + 1) * (domain.nz + 1) + 
                               j * (domain.nz + 1) + 0;
            fixed_dofs.push_back(node_idx * 3 + 0);  // UX
            fixed_dofs.push_back(node_idx * 3 + 1);  // UY
            fixed_dofs.push_back(node_idx * 3 + 2);  // UZ
        }
    }
    
    return fixed_dofs;
}

// =============================================================================
// File I/O
// =============================================================================

/**
 * Save density field to NumPy .npy format
 * Format spec: https://numpy.org/doc/stable/reference/generated/numpy.lib.format.html
 */
void saveNpy(const std::string& filename, const std::vector<Scalar>& data,
             int nx, int ny, int nz) {
    std::ofstream file(filename, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Cannot open file: " + filename);
    }
    
    // NumPy .npy header
    const uint8_t magic[] = {0x93, 'N', 'U', 'M', 'P', 'Y'};
    const uint8_t version[] = {0x01, 0x00};  // Version 1.0
    
    // Build header dictionary
    std::ostringstream header_stream;
    header_stream << "{'descr': '<f8', 'fortran_order': False, 'shape': ("
                  << nx << ", " << ny << ", " << nz << "), }";
    std::string header_dict = header_stream.str();
    
    // Pad to 64-byte alignment (including magic + version + header_len)
    size_t header_len = header_dict.size() + 1;  // +1 for newline
    size_t total_prefix = 10 + header_len;  // magic(6) + version(2) + len(2) + header
    size_t padding = (64 - (total_prefix % 64)) % 64;
    header_len += padding;
    
    // Write header
    file.write(reinterpret_cast<const char*>(magic), 6);
    file.write(reinterpret_cast<const char*>(version), 2);
    
    uint16_t hlen = static_cast<uint16_t>(header_len);
    file.write(reinterpret_cast<const char*>(&hlen), 2);
    
    file << header_dict;
    for (size_t i = 0; i < padding; ++i) file << ' ';
    file << '\n';
    
    // Write data (C-order: last index varies fastest)
    // Our data is stored as: index(i,j,k) = k*ny*nx + j*nx + i
    // NumPy C-order expects: index(i,j,k) = i*ny*nz + j*nz + k
    std::vector<Scalar> c_order(data.size());
    for (int i = 0; i < nx; ++i) {
        for (int j = 0; j < ny; ++j) {
            for (int k = 0; k < nz; ++k) {
                int64_t cpp_idx = k * ny * nx + j * nx + i;
                int64_t npy_idx = i * ny * nz + j * nz + k;
                c_order[npy_idx] = data[cpp_idx];
            }
        }
    }
    
    file.write(reinterpret_cast<const char*>(c_order.data()), 
               c_order.size() * sizeof(Scalar));
    
    std::cout << "Saved: " << filename << " (" << nx << "x" << ny << "x" << nz << ")" << std::endl;
}

/**
 * Save metadata to JSON format
 */
void saveMetadata(const std::string& filename, const Config& cfg,
                  const BracketDomain& domain, const OptimizationResult& result,
                  double elapsed_seconds) {
    std::ofstream file(filename);
    if (!file) {
        throw std::runtime_error("Cannot open file: " + filename);
    }
    
    file << "{\n"
         << "  \"case\": \"BRK-A-01\",\n"
         << "  \"mode\": \"cpp_standalone\",\n"
         << "  \"provenance\": \"cpp_hpc_engine\",\n"
         << "  \"resolution_mm\": " << cfg.resolution_mm << ",\n"
         << "  \"domain_shape\": [" << domain.nx << ", " << domain.ny << ", " << domain.nz << "],\n"
         << "  \"n_elements\": " << domain.grid.size() << ",\n"
         << "  \"n_design\": " << domain.n_design << ",\n"
         << "  \"n_fixed\": " << domain.n_fixed << ",\n"
         << "  \"n_void\": " << domain.n_void << ",\n"
         << "  \"volume_fraction_target\": " << cfg.volume_fraction << ",\n";
    
    if (!result.volume_history.empty()) {
        file << "  \"volume_fraction_final\": " << result.volume_history.back() << ",\n";
    }
    if (!result.compliance_history.empty()) {
        file << "  \"compliance_final\": " << result.compliance_history.back() << ",\n";
    }
    
    file << "  \"iterations\": " << result.iterations << ",\n"
         << "  \"converged\": " << (result.converged ? "true" : "false") << ",\n"
         << "  \"elapsed_time_s\": " << elapsed_seconds << ",\n";
    
#ifdef _OPENMP
    file << "  \"openmp_threads\": " << omp_get_max_threads() << ",\n";
#else
    file << "  \"openmp_threads\": 1,\n";
#endif
    
    file << "  \"simp_config\": {\n"
         << "    \"filter_radius\": " << cfg.filter_radius << ",\n"
         << "    \"E_min\": " << cfg.E_min << ",\n"
         << "    \"penalty\": " << cfg.penalty << ",\n"
         << "    \"move_limit\": " << cfg.move_limit << "\n"
         << "  }\n"
         << "}\n";
    
    std::cout << "Saved: " << filename << std::endl;
}

// =============================================================================
// Command Line Parsing
// =============================================================================

void printUsage(const char* prog) {
    std::cout << "Usage: " << prog << " [options]\n\n"
              << "Options:\n"
              << "  --resolution <mm>     Grid resolution (default: 2.0)\n"
              << "  --volume <fraction>   Target volume fraction (default: 0.25)\n"
              << "  --iterations <n>      Max iterations (default: 100)\n"
              << "  --filter <voxels>     Filter radius in voxels (default: 3.5)\n"
              << "  --move <limit>        Max density change per iter (default: 0.1)\n"
              << "  --emin <value>        Minimum modulus ratio (default: 1e-4)\n"
              << "  --output <dir>        Output directory (default: ./output)\n"
              << "  --output_dir <dir>    Alias for --output\n"
              << "  -o <dir>              Alias for --output\n"
              << "  --quiet               Disable verbose output\n"
              << "  --help                Show this help\n";
}

bool parseArgs(int argc, char* argv[], Config& cfg) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        
        if (arg == "--help" || arg == "-h") {
            printUsage(argv[0]);
            return false;
        }
        else if (arg == "--resolution" && i + 1 < argc) {
            cfg.resolution_mm = std::stod(argv[++i]);
        }
        else if (arg == "--volume" && i + 1 < argc) {
            cfg.volume_fraction = std::stod(argv[++i]);
        }
        else if (arg == "--iterations" && i + 1 < argc) {
            cfg.max_iterations = std::stoi(argv[++i]);
        }
        else if (arg == "--filter" && i + 1 < argc) {
            cfg.filter_radius = std::stod(argv[++i]);
        }
        else if (arg == "--move" && i + 1 < argc) {
            cfg.move_limit = std::stod(argv[++i]);
        }
        else if (arg == "--emin" && i + 1 < argc) {
            cfg.E_min = std::stod(argv[++i]);
        }
        else if (arg == "--output" && i + 1 < argc) {
            cfg.output_dir = argv[++i];
        }
        else if ((arg == "--output_dir" || arg == "-o") && i + 1 < argc) {
            cfg.output_dir = argv[++i];
        }
        else if (arg == "--quiet") {
            cfg.verbose = false;
        }
        else {
            std::cerr << "Unknown option: " << arg << std::endl;
            printUsage(argv[0]);
            return false;
        }
    }
    return true;
}

// =============================================================================
// Main
// =============================================================================

int main(int argc, char* argv[]) {
    Config cfg;
    if (!parseArgs(argc, argv, cfg)) {
        return 1;
    }
    
    std::cout << "================================================================\n"
              << "AM Topology Optimization - BRK-A-01 Bracket (C++ Standalone)\n"
              << "================================================================\n\n";
    
#ifdef _OPENMP
    std::cout << "OpenMP threads: " << omp_get_max_threads() << std::endl;
#endif
    
    // 1. Create domain
    std::cout << "\n--- Domain Setup ---\n";
    BracketDomain domain(cfg);
    std::cout << "Grid: " << domain.nx << " x " << domain.ny << " x " << domain.nz 
              << " = " << domain.grid.size() << " elements\n"
              << "  Design: " << domain.n_design << "\n"
              << "  Fixed:  " << domain.n_fixed << "\n"
              << "  Void:   " << domain.n_void << std::endl;
    
    // 2. Material (Ti6Al4V)
    MaterialProperties material;
    std::cout << "Material: Ti6Al4V (E=" << material.E / 1e9 << " GPa, ν=" 
              << material.nu << ")\n";
    
    // 3. SIMP configuration
    SIMPConfig simp_cfg;
    simp_cfg.volume_fraction = cfg.volume_fraction;
    simp_cfg.max_iterations = cfg.max_iterations;
    simp_cfg.convergence_tol = cfg.tolerance;
    simp_cfg.filter_radius = cfg.filter_radius;
    simp_cfg.rho_min = cfg.E_min;  // Use E_min for rho_min
    simp_cfg.penalty = cfg.penalty;
    simp_cfg.move_limit = cfg.move_limit;
    simp_cfg.verbose = cfg.verbose;
    
    std::cout << "SIMP: vf=" << cfg.volume_fraction 
              << ", filter=" << cfg.filter_radius 
              << ", E_min=" << cfg.E_min 
              << ", move=" << cfg.move_limit << std::endl;
    
    // 4. Initial density
    std::vector<Scalar> density(domain.grid.size());
    for (size_t i = 0; i < domain.grid.size(); ++i) {
        int8_t type = domain.grid[i];
        if (type == static_cast<int8_t>(ElementType::DESIGN)) {
            density[i] = cfg.volume_fraction;
        } else if (type == static_cast<int8_t>(ElementType::FIXED)) {
            density[i] = 1.0;
        } else {  // VOID
            density[i] = cfg.E_min;
        }
    }
    
    // 5. Force vector and constraints
    std::cout << "\n--- Load Case ---\n";
    Vector forces = createForceVector(domain, cfg);
    std::vector<int64_t> fixed_dofs = createFixedDofs(domain);
    std::cout << "Fixed DOFs: " << fixed_dofs.size() << " (base constraint)\n";
    
    // 6. Create optimizer
    Scalar element_size = cfg.resolution_mm / 1000.0;  // mm to m
    TopologyOptimizer optimizer(
        domain.nx, domain.ny, domain.nz,
        element_size, material, simp_cfg
    );
    
    // 7. Run optimization
    std::cout << "\n--- Optimization ---\n";
    auto start_time = std::chrono::high_resolution_clock::now();
    
    OptimizationResult result = optimizer.run(
        density, forces, fixed_dofs,
        nullptr  // Use optimizer's built-in verbose output
    );
    
    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(end_time - start_time).count();
    
    // 8. Summary
    std::cout << "\n--- Results ---\n"
              << "Iterations: " << result.iterations << "\n"
              << "Converged: " << (result.converged ? "yes" : "no") << "\n"
              << "Time: " << std::fixed << std::setprecision(2) << elapsed << " s\n";
    
    if (!result.compliance_history.empty()) {
        std::cout << "Final compliance: " << std::scientific << std::setprecision(4) 
                  << result.compliance_history.back() << "\n";
    }
    if (!result.volume_history.empty()) {
        std::cout << "Final volume: " << std::fixed << std::setprecision(4) 
                  << result.volume_history.back() << "\n";
    }
    
    // 9. Save results
    std::cout << "\n--- Saving ---\n";
    fs::create_directories(cfg.output_dir);
    
    saveNpy(cfg.output_dir + "/density_field.npy", result.final_density,
            domain.nx, domain.ny, domain.nz);
    saveMetadata(cfg.output_dir + "/metadata.json", cfg, domain, result, elapsed);
    
    // Save convergence history
    std::ofstream hist_file(cfg.output_dir + "/history.csv");
    hist_file << "iteration,compliance,volume\n";
    for (size_t i = 0; i < result.compliance_history.size(); ++i) {
        hist_file << i << "," << result.compliance_history[i] << ","
                  << result.volume_history[i] << "\n";
    }
    std::cout << "Saved: " << cfg.output_dir << "/history.csv\n";
    
    std::cout << "\n✓ Done!\n";
    return 0;
}
