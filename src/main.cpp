#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <algorithm>
#include <iterator>
#include "physics_utils.h"
#include "assembler.h"
#include "solver.h"
#include "analytical.h"
#include "exporter.h"
#include "config_types.h" 
#include <chrono>

SimulationConfig load_config(const std::string& filename);

// --- 辅助：网格生成 ---
std::vector<double> generate_stretched_coords(int n, double dx_min, double stretch_ratio) {
    std::vector<double> coords(n + 1);
    int center_idx = n / 2;
    coords[center_idx] = 0.0;
    
    // 向正方向拉伸
    double current_dx = dx_min;
    for (int i = center_idx; i < n; ++i) {
        coords[i+1] = coords[i] + current_dx;
        current_dx *= stretch_ratio;
    }
    
    // 向负方向拉伸
    current_dx = dx_min;
    for (int i = center_idx - 1; i >= 0; --i) {
        coords[i] = coords[i+1] - current_dx;
        current_dx *= stretch_ratio;
    }
    return coords;
}

// --- 辅助：寻找最近网格点 ---
int find_nearest_node(const std::vector<double>& nodes, double val) {
    auto it = std::min_element(nodes.begin(), nodes.end(), 
        [val](double a, double b) { return std::abs(a - val) < std::abs(b - val); });
    return std::distance(nodes.begin(), it);
}

// --- 核心：通用源施加 ---
void apply_source_generic(
    const SourceConfig& src,
    const GridInfo& grid,
    std::vector<cuComplexType>& rhs,
    double omega
) {
    std::cout << "\n=== apply_source_generic called ===\n";
    std::cout << "Source type: " << src.type << "\n";
    std::cout << "Source position: (" << src.position[0] << ", " << src.position[1] << ", " << src.position[2] << ")\n";
    std::cout << "Source direction: (" << src.direction[0] << ", " << src.direction[1] << ", " << src.direction[2] << ")\n";
    std::cout << "Source amplitude: " << src.amplitude << "\n";

    int ix = find_nearest_node(grid.x_nodes, src.position[0]);
    int iy = find_nearest_node(grid.y_nodes, src.position[1]);
    int iz = find_nearest_node(grid.z_nodes, src.position[2]);

    if (src.type == "magnetic_dipole") {
        double mz = src.direction[2];
        if (std::abs(mz) > 1e-12) {
            long long idx_ex = (long long)0 * grid.nx * grid.ny * grid.nz + iz * grid.nx * grid.ny + iy * grid.nx + ix;
            long long idx_ey = (long long)1 * grid.nx * grid.ny * grid.nz + iz * grid.nx * grid.ny + iy * grid.nx + ix;

            if (idx_ex < (long long)rhs.size()) rhs[idx_ex].x += (float)(src.amplitude * mz);
            if (idx_ey < (long long)rhs.size()) rhs[idx_ey].x -= (float)(src.amplitude * mz);
        }
    }
}

// --- 材料生成函数 ---
std::vector<MaterialProperty> generate_materials(const SimulationConfig& cfg, const GridInfo& grid, double omega) {
    std::vector<MaterialProperty> materials(grid.total_unknowns());
    
    double mu = 4.0 * M_PI * 1.0e-7;
    double bg_sigma = 1.0 / cfg.model.bg_rho;
    
    #pragma omp parallel for
    for (int k = 0; k < grid.nz; ++k) {
        for (int j = 0; j < grid.ny; ++j) {
            for (int i = 0; i < grid.nx; ++i) {
                int mat_idx = k * grid.nx * grid.ny + j * grid.nx + i;
                
                materials[mat_idx].sigma_eff.val[0] = Complex(bg_sigma, 0.0f);
                materials[mat_idx].mu_inv_eff.val[0] = Complex(1.0f / mu, 0.0f);
                materials[mat_idx].sigma_eff.val[1] = Complex(0.0f, 0.0f);
                materials[mat_idx].mu_inv_eff.val[1] = Complex(0.0f, 0.0f);
                materials[mat_idx].sigma_eff.val[2] = Complex(0.0f, 0.0f);
                materials[mat_idx].mu_inv_eff.val[2] = Complex(0.0f, 0.0f);
                materials[mat_idx].sigma_eff.val[3] = Complex(0.0f, 0.0f);
                materials[mat_idx].mu_inv_eff.val[3] = Complex(0.0f, 0.0f);
                materials[mat_idx].sigma_eff.val[4] = Complex(0.0f, 0.0f);
                materials[mat_idx].mu_inv_eff.val[4] = Complex(0.0f, 0.0f);
                materials[mat_idx].sigma_eff.val[5] = Complex(0.0f, 0.0f);
                materials[mat_idx].mu_inv_eff.val[5] = Complex(0.0f, 0.0f);
                materials[mat_idx].sigma_eff.val[6] = Complex(0.0f, 0.0f);
                materials[mat_idx].mu_inv_eff.val[6] = Complex(0.0f, 0.0f);
                materials[mat_idx].sigma_eff.val[7] = Complex(0.0f, 0.0f);
                materials[mat_idx].mu_inv_eff.val[7] = Complex(0.0f, 0.0f);
                materials[mat_idx].sigma_eff.val[8] = Complex(0.0f, 0.0f);
                materials[mat_idx].mu_inv_eff.val[8] = Complex(0.0f, 0.0f);
            }
        }
    }
    
    return materials;
}

// --- 主函数 ---
int main(int argc, char* argv[]) {
    std::string config_file = "test_500hz.json";
    if (argc > 1) {
        config_file = argv[1];
    }
    
    std::cout << "Loading config from: " << config_file << std::endl;
    SimulationConfig cfg = load_config(config_file);
    
    // Create grid
    GridInfo grid;
    grid.nx = cfg.grid.nx;
    grid.ny = cfg.grid.ny;
    grid.nz = cfg.grid.nz;
    grid.x_nodes = generate_stretched_coords(cfg.grid.nx, cfg.grid.dx_min, cfg.grid.stretch_ratio);
    grid.y_nodes = generate_stretched_coords(cfg.grid.ny, cfg.grid.dx_min, cfg.grid.stretch_ratio);
    grid.z_nodes = generate_stretched_coords(cfg.grid.nz, cfg.grid.dx_min, cfg.grid.stretch_ratio);
    
    // Compute omega
    double omega = 2.0 * M_PI * cfg.source.frequency;
    
    // Generate materials
    auto materials = generate_materials(cfg, grid, omega);
    
    // Assemble system matrix
    std::cout << "Assembling system matrix..." << std::endl;
    Assembler assembler(grid, omega);
    CsrMatrix A = assembler.assemble_system_matrix(materials);
    
    // Initialize RHS
    std::vector<cuComplexType> rhs(grid.total_unknowns(), {0.0f, 0.0f});
    
    // Apply source
    std::cout << "Applying source..." << std::endl;
    apply_source_generic(cfg.source, grid, rhs, omega);
    
    std::cout << "\n=== Diagnostic: RHS magnitudes ===" << std::endl;
    float max_rhs = 0.0f;
    int max_idx = 0;
    for (int i = 0; i < (int)rhs.size(); ++i) {
        float mag = std::sqrt(rhs[i].x * rhs[i].x + rhs[i].y * rhs[i].y);
        if (mag > max_rhs) {
            max_rhs = mag;
            max_idx = i;
        }
    }
    std::cout << "Max RHS magnitude: " << max_rhs << " at index " << max_idx << std::endl;
    std::cout << "RHS size: " << rhs.size() << std::endl;
    
    // Print first few RHS values
    std::cout << "\nFirst 5 RHS values:" << std::endl;
    for (int i = 0; i < std::min(5, (int)rhs.size()); ++i) {
        std::cout << "  rhs[" << i << "] = " << rhs[i].x << " + i" << rhs[i].y << std::endl;
    }

    // --- Debug: inspect assembled matrix near source DOFs (sample triplets + diagonal) ---
    int ix_src_dbg = find_nearest_node(grid.x_nodes, cfg.source.position[0]);
    int iy_src_dbg = find_nearest_node(grid.y_nodes, cfg.source.position[1]);
    int iz_src_dbg = find_nearest_node(grid.z_nodes, cfg.source.position[2]);
    long long base_dbg = (long long)iz_src_dbg * grid.nx * grid.ny + (long long)iy_src_dbg * grid.nx + ix_src_dbg;

    std::cout << "\n--- CSR debug: sample entries near source grid index (" << ix_src_dbg << "," << iy_src_dbg << "," << iz_src_dbg << ") ---" << std::endl;
    for (int comp = 0; comp < 3; ++comp) {
        long long row = (long long)comp * grid.nx * grid.ny * grid.nz + base_dbg;
        if (row < 0 || row >= (long long)A.num_rows) {
            std::cout << "Row " << row << " out of range" << std::endl;
            continue;
        }
        std::cout << "Component " << comp << " row=" << row << ":\n";
        int start = A.row_ptr[row];
        int end = A.row_ptr[row+1];
        int to_print = std::min(8, end - start);
        for (int p = start; p < start + to_print; ++p) {
            int col = A.col_ind[p];
            cuComplexType v = A.val[p];
            std::cout << "  col=" << col << " val=" << v.x << " + i" << v.y << std::endl;
        }
        // find diagonal
        double diag_mag = 0.0;
        for (int p = start; p < end; ++p) {
            if (A.col_ind[p] == row) {
                cuComplexType vd = A.val[p];
                diag_mag = std::sqrt((double)vd.x * vd.x + (double)vd.y * vd.y);
                break;
            }
        }
        std::cout << "  diagonal magnitude = " << diag_mag << std::endl;
    }
    
    // Frequency check
    std::cout << "\nFrequency: " << cfg.source.frequency << " Hz" << std::endl;
    std::cout << "Angular frequency (omega): " << omega << " rad/s" << std::endl;
    
    // Create solver and result vector
    Solver solver;
    std::vector<cuComplexType> x(rhs.size(), make_cuComplex(0.0f, 0.0f));
    
    std::cout << "Solving..." << std::endl;
    auto t_solve_start = std::chrono::high_resolution_clock::now();
    
    solver.solve(A, rhs, x);
    
    auto t_solve_end = std::chrono::high_resolution_clock::now();
    std::cout << "Solver time: "
              << std::chrono::duration<double>(t_solve_end - t_solve_start).count() << " s" << std::endl;
    
    // Field at source
    int ix_src = find_nearest_node(grid.x_nodes, cfg.source.position[0]);
    int iy_src = find_nearest_node(grid.y_nodes, cfg.source.position[1]);
    int iz_src = find_nearest_node(grid.z_nodes, cfg.source.position[2]);
    
    std::cout << "\n=== Field at source location ===" << std::endl;
    std::cout << "Source grid index: (" << ix_src << ", " << iy_src << ", " << iz_src << ")" << std::endl;
    for (int comp = 0; comp < 3; ++comp) {
        long long idx = (long long)comp * grid.nx * grid.ny * grid.nz 
                      + iz_src * grid.nx * grid.ny 
                      + iy_src * grid.nx + ix_src;
        if (idx < (long long)x.size()) {
            std::cout << "E[" << comp << "] = " << x[idx].x << " + i " << x[idx].y << std::endl;
        }
    }

    std::cout << "\n=== Field magnitudes ===" << std::endl;
    for (int comp = 0; comp < 3; ++comp) {
        float max_mag = 0.0f;
        for (int idx = 0; idx < (int)x.size(); ++idx) {
            float mag = std::sqrt(x[idx].x * x[idx].x + x[idx].y * x[idx].y);
            max_mag = std::max(max_mag, mag);
        }
        std::cout << "Component " << comp << ": max magnitude = " << max_mag << std::endl;
    }
    
    // Export results
    if (cfg.receiver.type == "full_grid") {
        std::string vtk_name = cfg.receiver.output_file + ".vtr";
        VtkExporter::save_to_vtr(vtk_name, grid, x);
        std::cout << "Exported to " << vtk_name << std::endl;
    }
    
    if (cfg.validation.enabled && cfg.validation.mode == "analytical") {
        auto x_analytical = AnalyticalSolution::compute_field(
            1.0 / cfg.model.bg_rho, cfg.source.frequency, grid,
            (double)ix_src, (double)iy_src, (double)iz_src,
            cfg.source.direction[0], cfg.source.direction[1], cfg.source.direction[2]
        );
        std::string analytical_name = cfg.receiver.output_file + "_analytical.vtr";
        VtkExporter::save_to_vtr(analytical_name, grid, x_analytical);
        std::cout << "Exported analytical to " << analytical_name << std::endl;
        
        std::cout << "\n=== Field Comparison: Numerical vs Analytical ===" << std::endl;
        
        int i_near = ix_src + 1, j_near = iy_src, k_near = iz_src;
        for (int comp = 0; comp < 3; ++comp) {
            long long idx_num = (long long)comp * grid.nx * grid.ny * grid.nz 
                              + k_near * grid.nx * grid.ny + j_near * grid.nx + i_near;
            long long idx_ana = (long long)comp * grid.nx * grid.ny * grid.nz 
                              + k_near * grid.nx * grid.ny + j_near * grid.nx + i_near;
            
            float mag_num = std::sqrt(x[idx_num].x * x[idx_num].x + x[idx_num].y * x[idx_num].y);
            float mag_ana = std::sqrt(x_analytical[idx_ana].x * x_analytical[idx_ana].x 
                                    + x_analytical[idx_ana].y * x_analytical[idx_ana].y);
            
            std::cout << "Comp " << comp << " near field: num=" << mag_num << ", ana=" << mag_ana 
                      << ", ratio=" << (mag_ana > 1e-15 ? mag_num/mag_ana : 0) << std::endl;
        }
        
        int i_far = ix_src + 20, j_far = iy_src, k_far = iz_src;
        if (i_far < grid.nx) {
            for (int comp = 0; comp < 3; ++comp) {
                long long idx_num = (long long)comp * grid.nx * grid.ny * grid.nz 
                                  + k_far * grid.nx * grid.ny + j_far * grid.nx + i_far;
                long long idx_ana = (long long)comp * grid.nx * grid.ny * grid.nz 
                                  + k_far * grid.nx * grid.ny + j_far * grid.nx + i_far;
                
                float mag_num = std::sqrt(x[idx_num].x * x[idx_num].x + x[idx_num].y * x[idx_num].y);
                float mag_ana = std::sqrt(x_analytical[idx_ana].x * x_analytical[idx_ana].x 
                                        + x_analytical[idx_ana].y * x_analytical[idx_ana].y);
                
                std::cout << "Comp " << comp << " far field: num=" << mag_num << ", ana=" << mag_ana 
                          << ", ratio=" << (mag_ana > 1e-15 ? mag_num/mag_ana : 0) << std::endl;
            }
        }
    }
    
    return 0;
}

