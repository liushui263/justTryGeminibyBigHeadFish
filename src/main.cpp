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
            // Distribute dipole source over 8 neighboring cells (short-support kernel)
            // Determine neighbor cell indices (prefer ix and ix-1, but stay inside domain)
            int ix0 = ix;
            int ix1 = (ix > 0) ? ix-1 : std::min(ix+1, grid.nx-1);
            int iy0 = iy;
            int iy1 = (iy > 0) ? iy-1 : std::min(iy+1, grid.ny-1);
            int iz0 = iz;
            int iz1 = (iz > 0) ? iz-1 : std::min(iz+1, grid.nz-1);

            // uniform split among 8 neighbors
            double Js_mag = src.amplitude * mz;
            const int Nparts = 8;
            double Js_part = Js_mag / (double)Nparts;

            for (int dz_off = 0; dz_off <= 1; ++dz_off) {
                for (int dy_off = 0; dy_off <= 1; ++dy_off) {
                    for (int dx_off = 0; dx_off <= 1; ++dx_off) {
                        int ci = dx_off ? ix1 : ix0;
                        int cj = dy_off ? iy1 : iy0;
                        int ck = dz_off ? iz1 : iz0;
                        if (ci < 0 || cj < 0 || ck < 0) continue;
                        if (ci >= grid.nx || cj >= grid.ny || ck >= grid.nz) continue;

                        long long idx_ex = (long long)0 * grid.nx * grid.ny * grid.nz + ck * grid.nx * grid.ny + cj * grid.nx + ci;
                        long long idx_ey = (long long)1 * grid.nx * grid.ny * grid.nz + ck * grid.nx * grid.ny + cj * grid.nx + ci;
                        long long idx_ez = (long long)2 * grid.nx * grid.ny * grid.nz + ck * grid.nx * grid.ny + cj * grid.nx + ci;

                        double dx_loc = grid.dx(ci);
                        double dy_loc = grid.dy(cj);
                        double dz_loc = grid.dz(ck);
                        double cell_vol = dx_loc * dy_loc * dz_loc;

                        double rhs_im = omega * Js_part / std::max(cell_vol, 1e-18);

                        if (idx_ex >= 0 && idx_ex < (long long)rhs.size()) rhs[idx_ex].y += (float)(rhs_im);
                        if (idx_ey >= 0 && idx_ey < (long long)rhs.size()) rhs[idx_ey].y -= (float)(rhs_im);
                        // Ez may be unused for this dipole orientation but keep zero change
                        (void)idx_ez;
                    }
                }
            }
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

    // If analytical validation is enabled, project analytic field onto DOFs
    // by computing b = A * x_analytical (exact discrete RHS = i*w*J_s)
    if (cfg.validation.enabled && cfg.validation.mode == "analytical") {
        std::cout << "Applying analytic->DOF projection: building RHS = A * E_analytical" << std::endl;
        auto x_analytical_proj = AnalyticalSolution::compute_field(
            1.0 / cfg.model.bg_rho, cfg.source.frequency, grid,
            (double)find_nearest_node(grid.x_nodes, cfg.source.position[0]),
            (double)find_nearest_node(grid.y_nodes, cfg.source.position[1]),
            (double)find_nearest_node(grid.z_nodes, cfg.source.position[2]),
            cfg.source.direction[0], cfg.source.direction[1], cfg.source.direction[2]
        );

        // Compute b = A * x_analytical_proj and overwrite rhs
        std::vector<cuComplexType> b_proj((size_t)A.num_rows, make_cuComplex(0.0f, 0.0f));
        for (int r = 0; r < A.num_rows; ++r) {
            int start = A.row_ptr[r];
            int end = A.row_ptr[r+1];
            float acc_r = 0.0f, acc_i = 0.0f;
            for (int p = start; p < end; ++p) {
                int c = A.col_ind[p];
                cuComplexType a = A.val[p];
                cuComplexType xb = x_analytical_proj[c];
                float ar = a.x, ai = a.y;
                float br = xb.x, bi = xb.y;
                acc_r += ar * br - ai * bi;
                acc_i += ar * bi + ai * br;
            }
            b_proj[r] = make_cuComplex(acc_r, acc_i);
        }
        rhs = std::move(b_proj);
    }
    
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
    
    // Debug: print RHS values at the 8 source support DOFs (Ex, Ey, Ez)
    std::cout << "\nRHS at source support DOFs:" << std::endl;
    std::vector<std::pair<std::string,long long>> src_dofs;
    {
        int ix = find_nearest_node(grid.x_nodes, cfg.source.position[0]);
        int iy = find_nearest_node(grid.y_nodes, cfg.source.position[1]);
        int iz = find_nearest_node(grid.z_nodes, cfg.source.position[2]);
        int ix0 = ix;
        int ix1 = (ix > 0) ? ix-1 : std::min(ix+1, grid.nx-1);
        int iy0 = iy;
        int iy1 = (iy > 0) ? iy-1 : std::min(iy+1, grid.ny-1);
        int iz0 = iz;
        int iz1 = (iz > 0) ? iz-1 : std::min(iz+1, grid.nz-1);
        for (int dz_off = 0; dz_off <= 1; ++dz_off) {
            for (int dy_off = 0; dy_off <= 1; ++dy_off) {
                for (int dx_off = 0; dx_off <= 1; ++dx_off) {
                    int ci = dx_off ? ix1 : ix0;
                    int cj = dy_off ? iy1 : iy0;
                    int ck = dz_off ? iz1 : iz0;
                    long long idx_ex = (long long)0 * grid.nx * grid.ny * grid.nz + ck * grid.nx * grid.ny + cj * grid.nx + ci;
                    long long idx_ey = (long long)1 * grid.nx * grid.ny * grid.nz + ck * grid.nx * grid.ny + cj * grid.nx + ci;
                    long long idx_ez = (long long)2 * grid.nx * grid.ny * grid.nz + ck * grid.nx * grid.ny + cj * grid.nx + ci;
                    src_dofs.push_back({"Ex", idx_ex});
                    src_dofs.push_back({"Ey", idx_ey});
                    src_dofs.push_back({"Ez", idx_ez});
                }
            }
        }
    }
    for (auto &p : src_dofs) {
        long long id = p.second;
        if (id >= 0 && id < (long long)rhs.size()) {
            std::cout << "  " << p.first << "[" << id << "] = " << rhs[id].x << " + i" << rhs[id].y << std::endl;
        }
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
        // --- Verification: build RHS = A * x_analytical and solve to recover x_analytical ---
        std::cout << "\nBuilding RHS = A * x_analytical for verification..." << std::endl;
        std::vector<cuComplexType> b_check((size_t)A.num_rows, make_cuComplex(0.0f, 0.0f));
        for (int r = 0; r < A.num_rows; ++r) {
            int start = A.row_ptr[r];
            int end = A.row_ptr[r+1];
            float acc_r = 0.0f, acc_i = 0.0f;
            for (int p = start; p < end; ++p) {
                int c = A.col_ind[p];
                cuComplexType a = A.val[p];
                cuComplexType xb = x_analytical[c];
                float ar = a.x, ai = a.y;
                float br = xb.x, bi = xb.y;
                acc_r += ar * br - ai * bi;
                acc_i += ar * bi + ai * br;
            }
            b_check[r] = make_cuComplex(acc_r, acc_i);
        }

        std::cout << "Solving verification system..." << std::endl;
        std::vector<cuComplexType> x_check(b_check.size(), make_cuComplex(0.0f, 0.0f));
        Solver verifier;
        verifier.solve(A, b_check, x_check);

        // compute max abs error between x_check and x_analytical
        double max_err = 0.0;
        for (int i = 0; i < (int)x_check.size(); ++i) {
            double dr = x_check[i].x - x_analytical[i].x;
            double di = x_check[i].y - x_analytical[i].y;
            double err = std::sqrt(dr*dr + di*di);
            if (err > max_err) max_err = err;
        }
        std::cout << "Verification solve max abs error: " << max_err << std::endl;

        std::cout << "\n=== Field Comparison: Numerical vs Analytical (line scans) ===" << std::endl;
        int n_pml = cfg.model.pml_layers;
        int ixc = ix_src, iyc = iy_src, izc = iz_src;

        auto print_scan = [&](char axis) {
            std::cout << "\nScan along " << axis << ": index, E0_num, E0_ana, E1_num, E1_ana, E2_num, E2_ana" << std::endl;
            std::vector<int> coords;
            if (axis == 'x') {
                int i_end = grid.nx - n_pml - 1;
                int N = std::max(2, i_end - ixc + 1);
                int steps = std::min(20, N);
                for (int s = 0; s < steps; ++s) coords.push_back(ixc + (int)((double)s * (i_end - ixc) / std::max(1, steps-1)));
                for (int i : coords) {
                    long long base = (long long)izc * grid.nx * grid.ny + (long long)iyc * grid.nx + i;
                    std::cout << "i=" << i << ", ";
                    for (int comp = 0; comp < 3; ++comp) {
                        long long idx_num = (long long)comp * grid.nx * grid.ny * grid.nz + base;
                        long long idx_ana = idx_num;
                        float mag_num = 0.0f, mag_ana = 0.0f;
                        if (idx_num >= 0 && idx_num < (long long)x.size()) mag_num = std::sqrt(x[idx_num].x * x[idx_num].x + x[idx_num].y * x[idx_num].y);
                        if (idx_ana >= 0 && idx_ana < (long long)x_analytical.size()) mag_ana = std::sqrt(x_analytical[idx_ana].x * x_analytical[idx_ana].x + x_analytical[idx_ana].y * x_analytical[idx_ana].y);
                        std::cout << "E" << comp << "_num=" << mag_num << ", E" << comp << "_ana=" << mag_ana << "; ";
                    }
                    std::cout << std::endl;
                }
            } else if (axis == 'y') {
                int j_end = grid.ny - n_pml - 1;
                int N = std::max(2, j_end - iyc + 1);
                int steps = std::min(20, N);
                for (int s = 0; s < steps; ++s) coords.push_back(iyc + (int)((double)s * (j_end - iyc) / std::max(1, steps-1)));
                for (int j : coords) {
                    long long base = (long long)izc * grid.nx * grid.ny + (long long)j * grid.nx + ixc;
                    std::cout << "j=" << j << ", ";
                    for (int comp = 0; comp < 3; ++comp) {
                        long long idx_num = (long long)comp * grid.nx * grid.ny * grid.nz + base;
                        long long idx_ana = idx_num;
                        float mag_num = 0.0f, mag_ana = 0.0f;
                        if (idx_num >= 0 && idx_num < (long long)x.size()) mag_num = std::sqrt(x[idx_num].x * x[idx_num].x + x[idx_num].y * x[idx_num].y);
                        if (idx_ana >= 0 && idx_ana < (long long)x_analytical.size()) mag_ana = std::sqrt(x_analytical[idx_ana].x * x_analytical[idx_ana].x + x_analytical[idx_ana].y * x_analytical[idx_ana].y);
                        std::cout << "E" << comp << "_num=" << mag_num << ", E" << comp << "_ana=" << mag_ana << "; ";
                    }
                    std::cout << std::endl;
                }
            } else if (axis == 'z') {
                int k_end = grid.nz - n_pml - 1;
                int N = std::max(2, k_end - izc + 1);
                int steps = std::min(20, N);
                for (int s = 0; s < steps; ++s) coords.push_back(izc + (int)((double)s * (k_end - izc) / std::max(1, steps-1)));
                for (int k : coords) {
                    long long base = (long long)k * grid.nx * grid.ny + (long long)iyc * grid.nx + ixc;
                    std::cout << "k=" << k << ", ";
                    for (int comp = 0; comp < 3; ++comp) {
                        long long idx_num = (long long)comp * grid.nx * grid.ny * grid.nz + base;
                        long long idx_ana = idx_num;
                        float mag_num = 0.0f, mag_ana = 0.0f;
                        if (idx_num >= 0 && idx_num < (long long)x.size()) mag_num = std::sqrt(x[idx_num].x * x[idx_num].x + x[idx_num].y * x[idx_num].y);
                        if (idx_ana >= 0 && idx_ana < (long long)x_analytical.size()) mag_ana = std::sqrt(x_analytical[idx_ana].x * x_analytical[idx_ana].x + x_analytical[idx_ana].y * x_analytical[idx_ana].y);
                        std::cout << "E" << comp << "_num=" << mag_num << ", E" << comp << "_ana=" << mag_ana << "; ";
                    }
                    std::cout << std::endl;
                }
            }
        };
        print_scan('x');
        print_scan('y');
        print_scan('z');
    }
    
    return 0;
}

