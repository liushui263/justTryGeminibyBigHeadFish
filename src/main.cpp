#include <iostream>
#include <vector>
#include <cmath>
#include <string>
#include <chrono>
#include <iomanip>
#include <algorithm>
#include "physics_utils.h"
#include "assembler.h"
#include "solver.h"
#include "analytical.h"
#include "exporter.h"

// 简单的 NaN 检查工具
bool has_nan(const std::vector<cuComplexType>& vec, std::string name) {
    for (size_t i = 0; i < vec.size(); ++i) {
        if (std::isnan(vec[i].x) || std::isnan(vec[i].y)) {
            std::cerr << "[ERROR] " << name << " contains NaN at index " << i << "!" << std::endl;
            return true;
        }
    }
    return false;
}

// 非均匀网格生成器
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

void print_grid_stats(const std::vector<double>& nodes, const std::string& name) {
    double total_len = nodes.back() - nodes.front();
    double min_dx = 1e9, max_dx = 0;
    for(size_t i=0; i<nodes.size()-1; ++i) {
        double d = nodes[i+1] - nodes[i];
        if(d < min_dx) min_dx = d;
        if(d > max_dx) max_dx = d;
    }
    std::cout << "  [" << name << "] Radius: " << total_len/2.0 << "m (Total: " << total_len << "m)" << std::endl;
    std::cout << "             Min dx: " << min_dx << "m (Center)" << std::endl;
    std::cout << "             Max dx: " << max_dx << "m (Boundary)" << std::endl;
}

// 适配非均匀网格的材料计算
std::vector<MaterialProperty> compute_materials_nonuniform(
    const GridInfo& grid, double omega, double rho_bg, int n_pml
) {
    long long total_cells = (long long)grid.nx * grid.ny * grid.nz;
    std::vector<MaterialProperty> materials(total_cells);
    double sigma_bg = 1.0 / rho_bg;
    
    // 对于 500Hz，波长极长，PML 导电率需要设置得更强一些
    double sigma_pml_max = sigma_bg * 50.0; 

    #pragma omp parallel for
    for (int idx = 0; idx < total_cells; ++idx) {
        int k = idx / (grid.nx * grid.ny);
        int rem = idx % (grid.nx * grid.ny);
        int j = rem / grid.nx;
        int i = rem % grid.nx;

        MaterialProperty mat;
        mat.sigma_eff.set_isotropic({(float)sigma_bg, 0.0f});
        float mu_inv = 1.0f / (4.0f * M_PI * 1e-7f);
        mat.mu_inv_eff.set_isotropic({mu_inv, 0.0f});

        double pml_x = 0, pml_y = 0, pml_z = 0;
        
        // 基于网格索引添加 PML (简单且鲁棒)
        auto calc_pml = [&](int idx, int max_idx) {
            if (idx < n_pml) {
                double r = (double)(n_pml - idx) / n_pml;
                return sigma_pml_max * r * r * r;
            } else if (idx >= max_idx - n_pml) {
                double r = (double)(idx - (max_idx - n_pml - 1)) / n_pml;
                return sigma_pml_max * r * r * r;
            }
            return 0.0;
        };

        pml_x = calc_pml(i, grid.nx);
        pml_y = calc_pml(j, grid.ny);
        pml_z = calc_pml(k, grid.nz);

        double total_pml = pml_x + pml_y + pml_z;
        if (total_pml > 0) {
            Complex current = mat.sigma_eff.val[0];
            mat.sigma_eff.set_isotropic({current.real() + (float)total_pml, current.imag()});
        }
        materials[idx] = mat;
    }
    return materials;
}

void run_simulation_case(const std::string& case_name, const GridInfo& grid, 
                        const std::vector<MaterialProperty>& materials, double omega) {
    std::cout << "\n>>> Running Case: " << case_name << " <<<" << std::endl;

    auto t1 = std::chrono::high_resolution_clock::now();
    // 使用 Direct Non-Uniform 方法 (use_coord_xform = false)
    Assembler assembler(grid, omega, false);
    CsrMatrix A = assembler.assemble_system_matrix(materials);
    std::cout << "    Matrix Assembled. NNZ: " << A.nnz << std::endl;

    if (has_nan(A.val, "Matrix A")) return;

    std::vector<cuComplexType> rhs(A.num_rows, {0.0f, 0.0f});
    int cx = grid.nx / 2;
    int cy = grid.ny / 2;
    int cz = grid.nz / 2;
    
    // Mz Source Loop
    auto add_edge = [&](int i, int j, int k, int comp, double val) {
        long long idx = (long long)comp * grid.nx * grid.ny * grid.nz 
                        + (long long)k * grid.nx * grid.ny 
                        + (long long)j * grid.nx + i;
        if(idx < (long long)rhs.size()) rhs[idx].x += (float)val;
    };
    
    double mz = 1.0;
    add_edge(cx, cy, cz, 0, -mz);    
    add_edge(cx, cy+1, cz, 0, mz);   
    add_edge(cx, cy, cz, 1, mz);     
    add_edge(cx+1, cy, cz, 1, -mz);  

    std::vector<cuComplexType> x;
    Solver solver;
    try {
        solver.solve(A, rhs, x);
    } catch (const std::exception& e) {
        std::cerr << "Solver failed: " << e.what() << std::endl;
        return;
    }
    
    if (has_nan(x, "Solution X")) {
        std::cout << "    [FAILURE] Solver produced NaNs (Frequency might be too low for float)." << std::endl;
    } else {
        std::cout << "    [SUCCESS] Solver result valid." << std::endl;
        
        // 简单输出中心附近的场值，确认不是全 0
        int center_idx = get_dof_lebedev(grid, cx+5, cy, cz, SubGrid::G000, 1); // Offset 5
        float val = std::sqrt(x[center_idx].x*x[center_idx].x + x[center_idx].y*x[center_idx].y);
        std::cout << "    Check Val (Offset 5): " << val << std::endl;
    }

    auto t2 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
    std::cout << "    Solved in " << ms << " ms." << std::endl;

    std::string filename = case_name + ".vtr";
    VtkExporter::save_to_vtr(filename, grid, x);
    std::cout << "    Result saved to " << filename << std::endl;
}

int main() {
    std::cout << "=== LWD Low Frequency Challenge (500 Hz) ===" << std::endl;
    
    // --- 目标参数 ---
    double freq = 500.0; 
    double omega = 2.0 * M_PI * freq;
    double rho = 20.0; 
    
    // --- 网格设置 ---
    // 增加一点网格数来覆盖更大的区域，同时保持中心细腻
    // 60x60x60 约 65万 单元，QR 分解还能扛得住
    int n_grid = 60;        
    double dx_min = 0.05;   // 中心 5cm
    double ratio = 1.15;    // 增长率 15% (拉伸得更快，以捕捉 500Hz 的超长波长)

    GridInfo grid;
    grid.nx = n_grid; grid.ny = n_grid; grid.nz = n_grid;
    grid.x_nodes = generate_stretched_coords(grid.nx, dx_min, ratio);
    grid.y_nodes = generate_stretched_coords(grid.ny, dx_min, ratio);
    grid.z_nodes = generate_stretched_coords(grid.nz, dx_min, ratio);

    print_grid_stats(grid.x_nodes, "Grid Info");
    // 预期半径应该在 100m - 200m 级别
    
    std::cout << "Generating Material Properties..." << std::endl;
    // 增加 PML 层数到 8，保证低频吸收
    auto materials = compute_materials_nonuniform(grid, omega, rho, 8);

    // 调用仿真
    run_simulation_case("Final_500Hz_Run", grid, materials, omega);

    return 0;
}