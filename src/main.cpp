#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <algorithm> // <--- 之前缺失的关键头文件 (用于 std::min_element)
#include <iterator>  // 用于 std::distance
#include "physics_utils.h"
#include "assembler.h"
#include "solver.h"
#include "analytical.h"
#include "exporter.h"
#include "config_types.h" 

// 声明解析函数 (在 config_loader.cpp 中实现)
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
    std::vector<cuComplexType>& rhs
) {
    // 1. 找到源在网格中的索引
    int ix = find_nearest_node(grid.x_nodes, src.position[0]);
    int iy = find_nearest_node(grid.y_nodes, src.position[1]);
    int iz = find_nearest_node(grid.z_nodes, src.position[2]);
    
    // 边界保护
    if (ix >= grid.nx) ix = grid.nx-1; 
    if (iy >= grid.ny) iy = grid.ny-1; 
    if (iz >= grid.nz) iz = grid.nz-1;

    double amp = src.amplitude;
    double nx = src.direction[0];
    double ny = src.direction[1];
    double nz = src.direction[2];

    auto add_rhs = [&](int i, int j, int k, int comp, double val) {
        long long idx = (long long)comp * grid.nx * grid.ny * grid.nz 
                        + (long long)k * grid.nx * grid.ny 
                        + (long long)j * grid.nx + i;
        if(idx < (long long)rhs.size()) rhs[idx].x += (float)val;
    };

    if (src.type == "magnetic_dipole") {
        // 磁偶极子 M (电流环)
        // M_z (在 XY 平面上的环)
        if (std::abs(nz) > 1e-6) {
            double mz = amp * nz;
            add_rhs(ix, iy, iz, 0, -mz);    // Ex(y_bot)
            add_rhs(ix, iy+1, iz, 0, mz);   // Ex(y_top)
            add_rhs(ix, iy, iz, 1, mz);     // Ey(x_left)
            add_rhs(ix+1, iy, iz, 1, -mz);  // Ey(x_right)
        }
        // M_x (在 YZ 平面上的环)
        if (std::abs(nx) > 1e-6) {
            double mx = amp * nx;
            add_rhs(ix, iy, iz, 1, -mx);    // Ey(z_bot)
            add_rhs(ix, iy, iz+1, 1, mx);   // Ey(z_top)
            add_rhs(ix, iy, iz, 2, mx);     // Ez(y_left)
            add_rhs(ix, iy+1, iz, 2, -mx);  // Ez(y_right)
        }
        // M_y (在 XZ 平面上的环)
        if (std::abs(ny) > 1e-6) {
            double my = amp * ny;
            add_rhs(ix, iy, iz, 2, -my);    // Ez(x_left)
            add_rhs(ix+1, iy, iz, 2, my);   // Ez(x_right)
            add_rhs(ix, iy, iz, 0, my);     // Ex(z_bot)
            add_rhs(ix, iy, iz+1, 0, -my);  // Ex(z_top)
        }
    } else if (src.type == "electric_dipole") {
        // 电偶极子 J (直接加在边上)
        // 注意：这里做最简单的最近邻插值。更精确的做法是将 J 分配到周围的边上。
        if (std::abs(nx) > 1e-6) add_rhs(ix, iy, iz, 0, amp * nx);
        if (std::abs(ny) > 1e-6) add_rhs(ix, iy, iz, 1, amp * ny);
        if (std::abs(nz) > 1e-6) add_rhs(ix, iy, iz, 2, amp * nz);
    }
}

// --- 材料生成 (支持简单模型) ---
std::vector<MaterialProperty> generate_materials(const SimulationConfig& cfg, const GridInfo& grid, double omega) {
    long long total_cells = (long long)grid.nx * grid.ny * grid.nz;
    std::vector<MaterialProperty> materials(total_cells);
    
    // 默认简单模型
    double rho = cfg.model.bg_rho;
    double sigma_bg = 1.0 / rho;
    
    // 自动调节 PML 强度: 频率越低(波长越长)，需要的 PML 越强
    double sigma_pml_max = sigma_bg * 50.0; 
    int n_pml = cfg.model.pml_layers;

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

        // PML Logic (Simplified Index Based)
        auto get_pml = [&](int idx, int max) {
            if(idx < n_pml) return sigma_pml_max * std::pow((double)(n_pml-idx)/n_pml, 3);
            if(idx >= max-n_pml) return sigma_pml_max * std::pow((double)(idx-(max-n_pml-1))/n_pml, 3);
            return 0.0;
        };
        double px = get_pml(i, grid.nx);
        double py = get_pml(j, grid.ny);
        double pz = get_pml(k, grid.nz);
        
        double total_pml = px + py + pz;
        if (total_pml > 0) {
            Complex c = mat.sigma_eff.val[0];
            mat.sigma_eff.set_isotropic({c.real() + (float)total_pml, c.imag()});
        }
        materials[idx] = mat;
    }
    return materials;
}

int main(int argc, char** argv) {
    std::string config_file = "test_500hz.json"; // 默认输入文件
    if (argc > 1) config_file = argv[1];

    std::cout << "=== Universal LWD Solver (BigHeadFish) ===" << std::endl;
    std::cout << "Loading config: " << config_file << std::endl;

    // 1. 加载配置
    SimulationConfig cfg = load_config(config_file);

    // 2. 生成网格
    GridInfo grid;
    grid.nx = cfg.grid.nx; grid.ny = cfg.grid.ny; grid.nz = cfg.grid.nz;
    grid.x_nodes = generate_stretched_coords(grid.nx, cfg.grid.dx_min, cfg.grid.stretch_ratio);
    grid.y_nodes = generate_stretched_coords(grid.ny, cfg.grid.dx_min, cfg.grid.stretch_ratio);
    grid.z_nodes = generate_stretched_coords(grid.nz, cfg.grid.dx_min, cfg.grid.stretch_ratio);

    std::cout << "Grid: " << grid.nx << "x" << grid.ny << "x" << grid.nz 
              << " (Min dx=" << cfg.grid.dx_min << ")" << std::endl;

    // 3. 准备物理参数
    double omega = 2.0 * M_PI * cfg.source.frequency;
    auto materials = generate_materials(cfg, grid, omega);

    // 4. 组装矩阵
    std::cout << "Assembling Matrix..." << std::endl;
    Assembler assembler(grid, omega, false); // Default direct method
    CsrMatrix A = assembler.assemble_system_matrix(materials);
    std::cout << "NNZ: " << A.nnz << std::endl;

    // 5. 施加源
    std::vector<cuComplexType> rhs(A.num_rows, {0.0f, 0.0f});
    apply_source_generic(cfg.source, grid, rhs);

    // 6. 求解
    std::cout << "Solving..." << std::endl;
    std::vector<cuComplexType> x;
    Solver solver;
    try {
        solver.solve(A, rhs, x);
        std::cout << "Solved." << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Solver Failed: " << e.what() << std::endl;
        return 1;
    }

    // 7. 输出
    if (cfg.receiver.type == "full_grid") {
        std::string vtk_name = cfg.receiver.output_file + ".vtr";
        VtkExporter::save_to_vtr(vtk_name, grid, x);
        std::cout << "Full field saved to " << vtk_name << std::endl;
    } 
    
    // 8. 验证
    if (cfg.validation.enabled && cfg.validation.mode == "analytical") {
        std::cout << "[Validation] Comparing with Analytical Solution..." << std::endl;
        // 计算源中心索引 (近似)
        int ix = find_nearest_node(grid.x_nodes, cfg.source.position[0]);
        int iy = find_nearest_node(grid.y_nodes, cfg.source.position[1]);
        int iz = find_nearest_node(grid.z_nodes, cfg.source.position[2]);
        
        auto exact = AnalyticalSolution::compute_field(
            1.0/cfg.model.bg_rho, cfg.source.frequency, grid, 
            ix+0.5, iy+0.5, iz+0.5, 
            cfg.source.direction[0], cfg.source.direction[1], cfg.source.direction[2]
        );
        VtkExporter::save_to_vtr(cfg.receiver.output_file + "_analytical.vtr", grid, exact);
        std::cout << "Analytical solution saved to " << cfg.receiver.output_file + "_analytical.vtr" << std::endl;
    }

    return 0;
}