#include "physics_utils.h"
#include <cmath>
#include <algorithm>
#include <iostream>
#include <complex>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// 辅助函数：计算 PML 导电率
Complex get_pml_sigma(double dist, double L_pml, double sigma_max) {
    if (dist <= 0) return {0.0f, 0.0f};
    // 使用三次多项式渐变 (Cubic Ramp)
    double ratio = dist / L_pml;
    if (ratio > 1.0) ratio = 1.0;
    double val = sigma_max * std::pow(ratio, 3);
    return { (float)val, 0.0f }; 
}

MaterialProperty Physics::compute_cell_material(
    int i, int j, int k, 
    const GridInfo& grid, 
    int n_pml, 
    double omega,
    double rho_t, double rho_n, 
    double theta_deg, double phi_deg
) {
    MaterialProperty mat;
    
    // 1. 基础 TTI 参数
    double sigma_t = 1.0 / rho_t;
    double sigma_n = 1.0 / rho_n;

    double theta = theta_deg * M_PI / 180.0;
    double phi = phi_deg * M_PI / 180.0;

    double nx_val = std::sin(theta) * std::cos(phi);
    double ny_val = std::sin(theta) * std::sin(phi);
    double nz_val = std::cos(theta);

    auto compute_tti_comp = [&](int r, int c) -> Complex {
        double delta = (r == c ? 1.0 : 0.0);
        double n_vec[3] = {nx_val, ny_val, nz_val};
        double val = sigma_t * delta + (sigma_n - sigma_t) * n_vec[r] * n_vec[c];
        return {(float)val, 0.0f};
    };

    for(int r=0; r<3; ++r) {
        for(int c=0; c<3; ++c) {
            mat.sigma_eff.val[r*3 + c] = compute_tti_comp(r, c);
        }
    }
    
    // 2. PML 处理 (适配非均匀网格)
    // 使用节点坐标计算物理距离
    
    // 计算当前单元中心的物理坐标
    double xc = (grid.x_nodes[i] + grid.x_nodes[i+1]) * 0.5;
    double yc = (grid.y_nodes[j] + grid.y_nodes[j+1]) * 0.5;
    double zc = (grid.z_nodes[k] + grid.z_nodes[k+1]) * 0.5;

    double x_dist = 0, y_dist = 0, z_dist = 0;
    double L_pml_x = 0, L_pml_y = 0, L_pml_z = 0;

    // --- X 方向 PML 计算 ---
    if (i < n_pml) {
        // 左边界：PML区域是 [x_nodes[0], x_nodes[n_pml]]
        double interface_x = grid.x_nodes[n_pml];
        L_pml_x = interface_x - grid.x_nodes[0];
        x_dist = interface_x - xc;
    } else if (i >= grid.nx - n_pml) {
        // 右边界：PML区域是 [x_nodes[nx-n_pml], x_nodes[nx]]
        double interface_x = grid.x_nodes[grid.nx - n_pml];
        L_pml_x = grid.x_nodes[grid.nx] - interface_x;
        x_dist = xc - interface_x;
    }

    // --- Y 方向 PML 计算 ---
    if (j < n_pml) {
        double interface_y = grid.y_nodes[n_pml];
        L_pml_y = interface_y - grid.y_nodes[0];
        y_dist = interface_y - yc;
    } else if (j >= grid.ny - n_pml) {
        double interface_y = grid.y_nodes[grid.ny - n_pml];
        L_pml_y = grid.y_nodes[grid.ny] - interface_y;
        y_dist = yc - interface_y;
    }

    // --- Z 方向 PML 计算 ---
    if (k < n_pml) {
        double interface_z = grid.z_nodes[n_pml];
        L_pml_z = interface_z - grid.z_nodes[0];
        z_dist = interface_z - zc;
    } else if (k >= grid.nz - n_pml) {
        // 注意这里：你需要的是 grid.z_nodes[...] 中的元素
        double interface_z = grid.z_nodes[grid.nz - n_pml];
        L_pml_z = grid.z_nodes[grid.nz] - interface_z;
        z_dist = zc - interface_z;
    }

    // 自动调整 Sigma_max
    // 在非均匀网格中，边界网格很大，所以 PML 必须很强才能吸收。
    double sigma_max_val = (1.0/rho_t) * 50.0 + 5.0; 

    Complex sx = get_pml_sigma(x_dist, L_pml_x, sigma_max_val);
    Complex sy = get_pml_sigma(y_dist, L_pml_y, sigma_max_val);
    Complex sz = get_pml_sigma(z_dist, L_pml_z, sigma_max_val);

    for(int d=0; d<3; ++d) {
        float real_sum = sx.real() + sy.real() + sz.real();
        Complex pml_term = {real_sum, 0.0f};
        Complex orig = mat.sigma_eff.val[d*3 + d];
        mat.sigma_eff.val[d*3 + d] = { orig.real() + pml_term.real(), orig.imag() };
    }

    float mu_inv_val = 1.0f / (4.0f * (float)M_PI * 1.0e-7f);
    mat.mu_inv_eff.set_isotropic({mu_inv_val, 0.0f});

    return mat;
}