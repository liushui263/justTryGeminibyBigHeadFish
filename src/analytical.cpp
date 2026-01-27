#include "analytical.h"
#include <iostream>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

inline Real to_real(double val) { return static_cast<Real>(val); }

Complex AnalyticalSolution::compute_k(double omega, double sigma) {
    double mu = 4.0 * M_PI * 1.0e-7;
    Real val = to_real(omega * mu * sigma);
    Complex i_complex(0.0f, 1.0f);
    return std::sqrt(i_complex * val);
}

void compute_vector_field(
    double x, double y, double z,
    double xs, double ys, double zs,
    Complex k, Complex pre_factor,
    double mx, double my, double mz,
    Complex& Ex_out, Complex& Ey_out, Complex& Ez_out
) {
    double dx = x - xs;
    double dy = y - ys;
    double dz = z - zs;
    double r2 = dx*dx + dy*dy + dz*dz;
    double r_dbl = std::sqrt(r2);

    if (r_dbl < 1e-9) {
        Ex_out = Ey_out = Ez_out = {0,0};
        return;
    }

    Real r = to_real(r_dbl);
    Complex i_unit(0.0f, 1.0f);
    Complex ikr = i_unit * k * r;
    Complex term = (Complex(1.0f, 0.0f) - ikr) * std::exp(ikr) / (Real)(r2 * r_dbl);
    Complex coeff = pre_factor * term;

    double cross_x = my * dz - mz * dy;
    double cross_y = mz * dx - mx * dz;
    double cross_z = mx * dy - my * dx;

    Ex_out = coeff * to_real(cross_x);
    Ey_out = coeff * to_real(cross_y);
    Ez_out = coeff * to_real(cross_z);
}

std::vector<cuComplexType> AnalyticalSolution::compute_field(
    double sigma, double freq, 
    const GridInfo& grid, 
    double src_x_idx, double src_y_idx, double src_z_idx,
    double mx, double my, double mz
) {
    long long n_total = grid.total_unknowns();
    std::vector<cuComplexType> E_exact(n_total, {0.0f, 0.0f});

    double omega_d = 2.0 * M_PI * freq;
    double mu_d = 4.0 * M_PI * 1.0e-7;
    Complex k = compute_k(omega_d, sigma);

    Real omega = to_real(omega_d);
    Real mu = to_real(mu_d);
    Real pi = to_real(M_PI);

    Complex i_unit(0.0f, 1.0f);
    Complex pre_factor = i_unit * omega * mu / (4.0f * pi);

    // Coordinate handling for non-uniform grid
    // For fractional indices (src_x_idx), we interpolate or just use node + offset
    // Assuming src is at a node index (integer)
    int sxi = (int)src_x_idx;
    int syi = (int)src_y_idx;
    int szi = (int)src_z_idx;
    
    // Safety check
    if (sxi >= grid.nx) sxi = grid.nx-1;
    if (syi >= grid.ny) syi = grid.ny-1;
    if (szi >= grid.nz) szi = grid.nz-1;

    double xs = grid.x_nodes[sxi];
    double ys = grid.y_nodes[syi];
    double zs = grid.z_nodes[szi];

    #pragma omp parallel for collapse(3)
    for (int k_idx = 0; k_idx < grid.nz; ++k_idx) {
        for (int j_idx = 0; j_idx < grid.ny; ++j_idx) {
            for (int i_idx = 0; i_idx < grid.nx; ++i_idx) {
                
                // Ex position: center of edge along x -> x_mid, y_node, z_node
                double x_pos = (grid.x_nodes[i_idx] + grid.x_nodes[i_idx+1]) * 0.5;
                double y_pos = grid.y_nodes[j_idx];
                double z_pos = grid.z_nodes[k_idx];

                Complex Ex_val, dy1, dz1;
                compute_vector_field(x_pos, y_pos, z_pos, xs, ys, zs, k, pre_factor, mx, my, mz, Ex_val, dy1, dz1);
                int idx_ex = get_dof_lebedev(grid, i_idx, j_idx, k_idx, SubGrid::G000, 0);
                if(idx_ex < n_total) E_exact[idx_ex] = {Ex_val.real(), Ex_val.imag()};

                // Ey position
                x_pos = grid.x_nodes[i_idx];
                y_pos = (grid.y_nodes[j_idx] + grid.y_nodes[j_idx+1]) * 0.5;
                z_pos = grid.z_nodes[k_idx];
                
                Complex dx2, Ey_val, dz2;
                compute_vector_field(x_pos, y_pos, z_pos, xs, ys, zs, k, pre_factor, mx, my, mz, dx2, Ey_val, dz2);
                int idx_ey = get_dof_lebedev(grid, i_idx, j_idx, k_idx, SubGrid::G000, 1);
                if(idx_ey < n_total) E_exact[idx_ey] = {Ey_val.real(), Ey_val.imag()};

                // Ez position
                x_pos = grid.x_nodes[i_idx];
                y_pos = grid.y_nodes[j_idx];
                z_pos = (grid.z_nodes[k_idx] + grid.z_nodes[k_idx+1]) * 0.5;

                Complex dx3, dy3, Ez_val;
                compute_vector_field(x_pos, y_pos, z_pos, xs, ys, zs, k, pre_factor, mx, my, mz, dx3, dy3, Ez_val);
                int idx_ez = get_dof_lebedev(grid, i_idx, j_idx, k_idx, SubGrid::G000, 2);
                if(idx_ez < n_total) E_exact[idx_ez] = {Ez_val.real(), Ez_val.imag()};
            }
        }
    }
    return E_exact;
}