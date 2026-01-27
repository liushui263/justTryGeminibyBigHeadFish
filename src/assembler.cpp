#include "assembler.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>

// 辅助宏
#define IDX(i, j, k, comp) (long long)((comp) * grid.nx * grid.ny * grid.nz + (k) * grid.nx * grid.ny + (j) * grid.nx + (i))

Assembler::Assembler(const GridInfo& grid_info, double w, bool use_coord_xform)
    : grid(grid_info), omega(w), use_coordinate_transformation(use_coord_xform) {
}

CsrMatrix Assembler::assemble_system_matrix(const std::vector<MaterialProperty>& materials) {
    std::vector<Triplet> triplets;
    // 预估大小：每个节点约 7-13 个非零元，3个分量
    triplets.reserve(grid.total_unknowns() * 15); 

    auto get_geom_coeff = [&](double phys_len) -> double {
        return 1.0 / phys_len;
    };
    
    // =============================
    // 1. Loop for Ex Component
    // =============================
    #pragma omp parallel for
    for (int k = 0; k < grid.nz; ++k) {
        for (int j = 0; j < grid.ny; ++j) {
            for (int i = 0; i < grid.nx; ++i) {
                long long row_idx = IDX(i, j, k, 0); 
                int mat_idx = k * grid.nx * grid.ny + j * grid.nx + i;
                const MaterialProperty& mat = materials[mat_idx];
                
                double inv_dy_d = get_geom_coeff(grid.dy_dual(j));
                double inv_dz_d = get_geom_coeff(grid.dz_dual(k));

                // Mass Term (Diagonal): -i * omega * sigma_xx
                Complex sigma_val = mat.sigma_eff.val[0]; 
                Complex mass_term = {0.0f, -1.0f}; 
                mass_term = mass_term * (float)omega * sigma_val;
                
                #pragma omp critical
                triplets.push_back({(int)row_idx, (int)row_idx, mass_term});

                // Curl-Curl Terms (Ex)
                Complex mu_inv = mat.mu_inv_eff.val[0];
                
                // - d/dy (1/mu dEx/dy)
                if (j + 1 < grid.ny) {
                    double inv_dy_p_next = get_geom_coeff(grid.dy(j));
                    double term = inv_dy_d * inv_dy_p_next * mu_inv.real(); 
                    int col_idx = IDX(i, j+1, k, 0);
                    #pragma omp critical 
                    {
                        triplets.push_back({(int)row_idx, col_idx, {(float)(-term), 0.0f}});
                        triplets.push_back({(int)row_idx, (int)row_idx, {(float)term, 0.0f}});
                    }
                }
                if (j - 1 >= 0) {
                    double inv_dy_p_prev = get_geom_coeff(grid.dy(j-1));
                    double term = inv_dy_d * inv_dy_p_prev * mu_inv.real();
                    int col_idx = IDX(i, j-1, k, 0);
                    #pragma omp critical
                    {
                        triplets.push_back({(int)row_idx, col_idx, {(float)(-term), 0.0f}});
                        triplets.push_back({(int)row_idx, (int)row_idx, {(float)term, 0.0f}});
                    }
                }

                // - d/dz (1/mu dEx/dz)
                if (k + 1 < grid.nz) {
                    double inv_dz_p_next = get_geom_coeff(grid.dz(k));
                    double term = inv_dz_d * inv_dz_p_next * mu_inv.real();
                    int col_idx = IDX(i, j, k+1, 0);
                    #pragma omp critical
                    {
                        triplets.push_back({(int)row_idx, col_idx, {(float)(-term), 0.0f}});
                        triplets.push_back({(int)row_idx, (int)row_idx, {(float)term, 0.0f}});
                    }
                }
                if (k - 1 >= 0) {
                    double inv_dz_p_prev = get_geom_coeff(grid.dz(k-1));
                    double term = inv_dz_d * inv_dz_p_prev * mu_inv.real();
                    int col_idx = IDX(i, j, k-1, 0);
                    #pragma omp critical
                    {
                        triplets.push_back({(int)row_idx, col_idx, {(float)(-term), 0.0f}});
                        triplets.push_back({(int)row_idx, (int)row_idx, {(float)term, 0.0f}});
                    }
                }
            }
        }
    }

    // =============================
    // 2. Loop for Ey Component
    // =============================
    #pragma omp parallel for
    for (int k = 0; k < grid.nz; ++k) {
        for (int j = 0; j < grid.ny; ++j) {
            for (int i = 0; i < grid.nx; ++i) {
                long long row_idx = IDX(i, j, k, 1); // Component 1
                int mat_idx = k * grid.nx * grid.ny + j * grid.nx + i;
                const MaterialProperty& mat = materials[mat_idx];

                // Ey involves d/dz and d/dx
                double inv_dz_d = get_geom_coeff(grid.dz_dual(k));
                double inv_dx_d = get_geom_coeff(grid.dx_dual(i));

                // Mass Term: sigma_yy (index 4)
                Complex sigma_val = mat.sigma_eff.val[4]; 
                Complex mass_term = {0.0f, -1.0f}; 
                mass_term = mass_term * (float)omega * sigma_val;
                
                #pragma omp critical
                triplets.push_back({(int)row_idx, (int)row_idx, mass_term});

                Complex mu_inv = mat.mu_inv_eff.val[4]; // mu_yy

                // - d/dz (1/mu dEy/dz)
                if (k + 1 < grid.nz) {
                    double inv_dz_p_next = get_geom_coeff(grid.dz(k));
                    double term = inv_dz_d * inv_dz_p_next * mu_inv.real();
                    int col_idx = IDX(i, j, k+1, 1);
                    #pragma omp critical
                    {
                        triplets.push_back({(int)row_idx, col_idx, {(float)(-term), 0.0f}});
                        triplets.push_back({(int)row_idx, (int)row_idx, {(float)term, 0.0f}});
                    }
                }
                if (k - 1 >= 0) {
                    double inv_dz_p_prev = get_geom_coeff(grid.dz(k-1));
                    double term = inv_dz_d * inv_dz_p_prev * mu_inv.real();
                    int col_idx = IDX(i, j, k-1, 1);
                    #pragma omp critical
                    {
                        triplets.push_back({(int)row_idx, col_idx, {(float)(-term), 0.0f}});
                        triplets.push_back({(int)row_idx, (int)row_idx, {(float)term, 0.0f}});
                    }
                }

                // - d/dx (1/mu dEy/dx)
                if (i + 1 < grid.nx) {
                    double inv_dx_p_next = get_geom_coeff(grid.dx(i));
                    double term = inv_dx_d * inv_dx_p_next * mu_inv.real();
                    int col_idx = IDX(i+1, j, k, 1);
                    #pragma omp critical
                    {
                        triplets.push_back({(int)row_idx, col_idx, {(float)(-term), 0.0f}});
                        triplets.push_back({(int)row_idx, (int)row_idx, {(float)term, 0.0f}});
                    }
                }
                if (i - 1 >= 0) {
                    double inv_dx_p_prev = get_geom_coeff(grid.dx(i-1));
                    double term = inv_dx_d * inv_dx_p_prev * mu_inv.real();
                    int col_idx = IDX(i-1, j, k, 1);
                    #pragma omp critical
                    {
                        triplets.push_back({(int)row_idx, col_idx, {(float)(-term), 0.0f}});
                        triplets.push_back({(int)row_idx, (int)row_idx, {(float)term, 0.0f}});
                    }
                }
            }
        }
    }

    // =============================
    // 3. Loop for Ez Component
    // =============================
    #pragma omp parallel for
    for (int k = 0; k < grid.nz; ++k) {
        for (int j = 0; j < grid.ny; ++j) {
            for (int i = 0; i < grid.nx; ++i) {
                long long row_idx = IDX(i, j, k, 2); // Component 2
                int mat_idx = k * grid.nx * grid.ny + j * grid.nx + i;
                const MaterialProperty& mat = materials[mat_idx];

                // Ez involves d/dx and d/dy
                double inv_dx_d = get_geom_coeff(grid.dx_dual(i));
                double inv_dy_d = get_geom_coeff(grid.dy_dual(j));

                // Mass Term: sigma_zz (index 8)
                Complex sigma_val = mat.sigma_eff.val[8]; 
                Complex mass_term = {0.0f, -1.0f}; 
                mass_term = mass_term * (float)omega * sigma_val;
                
                #pragma omp critical
                triplets.push_back({(int)row_idx, (int)row_idx, mass_term});

                Complex mu_inv = mat.mu_inv_eff.val[8]; // mu_zz

                // - d/dx (1/mu dEz/dx)
                if (i + 1 < grid.nx) {
                    double inv_dx_p_next = get_geom_coeff(grid.dx(i));
                    double term = inv_dx_d * inv_dx_p_next * mu_inv.real();
                    int col_idx = IDX(i+1, j, k, 2);
                    #pragma omp critical
                    {
                        triplets.push_back({(int)row_idx, col_idx, {(float)(-term), 0.0f}});
                        triplets.push_back({(int)row_idx, (int)row_idx, {(float)term, 0.0f}});
                    }
                }
                if (i - 1 >= 0) {
                    double inv_dx_p_prev = get_geom_coeff(grid.dx(i-1));
                    double term = inv_dx_d * inv_dx_p_prev * mu_inv.real();
                    int col_idx = IDX(i-1, j, k, 2);
                    #pragma omp critical
                    {
                        triplets.push_back({(int)row_idx, col_idx, {(float)(-term), 0.0f}});
                        triplets.push_back({(int)row_idx, (int)row_idx, {(float)term, 0.0f}});
                    }
                }

                // - d/dy (1/mu dEz/dy)
                if (j + 1 < grid.ny) {
                    double inv_dy_p_next = get_geom_coeff(grid.dy(j));
                    double term = inv_dy_d * inv_dy_p_next * mu_inv.real();
                    int col_idx = IDX(i, j+1, k, 2);
                    #pragma omp critical
                    {
                        triplets.push_back({(int)row_idx, col_idx, {(float)(-term), 0.0f}});
                        triplets.push_back({(int)row_idx, (int)row_idx, {(float)term, 0.0f}});
                    }
                }
                if (j - 1 >= 0) {
                    double inv_dy_p_prev = get_geom_coeff(grid.dy(j-1));
                    double term = inv_dy_d * inv_dy_p_prev * mu_inv.real();
                    int col_idx = IDX(i, j-1, k, 2);
                    #pragma omp critical
                    {
                        triplets.push_back({(int)row_idx, col_idx, {(float)(-term), 0.0f}});
                        triplets.push_back({(int)row_idx, (int)row_idx, {(float)term, 0.0f}});
                    }
                }
            }
        }
    }

    // === Manual Triplet to CSR Conversion ===
    std::sort(triplets.begin(), triplets.end(), [](const Triplet& a, const Triplet& b) {
        if (a.row != b.row) return a.row < b.row;
        return a.col < b.col;
    });

    std::vector<Triplet> merged;
    if (!triplets.empty()) {
        merged.reserve(triplets.size());
        int curr_row = triplets[0].row;
        int curr_col = triplets[0].col;
        Complex curr_sum = triplets[0].val;
        
        for(size_t i=1; i<triplets.size(); ++i) {
            if (triplets[i].row == curr_row && triplets[i].col == curr_col) {
                curr_sum += triplets[i].val;
            } else {
                merged.push_back({curr_row, curr_col, curr_sum});
                curr_row = triplets[i].row;
                curr_col = triplets[i].col;
                curr_sum = triplets[i].val;
            }
        }
        merged.push_back({curr_row, curr_col, curr_sum});
    }

    CsrMatrix csr;
    csr.num_rows = 3 * grid.nx * grid.ny * grid.nz;
    csr.num_cols = csr.num_rows;
    csr.nnz = merged.size();
    
    csr.row_ptr.assign(csr.num_rows + 1, 0);
    csr.col_ind.resize(csr.nnz);
    csr.val.resize(csr.nnz);
    
    for(const auto& t : merged) {
        csr.row_ptr[t.row + 1]++;
    }
    for(int i=0; i<csr.num_rows; ++i) {
        csr.row_ptr[i+1] += csr.row_ptr[i];
    }
    
    for(size_t i=0; i<merged.size(); ++i) {
        csr.col_ind[i] = merged[i].col;
        csr.val[i] = {merged[i].val.real(), merged[i].val.imag()};
    }
    
    return csr;
}