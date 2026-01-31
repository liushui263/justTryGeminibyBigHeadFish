#include "assembler.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <map>
#include <set>
#include <omp.h>
#include <chrono>

#define IDX(i, j, k, comp) (long long)((comp) * grid.nx * grid.ny * grid.nz + (k) * grid.nx * grid.ny + (j) * grid.nx + (i))

static CsrMatrix convert_to_csr(const std::vector<Triplet>& triplets, long long n);

Assembler::Assembler(const GridInfo& grid_info, double w, bool use_coord_xform)
    : grid(grid_info), omega(w), use_coordinate_transformation(use_coord_xform) {
}
CsrMatrix Assembler::assemble_system_matrix(const std::vector<MaterialProperty>& materials) {
    int num_threads = omp_get_max_threads();
    std::vector<std::vector<Triplet>> thread_triplets(num_threads);

    #pragma omp parallel for collapse(3)
    for (int k = 0; k < grid.nz; ++k) {
        for (int j = 0; j < grid.ny; ++j) {
            for (int i = 0; i < grid.nx; ++i) {
                int tid = omp_get_thread_num();
                auto& triplets = thread_triplets[tid];
                int m_idx = k * grid.nx * grid.ny + j * grid.nx + i;
                const auto& mat = materials[m_idx];

                // Geometric factors for non-uniform grid [cite: 11, 75]
                double dx = (i < grid.nx - 1) ? (grid.x_nodes[i+1] - grid.x_nodes[i]) : (grid.x_nodes[i] - grid.x_nodes[i-1]);
                double dy = (j < grid.ny - 1) ? (grid.y_nodes[j+1] - grid.y_nodes[j]) : (grid.y_nodes[j] - grid.y_nodes[j-1]);
                double dz = (k < grid.nz - 1) ? (grid.z_nodes[k+1] - grid.z_nodes[k]) : (grid.z_nodes[k] - grid.z_nodes[k-1]);
                
                // Use .real() for std::complex instead of .x
                float inv_mu = mat.mu_inv_eff.val[0].real(); 

                for (int comp = 0; comp < 3; ++comp) {
                    long long row = IDX(i, j, k, comp);
                    if (row >= grid.total_unknowns()) continue;

                    // 1. CURL-CURL OPERATOR (7-Point Stencil) [cite: 28, 81]
                    float stencil_diag = 0.0f;
                    if (comp == 0) { // Ex
                        stencil_diag = inv_mu * (2.0f/(dy*dy) + 2.0f/(dz*dz));
                        if (j > 0) triplets.push_back({(int)row, (int)IDX(i, j-1, k, 0), {-inv_mu/(float)(dy*dy), 0}});
                        if (j < grid.ny-1) triplets.push_back({(int)row, (int)IDX(i, j+1, k, 0), {-inv_mu/(float)(dy*dy), 0}});
                        if (k > 0) triplets.push_back({(int)row, (int)IDX(i, j, k-1, 0), {-inv_mu/(float)(dz*dz), 0}});
                        if (k < grid.nz-1) triplets.push_back({(int)row, (int)IDX(i, j, k+1, 0), {-inv_mu/(float)(dz*dz), 0}});
                    } 
                    else if (comp == 1) { // Ey
                        stencil_diag = inv_mu * (2.0f/(dx*dx) + 2.0f/(dz*dz));
                        if (i > 0) triplets.push_back({(int)row, (int)IDX(i-1, j, k, 1), {-inv_mu/(float)(dx*dx), 0}});
                        if (i < grid.nx-1) triplets.push_back({(int)row, (int)IDX(i+1, j, k, 1), {-inv_mu/(float)(dx*dx), 0}});
                        if (k > 0) triplets.push_back({(int)row, (int)IDX(i, j, k-1, 1), {-inv_mu/(float)(dz*dz), 0}});
                        if (k < grid.nz-1) triplets.push_back({(int)row, (int)IDX(i, j, k+1, 1), {-inv_mu/(float)(dz*dz), 0}});
                    } 
                    else if (comp == 2) { // Ez
                        stencil_diag = inv_mu * (2.0f/(dx*dx) + 2.0f/(dy*dy));
                        if (i > 0) triplets.push_back({(int)row, (int)IDX(i-1, j, k, 2), {-inv_mu/(float)(dx*dx), 0}});
                        if (i < grid.nx-1) triplets.push_back({(int)row, (int)IDX(i+1, j, k, 2), {-inv_mu/(float)(dx*dx), 0}});
                        if (j > 0) triplets.push_back({(int)row, (int)IDX(i, j-1, k, 2), {-inv_mu/(float)(dy*dy), 0}});
                        if (j < grid.ny-1) triplets.push_back({(int)row, (int)IDX(i, j+1, k, 2), {-inv_mu/(float)(dy*dy), 0}});
                    }

                    // 2. MASS MATRIX DIAGONAL [cite: 55, 85]
                    Complex mass_diag = Complex(0.0f, -1.0f) * (float)omega * mat.sigma_eff.val[comp * 4];
                    triplets.push_back({(int)row, (int)row, mass_diag + Complex(stencil_diag, 0.0f)});

                    // 3. TTI COUPLING (Off-diagonal mass matrix) [cite: 69, 71]
                    if (comp == 0) {
                        triplets.push_back({(int)row, (int)IDX(i,j,k,1), Complex(0.0f, -1.0f) * (float)omega * mat.sigma_eff.val[1]});
                        triplets.push_back({(int)row, (int)IDX(i,j,k,2), Complex(0.0f, -1.0f) * (float)omega * mat.sigma_eff.val[2]});
                    } else if (comp == 1) {
                        triplets.push_back({(int)row, (int)IDX(i,j,k,0), Complex(0.0f, -1.0f) * (float)omega * mat.sigma_eff.val[3]});
                        triplets.push_back({(int)row, (int)IDX(i,j,k,2), Complex(0.0f, -1.0f) * (float)omega * mat.sigma_eff.val[5]});
                    } else if (comp == 2) {
                        triplets.push_back({(int)row, (int)IDX(i,j,k,0), Complex(0.0f, -1.0f) * (float)omega * mat.sigma_eff.val[6]});
                        triplets.push_back({(int)row, (int)IDX(i,j,k,1), Complex(0.0f, -1.0f) * (float)omega * mat.sigma_eff.val[7]});
                    }
                }
            }
        }
    }

    // Merge and apply Boundary Conditions
    std::vector<Triplet> merged;
    for (auto& t : thread_triplets) merged.insert(merged.end(), t.begin(), t.end());

    std::vector<bool> is_bc(grid.total_unknowns(), false);
    for (int k = 0; k < grid.nz; ++k) {
        for (int j = 0; j < grid.ny; ++j) {
            for (int i = 0; i < grid.nx; ++i) {
                if (i == 0 || i == grid.nx - 1 || j == 0 || j == grid.ny - 1 || k == 0 || k == grid.nz - 1) {
                    for (int c = 0; c < 3; ++c) is_bc[IDX(i, j, k, c)] = true;
                }
            }
        }
    }

    auto new_end = std::remove_if(merged.begin(), merged.end(), [&](const Triplet& t) {
        return is_bc[t.row];
    });
    merged.erase(new_end, merged.end());

    for (long long i = 0; i < grid.total_unknowns(); ++i) {
        if (is_bc[i]) merged.push_back({(int)i, (int)i, {1.0f, 0.0f}});
    }

    return convert_to_csr(merged, grid.total_unknowns());
}
CsrMatrix Assembler::assemble_system_matrix_(const std::vector<MaterialProperty>& materials) {
    auto t_start = std::chrono::high_resolution_clock::now();
    
    std::vector<std::vector<Triplet>> thread_triplets(omp_get_max_threads());
    for (auto& tv : thread_triplets) {
        tv.reserve((size_t)std::max(1LL, grid.total_unknowns() / omp_get_max_threads()));
    }

    auto get_geom_coeff = [&](double phys_len) -> double {
        return 1.0 / phys_len;
    };
    
    // Ex Component
    #pragma omp parallel for collapse(3)
    for (int k = 0; k < grid.nz; ++k) {
        for (int j = 0; j < grid.ny; ++j) {
            for (int i = 0; i < grid.nx; ++i) {
                int tid = omp_get_thread_num();
                auto& triplets = thread_triplets[tid];
                
                long long row_idx = IDX(i, j, k, 0); 
                int mat_idx = k * grid.nx * grid.ny + j * grid.nx + i;
                const MaterialProperty& mat = materials[mat_idx];
                
                double inv_dy_d = get_geom_coeff(grid.dy_dual(j));
                double inv_dz_d = get_geom_coeff(grid.dz_dual(k));

                Complex sigma_val = mat.sigma_eff.val[0]; 
                Complex mass_term = Complex(0.0f, -1.0f) * (float)omega * sigma_val;
                Complex diag_sum = mass_term;
                Complex mu_inv = mat.mu_inv_eff.val[0];
                
                if (j + 1 < grid.ny) {
                    double inv_dy_p_next = get_geom_coeff(grid.dy(j));
                    double term = inv_dy_d * inv_dy_p_next * mu_inv.real(); 
                    int col_idx = IDX(i, j+1, k, 0);
                    triplets.push_back({(int)row_idx, col_idx, Complex((float)(-term), 0.0f)});
                    diag_sum += Complex((float)term, 0.0f);
                }
                if (j - 1 >= 0) {
                    double inv_dy_p_prev = get_geom_coeff(grid.dy(j-1));
                    double term = inv_dy_d * inv_dy_p_prev * mu_inv.real();
                    int col_idx = IDX(i, j-1, k, 0);
                    triplets.push_back({(int)row_idx, col_idx, Complex((float)(-term), 0.0f)});
                    diag_sum += Complex((float)term, 0.0f);
                }
                if (k + 1 < grid.nz) {
                    double inv_dz_p_next = get_geom_coeff(grid.dz(k));
                    double term = inv_dz_d * inv_dz_p_next * mu_inv.real();
                    int col_idx = IDX(i, j, k+1, 0);
                    triplets.push_back({(int)row_idx, col_idx, Complex((float)(-term), 0.0f)});
                    diag_sum += Complex((float)term, 0.0f);
                }
                if (k - 1 >= 0) {
                    double inv_dz_p_prev = get_geom_coeff(grid.dz(k-1));
                    double term = inv_dz_d * inv_dz_p_prev * mu_inv.real();
                    int col_idx = IDX(i, j, k-1, 0);
                    triplets.push_back({(int)row_idx, col_idx, Complex((float)(-term), 0.0f)});
                    diag_sum += Complex((float)term, 0.0f);
                }

                // 2. TTI COUPLING: Add off-diagonal mass terms [cite: 69, 71]
                // Ex depends on Ey via sigma_xy (val[1])
                Complex mass_xy = Complex(0.0f, -1.0f) * (float)omega * mat.sigma_eff.val[1];
                int col_ey = (int)IDX(i, j, k, 1);
                triplets.push_back({(int)row_idx, col_ey, mass_xy});

                // Ex depends on Ez via sigma_xz (val[2])
                Complex mass_xz = Complex(0.0f, -1.0f) * (float)omega * mat.sigma_eff.val[2];
                int col_ez = (int)IDX(i, j, k, 2);
                triplets.push_back({(int)row_idx, col_ez, mass_xz});

                triplets.push_back({(int)row_idx, (int)row_idx, diag_sum});
            }
        }
    }

    // Ey Component
    #pragma omp parallel for collapse(3)
    for (int k = 0; k < grid.nz; ++k) {
        for (int j = 0; j < grid.ny; ++j) {
            for (int i = 0; i < grid.nx; ++i) {
                int tid = omp_get_thread_num();
                auto& triplets = thread_triplets[tid];
                
                long long row_idx = IDX(i, j, k, 1);
                int mat_idx = k * grid.nx * grid.ny + j * grid.nx + i;
                const MaterialProperty& mat = materials[mat_idx];

                double inv_dz_d = get_geom_coeff(grid.dz_dual(k));
                double inv_dx_d = get_geom_coeff(grid.dx_dual(i));

                Complex sigma_val = mat.sigma_eff.val[4]; 
                Complex mass_term = Complex(0.0f, -1.0f) * (float)omega * sigma_val;
                Complex diag_sum = mass_term;
                Complex mu_inv = mat.mu_inv_eff.val[4];

                if (k + 1 < grid.nz) {
                    double inv_dz_p_next = get_geom_coeff(grid.dz(k));
                    double term = inv_dz_d * inv_dz_p_next * mu_inv.real();
                    int col_idx = IDX(i, j, k+1, 1);
                    triplets.push_back({(int)row_idx, col_idx, Complex((float)(-term), 0.0f)});
                    diag_sum += Complex((float)term, 0.0f);
                }
                if (k - 1 >= 0) {
                    double inv_dz_p_prev = get_geom_coeff(grid.dz(k-1));
                    double term = inv_dz_d * inv_dz_p_prev * mu_inv.real();
                    int col_idx = IDX(i, j, k-1, 1);
                    triplets.push_back({(int)row_idx, col_idx, Complex((float)(-term), 0.0f)});
                    diag_sum += Complex((float)term, 0.0f);
                }
                if (i + 1 < grid.nx) {
                    double inv_dx_p_next = get_geom_coeff(grid.dx(i));
                    double term = inv_dx_d * inv_dx_p_next * mu_inv.real();
                    int col_idx = IDX(i+1, j, k, 1);
                    triplets.push_back({(int)row_idx, col_idx, Complex((float)(-term), 0.0f)});
                    diag_sum += Complex((float)term, 0.0f);
                }
                if (i - 1 >= 0) {
                    double inv_dx_p_prev = get_geom_coeff(grid.dx(i-1));
                    double term = inv_dx_d * inv_dx_p_prev * mu_inv.real();
                    int col_idx = IDX(i-1, j, k, 1);
                    triplets.push_back({(int)row_idx, col_idx, Complex((float)(-term), 0.0f)});
                    diag_sum += Complex((float)term, 0.0f);
                }

                // --- TTI COUPLING FOR Ey ---
                // Ey depends on Ex via sigma_yx (val[3] in a row-major 3x3) [cite: 69, 71]
                Complex mass_yx = Complex(0.0f, -1.0f) * (float)omega * mat.sigma_eff.val[3];
                int col_ex = (int)IDX(i, j, k, 0);
                triplets.push_back({(int)row_idx, col_ex, mass_yx});

                // Ey depends on Ez via sigma_yz (val[5]) [cite: 69, 71]
                Complex mass_yz = Complex(0.0f, -1.0f) * (float)omega * mat.sigma_eff.val[5];
                int col_ez = (int)IDX(i, j, k, 2);
                triplets.push_back({(int)row_idx, col_ez, mass_yz});

                triplets.push_back({(int)row_idx, (int)row_idx, diag_sum});
            }
        }
    }

    // Ez Component
    #pragma omp parallel for collapse(3)
    for (int k = 0; k < grid.nz; ++k) {
        for (int j = 0; j < grid.ny; ++j) {
            for (int i = 0; i < grid.nx; ++i) {
                int tid = omp_get_thread_num();
                auto& triplets = thread_triplets[tid];
                
                long long row_idx = IDX(i, j, k, 2);
                int mat_idx = k * grid.nx * grid.ny + j * grid.nx + i;
                const MaterialProperty& mat = materials[mat_idx];

                double inv_dx_d = get_geom_coeff(grid.dx_dual(i));
                double inv_dy_d = get_geom_coeff(grid.dy_dual(j));

                Complex sigma_val = mat.sigma_eff.val[8]; 
                Complex mass_term = Complex(0.0f, -1.0f) * (float)omega * sigma_val;
                Complex diag_sum = mass_term;
                Complex mu_inv = mat.mu_inv_eff.val[8];

                if (i + 1 < grid.nx) {
                    double inv_dx_p_next = get_geom_coeff(grid.dx(i));
                    double term = inv_dx_d * inv_dx_p_next * mu_inv.real();
                    int col_idx = IDX(i+1, j, k, 2);
                    triplets.push_back({(int)row_idx, col_idx, Complex((float)(-term), 0.0f)});
                    diag_sum += Complex((float)term, 0.0f);
                }
                if (i - 1 >= 0) {
                    double inv_dx_p_prev = get_geom_coeff(grid.dx(i-1));
                    double term = inv_dx_d * inv_dx_p_prev * mu_inv.real();
                    int col_idx = IDX(i-1, j, k, 2);
                    triplets.push_back({(int)row_idx, col_idx, Complex((float)(-term), 0.0f)});
                    diag_sum += Complex((float)term, 0.0f);
                }
                if (j + 1 < grid.ny) {
                    double inv_dy_p_next = get_geom_coeff(grid.dy(j));
                    double term = inv_dy_d * inv_dy_p_next * mu_inv.real();
                    int col_idx = IDX(i, j+1, k, 2);
                    triplets.push_back({(int)row_idx, col_idx, Complex((float)(-term), 0.0f)});
                    diag_sum += Complex((float)term, 0.0f);
                }
                if (j - 1 >= 0) {
                    double inv_dy_p_prev = get_geom_coeff(grid.dy(j-1));
                    double term = inv_dy_d * inv_dy_p_prev * mu_inv.real();
                    int col_idx = IDX(i, j-1, k, 2);
                    triplets.push_back({(int)row_idx, col_idx, Complex((float)(-term), 0.0f)});
                    diag_sum += Complex((float)term, 0.0f);
                }

                // --- TTI COUPLING FOR Ez ---
                // Ez depends on Ex via sigma_zx (val[6]) [cite: 69, 71]
                Complex mass_zx = Complex(0.0f, -1.0f) * (float)omega * mat.sigma_eff.val[6];
                int col_ex = (int)IDX(i, j, k, 0);
                triplets.push_back({(int)row_idx, col_ex, mass_zx});

                // Ez depends on Ey via sigma_zy (val[7]) [cite: 69, 71]
                Complex mass_zy = Complex(0.0f, -1.0f) * (float)omega * mat.sigma_eff.val[7];
                int col_ey = (int)IDX(i, j, k, 1);
                triplets.push_back({(int)row_idx, col_ey, mass_zy});

                triplets.push_back({(int)row_idx, (int)row_idx, diag_sum});
            }
        }
    }

    auto t_assembly = std::chrono::high_resolution_clock::now();
    std::cout << "Assembly time: "
              << std::chrono::duration<double>(t_assembly - t_start).count() << " s" << std::endl;

    // Merge thread-local triplets
    std::vector<Triplet> merged;
    merged.reserve(1024);
    for (const auto& tv : thread_triplets) {
        merged.insert(merged.end(), tv.begin(), tv.end());
    }

    auto t_merge = std::chrono::high_resolution_clock::now();
    std::cout << "Merge time: "
              << std::chrono::duration<double>(t_merge - t_assembly).count() << " s" << std::endl;

    // FAST boundary condition application using boolean array
    std::cout << "Building boundary rows..." << std::endl;
    auto t_bc_start = std::chrono::high_resolution_clock::now();

    // 1. Pre-calculate boundary status for O(1) lookup
    std::vector<bool> is_bc_row(grid.total_unknowns(), false);
    #pragma omp parallel for collapse(3)
    for (int k = 0; k < grid.nz; ++k) {
        for (int j = 0; j < grid.ny; ++j) {
            for (int i = 0; i < grid.nx; ++i) {
                bool on_bd = (i == 0 || i == grid.nx - 1 || 
                            j == 0 || j == grid.ny - 1 || 
                            k == 0 || k == grid.nz - 1);
                if (on_bd) {
                    for (int comp = 0; comp < 3; ++comp) {
                        long long row = (long long)comp * grid.nx * grid.ny * grid.nz
                                    + k * grid.nx * grid.ny + j * grid.nx + i;
                        is_bc_row[row] = true;
                    }
                }
            }
        }
    }

    // 2. Efficiently remove boundary triplets using the Erase-Remove Idiom
    // This is O(N) instead of O(N^2)
    auto new_end = std::remove_if(merged.begin(), merged.end(), 
        [&is_bc_row](const Triplet& t) {
            return is_bc_row[t.row]; 
        });
    merged.erase(new_end, merged.end()); 

    // 3. Add Identity for boundary rows to maintain matrix non-singularity
    // Pre-reserve to avoid reallocations
    size_t bc_count = 0;
    for(bool b : is_bc_row) if(b) bc_count++;
    merged.reserve(merged.size() + bc_count);

    for (long long i = 0; i < grid.total_unknowns(); ++i) {
        if (is_bc_row[i]) {
            merged.push_back({(int)i, (int)i, Complex(1.0f, 0.0f)});
        }
    }

    auto t_bc = std::chrono::high_resolution_clock::now();
    std::cout << "Optimized BC time: " 
            << std::chrono::duration<double>(t_bc - t_bc_start).count() << " s" << std::endl;

    /*auto t_bc_start = std::chrono::high_resolution_clock::now();
    
    // Pre-allocate boolean array for boundary rows
    std::vector<bool> is_bc_row(grid.total_unknowns(), false);
    
    for (int k = 0; k < grid.nz; ++k) {
        for (int j = 0; j < grid.ny; ++j) {
            for (int i = 0; i < grid.nx; ++i) {
                bool on_bd = (i == 0 || i == grid.nx-1 || 
                              j == 0 || j == grid.ny-1 || 
                              k == 0 || k == grid.nz-1);
                if (on_bd) {
                    for (int comp = 0; comp < 3; ++comp) {
                        long long row = (long long)comp * grid.nx * grid.ny * grid.nz
                                      + k * grid.nx * grid.ny + j * grid.nx + i;
                        is_bc_row[row] = true;
                    }
                }
            }
        }
    }
    
    std::cout << "Removing boundary triplets..." << std::endl;
    // Remove triplets in boundary rows
    auto it = merged.begin();
    int removed = 0;
    while (it != merged.end()) {
        if (is_bc_row[it->row]) {
            it = merged.erase(it);
            removed++;
        } else {
            ++it;
        }
    }
    std::cout << "Removed " << removed << " triplets" << std::endl;
    
    std::cout << "Adding boundary identity rows..." << std::endl;
    // Add identity rows for boundary
    int bc_count = 0;
    for (long long i = 0; i < grid.total_unknowns(); ++i) {
        if (is_bc_row[i]) {
            merged.push_back({(int)i, (int)i, Complex(1.0f, 0.0f)});
            bc_count++;
        }
    }
    std::cout << "Added " << bc_count << " boundary identity rows" << std::endl;
    
    auto t_bc = std::chrono::high_resolution_clock::now();
    std::cout << "BC time: "
              << std::chrono::duration<double>(t_bc - t_bc_start).count() << " s" << std::endl;
    std::cout << "Total triplets (with BC): " << merged.size() << std::endl;*/

    CsrMatrix result = convert_to_csr(merged, grid.total_unknowns());

    auto t_csr = std::chrono::high_resolution_clock::now();
    std::cout << "CSR conversion time: "
              << std::chrono::duration<double>(t_csr - t_bc).count() << " s" << std::endl;
    
    return result;
}

static CsrMatrix convert_to_csr(const std::vector<Triplet>& triplets, long long n) {
    CsrMatrix mat;
    mat.num_rows = (int)n;
    mat.num_cols = (int)n;
    mat.nnz = (int)triplets.size();

    auto sorted = triplets;
    std::sort(sorted.begin(), sorted.end(),
        [](const Triplet& a, const Triplet& b) {
            return a.row < b.row || (a.row == b.row && a.col < b.col);
        });

    mat.row_ptr.assign((size_t)n + 1, 0);
    for (const auto& t : sorted) mat.row_ptr[t.row + 1]++;
    for (long long i = 1; i <= n; ++i) mat.row_ptr[i] += mat.row_ptr[i-1];

    mat.col_ind.resize(sorted.size());
    mat.val.resize(sorted.size());

    std::vector<long long> pos(n + 1, 0);
    for (long long i = 0; i <= n; ++i) pos[i] = mat.row_ptr[i];

    for (const auto& t : sorted) {
        long long idx = pos[t.row]++;
        mat.col_ind[(size_t)idx] = t.col;
        mat.val[(size_t)idx] = make_cuComplex((float)t.val.real(), (float)t.val.imag());
    }

    return mat;
}
