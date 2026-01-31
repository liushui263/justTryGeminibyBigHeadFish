#pragma once
#include <vector>
#include <cmath>
#include <complex>
#include <cuComplex.h>
#include <array>

// === 核心修复 ===
// 定义 cuComplexType 为 cuComplex (float2)
// 这与 std::vector<cuComplexType> 兼容
using cuComplexType = cuComplex;

using Real = float;
using Complex = std::complex<float>;

// 稀疏矩阵的三元组
struct Triplet {
    int row;
    int col;
    Complex val;
};

// Simple 3x3 matrix struct
template<typename T>
struct Matrix3x3 {
    std::array<T, 9> val;
    
    Matrix3x3() : val{} {}
    
    void set_isotropic(const T& diag_val) {
        val.fill(T{0, 0});
        val[0] = val[4] = val[8] = diag_val;
    }
};

// CSR 矩阵结构
struct CsrMatrix {
    int num_rows;
    int num_cols;
    int nnz;
    std::vector<int> row_ptr;
    std::vector<int> col_ind;
    std::vector<cuComplexType> val;
};

struct GridInfo {
    int nx, ny, nz;
    std::vector<double> x_nodes;
    std::vector<double> y_nodes;
    std::vector<double> z_nodes;
    
    long long total_unknowns() const { return (long long)3 * nx * ny * nz; }
    
    double dx(int i) const { return (i < nx) ? x_nodes[i+1] - x_nodes[i] : 1.0; }
    double dy(int j) const { return (j < ny) ? y_nodes[j+1] - y_nodes[j] : 1.0; }
    double dz(int k) const { return (k < nz) ? z_nodes[k+1] - z_nodes[k] : 1.0; }
    
    double dx_dual(int i) const { return (i > 0) ? x_nodes[i] - x_nodes[i-1] : dx(0); }
    double dy_dual(int j) const { return (j > 0) ? y_nodes[j] - y_nodes[j-1] : dy(0); }
    double dz_dual(int k) const { return (k > 0) ? z_nodes[k] - z_nodes[k-1] : dz(0); }
};

struct MaterialProperty {
    Matrix3x3<Complex> sigma_eff;
    Matrix3x3<Complex> mu_inv_eff;
};

enum class SubGrid { G000 }; 
inline int get_dof_lebedev(const GridInfo& g, int i, int j, int k, SubGrid, int comp) {
    return comp * g.nx * g.ny * g.nz + k * g.nx * g.ny + j * g.nx + i;
}