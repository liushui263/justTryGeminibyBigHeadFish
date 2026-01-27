#pragma once
#include <vector>
#include <cmath>
#include <complex>
#include <cuComplex.h> // 必须包含 CUDA 复数头文件

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
    
    // 存储节点坐标 (大小为 n+1)
    std::vector<double> x_nodes;
    std::vector<double> y_nodes;
    std::vector<double> z_nodes;

    GridInfo() : nx(0), ny(0), nz(0) {}
    
    // 兼容均匀网格构造
    GridInfo(int nx_, int ny_, int nz_, double dx_, double dy_, double dz_) 
        : nx(nx_), ny(ny_), nz(nz_) {
        x_nodes.resize(nx + 1);
        y_nodes.resize(ny + 1);
        z_nodes.resize(nz + 1);
        for(int i=0; i<=nx; ++i) x_nodes[i] = i * dx_;
        for(int j=0; j<=ny; ++j) y_nodes[j] = j * dy_;
        for(int k=0; k<=nz; ++k) z_nodes[k] = k * dz_;
    }

    double dx(int i) const { return x_nodes[i+1] - x_nodes[i]; }
    double dy(int j) const { return y_nodes[j+1] - y_nodes[j]; }
    double dz(int k) const { return z_nodes[k+1] - z_nodes[k]; }

    double dx_dual(int i) const {
        if (i == 0) return dx(0); 
        if (i == nx) return dx(nx-1);
        return (dx(i-1) + dx(i)) * 0.5;
    }
    double dy_dual(int j) const { 
        if (j == 0) return dy(0);
        if (j == ny) return dy(ny-1);
        return (dy(j-1) + dy(j)) * 0.5;
    }
    double dz_dual(int k) const { 
        if (k == 0) return dz(0);
        if (k == nz) return dz(nz-1);
        return (dz(k-1) + dz(k)) * 0.5;
    }
    
    long long total_unknowns() const { return (long long)nx * ny * nz * 3; }
};

struct MaterialProperty {
    struct Tensor3 {
        Complex val[9]; // 3x3 row-major
        void set_isotropic(Complex c) {
            for(int i=0; i<9; ++i) val[i] = {0,0};
            val[0] = val[4] = val[8] = c;
        }
    } sigma_eff, mu_inv_eff;
};

enum class SubGrid { G000 }; 
inline int get_dof_lebedev(const GridInfo& g, int i, int j, int k, SubGrid, int comp) {
    return comp * g.nx * g.ny * g.nz + k * g.nx * g.ny + j * g.nx + i;
}