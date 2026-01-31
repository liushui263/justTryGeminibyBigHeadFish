#pragma once
#include "types.h"

class Physics {
public:
    // 确保这里的参数类型与 .cpp 文件完全一致
    static MaterialProperty compute_cell_material(
        int i, int j, int k, 
        const GridInfo& grid, 
        int n_pml, 
        double omega,
        double rho_t, double rho_n, 
        double theta, double phi
    );

     static MaterialProperty compute_cell_material_(
        int i, int j, int k, 
        const GridInfo& grid, 
        int n_pml, 
        double omega,
        double rho_t, double rho_n, 
        double theta, double phi
    );
};