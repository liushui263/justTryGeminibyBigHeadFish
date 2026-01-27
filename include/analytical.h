#pragma once
#include "types.h"
#include <vector>

class AnalyticalSolution {
public:
    // 修改：moment 变为三个分量 mx, my, mz
    static std::vector<cuComplexType> compute_field(
        double sigma, double freq, 
        const GridInfo& grid, 
        double src_x_idx, double src_y_idx, double src_z_idx,
        double mx, double my, double mz
    );

private:
    static Complex compute_k(double omega, double sigma);
};