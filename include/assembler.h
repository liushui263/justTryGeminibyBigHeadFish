#pragma once
#include "types.h"
#include <vector>

class Assembler {
public:
    // 更新构造函数，增加 use_coord_xform 参数
    Assembler(const GridInfo& grid, double omega, bool use_coord_xform = false);

    CsrMatrix assemble_system_matrix(const std::vector<MaterialProperty>& materials);
    CsrMatrix assemble_system_matrix_(const std::vector<MaterialProperty>& materials);

private:
    GridInfo grid;
    double omega;
    bool use_coordinate_transformation;
};