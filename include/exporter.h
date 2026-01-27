#pragma once
#include "types.h"
#include <vector>
#include <string>

class VtkExporter {
public:
    // 将场数据导出为 .vtr 文件 (XML 格式)
    // filename: 输出文件名 (如 "solution.vtr")
    // grid: 网格信息
    // sol: 求解器得到的解向量 x
    static void save_to_vtr(const std::string& filename, const GridInfo& grid, const std::vector<cuComplexType>& sol);
};