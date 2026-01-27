#pragma once
#include "types.h"
#include <vector>

class Solver {
public:
    // 构造函数与析构函数
    Solver();
    ~Solver();

    // 核心求解接口：Ax = b
    // 使用 cuComplexType (即 cuComplex / float2) 以匹配 CUDA
    void solve(const CsrMatrix& A, const std::vector<cuComplexType>& rhs, std::vector<cuComplexType>& x);
};