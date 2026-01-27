#include "solver.h"
#include <iostream>
#include <cuda_runtime.h>
#include <cusparse.h>
#include <cusolverSp.h>
#include <vector>

// 构造与析构函数实现
Solver::Solver() {}
Solver::~Solver() {}

// 简单的 CUDA 错误检查宏
#define check_cuda(status, msg) check_cuda_impl(status, msg, __LINE__)

void check_cuda_impl(cudaError_t status, const char* msg, int line) {
    if (status != cudaSuccess) {
        std::cerr << "CUDA Error at line " << line << ": " << msg << " - " << cudaGetErrorString(status) << std::endl;
        throw std::runtime_error("CUDA Error");
    }
}

void check_cusparse(cusparseStatus_t status, const char* msg) {
    if (status != CUSPARSE_STATUS_SUCCESS) {
        std::cerr << "cuSPARSE Error: " << msg << std::endl;
        throw std::runtime_error("cuSPARSE Error");
    }
}

void check_cusolver(cusolverStatus_t status, const char* msg) {
    if (status != CUSOLVER_STATUS_SUCCESS) {
        std::cerr << "cuSolver Error: " << msg << std::endl;
        throw std::runtime_error("cuSolver Error");
    }
}

void Solver::solve(const CsrMatrix& A, const std::vector<cuComplexType>& rhs, std::vector<cuComplexType>& x) {
    int N = A.num_rows;
    int nnz = A.nnz;
    
    // 调整解向量大小并初始化
    x.assign(N, {0.0f, 0.0f});

    // 1. 初始化句柄
    cusolverSpHandle_t solver_handle;
    cusparseHandle_t sparse_handle;
    check_cusolver(cusolverSpCreate(&solver_handle), "Create solver handle");
    check_cusparse(cusparseCreate(&sparse_handle), "Create sparse handle");

    // 2. 分配设备内存
    int *d_row_ptr, *d_col_ind;
    cuComplexType *d_val, *d_rhs, *d_x;

    check_cuda(cudaMalloc(&d_row_ptr, (N + 1) * sizeof(int)), "Malloc row_ptr");
    check_cuda(cudaMalloc(&d_col_ind, nnz * sizeof(int)), "Malloc col_ind");
    check_cuda(cudaMalloc(&d_val, nnz * sizeof(cuComplexType)), "Malloc val");
    check_cuda(cudaMalloc(&d_rhs, N * sizeof(cuComplexType)), "Malloc rhs");
    check_cuda(cudaMalloc(&d_x, N * sizeof(cuComplexType)), "Malloc x");

    // 3. 拷贝数据 (Host -> Device)
    check_cuda(cudaMemcpy(d_row_ptr, A.row_ptr.data(), (N + 1) * sizeof(int), cudaMemcpyHostToDevice), "Copy row_ptr");
    check_cuda(cudaMemcpy(d_col_ind, A.col_ind.data(), nnz * sizeof(int), cudaMemcpyHostToDevice), "Copy col_ind");
    check_cuda(cudaMemcpy(d_val, A.val.data(), nnz * sizeof(cuComplexType), cudaMemcpyHostToDevice), "Copy val");
    check_cuda(cudaMemcpy(d_rhs, rhs.data(), N * sizeof(cuComplexType), cudaMemcpyHostToDevice), "Copy rhs");
    
    // 初始化解向量为0 (作为初始猜测)
    check_cuda(cudaMemset(d_x, 0, N * sizeof(cuComplexType)), "Memset x");

    // 4. 创建矩阵描述符
    cusparseMatDescr_t descrA;
    check_cusparse(cusparseCreateMatDescr(&descrA), "Create MatDescr");
    check_cusparse(cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL), "Set MatType");
    check_cusparse(cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO), "Set IndexBase");

    // 5. 求解
    // 使用 csrlsvqr (QR分解) 求解 Ax = b
    // 注意：QR分解极其消耗显存。如果 N 很大 (如 80^3)，可能会显存不足 (OOM)。
    // 如果遇到 OOM，需要换用迭代解法 (如 csrlsvlu 或外部迭代库)。
    // 这里先尝试直接求解。
    int singularity = 0;
    
    // 容差设为 1e-6
    check_cusolver(cusolverSpCcsrlsvqr(
        solver_handle, N, nnz, descrA, d_val, d_row_ptr, d_col_ind, 
        d_rhs, 1e-6f, 0, d_x, &singularity), 
        "Solver execution");

    if (singularity != -1) {
        std::cerr << "WARNING: Matrix is singular at index " << singularity << std::endl;
    }

    // 6. 拷贝回结果
    check_cuda(cudaMemcpy(x.data(), d_x, N * sizeof(cuComplexType), cudaMemcpyDeviceToHost), "Copy result");

    // 7. 清理资源
    check_cusparse(cusparseDestroyMatDescr(descrA), "Destroy MatDescr");
    cudaFree(d_row_ptr); cudaFree(d_col_ind); cudaFree(d_val); cudaFree(d_rhs); cudaFree(d_x);
    cusolverSpDestroy(solver_handle);
    cusparseDestroy(sparse_handle);
}