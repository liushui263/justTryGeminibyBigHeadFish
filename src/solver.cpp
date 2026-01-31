#include "solver.h"
#include <iostream>
#include <stdexcept>
#include <vector>
#include <cmath>
#include <cublas_v2.h>
#include <cusparse.h>

Solver::Solver() {}
Solver::~Solver() {}

// Helper for safe complex division on host
static cuComplex complex_div(cuComplex a, cuComplex b) {
    float denom = b.x * b.x + b.y * b.y;
    if (denom < 1e-25f) return make_cuComplex(0.0f, 0.0f);
    return make_cuComplex((a.x * b.x + a.y * b.y) / denom, (a.y * b.x - a.x * b.y) / denom);
}

void Solver::solve(const CsrMatrix& A, const std::vector<cuComplexType>& b, std::vector<cuComplexType>& x) {
    int N = (int)b.size();
    int nnz = (int)A.val.size();

    cusparseHandle_t sp_handle;
    cublasHandle_t cb_handle;
    cusparseCreate(&sp_handle);
    cublasCreate(&cb_handle);

    cuComplexType *d_val, *d_b, *d_x, *d_r, *d_r_hat, *d_p, *d_v, *d_s, *d_t;
    int *d_row_ptr, *d_col_ind;

    cudaMalloc((void**)&d_val, nnz * sizeof(cuComplexType));
    cudaMalloc((void**)&d_row_ptr, (N + 1) * sizeof(int));
    cudaMalloc((void**)&d_col_ind, nnz * sizeof(int));
    cudaMalloc((void**)&d_b, N * sizeof(cuComplexType));
    cudaMalloc((void**)&d_x, N * sizeof(cuComplexType));
    cudaMalloc((void**)&d_r, N * sizeof(cuComplexType));
    cudaMalloc((void**)&d_r_hat, N * sizeof(cuComplexType));
    cudaMalloc((void**)&d_p, N * sizeof(cuComplexType));
    cudaMalloc((void**)&d_v, N * sizeof(cuComplexType));
    cudaMalloc((void**)&d_s, N * sizeof(cuComplexType));
    cudaMalloc((void**)&d_t, N * sizeof(cuComplexType));

    cudaMemcpy(d_val, A.val.data(), nnz * sizeof(cuComplexType), cudaMemcpyHostToDevice);
    cudaMemcpy(d_row_ptr, A.row_ptr.data(), (N + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_ind, A.col_ind.data(), nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), N * sizeof(cuComplexType), cudaMemcpyHostToDevice);
    cudaMemset(d_x, 0, N * sizeof(cuComplexType));

    cusparseSpMatDescr_t matA;
    cusparseCreateCsr(&matA, N, N, nnz, d_row_ptr, d_col_ind, d_val,
                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_C_32F);

    cusparseDnVecDescr_t vecP, vecV, vecS, vecT, vecR, vecX;
    cusparseCreateDnVec(&vecP, N, d_p, CUDA_C_32F);
    cusparseCreateDnVec(&vecV, N, d_v, CUDA_C_32F);
    cusparseCreateDnVec(&vecS, N, d_s, CUDA_C_32F);
    cusparseCreateDnVec(&vecT, N, d_t, CUDA_C_32F);
    cusparseCreateDnVec(&vecR, N, d_r, CUDA_C_32F);
    cusparseCreateDnVec(&vecX, N, d_x, CUDA_C_32F);

    size_t bufferSize = 0;
    void* d_buffer = nullptr;
    cuComplexType one = {1.0f, 0.0f}, zero = {0.0f, 0.0f};
    cusparseSpMV_bufferSize(sp_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, matA, vecP, &zero, vecV, CUDA_C_32F, CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize);
    cudaMalloc(&d_buffer, bufferSize);

    std::cout << "Starting Robust BiCGStab Solver..." << std::endl;
    cudaMemcpy(d_r, d_b, N * sizeof(cuComplexType), cudaMemcpyDeviceToDevice);
    cudaMemcpy(d_r_hat, d_r, N * sizeof(cuComplexType), cudaMemcpyDeviceToDevice);
    cudaMemcpy(d_p, d_r, N * sizeof(cuComplexType), cudaMemcpyDeviceToDevice);

    cuComplexType rho = {1.0f, 0.0f}, alpha = {1.0f, 0.0f}, omega = {1.0f, 0.0f};
    cuComplexType rho_prev, r_hat_v, ts, tt;

    for (int iter = 0; iter < 1000; ++iter) {
        rho_prev = rho;
        cublasCdotu(cb_handle, N, d_r_hat, 1, d_r, 1, &rho); 

        if (std::isnan(rho.x) || (std::abs(rho.x) < 1e-25f && std::abs(rho.y) < 1e-25f)) break;

        if (iter > 0) {
            cuComplexType beta_val = complex_div(rho, rho_prev);
            cuComplexType ratio_ao = complex_div(alpha, omega);
            cuComplexType beta = make_cuComplex(beta_val.x * ratio_ao.x - beta_val.y * ratio_ao.y, beta_val.x * ratio_ao.y + beta_val.y * ratio_ao.x);

            cuComplexType neg_omega = {-omega.x, -omega.y};
            cublasCaxpy(cb_handle, N, &neg_omega, d_v, 1, d_p, 1); 
            cublasCscal(cb_handle, N, &beta, d_p, 1);             
            cublasCaxpy(cb_handle, N, &one, d_r, 1, d_p, 1);      
        }

        cusparseSpMV(sp_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, matA, vecP, &zero, vecV, CUDA_C_32F, CUSPARSE_SPMV_ALG_DEFAULT, d_buffer);
        cublasCdotu(cb_handle, N, d_r_hat, 1, d_v, 1, &r_hat_v);
        alpha = complex_div(rho, r_hat_v);

        cudaMemcpy(d_s, d_r, N * sizeof(cuComplexType), cudaMemcpyDeviceToDevice);
        cuComplexType neg_alpha = {-alpha.x, -alpha.y};
        cublasCaxpy(cb_handle, N, &neg_alpha, d_v, 1, d_s, 1);

        cusparseSpMV(sp_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, matA, vecS, &zero, vecT, CUDA_C_32F, CUSPARSE_SPMV_ALG_DEFAULT, d_buffer);
        cublasCdotu(cb_handle, N, d_t, 1, d_s, 1, &ts);
        cublasCdotu(cb_handle, N, d_t, 1, d_t, 1, &tt);
        omega = complex_div(ts, tt);

        cublasCaxpy(cb_handle, N, &alpha, d_p, 1, d_x, 1);
        cublasCaxpy(cb_handle, N, &omega, d_s, 1, d_x, 1);

        cudaMemcpy(d_r, d_s, N * sizeof(cuComplexType), cudaMemcpyDeviceToDevice);
        cuComplexType neg_omega = {-omega.x, -omega.y};
        cublasCaxpy(cb_handle, N, &neg_omega, d_t, 1, d_r, 1);

        float res_norm;
        cublasScnrm2(cb_handle, N, d_r, 1, &res_norm);
        if (res_norm < 1e-6f) {
            std::cout << "Converged at " << iter << " res: " << res_norm << std::endl;
            break;
        }
    }

    cudaMemcpy(x.data(), d_x, N * sizeof(cuComplexType), cudaMemcpyDeviceToHost);

    cudaFree(d_val); cudaFree(d_row_ptr); cudaFree(d_col_ind);
    cudaFree(d_b); cudaFree(d_x); cudaFree(d_r); cudaFree(d_r_hat);
    cudaFree(d_p); cudaFree(d_v); cudaFree(d_s); cudaFree(d_t); cudaFree(d_buffer);
    cusparseDestroySpMat(matA); cusparseDestroyDnVec(vecP); cusparseDestroyDnVec(vecV);
    cusparseDestroyDnVec(vecS); cusparseDestroyDnVec(vecT); cusparseDestroyDnVec(vecR);
    cusparseDestroyDnVec(vecX); cusparseDestroy(sp_handle); cublasDestroy(cb_handle);
}