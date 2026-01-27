/*
 * Copyright 2023-2025 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO LICENSEE:
 *
 * This source code and/or documentation ("Licensed Deliverables") are
 * subject to NVIDIA intellectual property rights under U.S. and
 * international Copyright laws.
 *
 * These Licensed Deliverables contained herein is PROPRIETARY and
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and
 * conditions of a form of NVIDIA software license agreement by and
 * between NVIDIA and Licensee ("License Agreement") or electronically
 * accepted by Licensee.  Notwithstanding any terms or conditions to
 * the contrary in the License Agreement, reproduction or disclosure
 * of the Licensed Deliverables to any third party without the express
 * written consent of NVIDIA is prohibited.
 *
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
 * SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
 * PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
 * NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
 * DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
 * NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
 * SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
 * DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
 * WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
 * ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
 * OF THESE LICENSED DELIVERABLES.
 *
 * U.S. Government End Users.  These Licensed Deliverables are a
 * "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
 * 1995), consisting of "commercial computer software" and "commercial
 * computer software documentation" as such terms are used in 48
 * C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
 * only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
 * 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
 * U.S. Government End Users acquire the Licensed Deliverables with
 * only those rights set forth herein.
 *
 * Any use of the Licensed Deliverables in individual and commercial
 * software must include, in the user documentation and internal
 * comments to the code, the above Disclaimer and U.S. Government End
 * Users Notice.
 */

#ifndef CUDSS_PUBLIC_HEADER_H
#define CUDSS_PUBLIC_HEADER_H

#include <stddef.h>        // size_t
#include <stdint.h>        // int64_t
#include <library_types.h> // for cudaDataType_t
#include <cuda_runtime.h>  // for cudaStream_t

#define CUDSS_VERSION_MAJOR 0
#define CUDSS_VERSION_MINOR 7
#define CUDSS_VERSION_PATCH 1
#define CUDSS_VERSION (CUDSS_VERSION_MAJOR * 10000 + \
                       CUDSS_VERSION_MINOR *  100 +  \
                       CUDSS_VERSION_PATCH)

#if !defined(CUDSSAPI)
#    if defined(_WIN32)
#        define CUDSSAPI __stdcall
#    else
#        define CUDSSAPI
#    endif
#endif

#include "cudss_distributed_interface.h"
#include "cudss_threading_interface.h"

#ifdef __cplusplus
extern "C" {
#endif

struct cudssContext;
typedef struct cudssContext *cudssHandle_t; // the library context handle

struct cudssMatrix;
typedef struct cudssMatrix* cudssMatrix_t; // opaque generic matrix struct (for both dense and sparse matrices)

struct cudssData;
typedef struct cudssData* cudssData_t; // opaque object type which stores internal data (like LU factors) as well as some user-provided pointers

struct cudssConfig;
typedef struct cudssConfig* cudssConfig_t; // opaque object type which stores solver settings (e.g., algorithmic knobs)

typedef enum cudssConfigParam_t {
    CUDSS_CONFIG_REORDERING_ALG,
    CUDSS_CONFIG_FACTORIZATION_ALG,
    CUDSS_CONFIG_SOLVE_ALG,
    CUDSS_CONFIG_USE_MATCHING,      // disabled by default
    CUDSS_CONFIG_MATCHING_ALG,      // not used by default
    CUDSS_CONFIG_SOLVE_MODE,        // not supported right now (for transpose, conj transpose solves) (only off)
    CUDSS_CONFIG_IR_N_STEPS,
    CUDSS_CONFIG_IR_TOL,            // not supported right now
    CUDSS_CONFIG_PIVOT_TYPE,
    CUDSS_CONFIG_PIVOT_THRESHOLD,
    CUDSS_CONFIG_PIVOT_EPSILON,
    CUDSS_CONFIG_MAX_LU_NNZ,              // only for CUDSS_ALG_1 and CUDSS_ALG_2 reordering algorithms
    CUDSS_CONFIG_HYBRID_MODE,             // by default: disabled
    CUDSS_CONFIG_HYBRID_DEVICE_MEMORY_LIMIT,
    CUDSS_CONFIG_USE_CUDA_REGISTER_MEMORY,// by default: enabled
    CUDSS_CONFIG_HOST_NTHREADS,
    CUDSS_CONFIG_HYBRID_EXECUTE_MODE,     // default: 0 - disabled
    CUDSS_CONFIG_PIVOT_EPSILON_ALG,
    CUDSS_CONFIG_ND_NLEVELS,
    CUDSS_CONFIG_UBATCH_SIZE, // "U" - stands for Uniform
    CUDSS_CONFIG_UBATCH_INDEX,
    CUDSS_CONFIG_USE_SUPERPANELS, // by default: enabled
    CUDSS_CONFIG_DEVICE_COUNT,
    CUDSS_CONFIG_DEVICE_INDICES,
    CUDSS_CONFIG_SCHUR_MODE,
    CUDSS_CONFIG_DETERMINISTIC_MODE
} cudssConfigParam_t;

typedef enum cudssDataParam_t {
    CUDSS_DATA_INFO,                 // (out)
    CUDSS_DATA_LU_NNZ,               // (out)
    CUDSS_DATA_NPIVOTS,              // (out)
    CUDSS_DATA_INERTIA,              // (out, non-trivial for non-positive-definite matrices)
    CUDSS_DATA_PERM_REORDER_ROW,     // (out)
    CUDSS_DATA_PERM_REORDER_COL,     // (out)
    CUDSS_DATA_PERM_ROW,             // (out, supported only for CUDSS_ALG_1 and CUDSS_ALG_2 reordering algorithms)
    CUDSS_DATA_PERM_COL,             // (out, supported only for CUDSS_ALG_1 and CUDSS_ALG_2 reordering algorithms)
    CUDSS_DATA_DIAG,                 // (out)
    CUDSS_DATA_USER_PERM,            // (in, out) for the user to provide a permutation or retrieve the provided permutation
    CUDSS_DATA_HYBRID_DEVICE_MEMORY_MIN,
    CUDSS_DATA_COMM,                 // (in) communicator
    CUDSS_DATA_MEMORY_ESTIMATES,
    CUDSS_DATA_PERM_MATCHING,        // (out, supported only when matching is enabled)
    CUDSS_DATA_SCALE_ROW,            // (out, supported only when matching is enabled)
    CUDSS_DATA_SCALE_COL,            // (out, supported only when matching with scaling is enabled)
    CUDSS_DATA_NSUPERPANELS,         // (out)
    CUDSS_DATA_USER_SCHUR_INDICES,   // (in) indices[n] (1s for Schur complement, 0s for the rest)
    CUDSS_DATA_SCHUR_SHAPE,          // (out) int64_t shape[3]
    CUDSS_DATA_SCHUR_MATRIX,         // (out) cudssMatrix_t
    // (in, out) for the user to provide or retrieve the auxiliary elimination tree information
    CUDSS_DATA_USER_ELIMINATION_TREE,
    CUDSS_DATA_ELIMINATION_TREE,     // (out) retrieves the elimination tree information
    CUDSS_DATA_USER_HOST_INTERRUPT
} cudssDataParam_t;

typedef enum cudssPhase_t {
    CUDSS_PHASE_REORDERING             = 1 << 0,
    CUDSS_PHASE_SYMBOLIC_FACTORIZATION = 1 << 1,
    CUDSS_PHASE_ANALYSIS               = CUDSS_PHASE_REORDERING | CUDSS_PHASE_SYMBOLIC_FACTORIZATION,
    CUDSS_PHASE_FACTORIZATION          = 1 << 2,
    CUDSS_PHASE_REFACTORIZATION        = 1 << 3,
    CUDSS_PHASE_SOLVE_FWD_PERM         = 1 << 4,
    CUDSS_PHASE_SOLVE_FWD              = 1 << 5,
    CUDSS_PHASE_SOLVE_DIAG             = 1 << 6,
    CUDSS_PHASE_SOLVE_BWD              = 1 << 7,
    CUDSS_PHASE_SOLVE_BWD_PERM         = 1 << 8,
    CUDSS_PHASE_SOLVE_REFINEMENT       = 1 << 9,
    CUDSS_PHASE_SOLVE                  = CUDSS_PHASE_SOLVE_FWD_PERM | CUDSS_PHASE_SOLVE_FWD | CUDSS_PHASE_SOLVE_DIAG | CUDSS_PHASE_SOLVE_BWD | CUDSS_PHASE_SOLVE_BWD_PERM | CUDSS_PHASE_SOLVE_REFINEMENT
} cudssPhase_t;

typedef enum cudssStatus_t {
    CUDSS_STATUS_SUCCESS,
    CUDSS_STATUS_NOT_INITIALIZED,
    CUDSS_STATUS_ALLOC_FAILED,
    CUDSS_STATUS_INVALID_VALUE,
    CUDSS_STATUS_NOT_SUPPORTED,
    CUDSS_STATUS_EXECUTION_FAILED,
    CUDSS_STATUS_INTERNAL_ERROR
} cudssStatus_t;

typedef enum cudssMatrixType_t {
    CUDSS_MTYPE_GENERAL,
    CUDSS_MTYPE_SYMMETRIC,
    CUDSS_MTYPE_HERMITIAN,
    CUDSS_MTYPE_SPD,
    CUDSS_MTYPE_HPD
} cudssMatrixType_t;

typedef enum cudssMatrixViewType_t {
    CUDSS_MVIEW_FULL,
    CUDSS_MVIEW_LOWER,
    CUDSS_MVIEW_UPPER
} cudssMatrixViewType_t;

typedef enum cudssIndexBase_t {
    CUDSS_BASE_ZERO,
    CUDSS_BASE_ONE
} cudssIndexBase_t;

typedef enum cudssLayout_t {
    CUDSS_LAYOUT_COL_MAJOR,
    CUDSS_LAYOUT_ROW_MAJOR
} cudssLayout_t;

typedef enum cudssAlgType_t {
    CUDSS_ALG_DEFAULT,
    CUDSS_ALG_1,
    CUDSS_ALG_2,
    CUDSS_ALG_3,
    CUDSS_ALG_4,
    CUDSS_ALG_5
} cudssAlgType_t;

typedef enum cudssPivotType_t {
    CUDSS_PIVOT_COL,
    CUDSS_PIVOT_ROW,
    CUDSS_PIVOT_NONE
} cudssPivotType_t;

typedef enum cudssMatrixFormat_t {
  CUDSS_MFORMAT_DENSE       = 1,
  CUDSS_MFORMAT_CSR         = 2,
  CUDSS_MFORMAT_BATCH       = 4,
  CUDSS_MFORMAT_DISTRIBUTED = 8
} cudssMatrixFormat_t;

#define CUDSS_ALLOCATOR_NAME_LEN 64

typedef struct {
  void* ctx;
  int (*device_alloc)(void* ctx, void** ptr, size_t size, cudaStream_t stream);
  int (*device_free)(void* ctx, void* ptr, size_t size, cudaStream_t stream);
  char name[CUDSS_ALLOCATOR_NAME_LEN];
} cudssDeviceMemHandler_t;

// Set/Get APIs for cudssConfig_t and cudssData_t

cudssStatus_t CUDSSAPI cudssConfigSet(cudssConfig_t config, cudssConfigParam_t param, void *value, size_t sizeInBytes);

cudssStatus_t CUDSSAPI cudssConfigGet(cudssConfig_t config, cudssConfigParam_t param,  void *value, size_t sizeInBytes, size_t *sizeWritten);

cudssStatus_t CUDSSAPI cudssDataSet(cudssHandle_t handle, cudssData_t data, cudssDataParam_t param,  void *value, size_t sizeInBytes);

cudssStatus_t CUDSSAPI cudssDataGet(cudssHandle_t handle, cudssData_t data, cudssDataParam_t param, void *value, size_t sizeInBytes, size_t *sizeWritten);

// Main cuDSS routine

cudssStatus_t CUDSSAPI cudssExecute(cudssHandle_t handle, int phase, cudssConfig_t solverConfig, cudssData_t solverData, cudssMatrix_t inputMatrix, cudssMatrix_t solution, cudssMatrix_t rhs);

// Setting the stream (in the library handle)

cudssStatus_t CUDSSAPI cudssSetStream(cudssHandle_t handle, cudaStream_t stream);

// Setting the communication layer library name (in the library handle)

cudssStatus_t CUDSSAPI cudssSetCommLayer(cudssHandle_t handle, const char* commLibFileName);

// Setting the threading layer library name (in the library handle)

cudssStatus_t CUDSSAPI cudssSetThreadingLayer(cudssHandle_t handle, const char* thrLibFileName);

// Create/Destroy APIs (allocating structures + set defaults)

cudssStatus_t CUDSSAPI cudssConfigCreate(cudssConfig_t *solverConfig);
cudssStatus_t CUDSSAPI cudssConfigDestroy(cudssConfig_t solverConfig);

cudssStatus_t CUDSSAPI cudssDataCreate(cudssHandle_t handle, cudssData_t *solverData);
cudssStatus_t CUDSSAPI cudssDataDestroy(cudssHandle_t handle, cudssData_t solverData);

cudssStatus_t CUDSSAPI cudssCreate(cudssHandle_t *handle);
cudssStatus_t CUDSSAPI cudssCreateMg(cudssHandle_t *handle_pt, int device_count, int *device_indices);
cudssStatus_t CUDSSAPI cudssDestroy(cudssHandle_t handle);

// Versioning

cudssStatus_t CUDSSAPI cudssGetProperty(libraryPropertyType propertyType, int* value);

// Create/Destroy API helpers for matrix wrappers

cudssStatus_t CUDSSAPI cudssMatrixCreateDn(cudssMatrix_t *matrix, int64_t nrows, int64_t ncols, int64_t ld, void *values, cudaDataType_t valueType,  cudssLayout_t layout);

cudssStatus_t CUDSSAPI cudssMatrixCreateCsr(cudssMatrix_t *matrix, int64_t nrows, int64_t ncols, int64_t nnz, void *rowStart, void *rowEnd, void *colIndices, void *values, cudaDataType_t indexType, cudaDataType_t valueType, cudssMatrixType_t mtype, cudssMatrixViewType_t mview, cudssIndexBase_t indexBase);

cudssStatus_t CUDSSAPI cudssMatrixCreateBatchDn(cudssMatrix_t *matrix, int64_t batchCount, void *nrows, void *ncols, void *ld, void **values, cudaDataType_t indexType, cudaDataType_t valueType,  cudssLayout_t layout);

cudssStatus_t CUDSSAPI cudssMatrixCreateBatchCsr(cudssMatrix_t *matrix, int64_t batchCount, void *nrows, void *ncols, void *nnz, void **rowStart, void **rowEnd, void **colIndices, void **values, cudaDataType_t indexType, cudaDataType_t valueType, cudssMatrixType_t mtype, cudssMatrixViewType_t mview, cudssIndexBase_t indexBase);

cudssStatus_t CUDSSAPI cudssMatrixDestroy(cudssMatrix_t matrix);

// Setters/Getters API helpers for matrix wrappers

cudssStatus_t CUDSSAPI cudssMatrixGetDn(cudssMatrix_t matrix,  int64_t* nrows, int64_t* ncols, int64_t* ld, void **values, cudaDataType_t* type, cudssLayout_t* layout);

cudssStatus_t CUDSSAPI cudssMatrixGetCsr(cudssMatrix_t matrix, int64_t* nrows, int64_t* ncols, int64_t* nnz, void **rowStart, void **rowEnd, void **colIndices, void **values, cudaDataType_t* indexType, cudaDataType_t* valueType, cudssMatrixType_t* mtype, cudssMatrixViewType_t* mview, cudssIndexBase_t* indexBase);

cudssStatus_t CUDSSAPI cudssMatrixSetValues(cudssMatrix_t matrix, void *values);

cudssStatus_t CUDSSAPI cudssMatrixSetCsrPointers(cudssMatrix_t matrix, void *rowOffsets, void *rowEnd, void *colIndices, void *values);

cudssStatus_t CUDSSAPI cudssMatrixGetBatchDn(cudssMatrix_t matrix, int64_t *batchCount, void **nrows, void **ncols, void **ld, void ***values, cudaDataType_t *indexType, cudaDataType_t *valueType, cudssLayout_t* layout);

cudssStatus_t CUDSSAPI cudssMatrixGetBatchCsr(cudssMatrix_t matrix, int64_t *batchCount, void **nrows, void **ncols, void **nnz, void ***rowStart, void ***rowEnd, void ***colIndices, void ***values, cudaDataType_t* indexType, cudaDataType_t* valueType, cudssMatrixType_t* mtype, cudssMatrixViewType_t* mview, cudssIndexBase_t* indexBase);

cudssStatus_t CUDSSAPI cudssMatrixSetBatchValues(cudssMatrix_t matrix, void **values);

cudssStatus_t CUDSSAPI cudssMatrixSetBatchCsrPointers(cudssMatrix_t matrix, void **rowOffsets, void **rowEnd, void **colIndices, void **values);

cudssStatus_t CUDSSAPI cudssMatrixGetFormat(cudssMatrix_t matrix, int* format);

cudssStatus_t CUDSSAPI cudssMatrixSetDistributionRow1d(cudssMatrix_t matrix, int64_t first_row, int64_t last_row);

cudssStatus_t CUDSSAPI cudssMatrixGetDistributionRow1d(cudssMatrix_t matrix, int64_t *first_row, int64_t *last_row);

// Memory allocator API

cudssStatus_t CUDSSAPI cudssGetDeviceMemHandler(cudssHandle_t handle, cudssDeviceMemHandler_t* handler);

cudssStatus_t CUDSSAPI cudssSetDeviceMemHandler(cudssHandle_t handle, const cudssDeviceMemHandler_t* handler);

#ifdef __cplusplus
} /* extern "C" */
#endif
#endif /* CUDSS_PUBLIC_HEADER_H */
