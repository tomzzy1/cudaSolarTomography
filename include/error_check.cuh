#pragma once
#include <cuda.h>
#include <cublas_v2.h>
#include <cusparse.h>

#define CHECK_CUDA(func)                                                       \
    {                                                                          \
        cudaError_t status = (func);                                           \
        if (status != ::cudaSuccess)                                           \
        {                                                                      \
            printf("CUDA API failed at file %s line %d with error: %s (%d)\n", \
                   __FILE__, __LINE__, cudaGetErrorString(status), status);    \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    }

#if CUDA_TOOLKIT_VERSION >= 12
#define CHECK_CUBLAS(func)                                                       \
    {                                                                            \
        cublasStatus_t status = (func);                                          \
        if (status != CUBLAS_STATUS_SUCCESS)                                     \
        {                                                                        \
            printf("CUBLAS API failed at file %s line %d with error: %s (%d)\n", \
                   __FILE__, __LINE__, cublasGetStatusString(status), status);   \
            std::exit(EXIT_FAILURE);                                             \
        }                                                                        \
    }
#else
#define CHECK_CUBLAS(func)              \
    {                                   \
        cublasStatus_t status = (func); \
    }
#endif

#define CHECK_CUSPARSE(func)                                                       \
    {                                                                              \
        cusparseStatus_t status = (func);                                          \
        if (status != CUSPARSE_STATUS_SUCCESS)                                     \
        {                                                                          \
            printf("CUSPARSE API failed at file %s line %d with error: %s (%d)\n", \
                   __FILE__, __LINE__, cusparseGetErrorString(status), status);    \
            std::exit(EXIT_FAILURE);                                               \
        }                                                                          \
    }

#define cudaCheckError()                \
    {                                   \
        cudaDeviceSynchronize();        \
        CHECK_CUDA(cudaGetLastError()); \
    }

#define cudaCheckStreamError(stream)    \
    {                                   \
        cudaStreamSynchronize(stream);  \
        CHECK_CUDA(cudaGetLastError()); \
    }
