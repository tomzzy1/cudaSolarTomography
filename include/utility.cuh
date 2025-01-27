#pragma once
#include <cuda.h>
#include <cublas_v2.h>
#include <cusparse.h>
#include "constants.hpp"
#include "type.hpp"
#include "error_check.cuh"
#include "gpu_matrix.hpp"

namespace cudaSolarTomography
{

#if __cplusplus > 201703L
    template <typename T1, typename T2>
    using IndexValuePair = std::pair<T1, T2>;
#else
    template <typename T1, typename T2>
    struct IndexValuePair
    {
        T1 first;
        T2 second;

        __device__ bool operator<(const IndexValuePair &other)
        {
            if (first < other.first)
            {
                return true;
            }
            else if (first == other.first)
            {
                return second < other.second;
            }
            else
            {
                return false;
            }
        }
    };

    template <typename T1, typename T2>
    IndexValuePair(T1 index, T2 value) -> IndexValuePair<T1, T2>;
#endif

    __global__ void scan(int *input, int *block_sums, int length)
    {
        __shared__ std::array<int, 2 * 512> temp;
        int x = blockIdx.x * blockDim.x * 2;
        const int t = threadIdx.x;
        if (x + 2 * t + 1 < length)
        {
            temp[2 * t] = input[x + 2 * t];
            temp[2 * t + 1] = input[x + 2 * t + 1];
        }
        else if (x + 2 * t < length)
        {
            temp[2 * t] = input[x + 2 * t];
            temp[2 * t + 1] = 0;
        }
        else
        {
            temp[2 * t] = 0;
            temp[2 * t + 1] = 0;
        }
        int stride = 1;
        while (stride < 2 * blockDim.x)
        {
            __syncthreads();
            int idx = (t + 1) * 2 * stride - 1;
            if (idx < 2 * blockDim.x && idx >= stride)
            {
                temp[idx] += temp[idx - stride];
            }
            stride *= 2;
        }

        stride = blockDim.x / 2;
        while (stride > 0)
        {
            __syncthreads();
            int idx = (t + 1) * 2 * stride - 1;
            if (idx + stride < 2 * blockDim.x)
            {
                temp[idx + stride] += temp[idx];
            }
            stride /= 2;
        }
        __syncthreads();
        if (x + 2 * t + 1 < length)
        {
            input[x + 2 * t] = temp[2 * t];
            input[x + 2 * t + 1] = temp[2 * t + 1];
        }
        else if (x + 2 * t < length)
        {
            input[x + 2 * t] = temp[2 * t];
        }

        __syncthreads();

        if (t == 0)
        {
            block_sums[blockIdx.x] = temp[2 * blockDim.x - 1];
            // printf("blockDim.x %d gridDim.x %d\n", blockDim.x, gridDim.x);
            // printf("thread 0 tb %d bs %d\n",blockIdx.x, scan_sums[blockIdx.x]);
        }
    }

    __global__ void single_scan(int *input, int length)
    {
#if __cplusplus > 201703L
        std::partial_sum(input, input + length, input);
#else
        for (int i = 1; i < length; ++i)
        {
            input[i] += input[i - 1];
        }
#endif
    }

    __global__ void add(int *input, int *output, int *scan_sums, int length)
    {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        if (x < length)
        {
            if (blockIdx.x == 0)
            {
                output[x] = input[x];
            }
            else
            {
                output[x] = input[x] + scan_sums[blockIdx.x - 1];
            }
        }
    }

    template <typename Iter>
    __device__ void insertion_sort(Iter begin, Iter end)
    {
        auto dist = end - begin;

        for (int i = 1; i < dist; ++i)
        {
            auto cur = *(begin + i);
            int j = i - 1;
            while (cur < *(begin + j) && j >= 0)
            {
                *(begin + j + 1) = *(begin + j);
                --j;
            }
            *(begin + j + 1) = cur;
        }
    }

    /*template <typename Iter>
    __device__ Iter partition(Iter begin, Iter end)
    {
        auto pivot = end - 1;
        auto i = begin;
        for (auto j = begin; j != pivot; ++j)
        {
            if (*j < *pivot)
            {
                swap(*i++, *j);
            }
        }
        swap(*i, *pivot);
        return i;
    }

    constexpr int THRESHOLD = 50;

    template <typename Iter>
    __device__ void introsort(Iter begin, Iter end, int depth)
    {
        printf("introsort depth = %d", depth);
        if (depth > 0)
        {
            printf("sort too deep\n");
            return;
        }
        if (begin < end)
        {
            auto dist = end - begin;
            if (dist < THRESHOLD) // do insertion sort when small
            {
                insertion_sort(begin, end);
            }
            else
            {
                auto pivot = partition(begin, end);
                introsort(begin, pivot, depth + 1);
                introsort(pivot + 1, end, depth + 1);
            }
        }
    }*/

    template <typename Iter>
    __device__ void gpu_sort(Iter begin, Iter end)
    {
#if __cplusplus > 201703L
        std::sort(begin, end);
#else
        insertion_sort(begin, end);
#endif
    }

    template <typename Iter>
    __device__ void gpu_merge(Iter begin1, Iter end1, Iter begin2, Iter end2, Iter output)
    {

#if __cplusplus > 201703L
        std::merge(begin1, end1, begin2, end2, output);
#else
        while (begin1 < end1 && begin2 < end2)
        {
            if (*begin1 < *begin2)
            {
                *output++ = *begin1++;
            }
            else
            {
                *output++ = *begin2++;
            }
        }
        while (begin1 < end1)
        {
            *output++ = *begin1++;
        }
        while (begin2 < end2)
        {
            *output++ = *begin2++;
        }
#endif
    }

    template <typename Iter>
    __device__ void gpu_copy(Iter begin, Iter end, Iter output)
    {

#if __cplusplus > 201703L
        std::copy(begin, end, output);
#else
        while (begin < end)
        {
            *begin++ = *output++;
        }
#endif
    }

    template <typename T>
    __device__ void swap(T &item1, T &item2)
    {
#if __cplusplus > 201703L
        std::swap(item1, item2);
#else
        T temp = item1;
        item1 = item2;
        item2 = temp;
#endif
    }

    template <typename Iter>
    __device__ void partial_sum(Iter begin, Iter end, Iter output)
    {
#if __cplusplus > 201703L
        std::partial_sum(begin, end, output);
#else
        Iter cur = begin + 1;
        while (cur < end)
        {
            *cur += *(cur - 1);
            ++cur;
        }
#endif
    }

    template <typename T>
    __device__ auto lower_bound(T *begin, T *end, const T &val)
    {
#if __cplusplus > 201703L
        return std::lower_bound(begin, end, val);
#else
        T *old_end = end;
        while (begin < end)
        {
            T *mid = begin + (end - begin) / 2;

            if (val <= *mid)
            {
                end = mid;
            }
            else
            {
                begin = mid + 1;
            }
        }

        if (begin < old_end && *begin < val)
        {
            return begin + 1;
        }
        else
        {
            return begin;
        }
#endif
    }

    template <typename T>
    cublasStatus_t dot(cublasHandle_t handle, int size, real *x, real *y, T *result)
    {
        if constexpr (std::is_same_v<float, T>)
        {
            return cublasSdot(handle, size, x, 1, y, 1, result);
        }
        else
        {
            return cublasDdot(handle, size, x, 1, y, 1, result);
        }
    }

    template <typename T>
    cublasStatus_t axpy(cublasHandle_t handle, int size, real alpha, real *x, T *y)
    {
        if constexpr (std::is_same_v<float, T>)
        {
            return cublasSaxpy(handle, size, &alpha, x, 1, y, 1);
        }
        else
        {
            return cublasDaxpy(handle, size, &alpha, x, 1, y, 1);
        }
    }

    template <typename T>
    cublasStatus_t scale(cublasHandle_t handle, int size, real alpha, T *y)
    {
        if constexpr (std::is_same_v<float, T>)
        {
            return cublasSscal(handle, size, &alpha, y, 1);
        }
        else
        {
            return cublasDscal(handle, size, &alpha, y, 1);
        }
    }

    class cusparseContext
    {
    public:
        cusparseContext()
        {
            cusparseCreate(&handle);
        }

        cusparseContext(const cusparseContext &other) = delete;

        ~cusparseContext()
        {
            for (void *buffer : buffers)
            {
                CHECK_CUDA(cudaFree(buffer));
            }
            cusparseDestroy(handle);
        }

        auto get_handle()
        {
            return handle;
        }

        void SpMV(cusparseOperation_t opA, cusparseSpMatDescr_t matA, cusparseDnVecDescr_t vecX, cusparseDnVecDescr_t vecY, real alpha, real beta, int call_n)
        {
            // size_t new_buffer_size = 0;
            size_t buffer_size = 0;
            if (call_n >= buffers.size())
            {
                // std::cerr << "initialize buffer\n";
                buffers.push_back(nullptr);
                CHECK_CUSPARSE(cusparseSpMV_bufferSize(handle, opA, &alpha, matA, vecX, &beta, vecY, CUDA_REAL, CUSPARSE_SPMV_CSR_ALG1, &buffer_size));
                CHECK_CUDA(cudaMalloc(&buffers[call_n], buffer_size));
            }
            else
            {
                // CHECK_CUSPARSE(cusparseSpMV_bufferSize(handle, opA, &alpha, matA, vecX, &beta, vecY, CUDA_REAL, CUSPARSE_SPMV_CSR_ALG1, &buffer_size));
            }
            // std::cerr << "call_n " << call_n << " buffer_size " << buffer_size << '\n';

            CHECK_CUSPARSE(cusparseSpMV(handle, opA, &alpha, matA, vecX, &beta, vecY, CUDA_REAL, CUSPARSE_SPMV_CSR_ALG1, buffers[call_n]));
        }

        // currently only non-transpose is supported by CuSparse
        // Do not reuse buffer
        void A_T_mult_A(cusparseSpMatDescr_t matA, cusparseSpMatDescr_t matB, cusparseSpMatDescr_t matC, real alpha, real beta, int col_size, SparseMatrixGPU<real> &res)
        {
            void *buffer1 = nullptr;
            void *buffer2 = nullptr;
            size_t buffer_size1 = 0;
            size_t buffer_size2 = 0;

            cusparseSpGEMMDescr_t spgemm_descr;
            CHECK_CUSPARSE(cusparseSpGEMM_createDescr(&spgemm_descr));

            constexpr cusparseOperation_t op = CUSPARSE_OPERATION_NON_TRANSPOSE;

            CHECK_CUSPARSE(
                cusparseSpGEMM_workEstimation(handle, op, op,
                                              &alpha, matA, matB, &beta, matC,
                                              CUDA_REAL, CUSPARSE_SPGEMM_DEFAULT,
                                              spgemm_descr, &buffer_size1, nullptr));
            CHECK_CUDA(cudaMalloc(&buffer1, buffer_size1));

            CHECK_CUSPARSE(
                cusparseSpGEMM_workEstimation(handle, op, op,
                                              &alpha, matA, matB, &beta, matC,
                                              CUDA_REAL, CUSPARSE_SPGEMM_DEFAULT,
                                              spgemm_descr, &buffer_size1, buffer1));

            CHECK_CUSPARSE(
                cusparseSpGEMM_compute(handle, op, op,
                                       &alpha, matA, matB, &beta, matC,
                                       CUDA_REAL, CUSPARSE_SPGEMM_DEFAULT,
                                       spgemm_descr, &buffer_size2, nullptr));
            CHECK_CUDA(cudaMalloc(&buffer2, buffer_size2));

            CHECK_CUSPARSE(
                cusparseSpGEMM_compute(handle, op, op,
                                       &alpha, matA, matB, &beta, matC,
                                       CUDA_REAL, CUSPARSE_SPGEMM_DEFAULT,
                                       spgemm_descr, &buffer_size2, buffer2));
            int64_t C_row_size = 0;
            int64_t C_col_size = 0;
            int64_t C_nnz = 0;
            CHECK_CUSPARSE(cusparseSpMatGetSize(matC, &C_row_size, &C_col_size, &C_nnz));

            CHECK_CUDA(cudaMalloc(&res.col_idx_d, C_nnz * sizeof(int)));
            CHECK_CUDA(cudaMalloc(&res.val_d, C_nnz * sizeof(int)));

            CHECK_CUSPARSE(
                cusparseCsrSetPointers(matC, res.row_ptr_d, res.col_idx_d, res.val_d));

            CHECK_CUSPARSE(cusparseSpGEMM_copy(handle, op, op,
                                               &alpha, matA, matB, &beta, matC,
                                               CUDA_REAL, CUSPARSE_SPGEMM_DEFAULT, spgemm_descr));

            CHECK_CUSPARSE(cusparseSpGEMM_destroyDescr(spgemm_descr));
        }

        void transpose(int col_size, const SparseMatrixGPU<real> &A, SparseMatrixGPU<real> &A_T, int call_n)
        {
            size_t buffer_size = 0;
            if (call_n >= buffers.size())
            {
                // std::cerr << "initialize buffer\n";
                buffers.push_back(nullptr);
                CHECK_CUSPARSE(cusparseCsr2cscEx2_bufferSize(handle, A.n_rows, col_size, A.nnz,
                                                             A.val_d, A.row_ptr_d, A.col_idx_d,
                                                             A_T.val_d, A_T.row_ptr_d, A_T.col_idx_d,
                                                             CUDA_REAL, CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO, CUSPARSE_CSR2CSC_ALG1, &buffer_size));
                CHECK_CUDA(cudaMalloc(&buffers[call_n], buffer_size));
            }
            else
            {
                // CHECK_CUSPARSE(cusparseSpMV_bufferSize(handle, opA, &alpha, matA, vecX, &beta, vecY, CUDA_REAL, CUSPARSE_SPMV_CSR_ALG1, &buffer_size));
            }
            CHECK_CUSPARSE(cusparseCsr2cscEx2(handle, A.n_rows, col_size, A.nnz,
                                              A.val_d, A.row_ptr_d, A.col_idx_d,
                                              A_T.val_d, A_T.row_ptr_d, A_T.col_idx_d,
                                              CUDA_REAL, CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO, CUSPARSE_CSR2CSC_ALG1, buffers[call_n]));
        }

    private:
        cusparseHandle_t handle;
        std::vector<void *> buffers;
    };

    __global__ void computeTrace(int n_rows, const int *row_idx, const int *col_idx, const real *val, real *trace)
    {
        __shared__ real trace_shared;
        if (threadIdx.x == 0)
            trace_shared = 0;
        __syncthreads();
        real trace_local = 0;
        for (int row = blockIdx.x * blockDim.x + threadIdx.x; row < n_rows; row += gridDim.x * blockDim.x)
        {
            for (int i = row_idx[row]; i < row_idx[row + 1]; ++i)
            {
                if (col_idx[i] == row)
                {
                    trace_local += val[i];
                }
            }
        }
        atomicAdd(&trace_shared, trace_local);
        __syncthreads();
        if (threadIdx.x == 0)
        {
            atomicAdd(trace, trace_shared);
        }
    }
}
