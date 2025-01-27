#pragma once
#include <cost_function.h>
#include <cusparse.h>
#include <cuda.h>
#include <numeric>
#include <lbfgs.h>
#include <iomanip>
#include "constants.hpp"
#include "type.hpp"
// #define culbfgsb
#ifdef culbfgsb
#include <culbfgsb.h>
#endif

namespace cudaSolarTomography
{
    constexpr bool COST_FUNC_DEBUG = false;

    class tikhonov_cost_function : public cost_function
    {
    public:
        tikhonov_cost_function(cusparseContext *context, cusparseSpMatDescr_t A_descr, cusparseDnVecDescr_t x_descr, cusparseDnVecDescr_t y_descr,
                               cusparseSpMatDescr_t D_descr, const real *y_all_d, const int n_rows, const int D_row_size, const real lambda_tik, int n_cols = N_BINS) : cost_function(n_cols),
                                                                                                                                                                             context(context), A_descr(A_descr), x_descr(x_descr), y_descr(y_descr), D_descr(D_descr), y_all_d(y_all_d),
                                                                                                                                                                             n_rows(n_rows), D_row_size(D_row_size), lambda_tik(lambda_tik), debug_buffer(n_cols, -1), n_cols(n_cols)
        {
            // Store A.* x

            CHECK_CUDA(cudaMalloc(&Ax_d, sizeof(real) * n_rows));

            CHECK_CUSPARSE(cusparseCreateDnVec(&Ax_descr, n_rows, Ax_d, CUDA_REAL));

            CHECK_CUDA(cudaMalloc(&Dx_d, sizeof(real) * D_row_size));

            CHECK_CUSPARSE(cusparseCreateDnVec(&Dx_descr, D_row_size, Dx_d, CUDA_REAL));

            CHECK_CUDA(cudaMalloc(&temp_d, sizeof(real) * n_rows));

            CHECK_CUDA(cudaMalloc(&temp_x_d, sizeof(real) * n_cols));

            CHECK_CUSPARSE(cusparseCreateDnVec(&temp_x_descr, n_cols, temp_x_d, CUDA_REAL));

            CHECK_CUDA(cudaMalloc(&yA_d, sizeof(real) * n_cols));

            CHECK_CUSPARSE(cusparseCreateDnVec(&yA_descr, n_cols, yA_d, CUDA_REAL));

            // 2 (y .* A) y (n_rows) A (n_rows, n_cols)
            context->SpMV(CUSPARSE_OPERATION_TRANSPOSE, A_descr, y_descr, yA_descr, 2, 0, 0);

            cublasCreate(&cublas_handle);

            std::cerr << "Cost Function Init " << cudaGetErrorName(cudaDeviceSynchronize()) << '\n';
        }

        virtual void f_gradf(const real *d_x, real *d_f, real *d_gradf) override
        {
            // std::cerr << "f_gradf called\n";
            // std::cerr << std::setprecision(20);
            // f = ||y - A.*x||^2 + ||lambda_tik * D.*x||^2.
            real new_f = 0;

            if (COST_FUNC_DEBUG)
            {
                cudaMemcpy(debug_buffer.data(), d_x, 8 * sizeof(real), cudaMemcpyDeviceToHost);
                std::cerr << "x ";
                for (int i = 0; i < 8; ++i)
                {
                    std::cerr << debug_buffer[i] << ' ';
                }
                std::cerr << "\n";
                std::cerr << cudaGetErrorName(cudaGetLastError()) << '\n';
            }
            // A .* x
            // std::cerr << "A .* x ";
            context->SpMV(CUSPARSE_OPERATION_NON_TRANSPOSE, A_descr, x_descr, Ax_descr, 1, 0, 0);
            if (COST_FUNC_DEBUG)
            {
                // std::cerr << cudaGetErrorName(cudaGetLastError()) << '\n';
                cudaMemcpy(debug_buffer.data(), Ax_d, 8 * sizeof(real), cudaMemcpyDeviceToHost);
                std::cerr << "Ax ";
                for (int i = 0; i < 8; ++i)
                {
                    std::cerr << debug_buffer[i] << ' ';
                }
                std::cerr << "\n";
                std::cerr << cudaGetErrorName(cudaGetLastError()) << '\n';
            }

            // temp_d = y, temp_d = -Ax + temp_d
            CHECK_CUDA(cudaMemcpy(temp_d, y_all_d, sizeof(real) * n_rows, cudaMemcpyDeviceToDevice));

            cudaDeviceSynchronize(); // axpy use Ax_d and temp_d

            axpy(cublas_handle, n_rows, -1, Ax_d, temp_d);

            cudaDeviceSynchronize(); // dot use temp_d

            real norm = 0;
            CHECK_CUBLAS(dot(cublas_handle, n_rows, temp_d, temp_d, &norm));

            cudaDeviceSynchronize(); // new_f use norm

            new_f += norm;

            if (COST_FUNC_DEBUG)
            {
                std::cerr << "new_f2 " << new_f << '\n';
                std::cerr << cudaGetErrorName(cudaGetLastError()) << '\n';
            }

            // D .* x
            // std::cerr << "D .* x\n";

            context->SpMV(CUSPARSE_OPERATION_NON_TRANSPOSE, D_descr, x_descr, Dx_descr, 1, 0, 1);

            cudaDeviceSynchronize(); // dot use Dx

            if (COST_FUNC_DEBUG)
            {
                cudaMemcpy(debug_buffer.data(), Dx_d, 20 * sizeof(real), cudaMemcpyDeviceToHost);
                std::cerr << "Dx ";
                for (int i = 0; i < 20; ++i)
                {
                    std::cerr << debug_buffer[i] << ' ';
                }
                std::cerr << "\n";
                std::cerr << cudaGetErrorName(cudaGetLastError()) << '\n';
            }

            CHECK_CUBLAS(dot(cublas_handle, D_row_size, Dx_d, Dx_d, &norm));

            cudaDeviceSynchronize(); // new_f use norm

            new_f += norm * lambda_tik * lambda_tik;

            CHECK_CUDA(cudaMemcpy(d_f, &new_f, sizeof(real), cudaMemcpyHostToDevice));

            // g = 2 * A.T .* (A .* x) - 2 * (y .* A) + 2 * (lambda_tik ^ 2) * (D.T .* (D .* x))

            // 2 * A.T .* (A .* x)

            // try to clear temp x
            // CHECK_CUDA(cudaMemset(temp_x_d, 0, sizeof(real) * n_cols));

            context->SpMV(CUSPARSE_OPERATION_TRANSPOSE, A_descr, Ax_descr, temp_x_descr, 2, 0, 0);
            // cudaCheckError();
            cudaDeviceSynchronize(); // use temp_x_d

            CHECK_CUDA(cudaMemcpy(d_gradf, temp_x_d, sizeof(real) * n_cols, cudaMemcpyDeviceToDevice));

            if (COST_FUNC_DEBUG)
            {
                cudaMemcpy(debug_buffer.data(), d_gradf, 8 * sizeof(real), cudaMemcpyDeviceToHost);
                std::cerr << "2 * A.T .* (A .* x)  ";
                for (int i = 0; i < 8; ++i)
                {
                    std::cerr << debug_buffer[i] << ' ';
                }
                std::cerr << "\n";
                std::cerr << cudaGetErrorName(cudaGetLastError()) << '\n';
            }

            // 2 * D.T .* (D .* x)
            context->SpMV(CUSPARSE_OPERATION_TRANSPOSE, D_descr, Dx_descr, temp_x_descr, 2, 0, 1);

            if (COST_FUNC_DEBUG)
            {
                cudaMemcpy(debug_buffer.data(), temp_x_d, 8 * sizeof(real), cudaMemcpyDeviceToHost);
                std::cerr << "2 D.T Dx ";
                for (int i = 0; i < 8; ++i)
                {
                    std::cerr << debug_buffer[i] << ' ';
                }
                std::cerr << "\n";
                std::cerr << cudaGetErrorName(cudaGetLastError()) << '\n';
            }

            cudaDeviceSynchronize();

            // 2 * A.T .* (A .* x) - 2 * (y .* A)
            axpy(cublas_handle, n_cols, -1, yA_d, d_gradf);

            cudaDeviceSynchronize();

            if (COST_FUNC_DEBUG)
            {
                cudaMemcpy(debug_buffer.data(), d_gradf, 8 * sizeof(real), cudaMemcpyDeviceToHost);
                std::cerr << "2 * A.T .* (A .* x) - 2 * (y .* A)  ";
                for (int i = 0; i < 8; ++i)
                {
                    std::cerr << debug_buffer[i] << ' ';
                }
                std::cerr << "\n";
                std::cerr << cudaGetErrorName(cudaGetLastError()) << '\n';
            }

            axpy(cublas_handle, n_cols, lambda_tik * lambda_tik, temp_x_d, d_gradf);

            cudaDeviceSynchronize();

            if (COST_FUNC_DEBUG)
            {
                cudaMemcpy(debug_buffer.data(), d_gradf, 8 * sizeof(real), cudaMemcpyDeviceToHost);
                std::cerr << "dgradf ";
                for (int i = 0; i < 8; ++i)
                {
                    std::cerr << debug_buffer[i] << ' ';
                }
                for (int i = 0; i < n_cols; ++i)
                {
                    if (std::isnan(debug_buffer[i]))
                    {
                        std::cerr << "NaN in gradient!\n";
                        std::terminate();
                    }
                }
                std::cerr << "\n";
                std::cerr << cudaGetErrorName(cudaGetLastError()) << '\n';
            }
        }

        virtual ~tikhonov_cost_function()
        {
            cusparseDestroyDnVec(yA_descr);
            cusparseDestroyDnVec(Ax_descr);
            cusparseDestroyDnVec(Dx_descr);
            cusparseDestroyDnVec(temp_x_descr);
            cudaFree(Ax_d);
            cudaFree(Dx_d);
            cudaFree(temp_d);
            cudaFree(temp_x_d);
            cudaFree(yA_d);
            // release cublas
            cublasDestroy(cublas_handle);
        }

    private:
        cusparseContext *context;
        cusparseSpMatDescr_t A_descr;
        cusparseSpMatDescr_t D_descr;
        cusparseDnVecDescr_t y_descr;
        cusparseDnVecDescr_t x_descr;

        const real *y_all_d;
        const int n_rows;
        const int D_row_size;
        const real lambda_tik;

        real *Ax_d;
        real *Dx_d;
        real *temp_d;
        real *temp_x_d;
        real *yA_d;

        cusparseDnVecDescr_t yA_descr;
        cusparseDnVecDescr_t Ax_descr;
        cusparseDnVecDescr_t Dx_descr;
        cusparseDnVecDescr_t temp_x_descr;

        std::vector<real> debug_buffer;

        cublasHandle_t cublas_handle;

        int n_cols;
    };

    struct optimize_result
    {
        std::string optimizer_status;
        int iters;
        real f;
        real grad;
        friend std::ostream &operator<<(std::ostream &os, const optimize_result &res)
        {
            os << "Termination Reason: " << res.optimizer_status << " Number of Iterations " << res.iters << " f = " << res.f << " grad = " << res.grad;
            return os;
        }
    };

    optimize_result optimize(cusparseContext *context, cusparseSpMatDescr_t A_descr, cusparseDnVecDescr_t x_descr, cusparseDnVecDescr_t y_descr, cusparseSpMatDescr_t D_descr, const real *y_all_d, real *x_result_d, const int n_rows, const int D_row_size, const real lambda_tik, const int n_cols = N_BINS)
    {
        std::cerr << "optimize\n";
        tikhonov_cost_function cost_func(context, A_descr, x_descr, y_descr, D_descr, y_all_d, n_rows, D_row_size, lambda_tik, n_cols);
        cost_func.m_maxIter = std::numeric_limits<size_t>::max();
        // cost_func.m_maxIter = 800;
        // cost_func.m_maxIter = 100;
        // cost_func.m_maxIter = 8302;
        cost_func.m_gradientEps = 1e-20f;
        lbfgs optimizer(cost_func);

        optimize_result res;
        optimizer.minimize(x_result_d);
        res.optimizer_status = optimizer.cur_stat;
        res.iters = optimizer.cur_iters;
        res.f = optimizer.cur_f;
        res.grad = optimizer.cur_grad;
        // std::cout << res << '\n';
        return res;
    }

    class dynamic_cost_function : public cost_function
    {
    public:
        dynamic_cost_function(cusparseContext *context, cusparseSpMatDescr_t A_descr, cusparseDnVecDescr_t x_descr, cusparseDnVecDescr_t y_descr,
                              const std::vector<cusparseSpMatDescr_t> D_descrs, const real *y_all_d, const int n_rows,
                              const std::vector<int> D_row_sizes,
                              const std::vector<real> D_lambdas, int n_cols) : cost_function(n_cols),
                                                                               context(context), A_descr(A_descr), x_descr(x_descr), y_descr(y_descr),
                                                                               D_descrs(D_descrs),
                                                                               y_all_d(y_all_d), n_rows(n_rows),
                                                                               D_row_sizes(D_row_sizes),
                                                                               D_lambdas(D_lambdas),
                                                                               n_cols(n_cols)
        {
            // Store A.* x

            CHECK_CUDA(cudaMalloc(&Ax_d, sizeof(real) * n_rows));

            CHECK_CUSPARSE(cusparseCreateDnVec(&Ax_descr, n_rows, Ax_d, CUDA_REAL));

            D_size = D_descrs.size();

            for (int i = 0; i < D_size; ++i)
            {
                CHECK_CUDA(cudaMalloc(&Dx_d[i], sizeof(real) * D_row_sizes[i]));
                CHECK_CUSPARSE(cusparseCreateDnVec(&Dx_descrs[i], D_row_sizes[i], Dx_d[i], CUDA_REAL));
            }

            CHECK_CUDA(cudaMalloc(&temp_d, sizeof(real) * n_rows));

            CHECK_CUDA(cudaMalloc(&temp_x_d, sizeof(real) * n_cols));

            CHECK_CUSPARSE(cusparseCreateDnVec(&temp_x_descr, n_cols, temp_x_d, CUDA_REAL));

            CHECK_CUDA(cudaMalloc(&yA_d, sizeof(real) * n_cols));

            CHECK_CUSPARSE(cusparseCreateDnVec(&yA_descr, n_cols, yA_d, CUDA_REAL));

            // 2 (y .* A) y (n_rows) A (n_rows, n_cols)
            context->SpMV(CUSPARSE_OPERATION_TRANSPOSE, A_descr, y_descr, yA_descr, 2, 0, 0);

            cublasCreate(&cublas_handle);

            std::cerr << "Cost Function Init " << cudaGetErrorName(cudaDeviceSynchronize()) << '\n';
        }

        virtual void f_gradf(const real *d_x, real *d_f, real *d_gradf) override
        {
            // std::cerr << "f_gradf called\n";
            // std::cerr << std::setprecision(20);
            // f = ||y - A.*x||^2 + (sum i from 1 to n){||lambda_i * D_i.*x||^2.}
            real new_f = 0;

            // A .* x
            // std::cerr << "A .* x ";
            context->SpMV(CUSPARSE_OPERATION_NON_TRANSPOSE, A_descr, x_descr, Ax_descr, 1, 0, 0);

            // temp_d = y, temp_d = -Ax + temp_d
            CHECK_CUDA(cudaMemcpy(temp_d, y_all_d, sizeof(real) * n_rows, cudaMemcpyDeviceToDevice));

            cudaDeviceSynchronize(); // axpy use Ax_d and temp_d

            axpy(cublas_handle, n_rows, -1, Ax_d, temp_d);

            cudaDeviceSynchronize(); // dot use temp_d

            real norm = 0;
            CHECK_CUBLAS(dot(cublas_handle, n_rows, temp_d, temp_d, &norm));

            cudaDeviceSynchronize(); // new_f use norm

            new_f += norm;

            // D_i .* x
            // std::cerr << "D .* x\n";

            for (int i = 0; i < D_size; ++i)
            {
                context->SpMV(CUSPARSE_OPERATION_NON_TRANSPOSE, D_descrs[i], x_descr, Dx_descrs[i], 1, 0, 1 + i);

                cudaDeviceSynchronize(); // dot use Dx

                CHECK_CUBLAS(dot(cublas_handle, D_row_sizes[i], Dx_d[i], Dx_d[i], &norm));

                cudaDeviceSynchronize(); // new_f use norm

                new_f += norm * D_lambdas[i] * D_lambdas[i];
            }

            CHECK_CUDA(cudaMemcpy(d_f, &new_f, sizeof(real), cudaMemcpyHostToDevice));

            // g = 2 * A.T .* (A .* x) - 2 * (y .* A) + (sum i from 0 to n) {2 * (lambda_i ^ 2) * (D_i.T .* (D_i .* x))

            // 2 * A.T .* (A .* x)

            // try to clear temp x
            // CHECK_CUDA(cudaMemset(temp_x_d, 0, sizeof(real) * n_cols));

            context->SpMV(CUSPARSE_OPERATION_TRANSPOSE, A_descr, Ax_descr, temp_x_descr, 2, 0, 0);
            // cudaCheckError();
            cudaDeviceSynchronize(); // use temp_x_d

            CHECK_CUDA(cudaMemcpy(d_gradf, temp_x_d, sizeof(real) * n_cols, cudaMemcpyDeviceToDevice));

            // 2 * A.T .* (A .* x) - 2 * (y .* A)
            axpy(cublas_handle, n_cols, -1, yA_d, d_gradf);

            cudaDeviceSynchronize();

            // + 2 * D_i.T .* (D_i .* x)
            for (int i = 0; i < D_size; ++i)
            {
                context->SpMV(CUSPARSE_OPERATION_TRANSPOSE, D_descrs[i], Dx_descrs[i], temp_x_descr, 2, 0, 1 + i);
                cudaDeviceSynchronize();

                axpy(cublas_handle, n_cols, D_lambdas[i] * D_lambdas[i], temp_x_d, d_gradf);

                cudaDeviceSynchronize();
            }
        }

        virtual ~dynamic_cost_function()
        {
            cusparseDestroyDnVec(yA_descr);
            cusparseDestroyDnVec(Ax_descr);
            cusparseDestroyDnVec(temp_x_descr);
            cudaFree(Ax_d);
            cudaFree(temp_d);
            cudaFree(temp_x_d);
            cudaFree(yA_d);

            for (int i = 0; i < D_size; ++i)
            {
                cusparseDestroyDnVec(Dx_descrs[i]);
                cudaFree(Dx_d[i]);
            }
            // release cublas
            cublasDestroy(cublas_handle);
        }

    private:
        cusparseContext *context;
        cusparseSpMatDescr_t A_descr;
        cusparseDnVecDescr_t y_descr;
        cusparseDnVecDescr_t x_descr;

        const real *y_all_d;
        const int n_rows;

        std::vector<cusparseSpMatDescr_t> D_descrs;
        const std::vector<int> D_row_sizes;
        const std::vector<real> D_lambdas;

        int D_size;

        real *Ax_d;
        real *yA_d;

        std::vector<real *> Dx_d;
        std::vector<cusparseDnVecDescr_t> Dx_descrs;
        real *temp_d;
        real *temp_x_d;

        cusparseDnVecDescr_t yA_descr;
        cusparseDnVecDescr_t Ax_descr;
        cusparseDnVecDescr_t temp_x_descr;

        cublasHandle_t cublas_handle;

        int n_cols;
    };

    // A program should only call either optimize or optimize_dynamic
    optimize_result optimize_dynamic(cusparseContext *context, cusparseSpMatDescr_t A_descr, cusparseDnVecDescr_t x_descr, cusparseDnVecDescr_t y_descr, const std::vector<cusparseSpMatDescr_t> D_descrs, const real *y_all_d, real *x_result_d, const int n_rows,
                                     const std::vector<int> D_row_sizes,
                                     const std::vector<real> D_lambdas, int n_cols)
    {
        std::cerr << "optimize for dynamic tomography\n";
        dynamic_cost_function cost_func(context, A_descr, x_descr, y_descr, D_descrs, y_all_d, n_rows, D_row_sizes, D_lambdas, n_cols);
        cost_func.m_maxIter = std::numeric_limits<size_t>::max();
        // cost_func.m_maxIter = 8302;
        cost_func.m_gradientEps = 1e-20f;
        lbfgs optimizer(cost_func);

        optimize_result res;
        optimizer.minimize(x_result_d);
        res.optimizer_status = optimizer.cur_stat;
        res.iters = optimizer.cur_iters;
        res.f = optimizer.cur_f;
        res.grad = optimizer.cur_grad;
        // std::cout << res << '\n';
        return res;
    }
#ifdef culbfgsb
    size_t buffer_size = 0;
    real *buffer = nullptr;

    cusparseStatus_t SpMV(cusparseHandle_t handle, cusparseOperation_t opA, cusparseSpMatDescr_t matA, cusparseDnVecDescr_t vecX, cusparseDnVecDescr_t vecY, cudaDataType computeType, real alpha, real beta)
    {
        size_t new_buffer_size = 0;
        auto status = cusparseSpMV_bufferSize(handle, opA, &alpha, matA, vecX, &beta, vecY, CUDA_REAL, CUSPARSE_SPMV_CSR_ALG1, &new_buffer_size);
        cudaDeviceSynchronize();
        // std::cerr << "new buffer size " << new_buffer_size << " buffer size " << buffer_size << '\n';
        if (new_buffer_size > buffer_size)
        {
            if (!buffer)
            {
                cudaFree(buffer);
            }
            cudaMalloc(&buffer, new_buffer_size);
            std::cerr << "Buffer Realloc " << cudaGetErrorName(cudaDeviceSynchronize()) << '\n';
            // std::cerr << (buffer == nullptr) << '\n';
            if (buffer == nullptr)
            {
                std::cerr << "All GPU memory used up!\n";
                std::terminate();
            }
            buffer_size = new_buffer_size;
        }
        // else
        // {
        //     cudaMemset(buffer, 0, new_buffer_size);
        // }
        // cusparseSpMV_preprocess(handle, opA, &alpha, matA, vecX, &beta, vecY, CUDA_REAL, CUSPARSE_SPMV_CSR_ALG1, buffer);
        status = cusparseSpMV(handle, opA, &alpha, matA, vecX, &beta, vecY, CUDA_REAL, CUSPARSE_SPMV_CSR_ALG1, buffer);
        cudaDeviceSynchronize();
        return status;
    }

    class lbfgsb_callback_function
    {
    public:
        lbfgsb_callback_function(cublasHandle_t cublas_handle, cusparseHandle_t handle, cusparseSpMatDescr_t A_descr, cusparseDnVecDescr_t x_descr, cusparseDnVecDescr_t y_descr,
                                 cusparseSpMatDescr_t D_descr, const real *y_all_d, const int n_rows, const int D_row_size, const real lambda_tik, real *Ax_d,
                                 real *Dx_d, real *temp_d, real *temp_x_d, real *yA_d,
                                 cusparseDnVecDescr_t yA_descr, cusparseDnVecDescr_t Ax_descr, cusparseDnVecDescr_t Dx_descr, cusparseDnVecDescr_t temp_x_descr) : cublas_handle(cublas_handle),
                                                                                                                                                                   handle(handle),
                                                                                                                                                                   A_descr(A_descr), x_descr(x_descr), y_descr(y_descr), D_descr(D_descr), y_all_d(y_all_d),
                                                                                                                                                                   n_rows(n_rows), D_row_size(D_row_size), lambda_tik(lambda_tik), grad(N_BINS, -1),
                                                                                                                                                                   Ax_d(Ax_d), Dx_d(Dx_d), temp_d(temp_d), temp_x_d(temp_x_d), yA_d(yA_d),
                                                                                                                                                                   yA_descr(yA_descr), Ax_descr(Ax_descr), Dx_descr(Dx_descr), temp_x_descr(temp_x_descr)
        {
        }

        int operator()(real *d_x, real &f, real *d_gradf, const cudaStream_t &stream, const LBFGSB_CUDA_SUMMARY<real> &summary)
        {
            // std::cerr << "f_gradf called\n";
            // std::cerr << std::setprecision(20);
            // f = ||y - A.*x||^2 + ||lambda_tik * D.*x||^2.

            real new_f = 0;
            // cudaMemcpy(Ax_d, y_all_d, sizeof(real) * n_rows, cudaMemcpyDeviceToDevice);

            alpha = 1;
            beta = 0;

            if (COST_FUNC_DEBUG)
            {
                cudaMemcpy(grad.data(), d_x, N_BINS * sizeof(real), cudaMemcpyDeviceToHost);
                std::cerr << "x ";
                for (int i = 0; i < 8; ++i)
                {
                    std::cerr << grad[i] << ' ';
                }
                std::cerr << "\n";
                std::cerr << cudaDeviceSynchronize() << '\n';
            }

            // A .* x
            // std::cerr << "A .* x ";
            SpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, A_descr, x_descr, Ax_descr, CUDA_REAL, alpha, beta);

            if (COST_FUNC_DEBUG)
            {
                cudaMemcpy(grad.data(), Ax_d, N_BINS * sizeof(real), cudaMemcpyDeviceToHost);
                std::cerr << "Ax ";
                for (int i = 0; i < 8; ++i)
                {
                    std::cerr << grad[i] << ' ';
                }
                std::cerr << "\n";
                std::cerr << cudaDeviceSynchronize() << '\n';
            }

            // temp_d = y, temp_d = -Ax + temp_d
            alpha = -1;
            cudaMemcpy(temp_d, y_all_d, sizeof(real) * n_rows, cudaMemcpyDeviceToDevice);
            cudaDeviceSynchronize();

            // cublasSaxpy(cublas_handle, n_rows, &alpha, Ax_d, 1, temp_d, 1);
            axpy(cublas_handle, n_rows, &alpha, Ax_d, temp_d);
            cudaDeviceSynchronize();

            real norm = 0;
            // cublasSnrm2(cublas_handle, n_rows, temp_d, 1, &norm);
            // new_f += norm * norm;
            // cublasSdot(cublas_handle, n_rows, temp_d, 1, temp_d, 1, &norm);
            dot(cublas_handle, n_rows, temp_d, temp_d, &norm);
            cudaDeviceSynchronize();
            new_f += norm;

            if (COST_FUNC_DEBUG)
            {
                std::cerr << "new_f2 " << new_f << '\n';
                std::cerr << cudaDeviceSynchronize() << '\n';
            }

            alpha = 1;

            // D .* x
            // std::cerr << "D .* x\n";

            SpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, D_descr, x_descr, Dx_descr, CUDA_REAL, alpha, beta);

            if (COST_FUNC_DEBUG)
            {
                cudaMemcpy(grad.data(), Dx_d, N_BINS * sizeof(real), cudaMemcpyDeviceToHost);
                std::cerr << "Dx ";
                for (int i = 0; i < 8; ++i)
                {
                    std::cerr << grad[i] << ' ';
                }
                std::cerr << "\n";
                std::cerr << cudaDeviceSynchronize() << '\n';
            }

            // cublasSnrm2(cublas_handle, D_row_size, Dx_d, 1, &norm);

            // new_f += norm * norm * lambda_tik * lambda_tik;
            // cublasSdot(cublas_handle, D_row_size, Dx_d, 1, Dx_d, 1, &norm);
            dot(cublas_handle, D_row_size, Dx_d, Dx_d, &norm);
            cudaDeviceSynchronize();
            new_f += norm * lambda_tik * lambda_tik;

            // cudaMemcpy(d_f, &new_f, sizeof(real), cudaMemcpyHostToDevice);
            // cudaDeviceSynchronize();
            f = new_f;

            // g = 2 * A.T .* (A .* x) - 2 * (y .* A) + 2 * (lambda_tik ^ 2) * (D.T .* (D .* x))

            // 2 * A.T .* (A .* x)
            // std::cerr << "2 * A.T .* (A .* x)\n";

            alpha = 2;
            SpMV(handle, CUSPARSE_OPERATION_TRANSPOSE, A_descr, Ax_descr, temp_x_descr, CUDA_REAL, alpha, beta);

            cudaMemcpy(d_gradf, temp_x_d, sizeof(real) * N_BINS, cudaMemcpyDeviceToDevice);
            cudaDeviceSynchronize();

            if (COST_FUNC_DEBUG)
            {
                cudaMemcpy(grad.data(), d_gradf, N_BINS * sizeof(real), cudaMemcpyDeviceToHost);
                std::cerr << "2 * A.T .* (A .* x)  ";
                for (int i = 0; i < 8; ++i)
                {
                    std::cerr << grad[i] << ' ';
                }
                std::cerr << "\n";
                std::cerr << cudaDeviceSynchronize() << '\n';
            }

            alpha = -1;
            // cublasSaxpy(cublas_handle, N_BINS, &alpha, yA_d, 1, d_gradf, 1);
            axpy(cublas_handle, N_BINS, &alpha, yA_d, d_gradf);
            cudaDeviceSynchronize();

            if (COST_FUNC_DEBUG)
            {
                cudaMemcpy(grad.data(), d_gradf, N_BINS * sizeof(real), cudaMemcpyDeviceToHost);
                std::cerr << "2 * A.T .* (A .* x) - 2 * (y .* A)  ";
                for (int i = 0; i < 8; ++i)
                {
                    std::cerr << grad[i] << ' ';
                }
                std::cerr << "\n";
                std::cerr << cudaDeviceSynchronize() << '\n';
            }

            // 2 * D.T .* (D .* x)
            alpha = 2;
            SpMV(handle, CUSPARSE_OPERATION_TRANSPOSE, D_descr, Dx_descr, temp_x_descr, CUDA_REAL, alpha, beta);

            if (COST_FUNC_DEBUG)
            {
                cudaMemcpy(grad.data(), temp_x_d, N_BINS * sizeof(real), cudaMemcpyDeviceToHost);
                std::cerr << "2 D.T Dx ";
                for (int i = 0; i < 8; ++i)
                {
                    std::cerr << grad[i] << ' ';
                }
                std::cerr << "\n";
                std::cerr << cudaDeviceSynchronize() << '\n';
            }

            alpha = lambda_tik * lambda_tik;
            std::cerr << "alpha " << alpha << '\n';
            // cublasSaxpy(cublas_handle, N_BINS, &alpha, temp_x_d, 1, d_gradf, 1);
            axpy(cublas_handle, N_BINS, &alpha, temp_x_d, d_gradf);
            cudaDeviceSynchronize();

            if (COST_FUNC_DEBUG)
            {
                cudaMemcpy(grad.data(), d_gradf, N_BINS * sizeof(real), cudaMemcpyDeviceToHost);
                std::cerr << "dgradf ";
                for (int i = 0; i < 8; ++i)
                {
                    std::cerr << grad[i] << ' ';
                }
                for (int i = 0; i < N_BINS; ++i)
                {
                    if (std::isnan(grad[i]))
                    {
                        std::cerr << "NaN in gradient!\n";
                        std::terminate();
                    }
                }
                std::cerr << "\n";
                std::cerr << cudaDeviceSynchronize() << '\n';
            }
            return 0;
        }

    private:
        cusparseHandle_t handle;
        cusparseSpMatDescr_t A_descr;
        cusparseSpMatDescr_t D_descr;
        cusparseDnVecDescr_t y_descr;

        cusparseDnVecDescr_t x_descr;

        const real *y_all_d;
        const int n_rows;
        const int D_row_size;
        const real lambda_tik;

        real *Ax_d;
        real *Dx_d;
        real *temp_d;
        real *temp_x_d;
        real *yA_d;

        cusparseDnVecDescr_t yA_descr;
        cusparseDnVecDescr_t Ax_descr;
        cusparseDnVecDescr_t Dx_descr;
        cusparseDnVecDescr_t temp_x_descr;

        std::vector<real> grad;

        real alpha;
        real beta;

        cublasHandle_t cublas_handle;
    };

    optimize_result optimize2(cusparseHandle_t handle, cusparseSpMatDescr_t A_descr, cusparseDnVecDescr_t x_descr, cusparseDnVecDescr_t y_descr, cusparseSpMatDescr_t D_descr, const real *y_all_d, real *x_result_d, const int n_rows, const int D_row_size, const real lambda_tik)
    {
        std::cerr << "optimize with lbfgsb\n";
        LBFGSB_CUDA_OPTION<real> lbfgsb_options;

        lbfgsbcuda::lbfgsbdefaultoption<real>(lbfgsb_options);
        lbfgsb_options.mode = LCM_CUDA;
        lbfgsb_options.eps_f = static_cast<real>(1e-20);
        lbfgsb_options.eps_g = static_cast<real>(1e-20);
        lbfgsb_options.eps_x = static_cast<real>(1e-20);
        lbfgsb_options.max_iteration = 1000;

        LBFGSB_CUDA_STATE<real> state;
        memset(&state, 0, sizeof(state));
        cublasStatus_t stat = cublasCreate(&(state.m_cublas_handle));

        real *Ax_d;
        real *Dx_d;
        real *temp_d;
        real *temp_x_d;
        real *yA_d;

        cusparseDnVecDescr_t yA_descr;
        cusparseDnVecDescr_t Ax_descr;
        cusparseDnVecDescr_t Dx_descr;
        cusparseDnVecDescr_t temp_x_descr;

        // Store A.* x

        cudaMalloc(&Ax_d, sizeof(real) * n_rows);

        cusparseCreateDnVec(&Ax_descr, n_rows, Ax_d, CUDA_REAL);

        cudaMalloc(&Dx_d, sizeof(real) * D_row_size);

        cusparseCreateDnVec(&Dx_descr, D_row_size, Dx_d, CUDA_REAL);

        cudaMalloc(&temp_d, sizeof(real) * n_rows);

        cudaMalloc(&temp_x_d, sizeof(real) * N_BINS);

        cusparseCreateDnVec(&temp_x_descr, N_BINS, temp_x_d, CUDA_REAL);

        cudaMalloc(&yA_d, sizeof(real) * N_BINS);

        cusparseCreateDnVec(&yA_descr, N_BINS, yA_d, CUDA_REAL);

        real alpha = 2;
        real beta = 0;
        SpMV(handle, CUSPARSE_OPERATION_TRANSPOSE, A_descr, y_descr, yA_descr, CUDA_REAL, alpha, beta);

        std::cerr << "Callback Function Init " << cudaGetErrorName(cudaDeviceSynchronize()) << '\n';

        lbfgsb_callback_function callback = lbfgsb_callback_function(state.m_cublas_handle, handle, A_descr, x_descr, y_descr, D_descr, y_all_d, n_rows, D_row_size, lambda_tik,
                                                                     Ax_d, Dx_d, temp_d, temp_x_d, yA_d,
                                                                     yA_descr, Ax_descr, Dx_descr, temp_x_descr);

        state.m_funcgrad_callback = callback;

        real *xl = nullptr;
        real *xu = nullptr;
        int *nbd = nullptr;

        cudaMalloc(&xl, N_BINS * sizeof(xl[0]));
        cudaMalloc(&xu, N_BINS * sizeof(xu[0]));

        cudaMemset(xl, 0, N_BINS * sizeof(xl[0]));
        cudaMemset(xu, 0, N_BINS * sizeof(xu[0]));

        cudaMalloc(&nbd, N_BINS * sizeof(nbd[0]));
        cudaMemset(nbd, 0, N_BINS * sizeof(nbd[0]));

        LBFGSB_CUDA_SUMMARY<real> summary;
        memset(&summary, 0, sizeof(summary));

        lbfgsbcuda::lbfgsbminimize<real>(N_BINS, state, lbfgsb_options, x_result_d, nbd, xl, xu, summary);

        std::cout << "minimize end\n";

        cudaFree(xl);
        cudaFree(xu);
        cudaFree(nbd);
        cublasDestroy(state.m_cublas_handle);

        cusparseDestroyDnVec(yA_descr);
        cusparseDestroyDnVec(Ax_descr);
        cusparseDestroyDnVec(Dx_descr);
        cusparseDestroyDnVec(temp_x_descr);
        cudaFree(Ax_d);
        cudaFree(Dx_d);
        cudaFree(temp_d);
        cudaFree(temp_x_d);
        cudaFree(yA_d);
        if (!buffer)
        {
            cudaFree(buffer);
        }

        optimize_result res;
        res.optimizer_status = "SUCCESS";
        res.iters = summary.num_iteration;
        // res.f = optimizer.cur_f;
        // res.grad = optimizer.cur_grad;
        std::cout << res << '\n';
        return res;
    }

#endif
}
