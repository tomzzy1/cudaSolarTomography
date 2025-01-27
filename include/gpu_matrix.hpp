#pragma once
#include <cuda.h>
#include "constants.hpp"
#include "error_check.cuh"
#include <fstream>

namespace cudaSolarTomography
{
    template <typename T>
    struct SparseMatrix;

    template <typename T>
    struct SparseMatrixGPU
    {
        SparseMatrixGPU(int n_rows, int nnz) : n_rows(n_rows), nnz(nnz)
        {
            CHECK_CUDA(cudaMalloc(&row_ptr_d, (n_rows + 1) * sizeof(int)));
            CHECK_CUDA(cudaMalloc(&col_idx_d, nnz * sizeof(int)));
            CHECK_CUDA(cudaMalloc(&val_d, nnz * sizeof(T)));
        }

        // SparseMatrixGPU(SparseMatrixGPU &&other) = default;
        SparseMatrixGPU(SparseMatrixGPU &&other) : row_ptr_d(other.row_ptr_d), col_idx_d(other.col_idx_d),
                                                   val_d(other.val_d), n_rows(other.n_rows), nnz(other.nnz)
        {
            other.row_ptr_d = nullptr;
            other.col_idx_d = nullptr;
            other.val_d = nullptr;
        }
        SparseMatrixGPU(const SparseMatrixGPU &other) = delete;
        cusparseSpMatDescr_t createSpMatDescr(int n_cols = N_BINS) const
        {
            cusparseSpMatDescr_t descr;
            constexpr auto CUDA_REAL = std::is_same_v<T, float> ? CUDA_R_32F : CUDA_R_64F;
            CHECK_CUSPARSE(cusparseCreateCsr(&descr, n_rows, n_cols, nnz, row_ptr_d, col_idx_d, val_d, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_REAL));
            return descr;
        }
        template <typename host_real>
        SparseMatrix<host_real> to_host()
        {
            SparseMatrix<host_real> result(n_rows, nnz);
            CHECK_CUDA(cudaMemcpy(result.row_ptr_h, row_ptr_d, (n_rows + 1) * sizeof(int), cudaMemcpyDeviceToHost));
            CHECK_CUDA(cudaMemcpy(result.col_idx_h, col_idx_d, nnz * sizeof(int), cudaMemcpyDeviceToHost));

            if constexpr (std::is_same_v<T, host_real>)
            {
                CHECK_CUDA(cudaMemcpy(result.val_h, val_d, nnz * sizeof(T), cudaMemcpyDeviceToHost));
            }
            else
            {
                std::vector<T> val_temp_h;
                val_temp_h.reserve(nnz);
                CHECK_CUDA(cudaMemcpy(val_temp_h.data(), result.val_d, nnz * sizeof(T), cudaMemcpyDeviceToHost));
                for (int i = 0; i < nnz; ++i)
                {
                    result.val_h[i] = static_cast<host_real>(val_temp_h[i]);
                }
            }
            return result;
        }
        ~SparseMatrixGPU()
        {
            if (row_ptr_d)
                CHECK_CUDA(cudaFree(row_ptr_d));
            if (col_idx_d)
                CHECK_CUDA(cudaFree(col_idx_d));
            if (val_d)
                CHECK_CUDA(cudaFree(val_d));
        }
        int *row_ptr_d;
        int *col_idx_d;
        T *val_d;
        int n_rows;
        int nnz;
    };

    template <typename T>
    struct SparseMatrixAndImage;

    template <typename T>
    struct SparseMatrixAndImageGPU : public SparseMatrixGPU<T>
    {
        SparseMatrixAndImageGPU(int n_rows, int nnz) : SparseMatrixGPU<T>(n_rows, nnz)
        {
            CHECK_CUDA(cudaMalloc(&y_d, n_rows * sizeof(T)));
        }
        SparseMatrixAndImageGPU(SparseMatrixGPU<T> &&other) : SparseMatrixGPU<T>(std::move(other))
        {
            CHECK_CUDA(cudaMalloc(&y_d, SparseMatrixGPU<T>::n_rows * sizeof(T)));
        }
        // SparseMatrixAndImageGPU(SparseMatrixAndImageGPU &&other) = default;
        SparseMatrixAndImageGPU(SparseMatrixAndImageGPU &&other) : SparseMatrixGPU<T>(std::move(other)), y_d(other.y_d)
        {
            other.y_d = nullptr;
        }
        SparseMatrixAndImageGPU(const SparseMatrixAndImageGPU &other) = delete;
        cusparseDnVecDescr_t createDnVecDescr() const
        {
            cusparseDnVecDescr_t y_descr;
            constexpr auto CUDA_REAL = std::is_same_v<T, float> ? CUDA_R_32F : CUDA_R_64F;
            CHECK_CUSPARSE(cusparseCreateDnVec(&y_descr, SparseMatrixGPU<T>::n_rows, y_d, CUDA_REAL));
            return y_descr;
        }
        template <typename host_real>
        SparseMatrixAndImage<host_real> to_host()
        {
            SparseMatrixAndImage<host_real> result(SparseMatrixGPU<T>::template to_host<host_real>());
            if constexpr (std::is_same_v<T, host_real>)
            {
                CHECK_CUDA(cudaMemcpy(result.y_h, y_d, SparseMatrixGPU<T>::n_rows * sizeof(T), cudaMemcpyDeviceToHost));
            }
            else
            {
                std::vector<T> y_temp_h;
                y_temp_h.reserve(SparseMatrixGPU<T>::n_rows);
                CHECK_CUDA(cudaMemcpy(y_temp_h.data(), result.y_d, SparseMatrixGPU<T>::n_rows * sizeof(T), cudaMemcpyDeviceToHost));
                for (int i = 0; i < SparseMatrixGPU<T>::n_rows; ++i)
                {
                    result.y_h[i] = static_cast<host_real>(y_temp_h[i]);
                }
            }
            return result;
        }
        ~SparseMatrixAndImageGPU()
        {
            if (y_d)
                CHECK_CUDA(cudaFree(y_d));
        }
        T *y_d = nullptr;
    };

    template <typename T>
    struct SparseMatrix
    {
        SparseMatrix() = default;
        SparseMatrix(int n_rows, int nnz) : n_rows(n_rows), nnz(nnz)
        {
            row_ptr_h = static_cast<int *>(malloc((n_rows + 1) * sizeof(int)));
            col_idx_h = static_cast<int *>(malloc(nnz * sizeof(int)));
            val_h = static_cast<T *>(malloc(nnz * sizeof(T)));
        }
        SparseMatrix(const SparseMatrix &other) = delete;
        SparseMatrix(SparseMatrix &&other) : n_rows(other.n_rows), nnz(other.nnz),
                                             row_ptr_h(other.row_ptr_h), col_idx_h(other.col_idx_h),
                                             val_h(other.val_h)
        {
            other.row_ptr_h = nullptr;
            other.col_idx_h = nullptr;
            other.val_h = nullptr;
        }
        // should only be called when default constructed
        SparseMatrix &operator=(SparseMatrix &&other)
        {
            if (this != &other)
            {
                std::swap(n_rows, other.n_rows);
                std::swap(nnz, other.nnz);
                std::swap(row_ptr_h, other.row_ptr_h);
                std::swap(col_idx_h, other.col_idx_h);
                std::swap(val_h, other.val_h);
            }
            return *this;
        }
        void release()
        {
            row_ptr_h = nullptr;
            col_idx_h = nullptr;
            val_h = nullptr;
        }
        ~SparseMatrix()
        {
            free(row_ptr_h);
            free(col_idx_h);
            free(val_h);
        }
        template <typename cuda_real>
        SparseMatrixGPU<cuda_real> to_cuda()
        {
            SparseMatrixGPU<cuda_real> result(n_rows, nnz);
            CHECK_CUDA(cudaMemcpy(result.row_ptr_d, row_ptr_h, (n_rows + 1) * sizeof(int), cudaMemcpyHostToDevice));
            CHECK_CUDA(cudaMemcpy(result.col_idx_d, col_idx_h, nnz * sizeof(int), cudaMemcpyHostToDevice));

            if constexpr (std::is_same_v<T, cuda_real>)
            {
                CHECK_CUDA(cudaMemcpy(result.val_d, val_h, nnz * sizeof(T), cudaMemcpyHostToDevice));
            }
            else
            {
                std::vector<cuda_real> val_temp_h;
                val_temp_h.reserve(nnz);
                for (int i = 0; i < nnz; ++i)
                {
                    val_temp_h[i] = static_cast<cuda_real>(val_h[i]);
                }
                CHECK_CUDA(cudaMemcpy(result.val_d, val_temp_h.data(), nnz * sizeof(cuda_real), cudaMemcpyHostToDevice));
            }
            return result;
        }

        int *row_ptr_h = nullptr;
        int *col_idx_h = nullptr;
        T *val_h = nullptr;
        int n_rows = 0;
        int nnz = 0;
    };

    template <typename T>
    struct SparseMatrixAndImage : public SparseMatrix<T>
    {
        SparseMatrixAndImage() = default;
        SparseMatrixAndImage(int n_rows, int nnz) : SparseMatrix<T>(n_rows, nnz)
        {
            y_h = static_cast<T *>(malloc(n_rows * sizeof(T)));
        }
        SparseMatrixAndImage(SparseMatrix<T> &&other) : SparseMatrix<T>(std::move(other))
        {
            y_h = static_cast<T *>(malloc(other.n_rows * sizeof(T)));
        }
        SparseMatrixAndImage(SparseMatrixAndImage &&other) : SparseMatrix<T>(std::move(other)), y_h(other.y_h)
        {
            other.y_h = nullptr;
        }
        SparseMatrixAndImage &operator=(SparseMatrixAndImage &&other)
        {
            this->SparseMatrix<T>::template operator=(std::move(other));
            std::swap(y_h, other.y_h);
            return *this;
        }
        SparseMatrixAndImage(const SparseMatrixAndImage &other) = delete;
        void release()
        {
            SparseMatrix<T>::release();
            y_h = nullptr;
        }
        ~SparseMatrixAndImage()
        {
            free(y_h);
        }
        template <typename cuda_real>
        SparseMatrixAndImageGPU<cuda_real> to_cuda()
        {
            SparseMatrixAndImageGPU<cuda_real> result(SparseMatrix<T>::template to_cuda<cuda_real>());

            if constexpr (std::is_same_v<T, cuda_real>)
            {
                CHECK_CUDA(cudaMemcpy(result.y_d, y_h, SparseMatrix<T>::n_rows * sizeof(T), cudaMemcpyHostToDevice));
            }
            else
            {
                std::vector<cuda_real> y_temp_h;
                y_temp_h.reserve(SparseMatrix<T>::n_rows);
                for (int i = 0; i < SparseMatrix<T>::n_rows; ++i)
                {
                    y_temp_h[i] = static_cast<cuda_real>(y_h[i]);
                }
                CHECK_CUDA(cudaMemcpy(result.y_d, y_temp_h.data(), SparseMatrix<T>::n_rows * sizeof(cuda_real), cudaMemcpyHostToDevice));
            }
            return result;
        }
        T *y_h = nullptr;
        // only used when delete_column is true
        // std::vector<int> bin_mask_h;
        // int delete_count;
    };

    template <typename T>
    struct SparseMatrixRaw
    {
        void load_raw_matrix(std::fstream &val_file, std::fstream &col_idx_file, std::fstream &row_ptr_file)
        {
            val.assign(std::istreambuf_iterator<char>(val_file), std::istreambuf_iterator<char>());
            col_idx.assign(std::istreambuf_iterator<char>(col_idx_file), std::istreambuf_iterator<char>());
            row_ptr.assign(std::istreambuf_iterator<char>(row_ptr_file), std::istreambuf_iterator<char>());
            std::cerr << "row size " << get_n_rows() << " nnz " << get_nnz() << '\n';
        }
        void load_regularization_matrix(std::string_view D_dir, std::string_view reg_type = "hlaplac")
        {
            std::string prefix = std::string(reg_type) + "_" + std::to_string(N_RAD_BINS) + "_" + std::to_string(N_THETA_BINS) + "_" + std::to_string(N_PHI_BINS);
            std::fstream reg_val_file;
            if (reg_val_file.open(std::string(D_dir) + "w" + prefix); !reg_val_file.is_open())
            {
                std::cerr << "Failed to open the value file\n";
                std::exit(-1);
            }
            std::fstream reg_col_idx_file;
            if (reg_col_idx_file.open(std::string(D_dir) + "j" + prefix); !reg_col_idx_file.is_open())
            {
                std::cerr << "Failed to open the column index file\n";
                std::exit(-1);
            }
            std::fstream reg_row_ptr_file;
            if (reg_row_ptr_file.open(std::string(D_dir) + "m" + prefix); !reg_row_ptr_file.is_open())
            {
                std::cerr << "Failed to open the row index file\n";
                std::exit(-1);
            }
            load_raw_matrix(reg_val_file, reg_col_idx_file, reg_row_ptr_file);
        }

        int get_n_rows() const
        {
            return row_ptr.size() / sizeof(int) - 1;
        }

        int get_nnz() const
        {
            return val.size() / sizeof(T);
        }

        template <typename cuda_real>
        SparseMatrixGPU<cuda_real> to_cuda()
        {
            SparseMatrixGPU<cuda_real> mat_d(get_n_rows(), get_nnz());
            CHECK_CUDA(cudaMemcpy(mat_d.col_idx_d, col_idx.data(), col_idx.size(), cudaMemcpyHostToDevice));
            CHECK_CUDA(cudaMemcpy(mat_d.row_ptr_d, row_ptr.data(), row_ptr.size(), cudaMemcpyHostToDevice));
            if constexpr (std::is_same_v<T, cuda_real>)
            {
                CHECK_CUDA(cudaMemcpy(mat_d.val_d, val.data(), val.size(), cudaMemcpyHostToDevice));
            }
            else
            {
                std::vector<cuda_real> val_temp_h;
                val_temp_h.resize(val.size() / sizeof(T));
                std::cerr << "val temp size " << val_temp_h.size() << '\n';
                for (int i = 0; i < val_temp_h.size(); ++i)
                {
                    val_temp_h[i] = reinterpret_cast<const T *>(val.data())[i];
                }
                CHECK_CUDA(cudaMemcpy(mat_d.val_d, val_temp_h.data(), val_temp_h.size() * sizeof(cuda_real), cudaMemcpyHostToDevice));
            }
            return mat_d;
        }

        std::vector<char> val;
        std::vector<char> col_idx;
        std::vector<char> row_ptr;
    };

    template <typename T>
    struct SparseMatrixAndIamgeRaw : public SparseMatrixRaw<T>
    {
        void load_raw_y(std::fstream &y_file)
        {
            y.assign(std::istreambuf_iterator<char>(y_file), std::istreambuf_iterator<char>());
        }

        template <typename cuda_real>
        SparseMatrixAndImageGPU<cuda_real> to_cuda()
        {
            SparseMatrixAndImageGPU<cuda_real> result(SparseMatrixRaw<T>::template to_cuda<cuda_real>());
            CHECK_CUDA(cudaMemcpy(result.y_d, y.data(), y.size(), cudaMemcpyHostToDevice));
            return result;
        }
        std::vector<char> y;
    };
}
