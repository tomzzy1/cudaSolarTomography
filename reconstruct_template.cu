#include <cmath>
#include <cstdio>
#include <iostream>
#include <fstream>
#include <string>
#include <string_view>
#include <iterator>
#include <vector>
#include <array>
#include <algorithm>
#include <numeric>
#include <chrono>
#include <fitsfile.h>
#include <cusparse.h>
#include "constants.hpp"
#include "output.hpp"
#include "build_A_matrix.cuh"
#include "optimizer.hpp"
#include "gpu_matrix.hpp"
#include <stdint.h>

// #define NDEBUG
#include <cassert>

constexpr std::string_view output_dir = "../output/gpu/";
//  location for regularization matrix
constexpr std::string_view D_dir = "../python/";
// constexpr std::string_view D_dir = "../tests/gold_prelim/";
/*
naming: _d means data on device, _h means data on host
For the detailed explanation of the computation, check the CPU version
*/

int main(int argc, char **argv)
{
    using namespace cudaSolarTomography;

    Timer timer;

    // float lambda_tik = 4e-5;
    float lambda_tik = 4e-5 * 2 / 3; // default lambda for new D matrix

    if (argc == 2)
    {
        lambda_tik = atof(argv[1]);
    }

    // fill in your own filenames for these files
    // assume to be plain binary format
    std::fstream val_file;
    if (val_file.open("../../SolarTom2/bindata/wSolarTom"); !val_file.is_open())
    {
        std::cerr << "Failed to open the value file\n";
        std::exit(-1);
    }
    std::fstream col_idx_file;
    if (col_idx_file.open("../../SolarTom2/bindata/jSolarTom"); !col_idx_file.is_open())
    {
        std::cerr << "Failed to open the column index file\n";
        std::exit(-1);
    }
    std::fstream row_ptr_file;
    if (row_ptr_file.open("../../SolarTom2/bindata/mSolarTom"); !row_ptr_file.is_open())
    {
        std::cerr << "Failed to open the row index file\n";
        std::exit(-1);
    }
    std::fstream y_file;
    if (y_file.open("../../SolarTom2/bindata/ySolarTom"); !y_file.is_open())
    {
        std::cerr << "Failed to open the value file\n";
        std::exit(-1);
    }
    // can be both float or double, just replace the template argument
    SparseMatrixAndIamgeRaw<float> A_y;
    A_y.load_raw_matrix(val_file, col_idx_file, row_ptr_file);
    A_y.load_raw_y(y_file);

    // SparseMatrixAndImage A_y = build_A_matrix(sub_matrix_filenames);

    std::cerr << "A and y loaded " << cudaGetErrorName(cudaGetLastError()) << '\n';
    std::cerr << "row size " << A_y.get_n_rows() << " nnz " << A_y.get_nnz() << '\n';

    timer.start();
    timer.start();

    SparseMatrixAndImageGPU<real> A_y_d = A_y.to_cuda<real>();

    std::cerr << "Create A Cuda Matrix " << cudaGetErrorName(cudaGetLastError()) << '\n';

    cudaDeviceSynchronize();

    // reconstruction begin
    cudaSolarTomography::cusparseContext context;

    std::cerr << "Cusparse Create " << cudaGetErrorName(cudaGetLastError()) << '\n';

    cusparseDnVecDescr_t y_descr = A_y_d.createDnVecDescr();

    std::cerr << "Before A matrix built " << cudaGetErrorName(cudaGetLastError()) << '\n';

    std::cerr << A_y_d.n_rows << ' ' << A_y_d.nnz << '\n';
    cusparseSpMatDescr_t A_descr = A_y_d.createSpMatDescr();

    std::cerr << "A matrix built " << cudaGetErrorName(cudaGetLastError()) << '\n';

    // should modify to real regularization matrix
    // load the current regularizaiton matrix

    SparseMatrixRaw<float> reg_matrix;

    // The "load_regularization_matrix" method assumes the names of the CSR matrix files follow some rules
    reg_matrix.load_regularization_matrix(D_dir);

    // or you can load this files similar to loading A, I leave it in the comments
    // std::fstream reg_val_file;
    // if (reg_val_file.open("../../SolarTom2/bindata/wSolarTom"); !reg_val_file.is_open())
    // {
    //     std::cerr << "Failed to open the value file\n";
    //     std::exit(-1);
    // }
    // std::fstream reg_col_idx_file;
    // if (reg_col_idx_file.open("../../SolarTom2/bindata/jSolarTom"); !reg_col_idx_file.is_open())
    // {
    //     std::cerr << "Failed to open the column index file\n";
    //     std::exit(-1);
    // }
    // std::fstream reg_row_ptr_file;
    // if (reg_row_ptr_file.open("../../SolarTom2/bindata/mSolarTom"); !reg_row_ptr_file.is_open())
    // {
    //     std::cerr << "Failed to open the row index file\n";
    //     std::exit(-1);
    // }
    // reg_matrix.load_raw_matrix(reg_val_file, reg_col_idx_file, reg_row_ptr_file);

    SparseMatrixGPU<real> reg_matrix_d = reg_matrix.to_cuda<real>();

    cusparseSpMatDescr_t D_descr = reg_matrix_d.createSpMatDescr();

    std::cerr << "D matrix built " << cudaGetErrorName(cudaGetLastError()) << '\n';

    real *x_result_d = nullptr;
    std::vector<real> x_result_h(N_BINS, 1e4);

    cudaMalloc(&x_result_d, N_BINS * sizeof(real));
    cudaMemset(x_result_d, 0, N_BINS * sizeof(real));

    // cudaMemcpy(x_result_d, x_result_h.data(), N_BINS * sizeof(float), cudaMemcpyHostToDevice);

    cusparseDnVecDescr_t x_descr;
    cusparseCreateDnVec(&x_descr, N_BINS, x_result_d, CUDA_REAL);

    std::cerr << "initialization end " << cudaGetErrorName(cudaDeviceSynchronize()) << '\n';

    timer.stop("Prepare Reconstruction");

    optimize(&context, A_descr, x_descr, y_descr, D_descr, A_y_d.y_d, x_result_d, A_y_d.n_rows, reg_matrix.get_n_rows(), lambda_tik);

    timer.stop("Reconstruction");
    timer.start();

    cudaMemcpy(x_result_h.data(), x_result_d, N_BINS * sizeof(real), cudaMemcpyDeviceToHost);

    for (int i = 0; i < 100; ++i)
    {
        std::cerr << x_result_h[i] << ' ';
    }
    std::cerr << '\n';

    cusparseDestroySpMat(A_descr);
    cusparseDestroySpMat(D_descr);
    cusparseDestroyDnVec(x_descr);
    cusparseDestroyDnVec(y_descr);

    cudaFree(x_result_d);

    // reconstruction end

    std::vector<real> x_result_full(N_BINS, 0);

    x_result_full = x_result_h;

    std::vector<float> x_result_float(N_BINS, 0);
    if constexpr (!std::is_same_v<real, float>)
    {
        for (int i = 0; i < N_BINS; ++i)
        {
            x_result_float[i] = static_cast<float>(x_result_full[i]);
        }
        write_vector(std::string(output_dir) + "x_result", "reconstructed x vector", x_result_float);
    }
    else
        write_vector(std::string(output_dir) + "x_result", "reconstructed x vector", x_result_full);

    timer.stop("Output x");
}
