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

constexpr bool store_matrix = true;

constexpr bool eliminate_background = false;


/*
naming: _d means data on device, _h means data on host
For the detailed explanation of the computation, check the CPU version
*/

int main(int argc, char **argv)
{
    using namespace cudaSolarTomography;

    std::cerr << "binning factor " << BINNING_FACTOR << " bin scale factor " << BIN_SCALE_FACTOR << '\n';
    std::cerr << "number of bins " << N_BINS << '\n';

    Timer timer;
    timer.start();

    std::fstream config_file;

    if (config_file.open(std::string(config_dir)); !config_file.is_open())
    {
        std::cerr << "Failed to open the config file " << config_dir << "\n";
        return -1;
    }

    std::string file_count_str;
    std::getline(config_file, file_count_str);
    int file_count = std::stoi(file_count_str);
    std::cerr << "file number " << file_count << '\n';

    std::vector<std::string> sub_matrix_filenames;
    for (int i = 0; i < file_count; ++i)
    {
        std::string sub_matrix_filename;
        std::getline(config_file, sub_matrix_filename);
        sub_matrix_filenames.push_back(sub_matrix_filename);
    }
    config_file.close();

    int file_begin = -1;
    int file_end = -1;

    // float lambda_tik = 4e-5;
    float lambda_tik = 4e-5 * 2 / 3; // default lambda for new D matrix

    if (argc == 3)
    {
        file_begin = atoi(argv[1]);
        file_end = atoi(argv[2]);
        file_count = file_end - file_begin;
    }
    else
    {
        file_begin = 0;
        file_end = file_count;
        if (argc == 2)
        {
            lambda_tik = atof(argv[1]);
        }
    }

    timer.stop("time for reading configuration");

    // unit solar pole (1, delta_pole, alpha_pole) (in spherical coordinate)
    SparseMatrixAndImage A_y = build_A_matrix(sub_matrix_filenames);

    // const int col_size = delete_column ? N_BINS - A_y.delete_count : N_BINS;
    const int col_size = N_BINS;

    std::cerr << "build A finish " << cudaGetErrorName(cudaGetLastError()) << '\n';

    timer.start();
    timer.start();

    if constexpr (store_matrix)
    {
        write_vector(std::string(output_dir) + "_row_index", "row_index", A_y.row_ptr_h, (A_y.n_rows + 1));
        write_vector(std::string(output_dir) + "y_data", "y_data", A_y.y_h, A_y.n_rows);
        write_vector(std::string(output_dir) + "_col_index", "col_index", A_y.col_idx_h, A_y.nnz);
        write_vector(std::string(output_dir) + "_val", "val", A_y.val_h, A_y.nnz);
    }

    // return 0;

    /* use gold prelim for test of reconstruction*/
    // std::fstream prelim_val_file;
    // if (prelim_val_file.open("../../SolarTom2/bindata/wSolarTom"); !prelim_val_file.is_open())
    // {
    //     std::cerr << "Failed to open the value file\n";
    //     std::exit(-1);
    // }
    // std::fstream prelim_col_idx_file;
    // if (prelim_col_idx_file.open("../../SolarTom2/bindata/jSolarTom"); !prelim_col_idx_file.is_open())
    // {
    //     std::cerr << "Failed to open the column index file\n";
    //     std::exit(-1);
    // }
    // std::fstream prelim_row_ptr_file;
    // if (prelim_row_ptr_file.open("../../SolarTom2/bindata/mSolarTom"); !prelim_row_ptr_file.is_open())
    // {
    //     std::cerr << "Failed to open the row index file\n";
    //     std::exit(-1);
    // }
    // std::fstream prelim_y_file;
    // if (prelim_y_file.open("../../SolarTom2/bindata/ySolarTom"); !prelim_y_file.is_open())
    // {
    //     std::cerr << "Failed to open the value file\n";
    //     std::exit(-1);
    // }

    // SparseMatrixAndIamgeRaw<float> prelim_h;
    // prelim_h.load_raw_matrix(prelim_val_file, prelim_col_idx_file, prelim_row_ptr_file);
    // prelim_h.load_raw_y(prelim_y_file);
    // SparseMatrixAndImageGPU<real> prelim_d = prelim_h.to_cuda<real>();

    auto A_y_d = A_y.to_cuda<real>();

    std::cerr << "Create A Cuda Matrix " << cudaGetErrorName(cudaGetLastError()) << '\n';

    cudaDeviceSynchronize();

    // reconstruction begin
    cudaSolarTomography::cusparseContext context;

    std::cerr << "Cusparse Create " << cudaGetErrorName(cudaGetLastError()) << '\n';

    cusparseDnVecDescr_t y_descr = A_y_d.createDnVecDescr();

    std::cerr << "Before A matrix built " << cudaGetErrorName(cudaGetLastError()) << '\n';

    std::cerr << A_y.n_rows << ' ' << A_y.nnz << '\n';
    cusparseSpMatDescr_t A_descr = A_y_d.createSpMatDescr();

    std::cerr << "A matrix built " << cudaGetErrorName(cudaGetLastError()) << '\n';

    // should modify to real regularization matrix
    // load the current regularizaiton matrix
    SparseMatrixRaw<float> reg_matrix;
    // reg_matrix.load_regularization_matrix(D_dir, "r3");
    // std::cerr << "read reg matrix r3\n";
    reg_matrix.load_regularization_matrix(D_dir);

    SparseMatrixGPU<real> reg_matrix_d = reg_matrix.to_cuda<real>();

    cusparseSpMatDescr_t D_descr = reg_matrix_d.createSpMatDescr();

    std::cerr << "D matrix built " << cudaGetErrorName(cudaGetLastError()) << '\n';

    real *x_result_d = nullptr;
    std::vector<real> x_result_h(col_size, 1e4);

    cudaMalloc(&x_result_d, col_size * sizeof(real));
    cudaMemset(x_result_d, 0, col_size * sizeof(real));

    // cudaMemcpy(x_result_d, x_result_h.data(), N_BINS * sizeof(float), cudaMemcpyHostToDevice);

    cusparseDnVecDescr_t x_descr;
    cusparseCreateDnVec(&x_descr, col_size, x_result_d, CUDA_REAL);

    std::cerr << "initialization end " << cudaGetErrorName(cudaDeviceSynchronize()) << '\n';

    std::vector<real> background_h(N_BINS, 0);
    if constexpr (eliminate_background)
    {
        for (int i = 0; i < N_RAD_BINS; ++i)
        {
            double r = (R_MIN + RAD_BIN_SIZE * (i + 1));
            // double N_e = (1.545 / pow(r, 16) + 0.079 / pow(r, 6)) * 1e8;
            double N_e = 1e6 - i * 2.5e4;
            std::cerr << "r " << r << " N_e " << N_e << '\n';
            for (int j = i; j < N_BINS; j += N_RAD_BINS)
            {
                background_h[j] = static_cast<real>(N_e);
            }
        }

        real *background_d = nullptr;
        cudaMalloc(&background_d, N_BINS * sizeof(real));
        cudaMemcpy(background_d, background_h.data(), N_BINS * sizeof(real), cudaMemcpyHostToDevice);

        cusparseDnVecDescr_t background_descr;
        cusparseCreateDnVec(&background_descr, N_BINS, background_d, CUDA_REAL);

        real *Ab_d = nullptr; // A .* background
        cudaMalloc(&Ab_d, A_y_d.n_rows * sizeof(real));
        cusparseDnVecDescr_t Ab_descr;
        cusparseCreateDnVec(&Ab_descr, A_y_d.n_rows, Ab_d, CUDA_REAL);

        context.SpMV(CUSPARSE_OPERATION_NON_TRANSPOSE, A_descr, background_descr, Ab_descr, -1, 1, 0);
        cudaCheckError();
        std::cerr << "background eliminated\n";
    }

    timer.stop("Prepare Reconstruction");

    optimize(&context, A_descr, x_descr, y_descr, D_descr, A_y_d.y_d, x_result_d, A_y_d.n_rows, reg_matrix.get_n_rows(), lambda_tik);

    timer.stop("Reconstruction");
    timer.start();

    cudaMemcpy(x_result_h.data(), x_result_d, col_size * sizeof(real), cudaMemcpyDeviceToHost);

    if constexpr (eliminate_background)
    {
        for (int i = 0; i < N_BINS; ++i)
        {
            x_result_h[i] += background_h[i];
        }
    }

    for (int i = 0; i < 100; ++i)
    {
        std::cerr << x_result_h[i] << ' ';
    }
    std::cerr << '\n';

    int dark_area = 0;
    for (int i = 0; i < N_BINS; ++i)
    {
        int r_idx = i % N_RAD_BINS;
        if (r_idx == 0 && r_idx == N_RAD_BINS - 1)
            continue;
        if (x_result_h[i] < 0)
            ++dark_area;
    }
    std::cerr << "dark area " << 100.0 * dark_area / (N_BINS - 2 * N_THETA_BINS * N_PHI_BINS) << "%\n";

    // compare the result of two regularization matrix
    // constexpr std::string_view new_D_dir = "../python/";
    // SparseMatrixRaw new_reg_matrix;
    // new_reg_matrix.load_regularization_matrix(new_D_dir);
    // SparseMatrixGPU<real> new_reg_matrix_d = new_reg_matrix.to_cuda<real>();
    // cusparseSpMatDescr_t new_D_descr = new_reg_matrix_d.createSpMatDescr();

    // std::vector<float> new_x_result_h(N_BINS, 0);
    // float *new_x_result_d = nullptr;
    // cudaMalloc(&new_x_result_d, N_BINS * sizeof(float));

    // cusparseDnVecDescr_t new_x_descr;
    // cusparseCreateDnVec(&new_x_descr, N_BINS, new_x_result_d, CUDA_R_32F);

    // cusparseCreateCsr(&A_descr, n_rows, N_BINS, nnz, row_ptr_all_d, col_idx_all_d, val_all_d, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
    // cusparseCreateDnVec(&y_descr, n_rows, y_all_d, CUDA_R_32F);

    // std::cerr << "lambda find initialization " << cudaGetErrorName(cudaDeviceSynchronize()) << "\n";

    // by testing, use lambda = 2.6-5 (apporx. 4 * (2 / 3)) for new D matrix
    // the D matrix is 3 / 2 times larger than the original one
    // notice that the factor is (2 / 3), not (2 / 3)^2
    // for (int i = 0; i <= 20; ++i)
    // {
    //     float new_lambda_tik = 4e-5 - i * 2e-6;
    //     cudaMemset(new_x_result_d, 0, N_BINS * sizeof(float));
    //     optimize(handle, A_descr, new_x_descr, y_descr, new_D_descr, y_all_d, new_x_result_d, n_rows, new_reg_matrix.get_n_rows(), new_lambda_tik);
    //     cudaMemcpy(new_x_result_h.data(), new_x_result_d, N_BINS * sizeof(float), cudaMemcpyDeviceToHost);
    //     float x_diff_norm = 0;
    //     for (int j = 0; j < N_BINS; ++j)
    //     {
    //         x_diff_norm += pow(new_x_result_h[j] - x_result_h[j], 2);
    //     }
    //     std::cout << "iter " << i << " lambda_tik " << new_lambda_tik << " norm of difference " << sqrt(x_diff_norm) << '\n';
    // }
    // float new_lambda_tik = 2.67e-5;
    // cudaMemset(new_x_result_d, 0, N_BINS * sizeof(float));
    // optimize(handle, A_descr, new_x_descr, y_descr, new_D_descr, y_all_d, new_x_result_d, n_rows, new_reg_matrix.get_n_rows(), new_lambda_tik);
    // cudaMemcpy(new_x_result_h.data(), new_x_result_d, N_BINS * sizeof(float), cudaMemcpyDeviceToHost);
    // float x_diff_norm = 0;
    // for (int j = 0; j < N_BINS; ++j)
    // {
    //     x_diff_norm += pow(new_x_result_h[j] - x_result_h[j], 2);
    // }
    // std::cout << "norm of difference of 2.67e-5 " << sqrt(x_diff_norm) << '\n';

    // cusparseDestroySpMat(new_D_descr);
    // cudaFree(new_x_result_d);

    // load result for bin_scale = 1
    // std::fstream small_x_file;
    // if (small_x_file.open(std::string(output_dir) + "x_result_small"); !small_x_file.is_open())
    // {
    //     std::cerr << "Failed to open the small x file\n";
    //     std::exit(-1);
    // }
    // std::vector<char> x_small{std::istreambuf_iterator<char>(small_x_file), std::istreambuf_iterator<char>()};
    // std::cerr << "x_small size " << x_small.size() << '\n';

    // auto norm = [](auto v, int size)
    // {
    //     double res = 0;
    //     for (int i = 0; i < size; ++i)
    //         res += v[i] * v[i];
    //     return sqrt(res);
    // };
    // double x_small_norm = norm(reinterpret_cast<float*>(x_small.data()), x_small.size() / sizeof(float));

    // // Find lambda for different bin scale
    // std::fstream lambda_file;
    // if (lambda_file.open("lambda.txt", std::fstream::out); !lambda_file.is_open())
    // {
    //     std::cerr << "Failed to open the lambda file\n";
    //     std::exit(-1);
    // }
    // for (float new_lambda_tik = 3e-5; new_lambda_tik >= 1e-6; new_lambda_tik -= 1e-6)
    // {
    //     cudaMemset(new_x_result_d, 0, N_BINS * sizeof(float));
    //     auto res = optimize(handle, A_descr, new_x_descr, y_descr, D_descr, y_all_d, new_x_result_d, n_rows, reg_matrix.get_n_rows(), new_lambda_tik);
    //     cudaMemcpy(new_x_result_h.data(), new_x_result_d, N_BINS * sizeof(float), cudaMemcpyDeviceToHost);
    //     double x_diff_norm = 0;
    //     double x_norm = 0;
    //     for (int j = 0; j < N_BINS; ++j)
    //     {
    //         // x_diff_norm += pow(new_x_result_h[j] - x_result_h[j], 2);
    //         x_norm += pow(new_x_result_h[j], 2);
    //     }
    //     x_norm = sqrt(x_norm);
    //     std::vector<float> new_x_result_zip(20 * 30 * 60, 0);
    //     for (int i = 0; i < 20; ++i)
    //     {
    //         for (int j = 0; j < 30; ++j)
    //         {
    //             for (int k = 0; k < 60; ++k)
    //             {
    //                 int old_idx = k * 30 * 20 + j * 20 + i;
    //                 double old_x = reinterpret_cast<float*>(x_small.data())[old_idx];
    //                 double new_x = 0;
    //                 for (int ii = 2 * i; ii < 2 * i + 2; ++ii)
    //                 {
    //                     for (int jj = 2 * j; jj < 2 * j + 2; ++jj)
    //                     {
    //                         for (int kk = 2 * k; kk < 2 * k + 2; ++kk)
    //                         {
    //                             int new_idx = kk * 60 * 40 + jj * 40 + ii;
    //                             new_x += new_x_result_h[new_idx];
    //                         }
    //                     }
    //                 }
    //                 new_x /= 8;
    //                 new_x_result_zip[old_idx] = new_x;
    //                 x_diff_norm += pow(new_x - old_x, 2);
    //             }
    //         }
    //     }
    //     x_diff_norm = sqrt(x_diff_norm);
    //     double cosine_similarity = 0;
    //     for (int i = 0; i < 20 * 30 * 60; ++i)
    //     {
    //         cosine_similarity += reinterpret_cast<float*>(x_small.data())[i] * new_x_result_zip[i];
    //     }
    //     double x_zip_norm = norm(new_x_result_zip, new_x_result_zip.size());
    //     cosine_similarity /= x_zip_norm * x_small_norm;
    //     std::cout << "x_zip_norm " << x_zip_norm << " x_small_norm " << x_small_norm << '\n';
    //     std::cout << " lambda_tik " << new_lambda_tik << " norm of difference " << x_diff_norm << " x norm " << x_norm << " cosine similarity " << cosine_similarity << '\n';
    //     std::cout << res << '\n';
    //     lambda_file << " lambda_tik " << new_lambda_tik << " norm of difference " << x_diff_norm << " x norm " << x_norm << " cosine similarity " << cosine_similarity <<'\n';
    //     lambda_file << res << '\n';
    // }
    // lambda_file.close();

    cusparseDestroySpMat(A_descr);
    cusparseDestroySpMat(D_descr);
    cusparseDestroyDnVec(x_descr);
    cusparseDestroyDnVec(y_descr);

    cudaFree(x_result_d);

    // reconstruction end

    std::vector<real> x_result_full(N_BINS, 0);
    // if constexpr (delete_column)
    // {
    //     int j = 0;
    //     for (int i = 0; i < N_BINS; ++i)
    //     {
    //         if (A_y.bin_mask_h[i] != -1)
    //         {
    //             x_result_full[i] = x_result_h[j++];
    //         }
    //         else
    //         {
    //             x_result_full[i] = 0;
    //         }
    //     }
    // }
    // else
    {
        x_result_full = x_result_h;
    }

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
