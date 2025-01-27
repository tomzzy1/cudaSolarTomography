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
#include <thread>
#include <random>
#include <unordered_map>
#include <map>
#include <cusparse.h>
#include "output.hpp"
#include "build_A_matrix.cuh"
#include "optimizer.hpp"
#include <stdint.h>

// #define NDEBUG
#include <cassert>

constexpr std::string_view D_dir = "../python/";

// constexpr std::string_view projection_dir = "../data/projection/lasco_2008/";
// constexpr std::string_view projection_output_dir = "../output/projection/lasco_2008/";
constexpr std::string_view projection_dir = "../data/projection/lasco_2023/";
constexpr std::string_view projection_output_dir = "../output/projection/lasco_2023/";
// constexpr std::string_view projection_dir = "../data/projection/lasco_2023/downsample2/";
// constexpr std::string_view projection_output_dir = "../output/projection/lasco_2023/downsample2/";
// constexpr std::string_view projection_dir = "../data/projection/lasco_2023/30_143_300/";
// constexpr std::string_view projection_output_dir = "../output/projection/lasco_2023/30_143_300/";
// constexpr std::string_view projection_dir = "../data/projection/lasco_2023/76_72_150/";
// constexpr std::string_view projection_output_dir = "../output/projection/lasco_2023/76_72_150/";


int main(int argc, char **argv)
{
    using namespace cudaSolarTomography;
    std::cerr << "binning factor " << BINNING_FACTOR << " bin scale factor " << BIN_SCALE_FACTOR << '\n';

    Timer timer;

    // float lambda_tik = 4e-5;
    float lambda_tik = 4e-5 * 2 / 3; // default lambda for new D matrix

    if (argc == 2)
    {
        lambda_tik = atof(argv[1]);
    }

    std::random_device rd;
    std::mt19937 gen(rd());

    std::fstream projection_config_file;

    if (projection_config_file.open(std::string(projection_dir) + "projection.conf"); !projection_config_file.is_open())
    {
        std::cerr << "Failed to open the projection config file "
                  << "\n";
        return -1;
    }

    std::string file_count_str;
    std::getline(projection_config_file, file_count_str);
    int file_count = std::stoi(file_count_str);
    std::cerr << "file number " << file_count << '\n';

    std::vector<std::string> sub_matrix_filenames;
    for (int fc = 0; fc < file_count; ++fc)
    {
        std::string sub_matrix_filename;
        std::getline(projection_config_file, sub_matrix_filename);
        sub_matrix_filenames.push_back(sub_matrix_filename);
    }

    cudaSolarTomography::cusparseContext context;

    cublasHandle_t cublas_handle;
    cublasCreate(&cublas_handle);

    timer.start();

    SparseMatrixAndImage A_y = build_A_matrix_with_projection(sub_matrix_filenames, projection_dir);

    auto A_y_d = A_y.to_cuda<real>();

    cusparseDnVecDescr_t y_descr = A_y_d.createDnVecDescr();

    cusparseSpMatDescr_t A_descr = A_y_d.createSpMatDescr();

    // load the current regularizaiton matrix
    SparseMatrixRaw<float> reg_matrix;
    reg_matrix.load_regularization_matrix(D_dir);
    SparseMatrixGPU<real> reg_matrix_d = reg_matrix.to_cuda<real>();

    cusparseSpMatDescr_t D_descr = reg_matrix_d.createSpMatDescr();

    std::vector<real> x_result_h(N_BINS, 1e4);
    real *x_result_d = nullptr;
    CHECK_CUDA(cudaMalloc(&x_result_d, N_BINS * sizeof(real)));

    // cudaMemcpy(x_result_d, x_result_h.data(), N_BINS * sizeof(float), cudaMemcpyHostToDevice);

    cusparseDnVecDescr_t x_descr;
    CHECK_CUSPARSE(cusparseCreateDnVec(&x_descr, N_BINS, x_result_d, CUDA_REAL));

    cudaCheckError();
    std::cerr << "initialization end\n";

    timer.stop("Prepare Projection Reconstruction");

    // reconstruction
    CHECK_CUDA(cudaMemset(x_result_d, 0, N_BINS * sizeof(real)));
    optimize(&context, A_descr, x_descr, y_descr, D_descr,
             A_y_d.y_d, x_result_d,
             A_y_d.n_rows, reg_matrix.get_n_rows(), lambda_tik);
    std::cerr << "reconstruction end " << '\n';
    cudaCheckError();
    CHECK_CUDA(cudaMemcpy(x_result_h.data(), x_result_d, N_BINS * sizeof(real), cudaMemcpyDeviceToHost));
    write_vector(std::string(projection_output_dir) + "x_result", "reconstructed x vector", x_result_h);

    cusparseDestroySpMat(A_descr);
    cusparseDestroyDnVec(x_descr);
    cusparseDestroyDnVec(y_descr);

    cusparseDestroySpMat(D_descr);

    cudaFree(x_result_d);

    cublasDestroy(cublas_handle);
}
