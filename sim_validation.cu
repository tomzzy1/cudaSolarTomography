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
constexpr std::string_view sim_dir = "../data/mhd_2008/";

// constexpr int epoch = mode == 0 ? 1 : 6;
constexpr int epoch = 1;

int main(int argc, char **argv)
{
    using namespace cudaSolarTomography;
    std::cerr << "binning factor " << BINNING_FACTOR << " bin scale factor " << BIN_SCALE_FACTOR << '\n';

    // float lambda_tik = 4e-5;
    float lambda_tik = 4e-5 * 2 / 3; // default lambda for new D matrix

    if (argc == 2)
    {
        lambda_tik = atof(argv[1]);
    }

    std::random_device rd;
    std::mt19937 gen(rd());

    std::fstream config_file;

    if (config_file.open(std::string(config_dir)); !config_file.is_open())
    {
        std::cerr << "Failed to open the config file " << config_dir << "\n";
        return -1;
    }

    std::string file_count_str;
    std::getline(config_file, file_count_str);
    int n_files = std::stoi(file_count_str);
    std::cerr << "file number " << n_files << '\n';

    std::vector<std::string> sub_matrix_filenames;
    for (int fc = 0; fc < n_files; ++fc)
    {
        std::string sub_matrix_filename;
        std::getline(config_file, sub_matrix_filename);
        sub_matrix_filenames.push_back(sub_matrix_filename);
    }

    Timer timer;

    cublasHandle_t cublas_handle;
    cublasCreate(&cublas_handle);

    cudaSolarTomography::cusparseContext context;

    for (int e = 0; e < epoch; ++e)
    {
        timer.start();

        SparseMatrixAndImage A_y = build_A_matrix(sub_matrix_filenames);

        auto A_y_d = A_y.to_cuda<real>();

        // cusparseHandle_t handle;
        // cusparseCreate(&handle);

        cudaDeviceSynchronize();

        cusparseDnVecDescr_t y_descr = A_y_d.createDnVecDescr();

        cusparseSpMatDescr_t A_descr = A_y_d.createSpMatDescr();

        // should modify to real regularization matrix
        // load the current regularizaiton matrix
        SparseMatrixRaw<float> reg_matrix;
        reg_matrix.load_regularization_matrix(D_dir);
        SparseMatrixGPU<real> reg_matrix_d = reg_matrix.to_cuda<real>();

        cusparseSpMatDescr_t D_descr = reg_matrix_d.createSpMatDescr();

        real *x_result_d = nullptr;
        // float *x_result_h = static_cast<float *>(malloc(N_BINS * sizeof(float)));
        std::vector<real> x_result_h(N_BINS, 1e4);

        cudaMalloc(&x_result_d, N_BINS * sizeof(real));

        // cudaMemcpy(x_result_d, x_result_h.data(), N_BINS * sizeof(float), cudaMemcpyHostToDevice);

        cusparseDnVecDescr_t x_descr;
        cusparseCreateDnVec(&x_descr, N_BINS, x_result_d, CUDA_REAL);

        std::cerr << cudaDeviceSynchronize() << " initialization end\n";

        timer.stop("Prepare Simulation Validation");

        timer.start();

        // real *SpMV_buffer = nullptr;
        // size_t buffer_size = 0;

        cudaMemset(x_result_d, 0, N_BINS * sizeof(real));
        optimize(&context, A_descr, x_descr, y_descr, D_descr,
                 A_y_d.y_d, x_result_d,
                 A_y_d.n_rows, reg_matrix.get_n_rows(), lambda_tik);
        std::cerr << cudaGetErrorName(cudaGetLastError()) << '\n';
        std::cerr << cudaGetErrorName(cudaDeviceSynchronize()) << '\n';

        // take a look on the x
        cudaMemcpy(x_result_h.data(), x_result_d, N_BINS * sizeof(real), cudaMemcpyDeviceToHost);
        for (int i = 0; i < 30; ++i)
        {
            std::cerr << x_result_h[i] << ' ';
        }
        std::cerr << "\nx output end\n";
        std::cerr << cudaGetErrorName(cudaGetLastError()) << '\n';

        // load the simulation result
        std::vector<char> x_sim_raw_h;
        std::fstream x_sim_file;
        if constexpr (std::is_same_v<real, float>)
        {
            if (x_sim_file.open(std::string(sim_dir) + "x_corhel"); !x_sim_file.is_open())
            {
                std::cerr << "Failed to open the corhel file\n";
                std::exit(-1);
            }
        }
        else
        {
            if (x_sim_file.open(std::string(D_dir) + "x_corhel_db"); !x_sim_file.is_open())
            {
                std::cerr << "Failed to open the corhel file\n";
                std::exit(-1);
            }
        }
        x_sim_raw_h.assign(std::istreambuf_iterator<char>(x_sim_file), std::istreambuf_iterator<char>());

        real *x_sim_d = nullptr;
        CHECK_CUDA(cudaMalloc(&x_sim_d, sizeof(real) * N_BINS));
        CHECK_CUDA(cudaMemcpy(x_sim_d, x_sim_raw_h.data(), sizeof(real) * N_BINS, cudaMemcpyHostToDevice));

        cusparseDnVecDescr_t x_sim_descr;
        cusparseCreateDnVec(&x_sim_descr, N_BINS, x_sim_d, CUDA_REAL);

        auto cosine_similarity = [cublas_handle](auto x1_d, auto x2_d, int size)
        {
            real x1_norm = 0;
            real x2_norm = 0;

            dot(cublas_handle, size, x1_d, x1_d, &x1_norm);
            dot(cublas_handle, size, x2_d, x2_d, &x2_norm);

            x1_norm = sqrt(x1_norm);
            x2_norm = sqrt(x2_norm);

            real x_product = 0;
            dot(cublas_handle, size, x1_d, x2_d, &x_product);

            return x_product / (x1_norm * x2_norm);
        };

        // compute the cosine similarity of x and x_sim

        std::cerr << "cosine similarity of x and x_sim is " << cosine_similarity(x_result_d, x_sim_d, N_BINS) << '\n';

        {
            std::vector<real> x_temp(N_BINS, 0);
            std::vector<real> x_sim_temp(N_BINS, 0);

            constexpr int partial_rad = 10;

            std::vector<real> x_partial(N_PHI_BINS * N_THETA_BINS * partial_rad, 0);
            std::vector<real> x_sim_partial(N_PHI_BINS * N_THETA_BINS * partial_rad, 0);

            cudaMemcpy(x_temp.data(), x_result_d, sizeof(real) * N_BINS, cudaMemcpyDeviceToHost);
            cudaMemcpy(x_sim_temp.data(), x_sim_d, sizeof(real) * N_BINS, cudaMemcpyDeviceToHost);

            for (int i = 0; i < N_PHI_BINS * N_THETA_BINS; ++i)
            {
                std::copy(x_temp.begin() + i * N_RAD_BINS, x_temp.begin() + i * N_RAD_BINS + partial_rad, x_partial.begin() + i * partial_rad);
                std::copy(x_sim_temp.begin() + i * N_RAD_BINS, x_sim_temp.begin() + i * N_RAD_BINS + partial_rad, x_sim_partial.begin() + i * partial_rad);
            }

            real *x_partial_d = nullptr;
            real *x_sim_partial_d = nullptr;
            cudaMalloc(&x_partial_d, sizeof(real) * N_PHI_BINS * N_THETA_BINS * partial_rad);
            cudaMalloc(&x_sim_partial_d, sizeof(real) * N_PHI_BINS * N_THETA_BINS * partial_rad);

            cudaMemcpy(x_partial_d, x_partial.data(), sizeof(real) * N_PHI_BINS * N_THETA_BINS * partial_rad, cudaMemcpyHostToDevice);
            cudaMemcpy(x_sim_partial_d, x_sim_partial.data(), sizeof(real) * N_PHI_BINS * N_THETA_BINS * partial_rad, cudaMemcpyHostToDevice);

            std::cerr << "cosine similarity of the first " << partial_rad << " bins of x and x_sim is " << cosine_similarity(x_partial_d, x_sim_partial_d, N_PHI_BINS * N_THETA_BINS * partial_rad) << '\n';

            cudaFree(x_partial_d);
            cudaFree(x_sim_partial_d);
        }

        // compute the cosine similarity of Ax, y
        real *Ax_d = nullptr;
        cudaMalloc(&Ax_d, sizeof(real) * A_y_d.n_rows);

        cusparseDnVecDescr_t Ax_descr;
        cusparseCreateDnVec(&Ax_descr, A_y_d.n_rows, Ax_d, CUDA_REAL);

        context.SpMV(CUSPARSE_OPERATION_NON_TRANSPOSE, A_descr, x_descr, Ax_descr, 1, 0, 0);

        std::cerr << "cosine similarity of Ax and y is " << cosine_similarity(A_y_d.y_d, Ax_d, A_y_d.n_rows) << '\n';

        context.SpMV(CUSPARSE_OPERATION_NON_TRANSPOSE, A_descr, x_sim_descr, Ax_descr, 1, 0, 0);

        std::cerr << "cosine similarity of Ax_sim and y is " << cosine_similarity(A_y_d.y_d, Ax_d, A_y_d.n_rows) << '\n';

        cusparseDestroyDnVec(x_sim_descr);
        cusparseDestroyDnVec(Ax_descr);
        cudaFree(x_sim_d);
        cudaFree(Ax_d);

        // compute the prediction error

        timer.stop("Cross Validation");

        cusparseDestroySpMat(A_descr);
        cusparseDestroyDnVec(x_descr);
        cusparseDestroyDnVec(y_descr);

        cusparseDestroySpMat(D_descr);
        cudaFree(x_result_d);
    }
    cublasDestroy(cublas_handle);
}
