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

// constexpr std::string_view sim_dir = "../data/mhd_2008/";
// constexpr std::string_view projection_dir = "../data/projection/lasco_2008/";
// constexpr std::string_view projection_output_dir = "../output/projection/lasco_2008/";
// constexpr std::string_view sim_dir = "../data/mhd/";
// constexpr std::string_view projection_dir = "../data/projection/lasco_2023/";
// constexpr std::string_view projection_output_dir = "../output/projection/lasco_2023/";
constexpr std::string_view sim_dir = "../data/mhd/downsample2/";
constexpr std::string_view projection_dir = "../data/projection/lasco_2023/downsample2/";
constexpr std::string_view projection_output_dir = "../output/projection/lasco_2023/downsample2/";
// constexpr std::string_view sim_dir = "../data/mhd/30_143_300/";
// constexpr std::string_view projection_dir = "../data/projection/lasco_2023/30_143_300/";
// constexpr std::string_view projection_output_dir = "../output/projection/lasco_2023/30_143_300/";
// constexpr std::string_view sim_dir = "../data/mhd/76_72_150/";
// constexpr std::string_view projection_dir = "../data/projection/lasco_2023/76_72_150/";
// constexpr std::string_view projection_output_dir = "../output/projection/lasco_2023/76_72_150/";

// constexpr int epoch = mode == 0 ? 1 : 6;
constexpr int epoch = 1;
constexpr int search_method = 0;

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

    std::map<double, std::vector<float>, std::greater<>> errors;
    std::map<double, std::vector<float>, std::greater<>> norm_errors;

    std::fstream x_sim_file;
    std::vector<char> x_sim_raw_h;
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
        if (x_sim_file.open(std::string(sim_dir) + "x_corhel_db"); !x_sim_file.is_open())
        {
            std::cerr << "Failed to open the corhel file\n";
            std::exit(-1);
        }
    }
    x_sim_raw_h.assign(std::istreambuf_iterator<char>(x_sim_file), std::istreambuf_iterator<char>());
    std::cerr << x_sim_raw_h.size() << " " << N_BINS * sizeof(real) << '\n';
    assert(x_sim_raw_h.size() == N_BINS * sizeof(real));

    real *x_sim_d = nullptr;
    CHECK_CUDA(cudaMalloc(&x_sim_d, sizeof(real) * N_BINS));
    CHECK_CUDA(cudaMemcpy(x_sim_d, x_sim_raw_h.data(), sizeof(real) * N_BINS, cudaMemcpyHostToDevice));

    cudaSolarTomography::cusparseContext context;

    cublasHandle_t cublas_handle;
    cublasCreate(&cublas_handle);

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

    // will modify x2_d
    auto norm_of_difference = [cublas_handle](auto x1_d, auto x2_d, int size)
    {
        axpy(cublas_handle, size, -1, x1_d, x2_d);

        real norm = 0;
        dot(cublas_handle, size, x2_d, x2_d, &norm);
        return sqrt(norm);
    };

    // validation with x_sim to find lambda
    for (int e = 0; e < epoch; ++e)
    {
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

        timer.stop("Prepare Projection Validation");

        // // reconstruction
        // CHECK_CUDA(cudaMemset(x_result_d, 0, N_BINS * sizeof(real)));
        // optimize(&context, A_descr, x_descr, y_descr, D_descr,
        //          A_y_d.y_d, x_result_d,
        //          A_y_d.n_rows, reg_matrix.get_n_rows(), lambda_tik);
        // std::cerr << "reconstruction end " << '\n';
        // cudaCheckError();
        // CHECK_CUDA(cudaMemcpy(x_result_h.data(), x_result_d, N_BINS * sizeof(real), cudaMemcpyDeviceToHost));
        // write_vector(std::string(projection_output_dir) + "x_result", "reconstructed x vector", x_result_h);

        timer.start();

        auto compute_error = [&](double new_lambda_tik)
        {
            std::cout << "current lambda " << new_lambda_tik << '\n';
            CHECK_CUDA(cudaMemset(x_result_d, 0, N_BINS * sizeof(real)));
            optimize(&context, A_descr, x_descr, y_descr, D_descr,
                     A_y_d.y_d, x_result_d,
                     A_y_d.n_rows, reg_matrix.get_n_rows(), new_lambda_tik);
            std::cerr << "optimize end " << '\n';
            cudaCheckError();
            return std::pair{1 - cosine_similarity(x_result_d, x_sim_d, N_BINS), norm_of_difference(x_sim_d, x_result_d, N_BINS)};
        };

        if constexpr (search_method == 0)
        {
            for (int i = 99; i <= 100; ++i)
            {
                double new_lambda_tik = 1e-4 - i * 1e-6;
                auto [error, norm_error] = compute_error(new_lambda_tik);
                std::cerr << "error " << error << ' ' << norm_error << '\n';
                // lambda_file << new_lambda_tik << ' ' << sqrt(error) << '\n';
                errors[new_lambda_tik].push_back(error);
                norm_errors[new_lambda_tik].push_back(norm_error);
            }
        }
        // else if constexpr (search_method == 1)
        // {
        //     // the lambda-error is convex, search the local minimum using golden-section search
        //     constexpr double phi = 1.618033988749894848;
        //     double tolerance = 5e-7;
        //     double left_bound = 1e-6;
        //     double right_bound = 1e-3;
        //     int num_of_evaluations = 0;
        //     while (fabs(right_bound - left_bound) > tolerance)
        //     {
        //         double cut = (right_bound - left_bound) / phi;
        //         double guess1 = right_bound - cut;
        //         double guess2 = left_bound + cut;
        //         double error1 = compute_error(guess1);
        //         double error2 = compute_error(guess2);
        //         // errors[guess1].push_back(sqrt(error1));
        //         // errors[guess2].push_back(sqrt(error2));
        //         num_of_evaluations += 2;
        //         if (error1 < error2)
        //         {
        //             right_bound = guess2;
        //         }
        //         else
        //         {
        //             left_bound = guess1;
        //         }
        //     }

        //     double final_guess = (left_bound + right_bound) / 2;
        //     double min_error = compute_error(final_guess);
        //     std::cout << "lambda of minimum error is " << final_guess << " with error = " << min_error << '\n';
        //     errors[final_guess].push_back(min_error);
        //     std::cout << "number of evaluations " << num_of_evaluations + 1 << '\n';
        // }

        // compute the prediction error

        timer.stop("Cross Validation");

        cusparseDestroySpMat(A_descr);
        cusparseDestroyDnVec(x_descr);
        cusparseDestroyDnVec(y_descr);

        cusparseDestroySpMat(D_descr);

        cudaFree(x_result_d);
    }

    std::fstream lambda_norm_file;
    if (lambda_norm_file.open("projection_lambda_norm.txt", std::fstream::out); !lambda_norm_file.is_open())
    {
        std::cerr << "Failed to open the lambda file\n";
        std::exit(-1);
    }

    for (const auto &[lambda_tik, error] : norm_errors)
    {
        lambda_norm_file << lambda_tik << ' ';
        float error_sum = 0;
        for (auto e : error)
        {
            error_sum += e;
            lambda_norm_file << e << ' ';
        }
        lambda_norm_file << error_sum / error.size() << '\n';
    }

    std::fstream lambda_file;
    if (lambda_file.open("projection_lambda.txt", std::fstream::out); !lambda_file.is_open())
    {
        std::cerr << "Failed to open the lambda file\n";
        std::exit(-1);
    }

    for (const auto &[lambda_tik, error] : errors)
    {
        lambda_file << lambda_tik << ' ';
        float error_sum = 0;
        for (auto e : error)
        {
            error_sum += e;
            lambda_file << e << ' ';
        }
        lambda_file << error_sum / error.size() << '\n';
    }

    cusparseDestroyDnVec(x_sim_descr);
    cudaFree(x_sim_d);

    cublasDestroy(cublas_handle);
}
