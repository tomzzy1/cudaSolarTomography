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

// constexpr std::string_view training_config_dir = "../config/lasco_c2_2023_training.conf";
// constexpr std::string_view validation_config_dir = "../config/lasco_c2_2023_validation.conf";
constexpr std::string_view D_dir = "../python/";

// mode 0
// you have more than one images from each day, choose the first image in each day as the validation set
// mode 1
// you have more than one images from each day, choose one random image in each day as the validation set
// mode 2
// you have only one image from each day, randomly choose some days as the validation set
constexpr int mode = 2;

// search method 0 - brute-force
// search method 1 - gold-section
constexpr int search_method = 0;

constexpr int epoch = mode == 0 ? 1 : 10;

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

    std::unordered_map<std::string, std::vector<std::string>> date_file;

    if constexpr (mode == 1)
    {
        for (int fc = 0; fc < n_files; ++fc)
        {
            std::string sub_matrix_filename;
            std::getline(config_file, sub_matrix_filename);
            auto date = sub_matrix_filename.substr(0, 10);
            date_file[date].push_back(sub_matrix_filename);
        }
    }

    std::vector<std::string> sub_matrix_filenames;
    for (int fc = 0; fc < n_files; ++fc)
    {
        std::string sub_matrix_filename;
        std::getline(config_file, sub_matrix_filename);
        sub_matrix_filenames.push_back(sub_matrix_filename);
    }

    std::map<double, std::vector<float>, std::greater<>> errors;

    cublasHandle_t cublas_handle;
    cublasCreate(&cublas_handle);

    cudaSolarTomography::cusparseContext context;

    for (int e = 0; e < epoch; ++e)
    {
        std::cerr << "Epoch " << e + 1 << "/" << epoch << '\n';
        Timer timer;
        timer.start();

        std::vector<std::string> training_sub_matrix_filenames;
        std::vector<std::string> validation_sub_matrix_filenames;

        if constexpr (mode == 0 || mode == 1)
        {
            // std::fstream training_config_file;
            // std::fstream validation_config_file;
            // if (training_config_file.open(std::string(training_config_dir)); !training_config_file.is_open())
            // {
            //     std::cerr << "Failed to open the training config file " << training_config_dir << "\n";
            //     return -1;
            // }

            // if (validation_config_file.open(std::string(validation_config_dir)); !validation_config_file.is_open())
            // {
            //     std::cerr << "Failed to open the validation config file " << validation_config_dir << "\n";
            //     return -1;
            // }

            // std::string training_file_count_str;
            // std::getline(training_config_file, training_file_count_str);
            // int training_file_count = std::stoi(training_file_count_str);
            // std::cerr << "training file number " << training_file_count << '\n';

            // std::string validation_file_count_str;
            // std::getline(validation_config_file, validation_file_count_str);
            // int validation_file_count = std::stoi(validation_file_count_str);
            // std::cerr << "validation file number " << validation_file_count << '\n';

            // std::vector<std::string> training_sub_matrix_filenames;
            // for (int i = 0; i < training_file_count; ++i)
            // {
            //     std::string sub_matrix_filename;
            //     std::getline(training_config_file, sub_matrix_filename);
            //     training_sub_matrix_filenames.push_back(sub_matrix_filename);
            // }
            // training_config_file.close();

            // std::vector<std::string> validation_sub_matrix_filenames;
            // for (int i = 0; i < validation_file_count; ++i)
            // {
            //     std::string sub_matrix_filename;
            //     std::getline(validation_config_file, sub_matrix_filename);
            //     validation_sub_matrix_filenames.push_back(sub_matrix_filename);
            // }
            // validation_config_file.close();

            for (const auto &[date, filenames] : date_file)
            {
                const int size = filenames.size();
                std::uniform_int_distribution<> distr(0, size - 1);
                int validation_n = mode == 0 ? 0 : distr(gen);
                for (int i = 0; i < size; ++i)
                {
                    if (i != validation_n)
                    {
                        training_sub_matrix_filenames.push_back(filenames[i]);
                    }
                    else
                    {
                        validation_sub_matrix_filenames.push_back(filenames[i]);
                    }
                }
            }
        }
        else if (mode == 2)
        {
            std::vector<int> validation_n;
            for (int i = 0; i < 3; ++i)
            {
                std::uniform_int_distribution<> distr(0, n_files - 1);
                bool repeated = false;
                do
                {
                    repeated = false;
                    int n = distr(gen);
                    for (int j : validation_n)
                    {
                        if (n == j)
                            repeated = true;
                    }
                } while (repeated);
                validation_n.push_back(i);
            }
            for (int fc = 0; fc < n_files; ++fc)
            {
                if (std::find(validation_n.begin(), validation_n.end(), fc) != validation_n.end())
                {
                    validation_sub_matrix_filenames.push_back(sub_matrix_filenames[fc]);
                }
                else
                {
                    training_sub_matrix_filenames.push_back(sub_matrix_filenames[fc]);
                }
            }
        }

        // first calculate some constants that are invariant in the loop

        timer.stop("time for reading configuration");

        timer.start();

        SparseMatrixAndImage training_A_y = build_A_matrix(training_sub_matrix_filenames);
        SparseMatrixAndImage validation_A_y = build_A_matrix(validation_sub_matrix_filenames);

        auto training_A_y_d = training_A_y.to_cuda<real>();
        auto validation_A_y_d = validation_A_y.to_cuda<real>();

        // predicted value for validation set
        real *y_predict_d = nullptr;
        cudaMalloc(&y_predict_d, validation_A_y_d.n_rows * sizeof(real));

        // cusparseHandle_t handle;
        // cusparseCreate(&handle);

        cusparseDnVecDescr_t y_predict_descr;
        cusparseCreateDnVec(&y_predict_descr, validation_A_y_d.n_rows, y_predict_d, CUDA_REAL);

        cudaDeviceSynchronize();

        cusparseDnVecDescr_t training_y_descr = training_A_y_d.createDnVecDescr();

        cusparseSpMatDescr_t training_A_descr = training_A_y_d.createSpMatDescr();

        cusparseSpMatDescr_t validation_A_descr = validation_A_y_d.createSpMatDescr();

        // should modify to real regularization matrix
        // load the current regularizaiton matrix
        SparseMatrixRaw<float> reg_matrix;
        // reg_matrix.load_regularization_matrix(D_dir, "r3");
        reg_matrix.load_regularization_matrix(D_dir);

        SparseMatrixGPU<real> reg_matrix_d = reg_matrix.to_cuda<real>();

        cusparseSpMatDescr_t D_descr = reg_matrix_d.createSpMatDescr();

        real *training_x_result_d = nullptr;
        // float *x_result_h = static_cast<float *>(malloc(N_BINS * sizeof(float)));
        std::vector<real> x_result_h(N_BINS, 1e4);

        cudaMalloc(&training_x_result_d, N_BINS * sizeof(real));

        // cudaMemcpy(x_result_d, x_result_h.data(), N_BINS * sizeof(float), cudaMemcpyHostToDevice);

        cusparseDnVecDescr_t training_x_descr;
        cusparseCreateDnVec(&training_x_descr, N_BINS, training_x_result_d, CUDA_REAL);

        std::cerr << cudaDeviceSynchronize() << " initialization end\n";

        timer.stop("Prepare Cross Validation");

        timer.start();

        // cross validation begin

        // brute-force lambda search

        auto compute_error = [&](double new_lambda_tik)
        {
            std::cout << "currrent lambda " << new_lambda_tik << '\n';
            cudaMemset(training_x_result_d, 0, N_BINS * sizeof(real));
            optimize(&context, training_A_descr, training_x_descr, training_y_descr, D_descr,
                     training_A_y_d.y_d, training_x_result_d,
                     training_A_y_d.n_rows, reg_matrix.get_n_rows(), new_lambda_tik);
            std::cerr << cudaGetErrorName(cudaDeviceSynchronize()) << '\n';
            // compute the prediction error

            // first err = y - Ax
            cudaMemcpy(y_predict_d, validation_A_y_d.y_d, validation_A_y_d.n_rows * sizeof(real), cudaMemcpyDeviceToDevice);
            std::cerr << cudaGetErrorName(cudaDeviceSynchronize()) << '\n';

            context.SpMV(CUSPARSE_OPERATION_NON_TRANSPOSE, validation_A_descr, training_x_descr, y_predict_descr, -1, 1, 2);

            real error = 0;
            // cublasSdot(cublas_handle, validation_A_y_d.n_rows, y_predict_d, 1, y_predict_d, 1, &error);
            dot(cublas_handle, validation_A_y_d.n_rows, y_predict_d, y_predict_d, &error);
            error = sqrt(error);
            std::cerr << cudaGetErrorName(cudaDeviceSynchronize()) << '\n';
            std::cout << "error of the prediction is " << error << '\n';
            return error;
        };

        if constexpr (search_method == 0)
        {
            for (int i = 0; i < 50; ++i)
            {
                double new_lambda_tik = (i + 1) * 1e-6;
                double error = compute_error(new_lambda_tik);
                // lambda_file << new_lambda_tik << ' ' << sqrt(error) << '\n';
                errors[new_lambda_tik].push_back(error);
            }
        }
        else if constexpr (search_method == 1)
        {
            // the lambda-error is convex, search the local minimum using golden-section search
            constexpr double phi = 1.618033988749894848;
            double tolerance = 5e-7;
            double left_bound = 1e-6;
            double right_bound = 1e-3;
            int num_of_evaluations = 0;
            while (fabs(right_bound - left_bound) > tolerance)
            {
                double cut = (right_bound - left_bound) / phi;
                double guess1 = right_bound - cut;
                double guess2 = left_bound + cut;
                double error1 = compute_error(guess1);
                double error2 = compute_error(guess2);
                // errors[guess1].push_back(sqrt(error1));
                // errors[guess2].push_back(sqrt(error2));
                num_of_evaluations += 2;
                if (error1 < error2)
                {
                    right_bound = guess2;
                }
                else
                {
                    left_bound = guess1;
                }
            }

            double final_guess = (left_bound + right_bound) / 2;
            double min_error = compute_error(final_guess);
            std::cout << "lambda of minimum error is " << final_guess << " with error = " << min_error << '\n';
            errors[final_guess].push_back(min_error);
            std::cout << "number of evaluations " << num_of_evaluations + 1 << '\n';
        }

        cudaFree(y_predict_d);
        cusparseDestroyDnVec(y_predict_descr);

        timer.stop("Cross Validation");

        cusparseDestroySpMat(training_A_descr);
        cusparseDestroyDnVec(training_x_descr);
        cusparseDestroyDnVec(training_y_descr);

        cusparseDestroySpMat(validation_A_descr);

        cusparseDestroySpMat(D_descr);

        // cusparseDestroy(handle);

        cudaFree(training_x_result_d);
    }

    cublasDestroy(cublas_handle);

    std::fstream lambda_file;
    if (lambda_file.open("lambda.txt", std::fstream::out); !lambda_file.is_open())
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
}
