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
#include <random>
#include <unordered_map>
#include <fitsfile.h>
#include <cusparse.h>
#include "output.hpp"
#include "build_A_matrix.cuh"

// #define NDEBUG
#include <cassert>

// constexpr std::string_view D_dir = "../python/";
// constexpr std::string_view defualt_output_dir = "../data/projection/lasco_2008/";
// constexpr std::string_view sim_dir = "../data/mhd_2008/";
constexpr std::string_view defualt_output_dir = "../data/projection/lasco_2023/";
constexpr std::string_view sim_dir = "../data/mhd/";
// constexpr std::string_view sim_dir = "../data/mhd/downsample2/";
// constexpr std::string_view defualt_output_dir = "../data/projection/lasco_2023/downsample2/";
// constexpr std::string_view sim_dir = "../data/mhd/30_143_300/";
// constexpr std::string_view defualt_output_dir = "../data/projection/lasco_2023/30_143_300/";
// constexpr std::string_view sim_dir = "../data/mhd/76_72_150/";
// constexpr std::string_view defualt_output_dir = "../data/projection/lasco_2023/76_72_150/";

// mode 0 first k images
// mode 1 randomly choose k images
constexpr int mode = 0;
constexpr size_t image_per_day = 1;

int main(int argc, char **argv)
{
    using namespace cudaSolarTomography;
    assert(BINNING_FACTOR == 1);

    std::cerr << "binning factor " << BINNING_FACTOR << " bin scale factor " << BIN_SCALE_FACTOR << '\n';

    std::string output_dir;
    if (argc == 2)
    {
        output_dir = std::string(argv[1]);
    }
    else
    {
        output_dir = defualt_output_dir;
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

    for (int fc = 0; fc < n_files; ++fc)
    {
        std::string sub_matrix_filename;
        std::getline(config_file, sub_matrix_filename);
        auto date = sub_matrix_filename.substr(0, 10);
        date_file[date].push_back(sub_matrix_filename);
    }

    cudaSolarTomography::cusparseContext context;

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
    assert(x_sim_raw_h.size() == sizeof(real) * N_BINS);

    real *x_sim_d = nullptr;
    CHECK_CUDA(cudaMalloc(&x_sim_d, sizeof(real) * N_BINS));
    CHECK_CUDA(cudaMemcpy(x_sim_d, x_sim_raw_h.data(), sizeof(real) * N_BINS, cudaMemcpyHostToDevice));

    cusparseDnVecDescr_t x_sim_descr;
    cusparseCreateDnVec(&x_sim_descr, N_BINS, x_sim_d, CUDA_REAL);

    real *Ax_d = nullptr;
    cudaMalloc(&Ax_d, sizeof(real) * IMAGE_SIZE * IMAGE_SIZE * n_files);

    std::vector<real> Ax_h;
    Ax_h.reserve(IMAGE_SIZE * IMAGE_SIZE * n_files);

    Timer timer;
    timer.start();

    std::vector<std::string> projection_sub_matrix_filenames;


    for (auto &[date, filenames] : date_file)
    {
        const int size = filenames.size();
        if constexpr (mode == 1)
            std::shuffle(filenames.begin(), filenames.end(), gen);
        for (int i = 0; i < min(image_per_day, filenames.size()); ++i)
            projection_sub_matrix_filenames.push_back(filenames[i]);
    }

    // first calculate some constants that are invariant in the loop

    timer.stop("time for reading configuration");

    timer.start();

    auto projection_params = get_all_parameters_from_files(projection_sub_matrix_filenames);
    std::vector<int> row_ptr_h(n_files * Y_SIZE + 1, 0);
    SparseMatrixAndImage projection_A_y = build_A_matrix_from_params(projection_params, row_ptr_h);

    auto projection_A_y_d = projection_A_y.to_cuda<real>();
    auto projection_A_descr = projection_A_y_d.createSpMatDescr();

    cusparseDnVecDescr_t Ax_descr;
    cusparseCreateDnVec(&Ax_descr, projection_A_y_d.n_rows, Ax_d, CUDA_REAL);

    context.SpMV(CUSPARSE_OPERATION_NON_TRANSPOSE, projection_A_descr, x_sim_descr, Ax_descr, 1, 0, 0);

    cudaMemcpy(Ax_h.data(), Ax_d, sizeof(real) * projection_A_y_d.n_rows, cudaMemcpyDeviceToHost);
    std::vector<Image_t> pb_vector;
    pb_vector.reserve(IMAGE_SIZE * IMAGE_SIZE);

    // store projection result
    std::fstream projection_config_file;
    if (projection_config_file.open(output_dir + "projection.conf", std::fstream::out); !projection_config_file.is_open())
    {
        std::cerr << "failed to open the projection config file\n";
        std::exit(-1);
    }
    projection_config_file << projection_sub_matrix_filenames.size() << '\n';

    int n_rows = 0;
    double y_norm = 0;
    for (int j = 0; j < projection_sub_matrix_filenames.size(); ++j)
    {
        pb_vector.clear();
        std::cerr << projection_params[j].b_zero << ' ' << projection_params[j].b_scale << '\n';
        for (int i = 0; i < Y_SIZE; ++i)
        {
            int row_n = j * Y_SIZE + i;
            int row_size = row_ptr_h[row_n + 1] - row_ptr_h[row_n];

            if (row_size <= 0) // skip the empty row
            {
                pb_vector.push_back(0);
            }
            else
            {
                // std::cerr << Ax_h[n_rows] << ' ' << projection_A_y.y_h[n_rows] << '\n';
                double y = Ax_h[n_rows++];
                assert(!std::isnan(y));
                y /= SCALE_FACTOR;
                y -= projection_params[j].b_zero;
                y /= projection_params[j].b_scale;
                y_norm += y * y;
                pb_vector.push_back(static_cast<Image_t>(y));
            }
        }
        assert(pb_vector.size() == IMAGE_SIZE * IMAGE_SIZE);
        write_vector(output_dir + projection_sub_matrix_filenames[j], "projection file", pb_vector);
        projection_config_file << projection_sub_matrix_filenames[j] << '\n';
    }
    std::cerr << "y_norm of projected pb vector is " << y_norm << '\n';
    std::cerr << "n_rows of projected pb vector is " << n_rows << '\n';

    cusparseDestroySpMat(projection_A_descr);
    cusparseDestroyDnVec(Ax_descr);
    cusparseDestroyDnVec(x_sim_descr);
    cudaFree(x_sim_d);
    cudaFree(Ax_d);
}
