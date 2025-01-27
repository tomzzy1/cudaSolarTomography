#pragma once
#include <fstream>
#include <iostream>
#include <vector>
#include <filesystem>
#include "constants.hpp"

namespace cudaSolarTomography
{
    int write_info(const std::string_view output_dir, const std::string &info_name, const std::string &config_file, const std::string &fits_dir, std::ios::openmode mode = std::ios::trunc)
    {
        std::fstream output_file;
        std::string output_filename = std::string(output_dir) + "block_idx";
        std::string info_filename = std::string(output_dir) + info_name;
        if (output_file.open(info_filename, std::fstream::out | mode); !output_file.is_open())
        {
            std::cerr << "Failed to open the info file " << info_filename << "\n";
            std::exit(-1);
        }
        output_file << "N_BINS = " << N_RAD_BINS * N_THETA_BINS * N_PHI_BINS << '\n';
        output_file << "N_RAD_BINS = " << N_RAD_BINS << '\n';
        output_file << "N_THETA_BINS = " << N_THETA_BINS << '\n';
        output_file << "N_PHI_BINS = " << N_PHI_BINS << '\n';
        output_file << "R_MAX = " << R_MAX << '\n';
        output_file << "R_MIN = " << R_MIN << '\n';
        output_file << "INSTR_R_MAX = " << INSTR_R_MAX << '\n';
        output_file << "INSTR_R_MIN = " << INSTR_R_MIN << '\n';
        output_file << "IMAGE_SIZE = " << IMAGE_SIZE << '\n';
        output_file << "BINNING_FACTOR = " << BINNING_FACTOR << '\n';
        // output_file << "CONFIG_FILE = " << std::filesystem::canonical(config_file) << '\n';
        // output_file << "FITS_PATH = " << std::filesystem::canonical(fits_dir) << '\n';
        output_file << "CONFIG_FILE = " << config_file << '\n';
        output_file << "FITS_PATH = " << fits_dir << '\n';
        output_file.close();
        return 0;
    }

    template <typename T>
    size_t write_vector(const std::string &output_filename,
                        const std::string &description,
                        const std::vector<T> &x,
                        std::ios::openmode mode = std::ios::trunc)
    {
        std::fstream output_file;
        if (output_file.open(output_filename, std::fstream::out | mode); !output_file.is_open())
        {
            std::cerr << "Failed to open the " << description << ' ' << output_filename << '\n';
            std::exit(-1);
        }
        output_file.write(reinterpret_cast<const char *>(x.data()), x.size() * sizeof(typename std::vector<T>::value_type) / sizeof(char));
        output_file.close();
        return x.size();
    }

    template <typename T>
    size_t write_vector(const std::string &output_filename,
                        const std::string &description,
                        const T *x,
                        int size,
                        std::ios::openmode mode = std::ios::trunc)
    {
        std::fstream output_file;
        if (output_file.open(output_filename, std::fstream::out | mode); !output_file.is_open())
        {
            std::cerr << "Failed to open the " << description << ' ' << output_filename << '\n';
            std::exit(-1);
        }
        output_file.write(reinterpret_cast<const char *>(x), size * sizeof(T) / sizeof(char));
        output_file.close();
        return size;
    }

}
