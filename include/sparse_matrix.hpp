#pragma once
#include <vector>
#include <fstream>
#include "constants.hpp"
#include "output.hpp"

namespace cudaSolarTomography
{
    // estimated of the ratio of maximum possible elements and real elemnts
    constexpr int ELEMENT_FACTOR = 1024 * BIN_SCALE_FACTOR;
    constexpr int MAX_RESERVED = 1 << 30;

    class SparseMatrixDyn // dynamic sparse matrix for CPU version
    {
    public:
        SparseMatrixDyn(int n_files)
        {
            row_index.reserve(n_files * Y_SIZE);
            row_index.push_back(0);
            long long n_reserved = n_files * (Y_SIZE / ELEMENT_FACTOR) * N_BINS;
            if (n_reserved < MAX_RESERVED && n_reserved > 0) // in case of overflow
            {
                // std::cerr << n_files * (Y_SIZE / ELEMENT_FACTOR) * N_BINS << ' ' <<  MAX_RESERVED << '\n';
                val.reserve(n_files * (Y_SIZE / ELEMENT_FACTOR) * N_BINS);
                col_index.reserve(n_files * (Y_SIZE / ELEMENT_FACTOR) * N_BINS);
            }
            else
            {
                val.reserve(MAX_RESERVED);
                col_index.reserve(MAX_RESERVED);
            }
        }

        template <size_t N>
        void add_row(const std::array<float, N> &arr, int scale)
        {
            int count = 0;
            for (size_t i = 0; i < N; ++i)
            {
                auto element = arr[i];
                element /= scale;
                if (element > 1e-16)
                {
                    val.push_back(element);
                    col_index.push_back(i);
                    ++count;
                }
            }
            row_index.push_back(row_index.back() + count);
        }

        template <typename Container>
        void add_row(const Container &arr, int scale)
        {
            int count = 0;
            for (size_t i = 0; i < arr.size(); ++i)
            {
                auto element = arr[i];
                // if (element > 1e-16)
                if (element != 0.0)
                {
                    element /= scale;
                    val.push_back(element);
                    col_index.push_back(i);
                    ++count;
                }
            }
            row_index.push_back(row_index.back() + count);
        }

        template <typename Container>
        void add_dense_row(const Container &arr, int scale)
        {
            int cur_index = -1;
            int count = 0;
            for (const auto &[index, value] : arr)
            {
                if (index != cur_index)
                {
                    cur_index = index;
                    ++count;
                    col_index.push_back(index);
                    val.push_back(value / scale);
                }
                else
                {
                    val.back() += value / scale;
                }
            }
            row_index.push_back(row_index.back() + count);
        }

        void print_status()
        {
            std::cout << "element count " << val.size() << " row count " << row_index.size() << '\n';
            /*for (int i = 0; i < 150; ++i)
            {
                std::cout << row_index[i + 1] - row_index[i] << '\n';
            }*/
            // for (int i = 0; i < 200; ++i)
            //{
            //    std::cout << val[i] << ' ';
            //}
        }

        void save(std::string_view filename)
        {
            write_vector(std::string(filename) + "_val", "matrix values", val);
            write_vector(std::string(filename) + "_col_index", "column index", col_index);
            write_vector(std::string(filename) + "_row_index", "row index", row_index);
        }

        void append_save(std::string_view filename)
        {
            write_vector(std::string(filename) + "_val", "matrix values", val, std::ios::app);
            write_vector(std::string(filename) + "_col_index", "column index", col_index, std::ios::app);
            // write_vector(std::string(filename) + "_row_index", "row index", row_index, std::ios::app);
        }

        auto get_row_index() const
        {
            return row_index;
        }

    private:
        std::vector<float> val;
        std::vector<int> col_index;
        std::vector<int> row_index;
    };

}
