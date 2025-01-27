#pragma once
#include "vector_maths.hpp"
#include "operators.hpp"
#include "utility.hpp"
#include "utility.cuh"
#include "gpu_matrix.hpp"
#include "timer.hpp"
#include "type.hpp"
#include <fitsfile.h>
#include <cassert>
#include <iomanip>
#include <thread>

/*
This version is deprecated in function, but for historical comments in implementation details,
you can check this version
*/

namespace cudaSolarTomography
{
    using Params_t = double;

    __global__ void precompute_nnz(ImageParameters<Params_t> *params, int *sizes)
    {

        // constexpr int mod_n = IMAGE_SIZE % BINNING_FACTOR == 0 ? 0 : BINNING_FACTOR; // throw the last bin if the bin is not full
        int x = blockDim.x * blockIdx.x + threadIdx.x;
        int y = blockDim.y * blockIdx.y + threadIdx.y;
        int idx = y * ROW_SIZE + x;

        int size = 0;

        float image_x = PIXEL_SIZE * (x - params->center_x);
        float image_y = PIXEL_SIZE * (y - params->center_y);

        double rho = static_cast<double>(
                         sqrt(static_cast<double>(image_x * image_x + image_y * image_y))) *
                     ARCSEC_TO_RAD;
        double eta = static_cast<double>(
            atan2(static_cast<double>(-image_x), static_cast<double>(image_y)) +
            static_cast<double>(params->roll_offset * DEG_TO_RAD));

        auto sin_rho = sin(rho);
        auto cos_rho = cos(rho);

        // if (i == 512 && j % 16 == 0)
        //     printf("sin_rho %f cos_rho %f\n", sin_rho, cos_rho);

        if (sin_rho * params->dist_to_sun < INSTR_R_MAX &&
            sin_rho * params->dist_to_sun > INSTR_R_MIN)
        {
            Rotation rx(Axis::x, eta);
            Vector3d<Params_t> nearest_point = {
                params->dist_to_sun * sin_rho * sin_rho,
                static_cast<Params_t>(0.0),
                params->dist_to_sun * sin_rho * cos_rho};

            nearest_point = rx.rotate(nearest_point);

            auto los = rx.rotate(Vector3d<Params_t>{-cos_rho, static_cast<Params_t>(0.0), sin_rho});
            nearest_point = params->r23.rotate(nearest_point);
            los = params->r23.rotate(los);

            auto impact2 = nearest_point.norm2();
            auto impact = sqrt(impact2);

            bool spacecraft_dir = los.dot(params->sun_to_obs_vec) > 0;

            if (impact > R_MAX || (params->dist_to_sun < R_MIN && impact <= 1 && !spacecraft_dir))
            {
                sizes[idx + 1] = 0;
                return;
            }

            bool obs_in_inner_sphere = params->dist_to_sun < R_MIN;
            bool impact_in_sun = impact < 1;

            real t_max = sqrt(R_MAX2 - impact2);
            real t_min = sqrt(R_MIN2 - impact2);

            real entry_time = 0;
            real exit_time = 0;

            if (!obs_in_inner_sphere && !impact_in_sun)
            {
                entry_time = -t_max;
                exit_time = t_max;
            }
            else if (!obs_in_inner_sphere && !spacecraft_dir)
            {
                entry_time = -t_max;
                exit_time = -t_min;
            }
            else // (!obs_in_inner_sphere && spacecraft_time > 0) || obs_in_inner_sphere
            {
                entry_time = t_min;
                exit_time = t_max;
            }

            auto entry_pos = nearest_point + los.scale(entry_time);
            auto exit_pos = nearest_point + los.scale(exit_time);

            real entry_phi = atan2(entry_pos.y, entry_pos.x);
            if (entry_phi < 0)
                entry_phi += PI_TWO;

            real exit_phi = atan2(exit_pos.y, exit_pos.x);
            if (exit_phi < 0)
                exit_phi += PI_TWO;

            int bin_of_min_rad = impact <= R_MIN ? 0 : floor((impact - R_MIN) / RAD_BIN_SIZE);
            int radius_bin_size = 2 * (R_DIFF / RAD_BIN_SIZE - bin_of_min_rad - 1);
            size += radius_bin_size;

            double los_xy_norm2 = los.x * los.x + los.y * los.y;
            double nearest_point_xy_norm2 = nearest_point.x * nearest_point.x + nearest_point.y * nearest_point.y;
            double xy_product = los.x * nearest_point.x + los.y * nearest_point.y;

            real entry_polar = atan(entry_pos.z / sqrt(entry_pos.x * entry_pos.x + entry_pos.y * entry_pos.y));
            real exit_polar = atan(exit_pos.z / sqrt(exit_pos.x * exit_pos.x + exit_pos.y * exit_pos.y));
            real t_extreme_polar = los.z * impact2 / nearest_point.z;
            real extreme_polar = atan((nearest_point.z + los.z * t_extreme_polar) / sqrt(nearest_point_xy_norm2 + 2 * xy_product * t_extreme_polar + los_xy_norm2 * t_extreme_polar * t_extreme_polar));

            // a loose bound for the polar bins

            size += std::isnan(extreme_polar) ? N_THETA_BINS : std::fabs(entry_polar + exit_polar - 2 * extreme_polar) / THETA_BIN_SIZE;

            auto phi_diff = fabs(entry_phi - exit_phi);

            size += (phi_diff > std::numbers::pi ? (PI_TWO - phi_diff) : phi_diff) / PHI_BIN_SIZE;

            // possible error term
            size += 4;
        }
        // printf("%d %d %d\n", i, j, size);

        sizes[idx + 1] = size;
    }

    __global__ void precompute_nnz_with_binning(ImageParameters<Params_t> *params, int *sizes)
    {
        // TODO: compute the minimum bin and maximum bin of all pixels in a bin
        // constexpr int mod_n = IMAGE_SIZE % BINNING_FACTOR == 0 ? 0 : BINNING_FACTOR; // throw the last bin if the bin is not full
        int x = blockDim.x * blockIdx.x + threadIdx.x;
        int y = blockDim.y * blockIdx.y + threadIdx.y;
        int r = blockIdx.y * ROW_SIZE + blockIdx.x;
        int t = threadIdx.y * BINNING_FACTOR + threadIdx.x;
        __shared__ int global_size;

        if (t == 0)
        {
            global_size = 0;
        }

        __syncthreads();

        float image_x = PIXEL_SIZE * (x - params->center_x);
        float image_y = PIXEL_SIZE * (y - params->center_y);

        double rho = static_cast<double>(
                         sqrt(static_cast<double>(image_x * image_x + image_y * image_y))) *
                     ARCSEC_TO_RAD;
        double eta = static_cast<double>(
            atan2(static_cast<double>(-image_x), static_cast<double>(image_y)) +
            static_cast<double>(params->roll_offset * DEG_TO_RAD));

        auto sin_rho = sin(rho);
        auto cos_rho = cos(rho);

        // if (i == 512 && j % 16 == 0)
        //     printf("sin_rho %f cos_rho %f\n", sin_rho, cos_rho);

        if (sin_rho * params->dist_to_sun < INSTR_R_MAX &&
            sin_rho * params->dist_to_sun > INSTR_R_MIN)
        {
            int size = 0;
            Rotation rx(Axis::x, eta);
            Vector3d<Params_t> nearest_point = {
                params->dist_to_sun * sin_rho * sin_rho,
                static_cast<Params_t>(0.0),
                params->dist_to_sun * sin_rho * cos_rho};

            nearest_point = rx.rotate(nearest_point);

            auto los = rx.rotate(Vector3d<Params_t>{-cos_rho, static_cast<Params_t>(0.0), sin_rho});
            nearest_point = params->r23.rotate(nearest_point);
            los = params->r23.rotate(los);

            double impact2 = nearest_point.norm2();
            double impact = sqrt(impact2);

            bool spacecraft_dir = los.dot(params->sun_to_obs_vec) > 0;

            if (impact <= R_MAX && !(params->dist_to_sun < R_MIN && impact <= 1 && !spacecraft_dir))
            {
                bool obs_in_inner_sphere = params->dist_to_sun < R_MIN;
                bool impact_in_sun = impact < 1;

                double t_max = sqrt(R_MAX2 - impact2);
                double t_min = sqrt(R_MIN2 - impact2);

                double entry_time = 0;
                double exit_time = 0;

                if (!obs_in_inner_sphere && !impact_in_sun)
                {
                    entry_time = -t_max;
                    exit_time = t_max;
                }
                else if (!obs_in_inner_sphere && !spacecraft_dir)
                {
                    entry_time = -t_max;
                    exit_time = -t_min;
                }
                else // (!obs_in_inner_sphere && spacecraft_time > 0) || obs_in_inner_sphere
                {
                    entry_time = t_min;
                    exit_time = t_max;
                }

                auto entry_pos = nearest_point + los.scale(entry_time);
                auto exit_pos = nearest_point + los.scale(exit_time);

                double entry_phi = atan2(entry_pos.y, entry_pos.x);
                if (entry_phi < 0)
                    entry_phi += PI_TWO;

                double exit_phi = atan2(exit_pos.y, exit_pos.x);
                if (exit_phi < 0)
                    exit_phi += PI_TWO;

                int bin_of_min_rad = impact <= R_MIN ? 0 : floor((impact - R_MIN) / RAD_BIN_SIZE);
                int radius_bin_size = 2 * (R_DIFF / RAD_BIN_SIZE - bin_of_min_rad - 1);
                size += radius_bin_size;

                double los_xy_norm2 = los.x * los.x + los.y * los.y;
                double nearest_point_xy_norm2 = nearest_point.x * nearest_point.x + nearest_point.y * nearest_point.y;
                double xy_product = los.x * nearest_point.x + los.y * nearest_point.y;

                double entry_polar = atan(entry_pos.z / sqrt(entry_pos.x * entry_pos.x + entry_pos.y * entry_pos.y));
                double exit_polar = atan(exit_pos.z / sqrt(exit_pos.x * exit_pos.x + exit_pos.y * exit_pos.y));
                double t_extreme_polar = (nearest_point_xy_norm2 * los.z - nearest_point.z * xy_product) / (nearest_point.z * los_xy_norm2 - los.z * xy_product);
                double extreme_polar = atan((nearest_point.z + los.z * t_extreme_polar) / sqrt(nearest_point_xy_norm2 + 2 * xy_product * t_extreme_polar + los_xy_norm2 * t_extreme_polar * t_extreme_polar));

                // a loose bound for the polar bins

                size += std::isnan(extreme_polar) ? N_THETA_BINS : std::fabs(entry_polar + exit_polar - 2 * extreme_polar) / THETA_BIN_SIZE;

                auto phi_diff = fabs(entry_phi - exit_phi);

                size += (phi_diff > std::numbers::pi ? (PI_TWO - phi_diff) : phi_diff) / PHI_BIN_SIZE;

                // possible error term
                // size += 4;

                atomicAdd(&global_size, size);
            }
        }
        __syncthreads();
        if (t == 0)
        {
            // if crash, adjust this error factor
            // global_size is the maximum possible size of all bins, the real size
            // is approximately 1 / binning_factor of it, since the index will overlap
            constexpr float error_factor = BINNING_FACTOR;
            sizes[r + 1] = global_size / BINNING_FACTOR * error_factor;
        }
    }

    template <typename T>
    __device__ void compute_crossing_times(ImageParameters<T> *params, const Vector3d<Params_t> &nearest_point, const Vector3d<Params_t> &los,
                                           const real impact, const real impact2, const real spacecraft_time,
                                           std::array<real, 2 * N_RAD_BINS + 2> &radius_times,
                                           int &radius_times_begin,
                                           int &radius_times_end,
                                           std::array<real, N_THETA_BINS + 3> &polar_times,
                                           int &polar_times_begin,
                                           int &polar_times_end,
                                           std::array<real, N_PHI_BINS + 3> &azimuthal_times,
                                           int &azimuthal_times_begin,
                                           int &azimuthal_times_end, real &entry_time,
                                           real &exit_time)
    {
        bool obs_in_inner_sphere = params->dist_to_sun < R_MIN;
        bool impact_in_sun = impact < 1;

        // float t_max = sqrt(R_MAX2 - impact2);
        // float t_min = sqrt(R_MIN2 - impact2);
        real t_max = sqrt(R_MAX2 - impact2);
        real t_min = sqrt(R_MIN2 - impact2);

        if (!obs_in_inner_sphere && !impact_in_sun)
        {
            entry_time = -t_max;
            exit_time = t_max;
        }
        else if (!obs_in_inner_sphere && spacecraft_time < 0)
        {
            entry_time = -t_max;
            exit_time = -t_min;
        }
        else // (!obs_in_inner_sphere && spacecraft_time > 0) || obs_in_inner_sphere
        {
            entry_time = t_min;
            exit_time = t_max;
        }

        auto entry_pos = nearest_point + los.scale(entry_time);
        auto exit_pos = nearest_point + los.scale(exit_time);

        // float entry_phi = atan2(entry_pos.y, entry_pos.x);
        real entry_phi = atan2(entry_pos.y, entry_pos.x);
        if (entry_phi < 0)
            entry_phi += PI_TWO;
        int phi_bin_begin = floor(entry_phi / PHI_BIN_SIZE);

        // float exit_phi = atan2(exit_pos.y, exit_pos.x);
        real exit_phi = atan2(exit_pos.y, exit_pos.x);
        if (exit_phi < 0)
            exit_phi += PI_TWO;
        int phi_bin_end = floor(exit_phi / PHI_BIN_SIZE);

        if (phi_bin_begin > phi_bin_end)
        {
            swap(phi_bin_begin, phi_bin_end);
            swap(entry_phi, exit_phi);
        }

        bool wrap = fabs(entry_phi - exit_phi) > std::numbers::pi;

        // radial bin crossings

        int bin_of_min_rad = impact <= R_MIN ? 0 : floor((impact - R_MIN) / RAD_BIN_SIZE);

        int radius_bin_size = N_RAD_BINS - bin_of_min_rad - 1;

        if (!obs_in_inner_sphere && !impact_in_sun)
        {
            int t1_times_idx = radius_bin_size;
            int t2_times_idx = radius_bin_size + 1;

            for (real rad = R_MIN + (bin_of_min_rad + 1) * RAD_BIN_SIZE; rad < R_MAX - RAD_BIN_SIZE + EPSILON; rad += RAD_BIN_SIZE)
            {
                // this fix the accuracy problem when using float
                real time = sqrt((rad - impact) * (rad + impact));
                radius_times[t1_times_idx--] = -time;
                radius_times[t2_times_idx++] = time;
            }

            radius_times_end = t2_times_idx;
            radius_times_begin = t1_times_idx + 1;
        }
        else if (!obs_in_inner_sphere && spacecraft_time < 0)
        {
            radius_times_end = radius_bin_size;
            int times_idx = radius_bin_size - 1;
            for (real rad = R_MIN + (bin_of_min_rad + 1) * RAD_BIN_SIZE; rad < R_MAX; rad += RAD_BIN_SIZE)
            {
                real time = sqrt(rad * rad - impact2);
                radius_times[times_idx--] = -time;
            }
            radius_times_begin = times_idx + 1;
        }
        else // (!obs_inner_bin && space_craft > 0) || obs_inner_bin
        {
            int times_idx = 0;
            for (real rad = R_MIN + (bin_of_min_rad + 1) * RAD_BIN_SIZE; rad < R_MAX; rad += RAD_BIN_SIZE)
            {
                real time = sqrt(rad * rad - impact2);
                radius_times[times_idx++] = time;
            }
            radius_times_end = times_idx;
        }

        // polar angle bin crossings
        // use double for coefficients, to avoid getting negative value under square root
        double los_xy_norm2 = los.x * los.x + los.y * los.y;
        double nearest_point_xy_norm2 = nearest_point.x * nearest_point.x + nearest_point.y * nearest_point.y;
        double los_z_norm2 = los.z * los.z;
        double nearest_point_z_norm2 = nearest_point.z * nearest_point.z;
        double xy_product = los.x * nearest_point.x + los.y * nearest_point.y;
        double z_product = los.z * nearest_point.z;

        // float entry_polar = atan(entry_pos.z / sqrt(entry_pos.x * entry_pos.x + entry_pos.y * entry_pos.y));
        // float exit_polar = atan(exit_pos.z / sqrt(exit_pos.x * exit_pos.x + exit_pos.y * exit_pos.y));
        real entry_polar = atan(entry_pos.z / sqrt(entry_pos.x * entry_pos.x + entry_pos.y * entry_pos.y));
        real exit_polar = atan(exit_pos.z / sqrt(exit_pos.x * exit_pos.x + exit_pos.y * exit_pos.y));
        // real t_extreme_polar = (nearest_point_xy_norm2 * los.z - nearest_point.z * xy_product) / (nearest_point.z * los_xy_norm2 - los.z * xy_product);
        // float t_extreme_polar = los.z * impact2 / nearest_point.z;
        real t_extreme_polar = los.z * impact2 / nearest_point.z;
        // float extreme_polar = atan((nearest_point.z + los.z * t_extreme_polar) / sqrt(nearest_point_xy_norm2 + 2 * xy_product * t_extreme_polar + los_xy_norm2 * t_extreme_polar * t_extreme_polar));
        real extreme_polar = atan((nearest_point.z + los.z * t_extreme_polar) / sqrt(nearest_point_xy_norm2 + 2 * xy_product * t_extreme_polar + los_xy_norm2 * t_extreme_polar * t_extreme_polar));

        // assert(!isnan(t_extreme_polar));

        // return the values for calculating the roots, solving at^2 + bt + c = 0
        auto calculate_roots = [&](real theta)
        {
            real theta2 = theta * theta;
            real a = theta2 * los_xy_norm2 - los_z_norm2;
            real b = 2 * (theta2 * xy_product - z_product);
            real c = theta2 * nearest_point_xy_norm2 - nearest_point_z_norm2;
            real delta = sqrt(b * b - 4 * a * c);
            a *= 2;
            // always make the second term positive, so that r1 - r2 < r1 + r2
            return std::pair<real, real>{-b / a, delta / fabs(a)};
        };

        // bin 0 is from -pi / 2 to -pi / 2 + THETA_BIN_SIZE

        // it's only used to calculate an estimate of # of bins
        int entry_polar_bin = static_cast<int>((entry_polar + PI_HALF) / THETA_BIN_SIZE);
        int exit_polar_bin = static_cast<int>((exit_polar + PI_HALF) / THETA_BIN_SIZE);

        bool concave = true;

        // if constexpr (std::is_same_v<float, Params_t>)
        // {
        //     if (std::fabs(entry_polar - extreme_polar) >= EPSILON)
        //     {
        //         concave = entry_polar > extreme_polar;
        //     }
        //     else
        //     {
        //         concave = exit_polar > extreme_polar;
        //     }
        // }
        // else if constexpr (std::is_same_v<double, Params_t>) // this method may fail when precision is low
        {
            // concave = xy_product * los.z > los_xy_norm2 * nearest_point.z;
            concave = nearest_point.z < 0;
        }
        // if (concave != concave2)
        // {
        //     printf("concave not equal np_xy_norm2 %f xy_product %f los_xy_norm %f vz %f nz %f\n",
        //     nearest_point_xy_norm2, xy_product, los_xy_norm2, los.z, nearest_point.z);
        // }

        if (concave)
        {
            int extreme_polar_bin = static_cast<int>((extreme_polar + PI_HALF - EPSILON) / THETA_BIN_SIZE);
            // float asymptotic_line = -fabs(atan(los.z / sqrt(los_xy_norm2)));
            real asymptotic_line = -fabs(atan(los.z / sqrt(los_xy_norm2)));

            if (entry_polar < exit_polar)
            {
                int t1_times_idx = entry_polar_bin - extreme_polar_bin + 2;
                int t2_times_idx = t1_times_idx + 1;

                real theta = 0;

                //  the equations have two valid solutions in this range

                if (entry_time < t_extreme_polar && t_extreme_polar < exit_time)
                {
                    theta = (extreme_polar_bin + 1) * THETA_BIN_SIZE - PI_HALF;
                    if (fabs(theta - extreme_polar) < 2 * EPSILON)
                    {
                        auto [r1, r2] = calculate_roots(tan(theta));
                        if (!std::isnan(r2))
                        {
                            polar_times[t1_times_idx--] = r1 - r2;
                            polar_times[t2_times_idx++] = r1 + r2;
                        }
                        theta += THETA_BIN_SIZE;
                    }
                    for (; theta < std::min(entry_polar, asymptotic_line); theta += THETA_BIN_SIZE)
                    {
                        auto [r1, r2] = calculate_roots(tan(theta));
                        polar_times[t1_times_idx--] = r1 - r2;
                        polar_times[t2_times_idx++] = r1 + r2;
                    }
                }
                else
                {
                    theta = (entry_polar_bin + 1) * THETA_BIN_SIZE - PI_HALF;
                }

                polar_times_begin = t1_times_idx + 1;

                for (; theta < std::min(exit_polar, asymptotic_line); theta += THETA_BIN_SIZE)
                {
                    auto [r1, r2] = calculate_roots(tan(theta));
                    polar_times[t2_times_idx++] = r1 + r2;
                }

                for (; theta < std::min(exit_polar, static_cast<real>(-EPSILON)); theta += THETA_BIN_SIZE)
                {
                    auto [r1, r2] = calculate_roots(tan(theta));
                    polar_times[t2_times_idx++] = r1 - r2;
                }

                if (fabs(theta) < EPSILON && theta < exit_polar)
                {
                    polar_times[t2_times_idx++] = -nearest_point.z / los.z;
                    theta += THETA_BIN_SIZE;
                }

                for (; theta < exit_polar; theta += THETA_BIN_SIZE)
                {
                    auto [r1, r2] = calculate_roots(tan(theta));
                    polar_times[t2_times_idx++] = r1 + r2;
                }
                polar_times_end = t2_times_idx;
            }
            else
            {
                int t1_times_idx = entry_polar_bin - extreme_polar_bin + 2;
                int t2_times_idx = t1_times_idx + 1;

                real theta = 0;

                if (entry_time < t_extreme_polar && t_extreme_polar < exit_time)
                {
                    theta = (extreme_polar_bin + 1) * THETA_BIN_SIZE - PI_HALF;
                    if (fabs(theta - extreme_polar) < 2 * EPSILON)
                    {
                        auto [r1, r2] = calculate_roots(tan(theta));
                        if (!std::isnan(r2))
                        {
                            polar_times[t1_times_idx--] = r1 - r2;
                            polar_times[t2_times_idx++] = r1 + r2;
                        }
                        theta += THETA_BIN_SIZE;
                    }
                    for (; theta < std::min(exit_polar, asymptotic_line); theta += THETA_BIN_SIZE)
                    {
                        auto [r1, r2] = calculate_roots(tan(theta));
                        polar_times[t1_times_idx--] = r1 - r2;
                        polar_times[t2_times_idx++] = r1 + r2;
                    }
                }
                else
                {
                    theta = (exit_polar_bin + 1) * THETA_BIN_SIZE - PI_HALF;
                }

                polar_times_end = t2_times_idx;
                for (; theta < std::min(entry_polar, asymptotic_line); theta += THETA_BIN_SIZE)
                {
                    auto [r1, r2] = calculate_roots(tan(theta));
                    polar_times[t1_times_idx--] = r1 - r2;
                }

                for (; theta < std::min(entry_polar, static_cast<real>(-EPSILON)); theta += THETA_BIN_SIZE)
                {
                    auto [r1, r2] = calculate_roots(tan(theta));
                    polar_times[t1_times_idx--] = r1 + r2;
                }

                if (fabs(theta) < EPSILON && theta < entry_polar)
                {
                    polar_times[t1_times_idx--] = -nearest_point.z / los.z;
                    theta += THETA_BIN_SIZE;
                }

                for (; theta < entry_polar; theta += THETA_BIN_SIZE)
                {
                    auto [r1, r2] = calculate_roots(tan(theta));
                    polar_times[t1_times_idx--] = r1 - r2;
                }
                polar_times_begin = t1_times_idx + 1;
            }
        }
        else
        {
            int extreme_polar_bin = static_cast<int>((extreme_polar + PI_HALF) / THETA_BIN_SIZE);
            // float asymptotic_line = fabs(atan(los.z / sqrt(los_xy_norm2)));
            real asymptotic_line = fabs(atan(los.z / sqrt(los_xy_norm2)));
            if (entry_polar < exit_polar)
            {
                int t1_times_idx = extreme_polar_bin - entry_polar_bin + 2;
                int t2_times_idx = t1_times_idx + 1;

                real theta = 0;

                if (entry_time < t_extreme_polar && t_extreme_polar < exit_time)
                {
                    theta = extreme_polar_bin * THETA_BIN_SIZE - PI_HALF;
                    if (fabs(theta - extreme_polar) < 2 * EPSILON)
                    {
                        auto [r1, r2] = calculate_roots(tan(theta));
                        if (!std::isnan(r2))
                        {
                            polar_times[t1_times_idx--] = r1 - r2;
                            polar_times[t2_times_idx++] = r1 + r2;
                        }
                        theta -= THETA_BIN_SIZE;
                    }
                    // the equations have two valid solutions in this range
                    for (; theta > std::max(exit_polar, asymptotic_line); theta -= THETA_BIN_SIZE)
                    {
                        auto [r1, r2] = calculate_roots(tan(theta));
                        polar_times[t1_times_idx--] = r1 - r2;
                        polar_times[t2_times_idx++] = r1 + r2;
                    }
                }
                else
                {
                    theta = exit_polar_bin * THETA_BIN_SIZE - PI_HALF;
                }

                polar_times_end = t2_times_idx;

                for (; theta > std::max(entry_polar, asymptotic_line); theta -= THETA_BIN_SIZE)
                {
                    auto [r1, r2] = calculate_roots(tan(theta));
                    polar_times[t1_times_idx--] = r1 - r2;
                }
                for (; theta > std::max(entry_polar, static_cast<real>(EPSILON)); theta -= THETA_BIN_SIZE)
                {
                    auto [r1, r2] = calculate_roots(tan(theta));
                    polar_times[t1_times_idx--] = r1 + r2;
                }
                if (fabs(theta) < EPSILON && theta > entry_polar)
                {
                    polar_times[t1_times_idx--] = -nearest_point.z / los.z;
                    theta -= THETA_BIN_SIZE;
                }
                for (; theta > entry_polar; theta -= THETA_BIN_SIZE)
                {
                    auto [r1, r2] = calculate_roots(tan(theta));
                    polar_times[t1_times_idx--] = r1 - r2;
                }

                polar_times_begin = t1_times_idx + 1;
            }
            else
            {
                int t1_times_idx = extreme_polar_bin - entry_polar_bin + 2;
                int t2_times_idx = t1_times_idx + 1;
                real theta = 0;

                if (entry_time < t_extreme_polar && t_extreme_polar < exit_time)
                {
                    theta = extreme_polar_bin * THETA_BIN_SIZE - PI_HALF;
                    if (fabs(theta - extreme_polar) < 2 * EPSILON)
                    {
                        auto [r1, r2] = calculate_roots(tan(theta));
                        if (!std::isnan(r2))
                        {
                            polar_times[t1_times_idx--] = r1 - r2;
                            polar_times[t2_times_idx++] = r1 + r2;
                        }
                        theta -= THETA_BIN_SIZE;
                    }

                    for (; theta > std::max(entry_polar, asymptotic_line); theta -= THETA_BIN_SIZE)
                    {
                        auto [r1, r2] = calculate_roots(tan(theta));
                        polar_times[t1_times_idx--] = r1 - r2;
                        polar_times[t2_times_idx++] = r1 + r2;
                    }
                }
                else
                {
                    theta = entry_polar_bin * THETA_BIN_SIZE - PI_HALF;
                }

                polar_times_begin = t1_times_idx + 1;

                for (; theta > std::max(exit_polar, asymptotic_line); theta -= THETA_BIN_SIZE)
                {
                    auto [r1, r2] = calculate_roots(tan(theta));
                    polar_times[t2_times_idx++] = r1 + r2;
                }

                for (; theta > std::max(exit_polar, static_cast<real>(EPSILON)); theta -= THETA_BIN_SIZE)
                {
                    auto [r1, r2] = calculate_roots(tan(theta));
                    polar_times[t2_times_idx++] = r1 - r2;
                }
                if (fabs(theta) < EPSILON && theta > exit_polar)
                {
                    polar_times[t2_times_idx++] = -nearest_point.z / los.z;
                    theta -= THETA_BIN_SIZE;
                }

                for (; theta > exit_polar; theta -= THETA_BIN_SIZE)
                {
                    auto [r1, r2] = calculate_roots(tan(theta));
                    polar_times[t2_times_idx++] = r1 + r2;
                }
                polar_times_end = t2_times_idx;
            }
        }

        while (polar_times_begin < polar_times_end && polar_times[polar_times_begin] <= entry_time)
            ++polar_times_begin;

        while (polar_times_begin < polar_times_end && polar_times[polar_times_end - 1] >= exit_time)
            --polar_times_end;

        bool increase = nearest_point.x * los.y > los.x * nearest_point.y;
        if (!wrap)
        {
            if (increase)
            {
                int times_idx = 0;
                real phi = (phi_bin_begin + 1) * PHI_BIN_SIZE;
                if (phi < entry_phi && entry_phi - phi < EPSILON)
                {
                    azimuthal_times[times_idx++] = (nearest_point.y - nearest_point.x * tan(entry_phi)) / (los.x * tan(entry_phi) - los.y);
                    phi += PHI_BIN_SIZE;
                }
                for (; phi < exit_phi; phi += PHI_BIN_SIZE)
                {
                    real tan_phi = tan(phi);
                    real time = (nearest_point.y - nearest_point.x * tan_phi) / (los.x * tan_phi - los.y);
                    // assert(!std::isnan(time) && "azimuthal time is nan");
                    // if (time > entry_time && time < exit_time) // is this necessary? No. The check is redundant.
                    azimuthal_times[times_idx++] = time;
                }
                if (fabs(exit_phi - phi) < EPSILON)
                    azimuthal_times[times_idx++] = (nearest_point.y - nearest_point.x * tan(exit_phi)) / (los.x * tan(exit_phi) - los.y);
                azimuthal_times_end = times_idx;
            }
            else
            {
                azimuthal_times_end = phi_bin_end - phi_bin_begin + 2;
                int times_idx = azimuthal_times_end - 1;
                real phi = (phi_bin_begin + 1) * PHI_BIN_SIZE;
                if (phi < entry_phi && entry_phi - phi < EPSILON)
                {
                    azimuthal_times[times_idx--] = (nearest_point.y - nearest_point.x * tan(entry_phi)) / (los.x * tan(entry_phi) - los.y);
                    phi += PHI_BIN_SIZE;
                }
                for (; phi < exit_phi; phi += PHI_BIN_SIZE)
                {
                    real tan_phi = tan(phi);
                    real time = (nearest_point.y - nearest_point.x * tan_phi) / (los.x * tan_phi - los.y);
                    // assert(!std::isnan(time) && "azimuthal time is nan");
                    azimuthal_times[times_idx--] = time;
                }
                if (fabs(exit_phi - phi) < EPSILON)
                    azimuthal_times[times_idx--] = (nearest_point.y - nearest_point.x * tan(exit_phi)) / (los.x * tan(exit_phi) - los.y);
                azimuthal_times_begin = times_idx + 1;
            }
        }
        else
        {
            if (increase)
            {
                int times_idx = 0;
                real phi = (phi_bin_end + 1) * PHI_BIN_SIZE;
                if (phi < exit_phi)
                {
                    if (fabs(phi - entry_phi) < EPSILON)
                    {
                        azimuthal_times[times_idx++] = (nearest_point.y - nearest_point.x * tan(entry_phi)) / (los.x * tan(entry_phi) - los.y);
                    }
                    phi += PHI_BIN_SIZE;
                }

                // Similar reason to the radius case for using eplison here
                // for (int i = phi_bin_end; i < N_PHI_BINS; ++i)
                for (; phi < PI_TWO; phi += PHI_BIN_SIZE)
                {
                    real tan_phi = tan(phi);
                    real time = (nearest_point.y - nearest_point.x * tan_phi) / (los.x * tan_phi - los.y);
                    // assert(!std::isnan(time) && "azimuthal time 1 is nan");
                    azimuthal_times[times_idx++] = time;
                }
                if (fabs(phi - PI_TWO) < EPSILON)
                    azimuthal_times[times_idx++] = nearest_point.y / (-los.y);

                phi = PHI_BIN_SIZE;
                for (; phi < entry_phi; phi += PHI_BIN_SIZE)
                {
                    real tan_phi = tan(phi);
                    real time = (nearest_point.y - nearest_point.x * tan_phi) / (los.x * tan_phi - los.y);
                    // assert(!std::isnan(time) && "azimuthal time 2 is nan");
                    azimuthal_times[times_idx++] = time;
                }
                if (fabs(entry_phi - phi) < EPSILON)
                {
                    real time = (nearest_point.y - nearest_point.x * tan(entry_phi)) / (los.x * tan(entry_phi) - los.y);
                    if (time < exit_time && exit_time - time > EPSILON)
                        azimuthal_times[times_idx++] = (nearest_point.y - nearest_point.x * tan(entry_phi)) / (los.x * tan(entry_phi) - los.y);
                }

                azimuthal_times_end = times_idx;
            }
            else
            {
                azimuthal_times_end = phi_bin_begin + N_PHI_BINS - phi_bin_end + 2;
                int times_idx = azimuthal_times_end - 1;
                real phi = (phi_bin_end + 1) * PHI_BIN_SIZE;
                if (phi < exit_phi)
                {
                    if (fabs(phi - exit_phi) < EPSILON)
                    {
                        azimuthal_times[times_idx--] = (nearest_point.y - nearest_point.x * tan(exit_phi)) / (los.x * tan(exit_phi) - los.y);
                        phi += PHI_BIN_SIZE;
                    }
                }
                for (; phi < PI_TWO; phi += PHI_BIN_SIZE)
                {
                    real tan_phi = tan(phi);
                    real time = (nearest_point.y - nearest_point.x * tan_phi) / (los.x * tan_phi - los.y);
                    // assert(!std::isnan(time) && "azimuthal time 1 is nan");
                    azimuthal_times[times_idx--] = time;
                }
                if (fabs(phi - PI_TWO) < EPSILON)
                    azimuthal_times[times_idx--] = nearest_point.y / (-los.y);

                phi = PHI_BIN_SIZE;
                for (; phi < entry_phi; phi += PHI_BIN_SIZE)
                {
                    real tan_phi = tan(phi);
                    real time = (nearest_point.y - nearest_point.x * tan_phi) / (los.x * tan_phi - los.y);
                    // assert(!std::isnan(time) && "azimuthal time 2 is nan");
                    azimuthal_times[times_idx--] = time;
                }
                if (fabs(entry_phi - phi) < EPSILON)
                    azimuthal_times[times_idx--] = (nearest_point.y - nearest_point.x * tan(entry_phi)) / (los.x * tan(entry_phi) - los.y);
                azimuthal_times_begin = times_idx + 1;
            }
        }

        while (azimuthal_times_begin < azimuthal_times_end && azimuthal_times[azimuthal_times_begin] <= entry_time)
            ++azimuthal_times_begin;

        while (azimuthal_times_begin < azimuthal_times_end && azimuthal_times[azimuthal_times_end - 1] >= exit_time)
            --azimuthal_times_end;
    }

    __device__ auto calculate_index_and_values(const Vector3d<Params_t> &nearest_point, const Vector3d<Params_t> &los, const real impact2,
                                               const std::array<real, 2 * N_RAD_BINS + N_THETA_BINS + N_PHI_BINS + 8> &times, int i)
    {
        auto time = 0.5 * (times[i] + times[i - 1]);
        auto arc_length = times[i] - times[i - 1];

        auto v = nearest_point + los.scale(time);
        auto r = v.norm();
        int r_idx = floor((r - R_MIN) / RAD_BIN_SIZE);
        if (r_idx == N_RAD_BINS)
            --r_idx;
        if (r_idx < 0)
            r_idx = 0;
        int theta_idx = floor((atan(v.z / sqrt(v.x * v.x + v.y * v.y)) + PI_HALF) / THETA_BIN_SIZE);
        if (theta_idx == N_THETA_BINS)
            --theta_idx;
        auto phi = atan2(v.y, v.x);
        if (phi < 0)
            phi += PI_TWO;
        int phi_idx = floor(phi / PHI_BIN_SIZE);
        if (phi_idx == N_PHI_BINS)
            --phi_idx;
        int arr_idx = phi_idx * N_RAD_BINS * N_THETA_BINS + theta_idx * N_RAD_BINS + r_idx;

        // Thomson scattering or Radon Transform
        return IndexValuePair{arr_idx, static_cast<float>(Operators<OperatorClass::Thompson, float>{}(impact2, r) * arc_length)};
    }

    template <typename T>
    __device__ void compute_crossing_times2(ImageParameters<T> *params, const Vector3d<Params_t> &nearest_point, const Vector3d<Params_t> &los,
                                            const real impact, const real impact2, const real spacecraft_time,
                                            std::array<real, 2 * N_RAD_BINS + 2> &radius_times,
                                            int &radius_times_begin,
                                            int &radius_times_end,
                                            std::array<real, N_THETA_BINS + 3> &polar_times,
                                            int &polar_times_begin,
                                            int &polar_times_end,
                                            std::array<real, N_PHI_BINS + 3> &azimuthal_times,
                                            int &azimuthal_times_begin,
                                            int &azimuthal_times_end, real &entry_time,
                                            real &exit_time)
    {
        bool obs_in_inner_sphere = params->dist_to_sun < R_MIN;
        bool impact_in_sun = impact < 1;

        real t_max = sqrt(R_MAX2 - impact2);
        real t_min = sqrt(R_MIN2 - impact2);

        if (!obs_in_inner_sphere && !impact_in_sun)
        {
            entry_time = -t_max;
            exit_time = t_max;
        }
        else if (!obs_in_inner_sphere && spacecraft_time < 0)
        {
            entry_time = -t_max;
            exit_time = -t_min;
        }
        else // (!obs_in_inner_sphere && spacecraft_time > 0) || obs_in_inner_sphere
        {
            entry_time = t_min;
            exit_time = t_max;
        }

        auto entry_pos = nearest_point + los.scale(entry_time);
        auto exit_pos = nearest_point + los.scale(exit_time);

        real entry_phi = atan2(entry_pos.y, entry_pos.x);
        if (entry_phi < 0)
            entry_phi += PI_TWO;
        int phi_bin_begin = floor(entry_phi / PHI_BIN_SIZE);

        real exit_phi = atan2(exit_pos.y, exit_pos.x);
        if (exit_phi < 0)
            exit_phi += PI_TWO;
        int phi_bin_end = floor(exit_phi / PHI_BIN_SIZE);

        if (phi_bin_begin > phi_bin_end)
        {
            swap(phi_bin_begin, phi_bin_end);
            swap(entry_phi, exit_phi);
        }

        bool wrap = fabs(entry_phi - exit_phi) > std::numbers::pi;

        // radial bin crossings

        int bin_of_min_rad = impact <= R_MIN ? 0 : floor((impact - R_MIN) / RAD_BIN_SIZE);

        int radius_bin_size = R_DIFF / RAD_BIN_SIZE - bin_of_min_rad - 1;

        if (!obs_in_inner_sphere && !impact_in_sun)
        {
            int t1_times_idx = radius_bin_size;
            int t2_times_idx = radius_bin_size + 1;

            for (real rad = R_MIN + (bin_of_min_rad + 1) * RAD_BIN_SIZE; rad < R_MAX - RAD_BIN_SIZE + EPSILON; rad += RAD_BIN_SIZE)
            {
                // this fix the accuracy problem when using float
                real time = sqrt((rad - impact) * (rad + impact));
                radius_times[t1_times_idx--] = -time;
                radius_times[t2_times_idx++] = time;
            }

            radius_times_end = t2_times_idx;
            radius_times_begin = t1_times_idx + 1;
        }
        else if (!obs_in_inner_sphere && spacecraft_time < 0)
        {
            radius_times_end = radius_bin_size;
            int times_idx = radius_bin_size - 1;
            for (real rad = R_MIN + (bin_of_min_rad + 1) * RAD_BIN_SIZE; rad < R_MAX; rad += RAD_BIN_SIZE)
            {
                real time = sqrt((rad - impact) * (rad + impact));
                radius_times[times_idx--] = -time;
            }
            radius_times_begin = times_idx + 1;
        }
        else // (!obs_inner_bin && space_craft > 0) || obs_inner_bin
        {
            int times_idx = 0;
            for (real rad = R_MIN + (bin_of_min_rad + 1) * RAD_BIN_SIZE; rad < R_MAX; rad += RAD_BIN_SIZE)
            {
                real time = sqrt((rad - impact) * (rad + impact));
                radius_times[times_idx++] = time;
            }
            radius_times_end = times_idx;
        }

        // polar angle bin crossings
        double los_xy_norm2 = los.x * los.x + los.y * los.y;
        double nearest_point_xy_norm2 = nearest_point.x * nearest_point.x + nearest_point.y * nearest_point.y;
        double los_z_norm2 = los.z * los.z;
        double nearest_point_z_norm2 = nearest_point.z * nearest_point.z;
        double xy_product = los.x * nearest_point.x + los.y * nearest_point.y;
        double z_product = los.z * nearest_point.z;

        real entry_polar = atan(entry_pos.z / sqrt(entry_pos.x * entry_pos.x + entry_pos.y * entry_pos.y));
        real exit_polar = atan(exit_pos.z / sqrt(exit_pos.x * exit_pos.x + exit_pos.y * exit_pos.y));

        // return the values for calculating the roots, solving at^2 + bt + c = 0
        // bin 0 is from -pi / 2 to -pi / 2 + THETA_BIN_SIZE

        // it's only used to calculate an estimate of # of bins
        int entry_polar_bin = static_cast<int>((entry_polar + PI_HALF) / THETA_BIN_SIZE);
        int exit_polar_bin = static_cast<int>((exit_polar + PI_HALF) / THETA_BIN_SIZE);

        polar_times_end = 0;
        polar_times_begin = 0;

        for (int i = 0; i < N_THETA_BINS / 2 - 1; ++i)
        {
            auto theta = tan((i + 1) * THETA_BIN_SIZE - PI_HALF);
            double theta2 = theta * theta;
            double a = theta2 * los_xy_norm2 - los_z_norm2;
            double b = 2 * (theta2 * xy_product - z_product);
            double c = theta2 * nearest_point_xy_norm2 - nearest_point_z_norm2;
            double delta = sqrt(b * b - 4 * a * c);
            a *= 2;
            // if (delta < 1e-6)
            // {
            //     auto t = -b / a;
            //     if (t > entry_time && t < exit_time)
            //         polar_times[polar_times_end++] = -b / a;
            // }
            // else
            {
                auto t1 = -b / a + delta / a;
                auto t2 = -b / a - delta / a;
                if (t1 > entry_time && t1 < exit_time)
                {
                    polar_times[polar_times_end++] = t1;
                }
                if (t2 > entry_time && t2 < exit_time)
                {
                    polar_times[polar_times_end++] = t2;
                }
            }
        }
        auto t = -nearest_point.z / los.z;
        if (t > entry_time && t < exit_time)
        {
            polar_times[polar_times_end++] = t;
        }

        insertion_sort(polar_times.begin() + polar_times_begin, polar_times.begin() + polar_times_end);
        azimuthal_times_begin = 0;
        azimuthal_times_end = 0;

        if (wrap == 0)
        {
            for (int i = phi_bin_begin; i < phi_bin_end; ++i)
            {
                auto tan_phi = tan((i + 1) * PHI_BIN_SIZE);
                auto t = (nearest_point.y - nearest_point.x * tan_phi) / (los.x * tan_phi - los.y);
                if (t > entry_time && t < exit_time)
                    azimuthal_times[azimuthal_times_end++] = t;
            }
        }
        else
        {
            for (int i = phi_bin_end; i < N_PHI_BINS; ++i)
            {
                auto tan_phi = tan((i + 1) * PHI_BIN_SIZE);
                auto t = (nearest_point.y - nearest_point.x * tan_phi) / (los.x * tan_phi - los.y);
                if (t > entry_time && t < exit_time)
                    azimuthal_times[azimuthal_times_end++] = t;
            }
            for (int i = 0; i < phi_bin_begin; ++i)
            {
                auto tan_phi = tan((i + 1) * PHI_BIN_SIZE);
                auto t = (nearest_point.y - nearest_point.x * tan_phi) / (los.x * tan_phi - los.y);
                if (t > entry_time && t < exit_time)
                    azimuthal_times[azimuthal_times_end++] = t;
            }
        }
        insertion_sort(azimuthal_times.begin() + azimuthal_times_begin, azimuthal_times.begin() + azimuthal_times_end);
    }

    template <typename T>
    __global__ void build_sub_matrix_with_binning(ImageParameters<T> *params, Image_t *pb_vector, float *y_output, float *val, int *row_nnzs, int *col_idx, int *precompute_row_ptr)
    {
        // mapping of index between the GPU version and the CPU version
        // blockDim.x * blockIdx.x == j (inner loop, looping the row)
        // blockDim.y * blockIdx.y == i (outer loop, looping the column)
        // threadIdx.x == l
        // threadIdx.y == k
        int x = blockDim.x * blockIdx.x + threadIdx.x;
        int y = blockDim.y * blockIdx.y + threadIdx.y;
        int t = threadIdx.y * BINNING_FACTOR + threadIdx.x;
        int r = blockIdx.y * ROW_SIZE + blockIdx.x;

        if (t == 0 && r == 0)
            printf("build_sub_matrix_with_binning\n");

        __shared__ int n_los;
        __shared__ float y_val;

        __shared__ std::array<IndexValuePair<int, float>, AROW_SIZE * BIN_SIZE> col_idx_and_val;
        __shared__ std::array<IndexValuePair<int, float>, AROW_SIZE * BIN_SIZE> col_idx_and_val2;
        __shared__ std::array<int, BIN_SIZE + 1> row_ptr;
        __shared__ int total_row_nnz;

        if (t == 0)
        {
            y_val = 0.0f;
            n_los = 0;
            row_ptr[0] = 0;
        }
        row_ptr[t + 1] = 0;
        __syncthreads();
        float image_x = PIXEL_SIZE * (x - params->center_x);
        float image_y = PIXEL_SIZE * (y - params->center_y);

        // image_x[x] = PIXEL_SIZE * (x - center_x)
        // image_y[y] = PIXEL_SIZE * (y - center_y)
        /* rhos[x][y] = static_cast<float>(
                                    sqrt(static_cast<double>(image_x[x] * image_x[x] + image_y[y] * image_y[y]))) *
                                 ARCSEC_TO_RAD;*/
        // rho = rhos[j + l][i + k] <=> rho = rhos[x][y]
        double rho = static_cast<double>(
                         sqrt(static_cast<double>(image_x * image_x + image_y * image_y))) *
                     ARCSEC_TO_RAD;
        double eta = static_cast<double>(
            atan2(static_cast<double>(-image_x), static_cast<double>(image_y)) +
            static_cast<double>(params->roll_offset * DEG_TO_RAD));

        std::array<real, 2 * N_RAD_BINS + 2> radius_times;
        int radius_times_begin = 0;
        int radius_times_end = 0;
        std::array<real, N_THETA_BINS + 3> polar_times;
        int polar_times_begin = 0;
        int polar_times_end = 0;
        std::array<real, N_PHI_BINS + 3> azimuthal_times;
        int azimuthal_times_begin = 0;
        int azimuthal_times_end = 0;

        real entry_time = 0;
        real exit_time = 0;

        std::array<real, 2 * N_RAD_BINS + N_THETA_BINS + N_PHI_BINS + 8> times;
        int times_size = 0;

        auto sin_rho = sin(rho);
        auto cos_rho = cos(rho);

        Rotation rx(Axis::x, eta);

        Vector3d<Params_t> nearest_point = {
            params->dist_to_sun * sin_rho * sin_rho,
            static_cast<Params_t>(0.0),
            params->dist_to_sun * sin_rho * cos_rho};

        nearest_point = rx.rotate(nearest_point);
        nearest_point = params->r23.rotate(nearest_point);

        Vector3d<Params_t> los = {-cos_rho, static_cast<Params_t>(0.0), sin_rho};
        los = rx.rotate(los);
        los = params->r23.rotate(los);

        auto impact = nearest_point.norm();
        auto impact2 = nearest_point.norm2();

        real spacecraft_time = sqrt(params->dist_to_sun * params->dist_to_sun - impact2);

        if (los.dot(params->sun_to_obs_vec) < 0)
            spacecraft_time *= -1;

        int t_idx = 0;

        if ((sin_rho * params->dist_to_sun < INSTR_R_MAX) &&
            (sin_rho * params->dist_to_sun > INSTR_R_MIN) &&
            !(impact > R_MAX || (params->dist_to_sun < R_MIN && impact < 1 && spacecraft_time < 0)))
        {
            // Image_t pB_val = (*(pb_vector + IMAGE_SIZE * (x + k) + (y + l)) * params->b_scale + params->b_zero) * SCALE_FACTOR * 0.79; // match the oldest gold prelim
            auto pB_val = (*(pb_vector + IMAGE_SIZE * y + x) * params->b_scale + params->b_zero) * SCALE_FACTOR; // match the oldest gold prelim
            compute_crossing_times(params, nearest_point, los, impact, impact2, spacecraft_time,
                                   radius_times, radius_times_begin, radius_times_end,
                                   polar_times, polar_times_begin, polar_times_end,
                                   azimuthal_times, azimuthal_times_begin, azimuthal_times_end,
                                   entry_time, exit_time);

            times_size += 2;

            atomicAdd(&y_val, pB_val);
            t_idx = atomicAdd(&n_los, 1);
        }

        // std::sort(times.begin(), times.begin() + times_idx);

        times_size += std::max(radius_times_end - radius_times_begin, 0) +
                      std::max(polar_times_end - polar_times_begin, 0) +
                      std::max(azimuthal_times_end - azimuthal_times_begin, 0);

        // printf("%d %d %d %d %d\n", x, y, k, l, times_size);

        int spacecraft_time_iter = 0;

        int row_nnz = 0;

        if (times_size > 0)
        {
            three_way_merge(times, radius_times, polar_times, azimuthal_times,
                            radius_times_begin, radius_times_end,
                            polar_times_begin, polar_times_end,
                            azimuthal_times_begin, azimuthal_times_end, entry_time, exit_time);
            spacecraft_time_iter = lower_bound(times.begin(), times.begin() + times_size, spacecraft_time) - times.begin();

            row_nnz = times_size - spacecraft_time_iter - 1;

            row_ptr[t_idx + 1] = row_nnz;
        }

        __syncthreads();

        bool odd = true;

        if (t == 0)
        {
            partial_sum(row_ptr.begin(), row_ptr.end(), row_ptr.begin());
        }

        __syncthreads();

        if (times_size > 0)
        {
            for (int i = 0; i < row_nnz; ++i)
            {
                col_idx_and_val[row_ptr[t_idx] + i] = calculate_index_and_values(nearest_point, los, impact2, times, i + spacecraft_time_iter + 1);
            }
        }

        __syncthreads();

        if (times_size > 0)
        {
            gpu_sort(col_idx_and_val.begin() + row_ptr[t_idx], col_idx_and_val.begin() + row_ptr[t_idx + 1]);
        }
        __syncthreads();
        int thread_count = BIN_SIZE / 2;
        int s = 1;

        while (thread_count > 0)
        {
            if (t < thread_count)
            {
                // use double buffer

                if (odd)
                {
                    if (2 * s * t + 2 * s <= n_los)
                    {
                        gpu_merge(col_idx_and_val.begin() + row_ptr[2 * s * t], col_idx_and_val.begin() + row_ptr[2 * s * t + s],
                                  col_idx_and_val.begin() + row_ptr[2 * s * t + s], col_idx_and_val.begin() + row_ptr[2 * s * t + 2 * s],
                                  col_idx_and_val2.begin() + row_ptr[2 * s * t]);
                    }
                    else if (2 * s * t + s <= n_los)
                    {
                        gpu_merge(col_idx_and_val.begin() + row_ptr[2 * s * t], col_idx_and_val.begin() + row_ptr[2 * s * t + s],
                                  col_idx_and_val.begin() + row_ptr[2 * s * t + s], col_idx_and_val.begin() + row_ptr[n_los],
                                  col_idx_and_val2.begin() + row_ptr[2 * s * t]);
                    }
                    else if (2 * s * t <= n_los)
                    {
                        gpu_copy(col_idx_and_val.begin() + row_ptr[2 * s * t], col_idx_and_val.begin() + row_ptr[n_los],
                                 col_idx_and_val2.begin() + row_ptr[2 * s * t]);
                    }
                }
                else
                {
                    if (2 * s * t + 2 * s <= n_los)
                    {
                        gpu_merge(col_idx_and_val2.begin() + row_ptr[2 * s * t], col_idx_and_val2.begin() + row_ptr[2 * s * t + s],
                                  col_idx_and_val2.begin() + row_ptr[2 * s * t + s], col_idx_and_val2.begin() + row_ptr[2 * s * t + 2 * s],
                                  col_idx_and_val.begin() + row_ptr[2 * s * t]);
                    }
                    else if (2 * s * t + s <= n_los)
                    {
                        gpu_merge(col_idx_and_val2.begin() + row_ptr[2 * s * t], col_idx_and_val2.begin() + row_ptr[2 * s * t + s],
                                  col_idx_and_val2.begin() + row_ptr[2 * s * t + s], col_idx_and_val2.begin() + row_ptr[n_los],
                                  col_idx_and_val.begin() + row_ptr[2 * s * t]);
                    }
                    else if (2 * s * t <= n_los)
                    {
                        gpu_copy(col_idx_and_val2.begin() + row_ptr[2 * s * t], col_idx_and_val2.begin() + row_ptr[n_los],
                                 col_idx_and_val.begin() + row_ptr[2 * s * t]);
                    }
                }
            }
            thread_count /= 2;
            s *= 2;
            odd = !odd;
            __syncthreads();
        }

        // merge same index
        if (t == 0 && n_los != 0)
        {
            int j = 0;
            int cur_idx = odd ? col_idx_and_val[0].first : col_idx_and_val2[0].first;
            if (!odd)
            {
                col_idx_and_val[0] = col_idx_and_val2[0];
            }
            for (int i = 1; i < row_ptr[n_los]; ++i)
            {
                int idx = odd ? col_idx_and_val[i].first : col_idx_and_val2[i].first;
                if (idx == cur_idx)
                {
                    col_idx_and_val[j].second += odd ? col_idx_and_val[i].second : col_idx_and_val2[i].second;
                }
                else
                {
                    ++j;
                    cur_idx = idx;
                    col_idx_and_val[j] = odd ? col_idx_and_val[i] : col_idx_and_val2[i];
                }
            }
            total_row_nnz = j + 1;
        }

        __syncthreads();

        if (n_los != 0)
        {
            if (t == 0)
            {
                int idx = params->file_idx * Y_SIZE + r;
                row_nnzs[idx] = total_row_nnz;
                assert(total_row_nnz != 0);
                y_output[r] = y_val / n_los;
            }

            int output_idx = -1;

            output_idx = precompute_row_ptr[r];

            for (int idx = t; idx < total_row_nnz; idx += BIN_SIZE)
            {
                col_idx[output_idx + idx] = col_idx_and_val[idx].first;
                val[output_idx + idx] = col_idx_and_val[idx].second / n_los;
                // assert(!std::isnan(val[output_idx + idx]));
            }
        }
        else
        {
            if (t == 0)
            {
                int idx = params->file_idx * Y_SIZE + r;
                row_nnzs[idx] = 0;
                y_output[r] = -1;
            }
        }
    }

    template <typename T>
    __global__ void build_sub_matrix_with_binning_global(ImageParameters<T> *params, Image_t *pb_vector, float *y_output, float *val, int *row_nnzs, int *col_idx, int *precompute_row_ptr, IndexValuePair<int, float> *buffer)
    {
        int x = blockDim.x * blockIdx.x + threadIdx.x;
        int y = blockDim.y * blockIdx.y + threadIdx.y;
        int t = threadIdx.y * BINNING_FACTOR + threadIdx.x;
        int r = blockIdx.y * ROW_SIZE + blockIdx.x;

        if (t == 0 && r == 0)
            printf("build_sub_matrix_with_binning_global\n");

        __shared__ int n_los;
        __shared__ float y_val;

        IndexValuePair<int, float> *col_idx_and_val = buffer + (AROW_SIZE * BIN_SIZE) * (2 * r);
        IndexValuePair<int, float> *col_idx_and_val2 = buffer + (AROW_SIZE * BIN_SIZE) * (2 * r + 1);

        __shared__ std::array<int, BIN_SIZE + 1> row_ptr;
        __shared__ int total_row_nnz;

        if (t == 0)
        {
            y_val = 0.0f;
            n_los = 0;
            row_ptr[0] = 0;
        }
        row_ptr[t + 1] = 0;
        __syncthreads();
        float image_x = PIXEL_SIZE * (x - params->center_x);
        float image_y = PIXEL_SIZE * (y - params->center_y);

        double rho = static_cast<double>(
                         sqrt(static_cast<double>(image_x * image_x + image_y * image_y))) *
                     ARCSEC_TO_RAD;
        double eta = static_cast<double>(
            atan2(static_cast<double>(-image_x), static_cast<double>(image_y)) +
            static_cast<double>(params->roll_offset * DEG_TO_RAD));

        std::array<real, 2 * N_RAD_BINS + 2> radius_times;
        int radius_times_begin = 0;
        int radius_times_end = 0;
        std::array<real, N_THETA_BINS + 3> polar_times;
        int polar_times_begin = 0;
        int polar_times_end = 0;
        std::array<real, N_PHI_BINS + 3> azimuthal_times;
        int azimuthal_times_begin = 0;
        int azimuthal_times_end = 0;

        real entry_time = 0;
        real exit_time = 0;

        std::array<real, 2 * N_RAD_BINS + N_THETA_BINS + N_PHI_BINS + 8> times;
        int times_size = 0;

        auto sin_rho = sin(rho);
        auto cos_rho = cos(rho);

        Rotation rx(Axis::x, eta);

        Vector3d<Params_t> nearest_point = {
            params->dist_to_sun * sin_rho * sin_rho,
            static_cast<Params_t>(0.0),
            params->dist_to_sun * sin_rho * cos_rho};

        nearest_point = rx.rotate(nearest_point);
        nearest_point = params->r23.rotate(nearest_point);

        Vector3d<Params_t> los = {-cos_rho, static_cast<Params_t>(0.0), sin_rho};
        los = rx.rotate(los);
        los = params->r23.rotate(los);

        auto impact = nearest_point.norm();
        auto impact2 = nearest_point.norm2();

        real spacecraft_time = sqrt(params->dist_to_sun * params->dist_to_sun - impact2);

        if (los.dot(params->sun_to_obs_vec) < 0)
            spacecraft_time *= -1;

        int t_idx = 0;

        if ((sin_rho * params->dist_to_sun < INSTR_R_MAX) &&
            (sin_rho * params->dist_to_sun > INSTR_R_MIN) &&
            !(impact > R_MAX || (params->dist_to_sun < R_MIN && impact < 1 && spacecraft_time < 0)))
        {
            // Image_t pB_val = (*(pb_vector + IMAGE_SIZE * (x + k) + (y + l)) * params->b_scale + params->b_zero) * SCALE_FACTOR * 0.79; // match the oldest gold prelim
            auto pB_val = (*(pb_vector + IMAGE_SIZE * y + x) * params->b_scale + params->b_zero) * SCALE_FACTOR; // match the oldest gold prelim
            compute_crossing_times(params, nearest_point, los, impact, impact2, spacecraft_time,
                                   radius_times, radius_times_begin, radius_times_end,
                                   polar_times, polar_times_begin, polar_times_end,
                                   azimuthal_times, azimuthal_times_begin, azimuthal_times_end,
                                   entry_time, exit_time);

            times_size += 2;

            atomicAdd(&y_val, pB_val);
            t_idx = atomicAdd(&n_los, 1);
        }

        // std::sort(times.begin(), times.begin() + times_idx);

        times_size += std::max(radius_times_end - radius_times_begin, 0) +
                      std::max(polar_times_end - polar_times_begin, 0) +
                      std::max(azimuthal_times_end - azimuthal_times_begin, 0);

        // printf("%d %d %d %d %d\n", x, y, k, l, times_size);

        int spacecraft_time_iter = 0;

        int row_nnz = 0;

        if (times_size > 0)
        {
            three_way_merge(times, radius_times, polar_times, azimuthal_times,
                            radius_times_begin, radius_times_end,
                            polar_times_begin, polar_times_end,
                            azimuthal_times_begin, azimuthal_times_end, entry_time, exit_time);
            spacecraft_time_iter = lower_bound(times.begin(), times.begin() + times_size, spacecraft_time) - times.begin();

            row_nnz = times_size - spacecraft_time_iter - 1;

            row_ptr[t_idx + 1] = row_nnz;
        }

        __syncthreads();

        bool odd = true;

        if (t == 0)
        {
            partial_sum(row_ptr.begin(), row_ptr.end(), row_ptr.begin());
        }

        __syncthreads();

        if (times_size > 0)
        {
            for (int i = 0; i < row_nnz; ++i)
            {
                col_idx_and_val[row_ptr[t_idx] + i] = calculate_index_and_values(nearest_point, los, impact2, times, i + spacecraft_time_iter + 1);
            }
        }

        __syncthreads();

        if (times_size > 0)
        {
            gpu_sort(col_idx_and_val + row_ptr[t_idx], col_idx_and_val + row_ptr[t_idx + 1]);
        }
        __syncthreads();
        int thread_count = BIN_SIZE / 2;
        int s = 1;

        while (thread_count > 0)
        {
            if (t < thread_count)
            {
                // use double buffer

                if (odd)
                {
                    if (2 * s * t + 2 * s <= n_los)
                    {
                        gpu_merge(col_idx_and_val + row_ptr[2 * s * t], col_idx_and_val + row_ptr[2 * s * t + s],
                                  col_idx_and_val + row_ptr[2 * s * t + s], col_idx_and_val + row_ptr[2 * s * t + 2 * s],
                                  col_idx_and_val2 + row_ptr[2 * s * t]);
                    }
                    else if (2 * s * t + s <= n_los)
                    {
                        gpu_merge(col_idx_and_val + row_ptr[2 * s * t], col_idx_and_val + row_ptr[2 * s * t + s],
                                  col_idx_and_val + row_ptr[2 * s * t + s], col_idx_and_val + row_ptr[n_los],
                                  col_idx_and_val2 + row_ptr[2 * s * t]);
                    }
                    else if (2 * s * t <= n_los)
                    {
                        gpu_copy(col_idx_and_val + row_ptr[2 * s * t], col_idx_and_val + row_ptr[n_los],
                                 col_idx_and_val2 + row_ptr[2 * s * t]);
                    }
                }
                else
                {
                    if (2 * s * t + 2 * s <= n_los)
                    {
                        gpu_merge(col_idx_and_val2 + row_ptr[2 * s * t], col_idx_and_val2 + row_ptr[2 * s * t + s],
                                  col_idx_and_val2 + row_ptr[2 * s * t + s], col_idx_and_val2 + row_ptr[2 * s * t + 2 * s],
                                  col_idx_and_val + row_ptr[2 * s * t]);
                    }
                    else if (2 * s * t + s <= n_los)
                    {
                        gpu_merge(col_idx_and_val2 + row_ptr[2 * s * t], col_idx_and_val2 + row_ptr[2 * s * t + s],
                                  col_idx_and_val2 + row_ptr[2 * s * t + s], col_idx_and_val2 + row_ptr[n_los],
                                  col_idx_and_val + row_ptr[2 * s * t]);
                    }
                    else if (2 * s * t <= n_los)
                    {
                        gpu_copy(col_idx_and_val2 + row_ptr[2 * s * t], col_idx_and_val2 + row_ptr[n_los],
                                 col_idx_and_val + row_ptr[2 * s * t]);
                    }
                }
            }
            thread_count /= 2;
            s *= 2;
            odd = !odd;
            __syncthreads();
        }

        // merge same index
        if (t == 0 && n_los != 0)
        {
            int j = 0;
            int cur_idx = odd ? col_idx_and_val[0].first : col_idx_and_val2[0].first;
            if (!odd)
            {
                col_idx_and_val[0] = col_idx_and_val2[0];
            }
            for (int i = 1; i < row_ptr[n_los]; ++i)
            {
                int idx = odd ? col_idx_and_val[i].first : col_idx_and_val2[i].first;
                if (idx == cur_idx)
                {
                    col_idx_and_val[j].second += odd ? col_idx_and_val[i].second : col_idx_and_val2[i].second;
                }
                else
                {
                    ++j;
                    cur_idx = idx;
                    col_idx_and_val[j] = odd ? col_idx_and_val[i] : col_idx_and_val2[i];
                }
            }
            total_row_nnz = j + 1;
        }

        __syncthreads();

        if (n_los != 0)
        {
            if (t == 0)
            {
                int idx = params->file_idx * Y_SIZE + r;
                row_nnzs[idx] = total_row_nnz;
                assert(total_row_nnz != 0);
                y_output[r] = y_val / n_los;
            }

            int output_idx = -1;

            output_idx = precompute_row_ptr[r];

            for (int idx = t; idx < total_row_nnz; idx += BIN_SIZE)
            {
                col_idx[output_idx + idx] = col_idx_and_val[idx].first;
                val[output_idx + idx] = col_idx_and_val[idx].second / n_los;
                // assert(!std::isnan(val[output_idx + idx]));
            }
        }
        else
        {
            if (t == 0)
            {
                int idx = params->file_idx * Y_SIZE + r;
                row_nnzs[idx] = 0;
                y_output[r] = -1;
            }
        }
    }

    // no binning
    template <typename T>
    __global__ void build_sub_matrix(ImageParameters<T> *params, float *val, int *row_nnzs, int *col_idx, int *precompute_row_ptr)
    {
        int x = blockDim.x * blockIdx.x + threadIdx.x;
        int y = blockDim.y * blockIdx.y + threadIdx.y;
        int r = y * ROW_SIZE + x;

        int n_los = 0;

        std::array<IndexValuePair<int, float>, AROW_SIZE> col_idx_and_val;

        float image_x = PIXEL_SIZE * (x - params->center_x);
        float image_y = PIXEL_SIZE * (y - params->center_y);

        double rho = static_cast<double>(
                         sqrt(static_cast<double>(image_x * image_x + image_y * image_y))) *
                     ARCSEC_TO_RAD;
        double eta = static_cast<double>(
            atan2(static_cast<double>(-image_x), static_cast<double>(image_y)) +
            static_cast<double>(params->roll_offset * DEG_TO_RAD));

        // float rho = static_cast<float>(
        //                  sqrt(static_cast<double>(image_x * image_x + image_y * image_y))) *
        //              ARCSEC_TO_RAD;
        // float eta = static_cast<float>(
        //     atan2(static_cast<double>(-image_x), static_cast<double>(image_y)) +
        //     static_cast<double>(params->roll_offset * DEG_TO_RAD));

        std::array<real, 2 * N_RAD_BINS + 2> radius_times;
        int radius_times_begin = 0;
        int radius_times_end = 0;
        std::array<real, N_THETA_BINS + 3> polar_times;
        int polar_times_end = 0;
        int polar_times_begin = 0;
        std::array<real, N_PHI_BINS + 3> azimuthal_times;
        int azimuthal_times_begin = 0;
        int azimuthal_times_end = 0;
        std::array<real, 2 * N_RAD_BINS + N_THETA_BINS + N_PHI_BINS + 8> times;
        int times_size = 0;

        // float sin_rho = sin(rho);
        // float cos_rho = cos(rho);
        auto sin_rho = sin(rho);
        auto cos_rho = cos(rho);

        Rotation rx(Axis::x, eta);

        Vector3d<Params_t> nearest_point = {
            static_cast<Params_t>(params->dist_to_sun * sin_rho * sin_rho),
            0.0f,
            static_cast<Params_t>(params->dist_to_sun * sin_rho * cos_rho)};

        nearest_point = rx.rotate(nearest_point);
        nearest_point = params->r23.rotate(nearest_point);

        Vector3d<Params_t> los = {static_cast<Params_t>(-cos_rho), 0.0f, static_cast<Params_t>(sin_rho)};
        los = rx.rotate(los);
        los = params->r23.rotate(los);

        auto impact = nearest_point.norm();
        auto impact2 = nearest_point.norm2();

        real spacecraft_time = sqrt(params->dist_to_sun * params->dist_to_sun - impact2);

        if (los.dot(params->sun_to_obs_vec) < 0)
            spacecraft_time *= -1;

        // float entry_time = 0;
        // float exit_time = 0;
        real entry_time = 0;
        real exit_time = 0;

        if ((sin_rho * params->dist_to_sun < INSTR_R_MAX) &&
            (sin_rho * params->dist_to_sun > INSTR_R_MIN) &&
            !(impact > R_MAX || (params->dist_to_sun < R_MIN && impact <= 1 && spacecraft_time < 0)))
        {
            compute_crossing_times(params, nearest_point, los, impact, impact2, spacecraft_time,
                                   radius_times, radius_times_begin, radius_times_end,
                                   polar_times, polar_times_begin, polar_times_end,
                                   azimuthal_times, azimuthal_times_begin, azimuthal_times_end,
                                   entry_time, exit_time);

            times_size += 2;
            n_los += 1;
        }

        // std::sort(times.begin(), times.begin() + times_idx);

        times_size += std::max(radius_times_end - radius_times_begin, 0) +
                      std::max(polar_times_end - polar_times_begin, 0) +
                      std::max(azimuthal_times_end - azimuthal_times_begin, 0);

        // printf("%d %d %d %d %d\n", x, y, k, l, times_size);

        int spacecraft_time_iter = 0;

        int row_nnz = 0;

        if (times_size > 0)
        {
            three_way_merge(times, radius_times, polar_times, azimuthal_times,
                            radius_times_begin, radius_times_end,
                            polar_times_begin, polar_times_end,
                            azimuthal_times_begin, azimuthal_times_end, entry_time, exit_time);
            spacecraft_time_iter = lower_bound(times.begin(), times.begin() + times_size, spacecraft_time) - times.begin();
            // assert(spacecraft_time_iter == 0);
            row_nnz = times_size - spacecraft_time_iter - 1;

            for (int i = 0; i < row_nnz; ++i)
            {
                col_idx_and_val[i] = calculate_index_and_values(nearest_point, los, impact2, times, i + spacecraft_time_iter + 1);
            }
        }
        gpu_sort(col_idx_and_val.begin(), col_idx_and_val.begin() + row_nnz);
        // assert(std::is_sorted(col_idx_and_val.begin(), col_idx_and_val.begin() + row_nnz));

        // merge same index
        // can be implemented together with output
        int j = 0;
        int cur_idx = col_idx_and_val[0].first;
        for (int i = 1; i < row_nnz; ++i)
        {
            int idx = col_idx_and_val[i].first;
            if (idx == cur_idx)
            {
                col_idx_and_val[j].second += col_idx_and_val[i].second;
            }
            else
            {
                ++j;
                cur_idx = idx;
                col_idx_and_val[j] = col_idx_and_val[i];
            }
        }
        row_nnz = j + 1;
        // assert(std::is_sorted(col_idx_and_val.begin(), col_idx_and_val.begin() + row_nnz));

        __syncthreads();
        if (n_los != 0)
        {
            int idx = params->file_idx * Y_SIZE + r;
            assert(row_nnz != 0);
            row_nnzs[idx] = row_nnz;

            int output_idx = precompute_row_ptr[r];

            for (int i = 0; i < row_nnz; ++i)
            {
                col_idx[output_idx + i] = col_idx_and_val[i].first;
                val[output_idx + i] = col_idx_and_val[i].second / n_los;
            }

            row_nnzs[idx] = row_nnz;
        }
        else
        {
            int idx = params->file_idx * Y_SIZE + r;
            row_nnzs[idx] = 0;
        }
    }

    template <typename T>
    __global__ void build_sub_matrix_with_mask(ImageParameters<T> *params, Image_t *pb_vector, float *y_output, float *val, int *row_nnzs, int *col_idx, int *precompute_row_ptr, int *bin_mask)
    {
        int x = blockDim.x * blockIdx.x + threadIdx.x;
        int y = blockDim.y * blockIdx.y + threadIdx.y;
        int t = threadIdx.y * BINNING_FACTOR + threadIdx.x;
        int r = blockIdx.y * ROW_SIZE + blockIdx.x;
        if (t == 0 && r == 0)
            printf("build_sub_matrix_with_mask\n");

        __shared__ int n_los;
        __shared__ float y_val;

        __shared__ std::array<IndexValuePair<int, float>, AROW_SIZE * BIN_SIZE> col_idx_and_val;
        __shared__ std::array<IndexValuePair<int, float>, AROW_SIZE * BIN_SIZE> col_idx_and_val2;
        __shared__ std::array<int, BIN_SIZE + 1> row_ptr;
        __shared__ int total_row_nnz;

        auto time_diff = std::abs(params->cur_time - params->ref_time);
        constexpr double seconds_of_day = 86400;
        // double time_scale_factor = std::exp(-time_diff / seconds_of_day); // exp1
        double time_scale_factor = std::exp(-time_diff / seconds_of_day / 2.0); // exp2
        // double time_scale_factor = 1 - time_diff / seconds_of_day * 0.15; // lin1
        // double time_scale_factor = 1; 
        // if (params->file_idx == 0)
        // {
        //     printf("time scale factor %f time diff %d\n", time_scale_factor, time_diff);
        // }
        if (t == 0)
        {
            y_val = 0.0f;
            n_los = 0;
            row_ptr[0] = 0;
        }
        row_ptr[t + 1] = 0;
        __syncthreads();
        float image_x = PIXEL_SIZE * (x - params->center_x);
        float image_y = PIXEL_SIZE * (y - params->center_y);

        double rho = static_cast<double>(
                         sqrt(static_cast<double>(image_x * image_x + image_y * image_y))) *
                     ARCSEC_TO_RAD;
        double eta = static_cast<double>(
            atan2(static_cast<double>(-image_x), static_cast<double>(image_y)) +
            static_cast<double>(params->roll_offset * DEG_TO_RAD));

        std::array<real, 2 * N_RAD_BINS + 2> radius_times;
        int radius_times_begin = 0;
        int radius_times_end = 0;
        std::array<real, N_THETA_BINS + 3> polar_times;
        int polar_times_begin = 0;
        int polar_times_end = 0;
        std::array<real, N_PHI_BINS + 3> azimuthal_times;
        int azimuthal_times_begin = 0;
        int azimuthal_times_end = 0;

        real entry_time = 0;
        real exit_time = 0;

        std::array<real, 2 * N_RAD_BINS + N_THETA_BINS + N_PHI_BINS + 8> times;
        int times_size = 0;

        auto sin_rho = sin(rho);
        auto cos_rho = cos(rho);

        Rotation rx(Axis::x, eta);

        Vector3d<Params_t> nearest_point = {
            params->dist_to_sun * sin_rho * sin_rho,
            static_cast<Params_t>(0.0),
            params->dist_to_sun * sin_rho * cos_rho};

        nearest_point = rx.rotate(nearest_point);
        nearest_point = params->r23.rotate(nearest_point);

        Vector3d<Params_t> los = {-cos_rho, static_cast<Params_t>(0.0), sin_rho};
        los = rx.rotate(los);
        los = params->r23.rotate(los);

        auto impact = nearest_point.norm();
        auto impact2 = nearest_point.norm2();

        real spacecraft_time = sqrt(params->dist_to_sun * params->dist_to_sun - impact2);

        if (los.dot(params->sun_to_obs_vec) < 0)
            spacecraft_time *= -1;

        int t_idx = 0;

        if ((sin_rho * params->dist_to_sun < INSTR_R_MAX) &&
            (sin_rho * params->dist_to_sun > INSTR_R_MIN) &&
            !(impact > R_MAX || (params->dist_to_sun < R_MIN && impact < 1 && spacecraft_time < 0)))
        {
            // Image_t pB_val = (*(pb_vector + IMAGE_SIZE * (x + k) + (y + l)) * params->b_scale + params->b_zero) * SCALE_FACTOR * 0.79; // match the oldest gold prelim
            auto pB_val = (*(pb_vector + IMAGE_SIZE * y + x) * params->b_scale + params->b_zero) * SCALE_FACTOR * time_scale_factor; // match the oldest gold prelim
            compute_crossing_times(params, nearest_point, los, impact, impact2, spacecraft_time,
                                   radius_times, radius_times_begin, radius_times_end,
                                   polar_times, polar_times_begin, polar_times_end,
                                   azimuthal_times, azimuthal_times_begin, azimuthal_times_end,
                                   entry_time, exit_time);

            times_size += 2;

            atomicAdd(&y_val, pB_val);
            t_idx = atomicAdd(&n_los, 1);
        }

        // std::sort(times.begin(), times.begin() + times_idx);

        times_size += std::max(radius_times_end - radius_times_begin, 0) +
                      std::max(polar_times_end - polar_times_begin, 0) +
                      std::max(azimuthal_times_end - azimuthal_times_begin, 0);

        // printf("%d %d %d %d %d\n", x, y, k, l, times_size);
        // if (x == 200 && y >= 300 && y <= 356)
        //     printf("%d %d %d %d %f %d\n", x, y, k, l, impact, times_size);

        int spacecraft_time_iter = 0;

        int row_nnz = 0;

        if (times_size > 0)
        {
            three_way_merge(times, radius_times, polar_times, azimuthal_times,
                            radius_times_begin, radius_times_end,
                            polar_times_begin, polar_times_end,
                            azimuthal_times_begin, azimuthal_times_end, entry_time, exit_time);
            spacecraft_time_iter = lower_bound(times.begin(), times.begin() + times_size, spacecraft_time) - times.begin();

            row_nnz = times_size - spacecraft_time_iter - 1;

            row_ptr[t_idx + 1] = row_nnz;
        }

        __syncthreads();

        bool odd = true;

        if (t == 0)
        {
            partial_sum(row_ptr.begin(), row_ptr.end(), row_ptr.begin());
        }

        __syncthreads();

        if (times_size > 0)
        {
            for (int i = 0; i < row_nnz; ++i)
            {
                col_idx_and_val[row_ptr[t_idx] + i] = calculate_index_and_values(nearest_point, los, impact2, times, i + spacecraft_time_iter + 1);
            }
        }

        __syncthreads();

        if (times_size > 0)
        {
            gpu_sort(col_idx_and_val.begin() + row_ptr[t_idx], col_idx_and_val.begin() + row_ptr[t_idx + 1]);
        }
        __syncthreads();
        int thread_count = BIN_SIZE / 2;
        int s = 1;

        while (thread_count > 0)
        {
            if (t < thread_count)
            {
                // use double buffer

                if (odd)
                {
                    if (2 * s * t + 2 * s <= n_los)
                    {
                        gpu_merge(col_idx_and_val.begin() + row_ptr[2 * s * t], col_idx_and_val.begin() + row_ptr[2 * s * t + s],
                                  col_idx_and_val.begin() + row_ptr[2 * s * t + s], col_idx_and_val.begin() + row_ptr[2 * s * t + 2 * s],
                                  col_idx_and_val2.begin() + row_ptr[2 * s * t]);
                    }
                    else if (2 * s * t + s <= n_los)
                    {
                        gpu_merge(col_idx_and_val.begin() + row_ptr[2 * s * t], col_idx_and_val.begin() + row_ptr[2 * s * t + s],
                                  col_idx_and_val.begin() + row_ptr[2 * s * t + s], col_idx_and_val.begin() + row_ptr[n_los],
                                  col_idx_and_val2.begin() + row_ptr[2 * s * t]);
                    }
                    else if (2 * s * t <= n_los)
                    {
                        gpu_copy(col_idx_and_val.begin() + row_ptr[2 * s * t], col_idx_and_val.begin() + row_ptr[n_los],
                                 col_idx_and_val2.begin() + row_ptr[2 * s * t]);
                    }
                }
                else
                {
                    if (2 * s * t + 2 * s <= n_los)
                    {
                        gpu_merge(col_idx_and_val2.begin() + row_ptr[2 * s * t], col_idx_and_val2.begin() + row_ptr[2 * s * t + s],
                                  col_idx_and_val2.begin() + row_ptr[2 * s * t + s], col_idx_and_val2.begin() + row_ptr[2 * s * t + 2 * s],
                                  col_idx_and_val.begin() + row_ptr[2 * s * t]);
                    }
                    else if (2 * s * t + s <= n_los)
                    {
                        gpu_merge(col_idx_and_val2.begin() + row_ptr[2 * s * t], col_idx_and_val2.begin() + row_ptr[2 * s * t + s],
                                  col_idx_and_val2.begin() + row_ptr[2 * s * t + s], col_idx_and_val2.begin() + row_ptr[n_los],
                                  col_idx_and_val.begin() + row_ptr[2 * s * t]);
                    }
                    else if (2 * s * t <= n_los)
                    {
                        gpu_copy(col_idx_and_val2.begin() + row_ptr[2 * s * t], col_idx_and_val2.begin() + row_ptr[n_los],
                                 col_idx_and_val.begin() + row_ptr[2 * s * t]);
                    }
                }
            }
            thread_count /= 2;
            s *= 2;
            odd = !odd;
            __syncthreads();
        }

        // merge same index
        if (t == 0 && n_los != 0)
        {
            int j = 0;
            int cur_idx = odd ? col_idx_and_val[0].first : col_idx_and_val2[0].first;
            if (!odd)
            {
                col_idx_and_val[0] = col_idx_and_val2[0];
            }
            for (int i = 1; i < row_ptr[n_los]; ++i)
            {
                int idx = odd ? col_idx_and_val[i].first : col_idx_and_val2[i].first;
                if (idx == cur_idx)
                {
                    col_idx_and_val[j].second += odd ? col_idx_and_val[i].second : col_idx_and_val2[i].second;
                }
                else
                {
                    ++j;
                    cur_idx = idx;
                    col_idx_and_val[j] = odd ? col_idx_and_val[i] : col_idx_and_val2[i];
                }
            }
            total_row_nnz = j + 1;
        }

        __syncthreads();

        if (n_los != 0)
        {
            // if (t == 0)
            // {
            //     int idx = params->file_idx * Y_SIZE + r;
            //     row_nnzs[idx] = total_row_nnz;
            //     assert(total_row_nnz != 0);
            //     y_output[r] = y_val / n_los;
            // }

            // int output_idx = -1;

            // output_idx = precompute_row_ptr[r];

            // for (int idx = t; idx < total_row_nnz; idx += BIN_SIZE)
            // {
            //     int bin_idx = col_idx_and_val[idx].first;
            //     int r_idx = bin_idx % N_RAD_BINS;
            //     int old_bin_count = atomicAdd(&bin_mask[bin_idx], 1);
            //     col_idx[output_idx + idx] = col_idx_and_val[idx].first;
            //     val[output_idx + idx] = col_idx_and_val[idx].second / n_los;
            //     assert(!std::isnan(val[output_idx + idx]));
            // }
            if (t == 0)
            {
                // assert(total_row_nnz != 0);
                y_output[r] = y_val / n_los;
                int output_idx = -1;

                int row_nnz2 = 0;

                output_idx = precompute_row_ptr[r];
                for (int idx = 0; idx < total_row_nnz; idx += 1)
                {
                    int bin_idx = col_idx_and_val[idx].first;
                    // int r_idx = bin_idx % N_RAD_BINS;
                    // int old_bin_count = atomicAdd(&bin_mask[bin_idx], 1);
                    // if (old_bin_count > 100)
                    // {
                    //     // printf("skip\n");
                    //     continue;
                    // }
                    // if (params->file_idx > static_cast<int>(params->n_files / 14.0 * 5) && old_bin_count > 500)
                    // {
                    //     continue;
                    // }
                    // if (params->file_idx > 14 && r_idx >= 10)
                    // {
                    //     continue;
                    // }
                    // if (params->file_idx > 28 && r_idx >= 5)
                    // {
                    //     continue;
                    // }
                    // if (old_bin_count > 500)
                    // if (old_bin_count > 8 * r_idx + 48)
                    // {
                    //     continue;
                    // }
                    col_idx[output_idx + idx] = col_idx_and_val[idx].first;
                    val[output_idx + idx] = col_idx_and_val[idx].second / n_los * time_scale_factor;
                    // assert(!std::isnan(val[output_idx + idx]));
                    ++row_nnz2;

                    // if (bin_mask[bin_idx] == 0)
                    // {
                    //     atomicAdd(&bin_mask[bin_idx], 1);
                    //     // printf("bin mask %d is %d\n", bin_idx, bin_mask[bin_idx]);
                    // }
                }
                int idx = params->file_idx * Y_SIZE + r;
                row_nnzs[idx] = row_nnz2;
            }
        }
        else
        {
            if (t == 0)
            {
                int idx = params->file_idx * Y_SIZE + r;
                row_nnzs[idx] = 0;
                y_output[r] = -1;
            }
        }
    }

    template <typename T>
    void y_computation(ImageParameters<T> *params, float *y_output)
    {
        // auto begin = stdc::stdc::high_resolution_clock::now();
        //  std::cerr << "y_computation\n";
        float y_norm = 0;
        for (int i = 0; i < ROW_SIZE; ++i)
        {
            for (int j = 0; j < ROW_SIZE; ++j)
            {
                int r = i * ROW_SIZE + j;
                Image_t pB_val = (*(params->pb_vector + r) * params->b_scale + params->b_zero) * SCALE_FACTOR;
                int idx = params->file_idx * Y_SIZE + r;
                y_output[idx] = pB_val;
                // y_norm += pB_val * pB_val;
            }
        }
        // std::cerr << "y_norm " << y_norm << '\n';

        // auto end = stdc::high_resolution_clock::now();
        // std::cerr << "time for CPU y computation " << stdc::duration_cast<stdc::milliseconds>(end - begin).count() << "ms\n";
    }

#if defined(COR)
    static constexpr bool binning_global = BIN_SCALE_FACTOR >= 2 && BINNING_FACTOR >= 4;
#elif defined(LASCO_C2)
    // static constexpr bool binning_global = BIN_SCALE_FACTOR >= 2 && BINNING_FACTOR >= 4;
    static constexpr bool binning_global = 2 * AROW_SIZE * BINNING_FACTOR * BINNING_FACTOR * sizeof(IndexValuePair<int, float>) >= 0xc000;
// static_assert(2 * AROW_SIZE * BINNING_FACTOR * BINNING_FACTOR * sizeof(IndexValuePair<int ,float>) == 0x10400);
#endif

#if defined(COR)
    using nnz_t = int32_t;
    using row_ptr_t = int32_t;
    using col_idx_t = int32_t;
#elif defined(LASCO_C2)
    using nnz_t = int32_t;
    using row_ptr_t = int32_t;
    using col_idx_t = int32_t;
#endif

    static constexpr bool benchmark = true;
    static constexpr bool compute_mask = false;
    static constexpr int N_STREAMS = !compute_mask ? 2 : 1;


    SparseMatrixAndImage<float> build_A_matrix_from_params(const std::vector<ImageParameters<Params_t>> &parameters, std::vector<int> &row_ptr_h, bool is_time_dependent = false)
    {
        Timer timer;
        timer.start();

        int n_files = parameters.size();

        std::vector<float *> val_h(n_files, nullptr);
        std::vector<int *> col_idx(n_files, nullptr);

        std::vector<int> precomputed_nnzs = {0};
        std::vector<int *> all_precompute_row_ptr;

        std::vector<Image_t *> pb_vector(n_files, nullptr);

        std::vector<ImageParameters<Params_t> *> params(n_files, nullptr);
        for (int fc = 0; fc < n_files; ++fc)
        {
            cudaMallocHost(&params[fc], sizeof(ImageParameters<Params_t>));
            cudaMemcpy(params[fc], &parameters[fc], sizeof(ImageParameters<Params_t>), cudaMemcpyHostToHost);
        }

        float *y_output_h = nullptr;
        if constexpr (BINNING_FACTOR != 1)
        {
            cudaMallocHost(&y_output_h, sizeof(float) * n_files * Y_SIZE);
        }
        else
        {
            y_output_h = static_cast<float *>(malloc(sizeof(float) * n_files * Y_SIZE));
        }

        for (int i = 0; i < n_files; ++i)
        {
            int *mem = nullptr;
            cudaMallocHost(&mem, sizeof(int) * (Y_SIZE + 1));
            all_precompute_row_ptr.push_back(mem);
        }

        timer.stop("CPU memory allocation");

        timer.start();

        if constexpr (BINNING_FACTOR == 1)
        {
            for (int fc = 0; fc < n_files; ++fc)
            {
                y_computation(params[fc], y_output_h);
                free(params[fc]->pb_vector);
                params[fc]->pb_vector = nullptr;
            }
        }

        timer.stop("Y normalization");

        timer.start();

        timer.start();
        // GPU allocation
        constexpr int BLOCK_SIZE = BINNING_FACTOR == 1 ? 8 : BINNING_FACTOR;
        dim3 grid(IMAGE_SIZE / BLOCK_SIZE, IMAGE_SIZE / BLOCK_SIZE, 1);
        dim3 block(BLOCK_SIZE, BLOCK_SIZE, 1);

        std::cerr << BLOCK_SIZE << '\n';

        int device_count = 1;
        cudaGetDeviceCount(&device_count);
        std::cerr << "device count " << device_count << '\n';

        int *row_nnzs_d_public = nullptr;

        cudaMalloc(&row_nnzs_d_public, sizeof(int) * n_files * Y_SIZE);

        std::vector<int> bin_mask_h(N_BINS, 0);

        timer.stop("GPU0 memory allocation");

        auto gpu_computation = [&](int begin, int end, int device_id)
        {
            cudaSetDevice(device_id);
            std::cerr << "set device " << device_id << '\n';
            cudaCheckError();

            constexpr int grid_size = Y_SIZE / 1024 + 1;

            cudaEvent_t event_begin;
            cudaEvent_t event_end;

            auto timing_begin = [&](cudaStream_t stream = static_cast<cudaStream_t>(0))
            {
                if (benchmark)
                {
                    cudaEventRecord(event_begin, stream);
                }
            };

            auto timing_end = [&](std::string_view message, cudaStream_t stream = static_cast<cudaStream_t>(0))
            {
                if (benchmark)
                {
                    // cudaStreamSynchronize(stream);
                    float gpu_time;
                    cudaEventRecord(event_end, stream);
                    cudaEventSynchronize(event_end);
                    cudaEventElapsedTime(&gpu_time, event_begin, event_end);
                    std::cerr << message << gpu_time << "ms\n";
                }
            };

            if (benchmark)
            {
                cudaEventCreate(&event_begin);
                cudaEventCreate(&event_end);
            }

            timing_begin();

            std::array<Image_t *, N_STREAMS> pb_vector_d;

            std::array<float *, N_STREAMS> y_output_d;

            std::array<float *, N_STREAMS> val_d;
            std::array<int *, N_STREAMS> col_idx_d;

            std::array<int *, N_STREAMS> row_nnzs_d;

            std::array<int *, N_STREAMS> precompute_row_ptr_d;

            std::array<ImageParameters<Params_t> *, N_STREAMS> params_d;

            std::array<IndexValuePair<int, float> *, N_STREAMS> buffer_d;

            std::array<int *, N_STREAMS> block_sums_d;

            int *bin_mask_d = nullptr;

            if constexpr (compute_mask)
            {
                cudaMalloc(&bin_mask_d, sizeof(int) * N_BINS);
                cudaMemset(bin_mask_d, 0, sizeof(int) * N_BINS);
            }

            for (int i = 0; i < N_STREAMS; ++i)
            {
                cudaMalloc(&params_d[i], sizeof(ImageParameters<Params_t>));

                cudaMalloc(&row_nnzs_d[i], sizeof(int) * n_files * Y_SIZE);
                cudaMemset(row_nnzs_d[i], 0, sizeof(int) * n_files * Y_SIZE);

                cudaMalloc(&precompute_row_ptr_d[i], sizeof(int) * (Y_SIZE + 1));
                cudaMemset(precompute_row_ptr_d[i], 0, sizeof(int));

                if constexpr (BINNING_FACTOR != 1)
                {
                    cudaMalloc(&pb_vector_d[i], sizeof(Image_t) * IMAGE_SIZE * IMAGE_SIZE);
                    cudaMalloc(&y_output_d[i], sizeof(float) * Y_SIZE);
                }

                if constexpr (binning_global)
                {
                    // CHECK_CUDA() ;
                    cudaMalloc(&buffer_d[i], sizeof(IndexValuePair<int, float>) * Y_SIZE * 2 * AROW_SIZE * BIN_SIZE);
                }

                cudaMalloc(&block_sums_d[i], sizeof(int) * grid_size);
            }

            val_d.fill(nullptr);
            col_idx_d.fill(nullptr);

            // stream initilization
            std::array<cudaStream_t, N_STREAMS> streams;
            for (int i = 0; i < N_STREAMS; ++i)
            {
                CHECK_CUDA(cudaStreamCreate(&streams[i]));
            }

            auto error = cudaDeviceSynchronize();
            std::cerr << "device id " << device_id << " " << cudaGetErrorName(error) << '\n';

            timing_end("GPU memory allocation ");
            std::array<int, N_STREAMS> cur_nnz;

            for (int fc = begin; fc < end; ++fc)
            {
                std::cerr << "GPU loop " << fc << " device id " << device_id << '\n';
                int stream_idx = fc % N_STREAMS;
                auto cur_stream = streams[stream_idx];
                int total_precomputed_nnz = 0;

                timing_begin(cur_stream);
                error = cudaMemcpyAsync(params_d[stream_idx], params[fc], sizeof(ImageParameters<Params_t>), cudaMemcpyHostToDevice, cur_stream);

                // std::vector<int> test(Y_SIZE + 1, 0);

                if constexpr (BINNING_FACTOR == 1)
                    precompute_nnz<<<grid, block, 0, cur_stream>>>(params_d[stream_idx], precompute_row_ptr_d[stream_idx]);
                else
                {
                    cudaMemsetAsync(precompute_row_ptr_d[stream_idx], 0, (Y_SIZE + 1) * sizeof(int), cur_stream);
                    precompute_nnz_with_binning<<<grid, block, 0, cur_stream>>>(params_d[stream_idx], precompute_row_ptr_d[stream_idx]);
                    cudaMemcpyAsync(pb_vector_d[stream_idx], params[fc]->pb_vector, sizeof(Image_t) * IMAGE_SIZE * IMAGE_SIZE, cudaMemcpyHostToDevice, cur_stream);
                }

                scan<<<grid_size, 512, 0, cur_stream>>>(precompute_row_ptr_d[stream_idx], block_sums_d[stream_idx], Y_SIZE + 1);

                single_scan<<<1, 1, 0, cur_stream>>>(block_sums_d[stream_idx], grid_size);

                add<<<grid_size, 1024, 0, cur_stream>>>(precompute_row_ptr_d[stream_idx], precompute_row_ptr_d[stream_idx], block_sums_d[stream_idx], Y_SIZE + 1);

                cudaStreamSynchronize(cur_stream);

                cudaMemcpyAsync(&total_precomputed_nnz, precompute_row_ptr_d[stream_idx] + Y_SIZE, sizeof(int), cudaMemcpyDeviceToHost, cur_stream);
                timing_end("time for memcpy from host to device ", cur_stream);

                // std::cerr << '\n';
                if (fc == begin)
                    timing_begin(cur_stream);
                std::cerr << "precompute non zero " << total_precomputed_nnz << '\n';
                cudaMallocHost(&val_h[fc], sizeof(float) * total_precomputed_nnz);
                cudaMallocHost(&col_idx[fc], sizeof(int) * total_precomputed_nnz);

                if (!val_d[stream_idx] && !col_idx_d[stream_idx])
                {
                    // if crash, multiply a coefficient to the allocated memory
                    constexpr float precompute_scale_factor = 1.2;
                    cudaMallocAsync(&val_d[stream_idx], sizeof(float) * total_precomputed_nnz * precompute_scale_factor, cur_stream);
                    cudaMallocAsync(&col_idx_d[stream_idx], sizeof(int) * total_precomputed_nnz * precompute_scale_factor, cur_stream);
                    cur_nnz[stream_idx] = 2 * total_precomputed_nnz;
                }
                else
                {
                    if (total_precomputed_nnz > cur_nnz[stream_idx])
                    {
                        std::cerr << "previous nnz " << cur_nnz[stream_idx] << " total_precomputed_nnz " << total_precomputed_nnz << '\n';
                        cudaFree(val_d[stream_idx]);
                        cudaFree(col_idx_d[stream_idx]);
                        int new_nnz = total_precomputed_nnz * 1.5;
                        cudaMallocAsync(&val_d[stream_idx], sizeof(float) * new_nnz, cur_stream);
                        cudaMallocAsync(&col_idx_d[stream_idx], sizeof(int) * new_nnz, cur_stream);
                        cur_nnz[stream_idx] = new_nnz;
                    }
                }
                if (fc == begin)
                    timing_end("First iteration precomputation allocation ", cur_stream);

                timing_begin(cur_stream);

                cudaCheckStreamError(cur_stream);

                if constexpr (BINNING_FACTOR == 1)
                {
                    build_sub_matrix<<<grid, block, 0, cur_stream>>>(params_d[stream_idx], val_d[stream_idx], row_nnzs_d[stream_idx], col_idx_d[stream_idx], precompute_row_ptr_d[stream_idx]);
                }
                else
                {
                    if constexpr (compute_mask)
                    {
                        build_sub_matrix_with_mask<<<grid, block, 0, cur_stream>>>(params_d[stream_idx], pb_vector_d[stream_idx], y_output_d[stream_idx], val_d[stream_idx], row_nnzs_d[stream_idx], col_idx_d[stream_idx], precompute_row_ptr_d[stream_idx], bin_mask_d);
                    }
                    else if constexpr (binning_global)
                    {
                        build_sub_matrix_with_binning_global<<<grid, block, 0, cur_stream>>>(params_d[stream_idx], pb_vector_d[stream_idx], y_output_d[stream_idx], val_d[stream_idx], row_nnzs_d[stream_idx], col_idx_d[stream_idx], precompute_row_ptr_d[stream_idx], buffer_d[stream_idx]);
                    }
                    else
                    {
                        build_sub_matrix_with_binning<<<grid, block, 0, cur_stream>>>(params_d[stream_idx], pb_vector_d[stream_idx], y_output_d[stream_idx], val_d[stream_idx], row_nnzs_d[stream_idx], col_idx_d[stream_idx], precompute_row_ptr_d[stream_idx]);
                    }
                }
                error = cudaStreamSynchronize(cur_stream);
                std::cerr << cudaGetErrorName(error) << '\n';
                timing_end("time for kernel run ", cur_stream);

                timing_begin(cur_stream);

                cudaMemcpyAsync(val_h[fc], val_d[stream_idx], total_precomputed_nnz * sizeof(float), cudaMemcpyDeviceToHost, cur_stream);
                cudaMemcpyAsync(col_idx[fc], col_idx_d[stream_idx], total_precomputed_nnz * sizeof(int), cudaMemcpyDeviceToHost, cur_stream);
                cudaMemcpyAsync(all_precompute_row_ptr[fc], precompute_row_ptr_d[stream_idx], sizeof(int) * (Y_SIZE + 1), cudaMemcpyDeviceToHost, cur_stream);
                if constexpr (BINNING_FACTOR != 1)
                {
                    cudaMemcpyAsync(y_output_h + fc * Y_SIZE, y_output_d[stream_idx], sizeof(float) * Y_SIZE, cudaMemcpyDeviceToHost, cur_stream);
                }
                timing_end("time for copying memory from device to host in loop ", cur_stream);
                // std::cerr << "GPU loop " << fc << " device id " << device_id << " " << cudaGetErrorName(cudaGetLastError()) << '\n';
            }

            timing_begin();
            error = cudaDeviceSynchronize();
            std::cerr << "device id " << device_id << " " << cudaGetErrorName(error) << '\n';

            std::cerr << "compute kernel ends\n";

            // merge row_nnzs_d
            for (int fc = begin; fc < end; ++fc)
            {
                int stream_idx = fc % N_STREAMS;
                auto error = cudaMemcpyPeer(row_nnzs_d_public + fc * Y_SIZE, 0, row_nnzs_d[stream_idx] + fc * Y_SIZE, device_id, sizeof(int) * Y_SIZE);
                // std::cerr << "cudaMemcpyPeer result " << cudaGetErrorName(error) << "\n";
            }
            timing_end("Sync + MemcpyPeer ");

            // GPU deallocation
            // timing_begin();
            timing_begin();

            if constexpr (compute_mask)
            {
                cudaMemcpy(bin_mask_h.data(), bin_mask_d, sizeof(int) * N_BINS, cudaMemcpyDeviceToHost);
                cudaFree(bin_mask_d);
                // for (int i : bin_mask_h)
                // {
                //     std::cerr << i << ' ';
                // }
                // std::cerr << '\n';
            }

            for (int i = 0; i < N_STREAMS; ++i)
            {
                cudaFree(params_d[i]);
                cudaFree(precompute_row_ptr_d[i]);
                cudaFree(val_d[i]);
                cudaFree(col_idx_d[i]);
                cudaFree(row_nnzs_d[i]);
                if constexpr (BINNING_FACTOR != 1)
                {
                    cudaFree(pb_vector_d[i]);
                    cudaFree(y_output_d[i]);
                }
                if constexpr (binning_global)
                {
                    cudaFree(buffer_d[i]);
                }
                cudaFree(block_sums_d[i]);
            }

            for (int i = 0; i < N_STREAMS; ++i)
            {
                cudaStreamDestroy(streams[i]);
            }
            timing_end("deallocation ");

            if (benchmark)
            {
                cudaEventDestroy(event_begin);
                cudaEventDestroy(event_end);
            }

            cudaCheckError();
        };

        timer.start();

        std::vector<std::thread> thread_pool;
        for (int i = 0; i < device_count; ++i)
        {
            int per_thread_file_count = (n_files - 1) / device_count + 1;
            int begin = per_thread_file_count * i;
            int end = std::min(n_files, begin + per_thread_file_count);
            thread_pool.push_back(std::thread(gpu_computation, begin, end, i));
        }

        timer.stop("Thread Spawn");

        for (auto &t : thread_pool)
        {
            t.join();
        }

        cudaCheckError();

        if constexpr (BINNING_FACTOR != 1)
        {
            for (int fc = 0; fc < n_files; ++fc)
            {
                free(params[fc]->pb_vector);
            }
        }

        timer.stop("GPU compuation");

        timer.start();

        // std::vector<int> row_ptr_h(n_files * Y_SIZE + 1, 0);

        int grid_size = (n_files * Y_SIZE - 1) / 1024 + 1;
        std::cerr << "grid size " << grid_size << '\n';
        int *block_sums_d_public = nullptr;
        cudaMalloc(&block_sums_d_public, sizeof(int) * grid_size);
        cudaMemset(block_sums_d_public, 0, sizeof(int) * grid_size);

        scan<<<grid_size, 512>>>(row_nnzs_d_public, block_sums_d_public, n_files * Y_SIZE);
        cudaCheckError();

        single_scan<<<1, 1>>>(block_sums_d_public, grid_size);
        cudaCheckError();

        add<<<grid_size, 1024>>>(row_nnzs_d_public, row_nnzs_d_public, block_sums_d_public, n_files * Y_SIZE);
        cudaCheckError();

        // leave a zero at head for convenience
        cudaMemcpy(row_ptr_h.data() + 1, row_nnzs_d_public, sizeof(int) * (n_files * Y_SIZE), cudaMemcpyDeviceToHost);

        assert(std::is_sorted(row_ptr_h.begin(), row_ptr_h.begin() + n_files * Y_SIZE + 1));

        // std::cerr << "copy row idx back " << cudaGetErrorName(cudaDeviceSynchronize()) << '\n';

        // all GPU data has been copied to CPU here, free GPU resource
        cudaFree(block_sums_d_public);
        cudaFree(row_nnzs_d_public);

        // number of non-zero
        long long total_precomputed_nnz = 0;
        for (int i = 0; i < n_files; ++i)
        {
            total_precomputed_nnz += all_precompute_row_ptr[i][Y_SIZE];
            // std::cerr << "i " << i << " nnz " <<  total_precomputed_nnz << '\n';
            if (!std::is_sorted(all_precompute_row_ptr[i], all_precompute_row_ptr[i] + Y_SIZE + 1))
            {
                std::cerr << "precompute indices " << i << " unordered\n";
                for (int j = 0; j < Y_SIZE; ++j)
                {
                    if (all_precompute_row_ptr[i][j] > all_precompute_row_ptr[i][j + 1])
                    {
                        std::cerr << "unorderd position j " << j << '\n';
                        break;
                    }
                }
                std::exit(-1);
            }
        }

        std::cerr << "total precompute nnz " << total_precomputed_nnz << '\n';

        SparseMatrixAndImage<float> result(n_files * Y_SIZE, total_precomputed_nnz);

        int empty_count = 0;
        std::vector<int> hist(N_RAD_BINS, 0);
        if constexpr (compute_mask)
        {
            for (int i = 0; i < N_BINS; ++i)
            {
                if (bin_mask_h[i] == 0)
                {
                    ++empty_count;
                    bin_mask_h[i] = -1;
                    hist[i % N_RAD_BINS]++;
                    // std::cerr << "bin i not passed by any LOS\n";
                }
                else
                {
                    bin_mask_h[i] = empty_count;
                }
            }
            // std::cerr << empty_count << " bins not passed by any LOS\n";
            // for (int i = 0; i < N_RAD_BINS; ++i)
            // {
            //     std::cerr << hist[i] << " empty bins in rad " << i << '\n';
            // }
        }

        int n_rows = 0;
        nnz_t nnz = 0;
        nnz_t prev_nnz = 0;

        result.row_ptr_h[0] = 0;

        float norm_y_all = 0;

        for (int j = 0; j < n_files; ++j)
        {
            for (int i = 0; i < Y_SIZE; ++i)
            {
                int row_n = j * Y_SIZE + i;
                int row_nnz = row_ptr_h[row_n + 1] - row_ptr_h[row_n];

                if (row_nnz <= 0) // skip the empty row
                    continue;

                result.y_h[n_rows] = y_output_h[row_n];
                norm_y_all += y_output_h[row_n] * y_output_h[row_n];
                // std::cerr << "Row " << row_n << '\n';
                result.row_ptr_h[n_rows + 1] = row_ptr_h[row_n + 1];
                ++n_rows;
                // std::cerr << "row_nnz " << row_nnz << " nnz " << nnz << '\n';
                memcpy(result.col_idx_h + nnz, reinterpret_cast<char *>(col_idx[j] + all_precompute_row_ptr[j][i]), sizeof(int) * row_nnz);
                memcpy(result.val_h + nnz, reinterpret_cast<char *>(val_h[j] + all_precompute_row_ptr[j][i]), sizeof(float) * row_nnz);
                // if constexpr (compute_mask)
                // {
                //     for (int k = nnz; k < row_nnz + nnz; ++k)
                //     {
                //         int col_idx = result.col_idx_h[k];
                //         if (bin_mask_h[col_idx] == -1)
                //         {
                //             std::cerr << "wrong deleted column\n";
                //             std::exit(-1);
                //         }
                //         else
                //         {
                //             result.col_idx_h[k] -= bin_mask_h[col_idx];
                //         }
                //     }
                // }
                nnz += row_nnz;
            }
            if (is_time_dependent && j > 0)
            {
                for (int i = prev_nnz; i < nnz; ++i)
                {
                    result.col_idx_h[i] += j * N_BINS;
                }
            }
            prev_nnz = nnz;
        }

        std::cerr << "Norm of y_all " << norm_y_all << '\n';
        std::cerr << n_rows << ' ' << nnz << '\n';

        timer.start();

        // check if there is NaN in result
        // for (int i = 0; i < nnz; ++i)
        // {
        //     if (std::isnan(result.val_h[i]))
        //     {
        //         std::cerr << "NaN in matrix!\n";
        //         std::exit(-1);
        //     }
        // }

        if constexpr (BINNING_FACTOR != 1)
        {
            cudaFreeHost(y_output_h);
        }
        else
        {
            free(y_output_h);
        }

        for (int i = 0; i < n_files; ++i)
        {
            cudaFreeHost(val_h[i]);
            cudaFreeHost(col_idx[i]);
        }

        for (auto i : all_precompute_row_ptr)
        {
            cudaFreeHost(i);
        }

        for (int i = 0; i < n_files; ++i)
        {
            cudaFreeHost(params[i]);
        }

        cudaCheckError();

        timer.stop("free the memory");

        timer.stop("Build the sparse matrix");

        // set the real nnz and n_rows
        result.nnz = nnz;
        result.n_rows = n_rows;
        return result;
    }

    SparseMatrixAndImage<float> build_A_matrix_with_time_factor(const std::vector<std::string> &sub_matrix_filenames, std::string_view ref_filename)
    {
        int n_files = sub_matrix_filenames.size();
        std::vector<ImageParameters<Params_t>> params(n_files);
        compute_parameters_from_files(sub_matrix_filenames, params, ref_filename);
        fetch_pb_vector_from_files(sub_matrix_filenames, params);
        std::vector<int> row_ptr_h(n_files * Y_SIZE + 1, 0);
        return build_A_matrix_from_params(params, row_ptr_h);
    }

    SparseMatrixAndImage<float> build_A_matrix(const std::vector<std::string> &sub_matrix_filenames)
    {
        int n_files = sub_matrix_filenames.size();
        Timer timer;
        timer.start();
        auto params = get_all_parameters_from_files(sub_matrix_filenames);
        timer.stop("Parameter computations");
        std::vector<int> row_ptr_h(n_files * Y_SIZE + 1, 0);
        return build_A_matrix_from_params(params, row_ptr_h);
    }

    SparseMatrixAndImage<float> build_A_matrix_with_projection(const std::vector<std::string> &sub_matrix_filenames, std::string_view projection_dir)
    {
        int n_files = sub_matrix_filenames.size();
        std::vector<ImageParameters<Params_t>> params(n_files);
        compute_parameters_from_files(sub_matrix_filenames, params);
        for (int i = 0; i < sub_matrix_filenames.size(); ++i)
        {
            std::fstream projection_file;
            if (projection_file.open(std::string(projection_dir) + sub_matrix_filenames[i]); !projection_file.is_open())
            {
                std::cerr << "failed to open the projection file!\n";
                std::exit(-1);
            }
            params[i].pb_vector = static_cast<Image_t *>(malloc(sizeof(Image_t) * IMAGE_SIZE * IMAGE_SIZE));
            projection_file.read(reinterpret_cast<char *>(params[i].pb_vector), sizeof(Image_t) * IMAGE_SIZE * IMAGE_SIZE);
        }
        std::vector<int> row_ptr_h(n_files * Y_SIZE + 1, 0);
        return build_A_matrix_from_params(params, row_ptr_h);
    }

    SparseMatrixAndImage<float> build_time_dependent_A_matrix(const std::vector<std::string> &sub_matrix_filenames)
    {
        int n_files = sub_matrix_filenames.size();
        auto params = get_all_parameters_from_files(sub_matrix_filenames);
        std::vector<int> row_ptr_h(n_files * Y_SIZE + 1, 0);
        return build_A_matrix_from_params(params, row_ptr_h, true);
    }
}
