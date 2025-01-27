#include <cmath>
#include <cstdio>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
#include <string_view>
#include <vector>
#include <array>
#include <algorithm>
#include <unordered_map>
#include <chrono>
#include <numeric>
#include <fitsfile.h>
#include "sparse_matrix.hpp"
#include "vector_maths.hpp"
#include "constants.hpp"
#include "output.hpp"
#include "utility.hpp"
#include "operators.hpp"
#include "timer.hpp"

// #define NDEBUG
#include <cassert>

constexpr std::string_view output_dir = "../output/spherical_v3/";

// constexpr bool high_resolution = cudaSolarTomography::N_BINS > 100000;
// constexpr bool high_resolution = true;
constexpr bool high_resolution = true;

constexpr int algorithm = 0;

int main(int argc, char **argv)
{
    using namespace cudaSolarTomography;

    Timer timer;

    std::fstream config_file;
    // std::cerr << std::setprecision(20) << std::numbers::pi << '\n';
    int file_begin = -1;
    int file_end = -1;
    if (argc == 3)
    {
        file_begin = atoi(argv[1]);
        file_end = atoi(argv[2]);
    }

    if (config_file.open(std::string(config_dir)); !config_file.is_open())
    {
        std::cerr << "Failed to open the config file " << config_dir << "\n";
        return -1;
    }

    std::string file_count_str;
    std::getline(config_file, file_count_str);
    int file_count = std::stoi(file_count_str);
    std::cerr << "total number of files " << file_count << '\n';

    std::vector<std::string> sub_matrix_filenames;
    for (int i = 0; i < file_count; ++i)
    {
        std::string sub_matrix_filename;
        std::getline(config_file, sub_matrix_filename);
        sub_matrix_filenames.push_back(sub_matrix_filename);
    }
    config_file.close();

    // first calculate some constants that are invariant in the loop

    // unit solar pole (1, delta_pole, alpha_pole) (in spherical coordinate)

    Vector3d<double> solar_pole = {
        cos(DELTA_POLE) * cos(ALPHA_POLE),
        cos(DELTA_POLE) * sin(ALPHA_POLE),
        sin(DELTA_POLE)};

// the final A matrix and y vector of all images
#if defined(COR) || defined(LASCO_C2)
    SparseMatrixDyn A(file_count);
#endif
    std::vector<float> all_y;
    all_y.reserve(file_count * Y_SIZE); // maximum possible y
    std::vector<int> all_y_idx;
    all_y_idx.reserve(file_count * Y_SIZE);
    // std::vector<float> all_delta;
    std::vector<int> block_idx;
    block_idx.reserve(file_count + 1);
    block_idx.push_back(0);

    static std::array<std::array<float, IMAGE_SIZE>, IMAGE_SIZE> rhos;
    static std::array<std::array<float, IMAGE_SIZE>, IMAGE_SIZE> etas;

    if (argc != 3)
    {
        file_begin = 0;
        file_end = file_count;
    }

    int count = 1;

    int max_phi_buckets = -1;

    // static std::array<float, N_BINS> A_row = {0.0};
    // std::vector<float> A_row(N_BINS, 0.0);
    std::vector<float> A_row;
    std::vector<std::pair<int, float>> A_row_index_and_value;
    if constexpr (high_resolution)
    {
        A_row_index_and_value.reserve(AROW_SIZE);
    }
    else
    {
        A_row.resize(N_BINS);
    }
    static std::array<double, 2 * N_RAD_BINS + 2> radius_times;
    static std::array<double, N_THETA_BINS + 3> polar_times;
    static std::array<double, N_PHI_BINS + 3> azimuthal_times;

    std::vector<int> bin_count(N_BINS, 0);
    int cnt = 0;
    int cnt2 = 0;
    int row_cnt = 0;

    for (int fc = file_begin; fc < file_end; ++fc)
    {
        timer.start();
        /*std::fstream phi_buckets_file;

        if (phi_buckets_file.open("buckets_count_" + std::to_string(fc) + ".txt", std::fstream::out | std::fstream::trunc); !phi_buckets_file.is_open())
        {
            std::cerr << "Failed to open the buckets count file " << fc << '\n';
            return -1;
        }
        std::array<int, N_PHI_BINS> buckets;*/
        // static std::array<std::vector<std::pair<int, int>>, N_BINS> bin_to_pixels;

        // std::fstream bin_pixel_file;
        // if (bin_pixel_file.open("bin_pixel_" + std::to_string(fc), std::fstream::out | std::fstream::trunc); !phi_buckets_file.is_open())
        // {
        //     std::cerr << "Failed to open the buckets count file " << fc << '\n';
        //     return -1;
        // }

        std::cerr << fc << std::endl;
        std::fstream fits_file;

        std::cerr << "current file " << sub_matrix_filenames[fc] << " " << fc + 1 << " of " << file_count << " files\n";

        std::string fits_filename = std::string(fits_dir) + sub_matrix_filenames[fc];
        if (fits_file.open(fits_filename); !fits_file.is_open())
        {
            std::cerr << "Failed to open the FITS file " << sub_matrix_filenames[fc] << "\n";
            return -1;
        }
        fits_file.close();

        // build the sub matrix here
        char *fits_image = nullptr;
        char *fits_header = nullptr;

        int max_header_size = 0; // maximum number of bytes in FITS header
        int header_size = 0;     // number of bytes in FITS header

        // using Image_t = float;

        if (fits_header = fitsrhead(const_cast<char *>(fits_filename.c_str()), &max_header_size, &header_size); fits_header)
        {
            if (fits_image = fitsrimage(const_cast<char *>(fits_filename.c_str()), header_size, fits_header); !fits_image)
            {
                std::cerr << "failed to read the FITS iamge " << sub_matrix_filenames[fc] << '\n';
                free(fits_image);
                return -1;
            }
        }
        else
        {
            std::cerr << "failed to read the FITS header " << sub_matrix_filenames[fc] << '\n';
            free(fits_header);
            return -1;
        }

        auto pb_vector = reinterpret_cast<Image_t *>(fits_image);
        double y_norm = 0;
        for (int i = 0; i < ROW_SIZE * ROW_SIZE; ++i)
        {
            y_norm += pb_vector[i] * pb_vector[i];
            if (pb_vector[i] < 0)
            {
                std::cerr << "negative pB!\n";
            }
        }
        std::cerr << "y_norm " << y_norm << '\n';

        double roll_offset = 180;
        if (hgetr8(fits_header, "CROTA1", &roll_offset))
        {
            std::cerr << "CROTA1 roll_offset read\n";
            std::cerr << roll_offset << '\n';
        }
        else if (hgetr8(fits_header, "INITANG1", &roll_offset))
        {
            std::cerr << "INITANG1 roll_offset read\n";
            std::cerr << roll_offset << '\n';
            // roll_offset -= 0.5;
        }
        else if (hgetr8(fits_header, "ROLLANGL", &roll_offset))
        {
            std::cerr << "ROLLANGL roll_offset read\n";
            std::cerr << roll_offset << '\n';
            // roll_offset -= 0.5;
        }
        else if (hgetr8(fits_header, "CROTA", &roll_offset))
        {
            std::cerr << "CROTA roll_offset read\n";
            std::cerr << roll_offset << '\n';
        }

#if defined(LASCO_C2)
        // roll_offset -= 0.5;
#endif
        std::cerr << "roll_offset " << roll_offset << '\n';

        // get the arcsec/pixel conversion factor from the FITS file
        double pixel_x_size = 0;
        double pixel_n_rows = 0;
        double b_scale = 1;
        double b_zero = 0;

        hgetr8(fits_header, "CDELT1", &pixel_x_size);
        hgetr8(fits_header, "CDELT2", &pixel_n_rows);
        hgetr8(fits_header, "BSCALE", &b_scale);
        hgetr8(fits_header, "BZERO", &b_zero);

        if (fabs(pixel_x_size - pixel_n_rows) > 0.0001)
        {
            std::cerr << "the width and length of pixel don't match\n";
            return -1;
        }

        double pixel_size = pixel_x_size;
// get the orbit here
#if defined(COR) || defined(LASCO_C2)
        auto [sun_to_obs_vec, carrington_longitude] = get_orbit(fits_header);
        std::cerr << "sun-observer vector " << sun_to_obs_vec << " carrington longitude " << carrington_longitude << '\n';
#elif defined(LASCO_C2)
        auto [sun_to_obs_vec, carrington_longitude] = get_orbit(fits_header, fits_filename, solar_pole);
        std::cerr << "sun-observer vector " << sun_to_obs_vec << " carrington longitude " << carrington_longitude << '\n';
        double header_dist_to_sun = 0;
        hgetr8(fits_header, "R_SOHO", &header_dist_to_sun);
#endif
        double dist_to_sun = sun_to_obs_vec.norm();
        std::cerr << "dist_to_sun " << dist_to_sun << '\n';
#ifdef LASCO_C2
        // std::cerr << "header dist_to_sun " << header_dist_to_sun << '\n';
#endif

        double center_x = 0;
        double center_y = 0;
        hgetr8(fits_header, "CRPIX1", &center_x);
        hgetr8(fits_header, "CRPIX2", &center_y);

        // change the index from 1-based to 0-based
        --center_x;
        --center_y;

        std::cerr << "pixel_size " << pixel_size << '\n';
        std::cerr << "center_x " << center_x << " center_y " << center_y << '\n';

        std::array<float, IMAGE_SIZE> image_x;
        std::array<float, IMAGE_SIZE> image_y;

        // calculate the arcsec from the center to each pixel

        for (int i = 0; i < IMAGE_SIZE; ++i)
        {
            image_x[i] = PIXEL_SIZE * (i - center_x);
            image_y[i] = PIXEL_SIZE * (i - center_y);
            // image_x[i] = pixel_size * (i - center_x);
            // image_y[i] = pixel_size * (i - center_y);
        }

        for (int x = 0; x < IMAGE_SIZE; ++x)
        {
            for (int y = 0; y < IMAGE_SIZE; ++y)
            {
                rhos[x][y] = static_cast<float>(
                                 sqrt(static_cast<double>(image_x[x] * image_x[x] + image_y[y] * image_y[y]))) *
                             ARCSEC_TO_RAD;
                // std::cerr << i << ' ' << j << ' ' << image_x[i] << ' ' << image_y[j] << ' ' << sqrt(static_cast<double>(image_x[i] * image_x[i] + image_y[j] * image_y[j])) / 3600 << '\n';
                etas[x][y] = static_cast<float>(
                    atan2(static_cast<double>(-image_x[x]), static_cast<double>(image_y[y])) +
                    static_cast<float>(roll_offset * DEG_TO_RAD));
            }
        }

        /*
            the rho is the angle between the observer to the sun vector and the pixel to the observer vector
            (pixel is in the image plane behind the observer, which inverse)

                    pixel
                    |\
                    | \
                    |  \  / eta
                    |   \/
                    ------center (on the image plane)

            eta is the angle in the image, between the horizon vector from center and the pixel to the center vector
        */

        // zero the y component of sun_to_obs_vec
        std::cerr << "sun_to_obs_vec1 " << sun_to_obs_vec << '\n';
        Rotation rz(Axis::z, -atan2(sun_to_obs_vec.y, sun_to_obs_vec.x));
        auto sun_to_obs_vec1 = rz.rotate(sun_to_obs_vec);

        // zero the z componenet of rz * sun_to_obs_vec
        Rotation ry(Axis::y, -atan2(sun_to_obs_vec1.z, sun_to_obs_vec1.x));
        Rotation ryz = ry.compose(rz);
        // zero the y component of solor_pole
        auto solar_pole1 = ryz.rotate(solar_pole);
        Rotation rx(Axis::x, atan2(solar_pole1.y, solar_pole1.z));
        Rotation r12 = rx.compose(ryz);
        auto sun_to_obs_vec2 = r12.rotate(sun_to_obs_vec);
        auto solar_pole2 = r12.rotate(solar_pole);

        std::cerr << "solar pole2 " << solar_pole2 << "\nsun_to_obs_vec2 " << sun_to_obs_vec2 << '\n';
        std::cerr << "bscale " << b_scale << " bzero " << b_zero << '\n';
        // after rotation, the final result makes the sun_to_obs_vec lies on the x-axis, solar pole lies on the xz-plane

        /*
                         (z axis)
                            ^
                            |
                            |   ^(y axis)
                            |  /
                            | /
                            |/
            observer--------/--------------> (x axis)
                          (sun)

        */

        ry = Rotation(Axis::y, atan2(solar_pole2.x, solar_pole2.z));
        rz = Rotation(Axis::z, carrington_longitude);

        Rotation r23 = rz.compose(ry);
        sun_to_obs_vec = r23.rotate(sun_to_obs_vec2);

        std::cerr << "solar pole 3 " << r23.rotate(solar_pole2) << '\n'
                  << "sun_to_obs_vec3 " << r23.rotate(sun_to_obs_vec2) << '\n';

        std::cerr << "Rotation matrix r23\n"
                  << r23 << '\n';

        // start to build the y and the matrix A (in row sparse format)
        // this part should be parallelized

        /*
        Use the pixel to find out all rays we want to use for calculating the matrix H (called A here)
        */

        constexpr int mod_n = IMAGE_SIZE % BINNING_FACTOR == 0 ? 0 : BINNING_FACTOR; // throw the last bin if the bin is not full

        using namespace std::chrono;
        int instr_lim = 0;
        // i iterates the y dimension
        // j iterates the x diemnsion
        for (int i = 0; i < IMAGE_SIZE - mod_n; i += BINNING_FACTOR)
        {
            for (int j = 0; j < IMAGE_SIZE - mod_n; j += BINNING_FACTOR)
            {

                int n_los = 0;
                float y = 0.0;
                // buckets.fill(0);
                if constexpr (high_resolution)
                {
                    A_row_index_and_value.clear();
                }
                else
                {
                    std::fill(A_row.begin(), A_row.end(), 0.0);
                }

                // timer.start();
                // bool choose = false;
                for (int k = 0; k < BINNING_FACTOR; ++k)
                {
                    for (int l = 0; l < BINNING_FACTOR; ++l)
                    {

                        // keep the data in radius range
                        double rho = rhos[j + l][i + k];
                        double eta = etas[j + l][i + k];
                        double sin_rho = sin(rho);
                        double cos_rho = cos(rho);

                        // sin(rho) is corrected from the tan(rho).
                        if (sin_rho * dist_to_sun < INSTR_R_MAX &&
                            sin_rho * dist_to_sun > INSTR_R_MIN)
                        {
                            auto pB_val = (*(pb_vector + IMAGE_SIZE * (i + k) + (j + l)) * b_scale + b_zero) * SCALE_FACTOR;
                            // if (pB_val <= 0 || pB_val >= OUTLIER_MAX)

                            // only drop negative valuse
                            // if (pB_val < 0)
                            // continue;

                            Rotation rx(Axis::x, eta);

                            // the nearest point is perdencicular to the LOS, which is the s in the formula (4)
                            /*
                            Assume eta = 0 and the distance to sun is 1, the pixel to the focus is perpendicular to the z axis
                            */

                            // the corrected nearest_point. Make a line from observer, perpendicular to the los, the
                            // intersection is the nearest_point, and the los is symmetric to the nearest point
                            // norm(nearest_point) = dist_to_sun * sin_rho
                            // This is what defined in Frazin (2002)

                            Vector3d<double> nearest_point = {
                                dist_to_sun * sin_rho * sin_rho,
                                0.0,
                                dist_to_sun * sin_rho * cos_rho};

                            // then rotate eta along x axis to the actual plane it's in
                            nearest_point = rx.rotate(nearest_point);

                            // los is the unit vector pointing in from the pixel to the sun (line of sight)
                            auto los = rx.rotate(Vector3d<double>{-cos_rho, 0.0, sin_rho});

                            // a final rotation to make the frame be the heliocentric carrington coordinate system
                            nearest_point = r23.rotate(nearest_point);
                            los = r23.rotate(los);

                            // std::cout << nearest_point << ' ' << los << '\n';

                            // the impact is used to check whether the position of the nearest point is inside the hollow sphere and is also used in the scattering factor calculation
                            double impact2 = nearest_point.norm2();
                            double impact = sqrt(impact2);
                            if (row_cnt > 7464686 / 14 * 9.5 && row_cnt < 7464686 / 14 * 9.6)
                                std::cerr << "impact " << impact << '\n';

                            // the signed time (position) of the spacecraft along the LOS
                            double spacecraft_time = sqrt(dist_to_sun * dist_to_sun - impact2);

                            if (los.dot(sun_to_obs_vec) < 0)
                                spacecraft_time *= -1;

                            bool obs_in_inner_sphere = dist_to_sun < R_MIN;
                            bool impact_in_sun = impact < 1;

                            // std::cerr << "nearest point " << nearest_point << ' ' << spacecraft_time << '\n';
                            // the ray reaches the nearest point before it enter the bin, skip
                            // impact <= 1 and spacecraft_time < 0 means the LOS is blocked by the sun, so
                            // only the spacecarft at the same direction with the postivie direction of the LOS
                            // can receive the LOS
                            if (impact > R_MAX || (obs_in_inner_sphere && impact_in_sun && spacecraft_time < 0))
                            {
                                // std::cerr << "skip, nearest point: " << nearest_point << "\nimpact " << impact << '\n';
                                // choose = false;
                                continue;
                            }
                            else
                            {
                                // simulate the behavior of the old solartom code
                                // choose = true;
                                // std::cerr << nearest_point << ' ' << impact << '\n';
                            }

                            // std::cerr << "spacecraft time " << spacecraft_time << '\n';
                            // increase the number of non-empty pixel
                            n_los += 1;

                            // std::cerr << i << ' ' << j << ' ' << k << ' ' << l << ' ' << impact << std::endl;

                            /*
                            How the time is defined: when the ray reaches the nearest point, the time is 0. The time is the same as the distance (since the los is the
                            speed vector and has the norm 1)
                            */

                            double t_max = sqrt(R_MAX2 - impact2);
                            double t_min = sqrt(R_MIN2 - impact2);

                            double entry_time = 0;
                            double exit_time = 0;

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

                            if (i == 256 && j == 512)
                                std::cerr << "rho " << rho << " eta " << eta << " impact " << impact << " t1 " << entry_time << " t2 " << exit_time << '\n';

                            auto entry_pos = nearest_point + los.scale(entry_time);
                            auto exit_pos = nearest_point + los.scale(exit_time);

                            if (entry_pos.z * exit_pos.z < 0)
                                ++cnt;

                            double entry_phi = atan2(entry_pos.y, entry_pos.x);
                            if (entry_phi < 0)
                                entry_phi += PI_TWO;
                            int phi_bin_begin = floor(entry_phi / PHI_BIN_SIZE);

                            double exit_phi = atan2(exit_pos.y, exit_pos.x);
                            if (exit_phi < 0)
                                exit_phi += PI_TWO;
                            int phi_bin_end = floor(exit_phi / PHI_BIN_SIZE);

                            if (phi_bin_begin > phi_bin_end)
                            {
                                std::swap(phi_bin_begin, phi_bin_end);
                                std::swap(entry_phi, exit_phi);
                            }

                            bool wrap = fabs(entry_phi - exit_phi) > std::numbers::pi;

                            std::cout << std::fixed << std::setprecision(20);

                            // std::cout << impact2 << ' ' << entry_time << ' ' << exit_time << '\n';

                            // Find bin crossings
                            /*
                            The idea here is that to find the change in any of the coordinate, since the ray enter
                            a new bin if any of these change

                            Maybe it's possible to use a fixed time interval to calculate the position instead of solving the
                            intersection with every bin?
                            */

                            /*
                            Geometry here
                                       entry_pos (also exit_pos in symmetry)
                                           |\
                                      los  | \
                                           |  \
                                           |   \
                            nearest_point -----sun (perpendicular to the image plane)
                            */

                            // radial bin crossings

                            int bin_of_min_rad = (impact <= R_MIN) ? 0 : floor((impact - R_MIN) / RAD_BIN_SIZE);
                            // std::cerr << "binrmin " << bin_of_min_rad << " impact " << impact << '\n';
                            assert(!std::isnan(entry_time) && "entry time is nan");
                            assert(!std::isnan(exit_time) && "exit time is nan");

                            radius_times.fill(0);
                            int radius_bin_size = N_RAD_BINS - bin_of_min_rad - 1;
                            int radius_times_begin = 0;
                            int radius_times_end = 0;

                            if (!obs_in_inner_sphere && !impact_in_sun)
                            {
                                // std::cerr << "radius case 1\n";
                                // std::cerr << radius_bin_size << '\n';
                                if constexpr (algorithm == 0)
                                {
                                    int t1_times_idx = radius_bin_size;
                                    int t2_times_idx = radius_bin_size + 1;
                                    for (double rad = R_MIN + (bin_of_min_rad + 1) * RAD_BIN_SIZE; rad < R_MAX - RAD_BIN_SIZE + EPSILON; rad += RAD_BIN_SIZE)
                                    {
                                        float time = sqrt(rad * rad - impact2);
                                        // std::cerr << time << '\n';
                                        assert(!std::isnan(time) && "r time is nan");
                                        radius_times[t1_times_idx--] = -time;
                                        radius_times[t2_times_idx++] = time;
                                        // std::cerr << radius_times[t1_times_idx + 1] << " " << radius_times[t2_times_idx - 1] << '\n';
                                    }
                                    radius_times_end = t2_times_idx;
                                    radius_times_begin = t1_times_idx + 1;
                                }
                                else if constexpr (algorithm == 1)
                                {
                                    for (int i = bin_of_min_rad; i <= N_RAD_BINS - 2; ++i)
                                    {
                                        auto rad = R_MIN + (i + 1) * RAD_BIN_SIZE;
                                        float time = -sqrt(rad * rad - impact2);
                                        assert(!std::isnan(time) && "r time is nan");
                                        radius_times[radius_times_end++] = time;
                                        radius_times[radius_times_end++] = -time;
                                    }
                                    std::sort(radius_times.begin() + radius_times_begin, radius_times.begin() + radius_times_end);
                                }
                            }
                            else if (!obs_in_inner_sphere && spacecraft_time < 0)
                            {
                                std::cerr << "radius case 2\n";
                                radius_times_end = radius_bin_size;
                                int times_idx = radius_bin_size - 1;
                                for (double rad = R_MIN + (bin_of_min_rad + 1) * RAD_BIN_SIZE; rad < R_MAX; rad += RAD_BIN_SIZE)
                                {
                                    float time = sqrt(rad * rad - impact2);
                                    assert(!std::isnan(time) && "r time is nan");
                                    radius_times[times_idx--] = -time;
                                }
                                radius_times_begin = times_idx + 1;
                            }
                            else // (!obs_inner_bin && space_craft > 0) || obs_inner_bin
                            {
                                // if (obs_in_inner_sphere)
                                //     std::cerr << "radius case 4\n";
                                // else
                                //     std::cerr << "radius case 2\n";
                                int times_idx = 0;
                                for (double rad = R_MIN + (bin_of_min_rad + 1) * RAD_BIN_SIZE; rad < R_MAX; rad += RAD_BIN_SIZE)
                                {
                                    float time = sqrt(rad * rad - impact2);
                                    assert(!std::isnan(time) && "r time is nan");
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

                            double entry_polar = atan(entry_pos.z / sqrt(entry_pos.x * entry_pos.x + entry_pos.y * entry_pos.y));
                            double exit_polar = atan(exit_pos.z / sqrt(exit_pos.x * exit_pos.x + exit_pos.y * exit_pos.y));

                            /*
                            From Prof. Lumetta's note (Verified)

                            For simplicity, in the comments below, let

                            nz = nearest_point.z
                            lz = los.z
                            c = nearest_point_xy_norm2
                            d = 2 * xy_product
                            e = los_xy_norm2

                            By taking the derivative of Theta(t) to 0
                            t_extreme_polar = (2 * c * lz - nz * d) / (2 * nz * e - d * lz)
                                            = (lz * impact2) / (nz * norm(los))
                            by using los dot nearest_point = 0

                            = (2 * c * lz - 2 * nz * xy_product) / (2 * nz * e - 2 * lz * xy_product)

                            = (c * lz - nz * xy_product) / (nz * e - lz * xy_product)

                            The final expression looks better

                            thete_max =
                            atan((nz + lz * t_extreme_polar) / sqrt(c + d * t_extreme_polar + e * t_extreme_polar * t_extreme_polar))

                            */

                            // std::cerr << nearest_point_xy_norm2 << ' ' << 2 * xy_product << ' ' << los_xy_norm2 << ' ' << los.z << ' ' << nearest_point.z << '\n';
                            double t_extreme_polar = los.z * impact2 / nearest_point.z;
                            // double t_extreme_polar = (nearest_point_xy_norm2 * los.z - nearest_point.z * xy_product) / (nearest_point.z * los_xy_norm2 - los.z * xy_product);
                            // double t_extreme_polar = los.z * impact2 / nearest_point.z;
                            double extreme_polar = atan((nearest_point.z + los.z * t_extreme_polar) / sqrt(nearest_point_xy_norm2 + 2 * xy_product * t_extreme_polar + los_xy_norm2 * t_extreme_polar * t_extreme_polar));

                            assert(!std::isnan(extreme_polar));

                            // At the calculated extreme point we will get nan sometimes

                            // return the values for calculating the roots, solving at^2 + bt + c = 0
                            auto calculate_roots = [&](double theta)
                            {
                                // if (fabs(theta) < EPSILON)
                                //{
                                //     return std::pair{-nearest_point.z / los.z, 0.0};
                                // }
                                double theta2 = theta * theta;
                                double a = theta2 * los_xy_norm2 - los_z_norm2;
                                double b = 2 * (theta2 * xy_product - z_product);
                                double c = theta2 * nearest_point_xy_norm2 - nearest_point_z_norm2;
                                double delta = sqrt(b * b - 4 * a * c);
                                a *= 2;
                                // always make the second term positive, so that r1 - r2 < r1 + r2
                                return std::pair{-b / a, delta / fabs(a)};
                            };

                            // bin 0 is from -pi / 2 to -pi / 2 + THETA_BIN_SIZE

                            polar_times.fill(0);
                            int polar_times_end = 0;
                            int polar_times_begin = 0;

                            auto polar_debug_print = [&]
                            {
                                std::cerr << "file count " << fc << '\n';
                                std::cerr << "theta_time " << t_extreme_polar << " entry time " << entry_time << " exit time " << exit_time << '\n';
                                std::cerr << "theta_max " << extreme_polar << " entry polar " << entry_polar << " exit polar " << exit_polar << '\n';
                                // std::cerr << "theta extreme bin " << extreme_polar_bin << " entry bin " << entry_polar_bin << " exit bin " << exit_polar_bin << '\n';
                                std::cerr << "Nx^2 + Ny^ 2 " << nearest_point_xy_norm2 << " 2 Nx vx + 2 Ny vy " << 2 * xy_product << " vx&2 + vy^2 " << los_xy_norm2 << " vz " << los.z << " Nz " << nearest_point.z << '\n';
                                std::cerr << std::flush;
                            };

                            // it's only used to calculate an estimate of # of bins
                            int entry_polar_bin = static_cast<int>((entry_polar + PI_HALF) / THETA_BIN_SIZE);
                            int exit_polar_bin = static_cast<int>((exit_polar + PI_HALF) / THETA_BIN_SIZE);

                            int case_n = -1;

                            // There are 8 cases
                            // entry_polar > t_extreme_polar (theta(t) first decrease then increase)
                            //      entry_time < t_extreme < exit_time
                            //          entry_polar > exit_polar or otherwise
                            //      exit_time < t_extreme
                            //      t_extreme < entry_time
                            // entry_polar < t_extreme_polar (theta(t) first increase then decrease)
                            // ...

                            // TODO: Check entry_polar < exit_polar first and merge case 1 and case 4, case 2 and case 3 (finished)

                            if constexpr (algorithm == 0)
                            {
                                bool concave = true;

                                if (std::fabs(entry_polar - extreme_polar) >= EPSILON)
                                {
                                    concave = entry_polar > extreme_polar;
                                }
                                else
                                {
                                    concave = exit_polar > extreme_polar;
                                }
                                // std::cerr << xy_product * los.z << ' ' << los_xy_norm2 << '\n';
                                // bool concave2 = xy_product * los.z > los_xy_norm2 * nearest_point.z;
                                // can be simplified to -nearest_point.z ^ |v|^2
                                bool concave2 = nearest_point.z < 0;
                                if (concave != concave2)
                                {
                                    printf("concave not equal! %f %f\n", xy_product * los.z, los_xy_norm2 * nearest_point.z);
                                    polar_debug_print();
                                    return 0;
                                }
                                // if (concave && los.z < 0)
                                // {
                                //     polar_debug_print();
                                // }

                                // if (std::max(entry_polar, exit_polar) > extreme_polar) // t_extreme_polar is the minimum
                                if (concave)
                                {
                                    /*
                                        theta(t) = tan^-1((nz + lz * t) / sqrt(c + dt + et^2))
                                        when t->inf, lim theta(t) -> tan^-1 (lim (nz + lz * t) / sqrt(c + dt + et^2))

                                        sqrt(c + dt + et^2) = sqrt(e)* t + d / (2 * sqrt(e)) + O(t^(-2)), exapnsion at t = inf

                                        (nz + lz * t) / (sqrt(e) * t + d / (2 * sqrt(e)) + O(t^(-2)))
                                        = (nz / t + lz) / (sqrt(e) + O(t^(-1))) -> nz / sqrt(e)

                                        so the asymptotic line is atan(nz / sqrt(e))

                                        By symmetry, at t = -inf, another asymtopic line is -atan(nz / sqrt(e))

                                    */
                                    int extreme_polar_bin = static_cast<int>((extreme_polar + PI_HALF - EPSILON) / THETA_BIN_SIZE);
                                    double asymptotic_line = -fabs(atan(los.z / sqrt(los_xy_norm2)));
                                    // std::cerr << "asymptotic " << asymptotic_line << '\n';
                                    // std::cerr << "theta extreme bin " << extreme_polar_bin << " entry bin " << entry_polar_bin << " exit bin " << exit_polar_bin << " file count " << fc << std::endl;

                                    if (entry_polar < exit_polar)
                                    {
                                        // std::cerr << "polar case 1\n"; // should be fixed
                                        case_n = 1;
                                        int t1_times_idx = entry_polar_bin - extreme_polar_bin + 2;
                                        int t2_times_idx = t1_times_idx + 1;

                                        double theta = 0;

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
                                                // std::cerr << theta << ' ' << r1 - r2 << ' ' << r1 + r2 << '\n';
                                            }
                                        }
                                        else
                                        {
                                            case_n = 2;
                                            theta = (entry_polar_bin + 1) * THETA_BIN_SIZE - PI_HALF;
                                        }

                                        polar_times_begin = t1_times_idx + 1;

                                        for (; theta < std::min(exit_polar, asymptotic_line); theta += THETA_BIN_SIZE)
                                        {
                                            auto [r1, r2] = calculate_roots(tan(theta));
                                            polar_times[t2_times_idx++] = r1 + r2;
                                        }

                                        for (; theta < std::min(exit_polar, -EPSILON); theta += THETA_BIN_SIZE)
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
                                        // std::cerr << "polar case 2\n";
                                        case_n = 3;
                                        int t1_times_idx = entry_polar_bin - extreme_polar_bin + 2;
                                        int t2_times_idx = t1_times_idx + 1;

                                        double theta = 0;

                                        // the equations have two valid solutions in this range
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
                                            case_n = 4;
                                            theta = (exit_polar_bin + 1) * THETA_BIN_SIZE - PI_HALF;
                                        }

                                        polar_times_end = t2_times_idx;
                                        for (; theta < std::min(entry_polar, asymptotic_line); theta += THETA_BIN_SIZE)
                                        {
                                            auto [r1, r2] = calculate_roots(tan(theta));
                                            polar_times[t1_times_idx--] = r1 - r2;
                                        }

                                        for (; theta < std::min(entry_polar, -EPSILON); theta += THETA_BIN_SIZE)
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
                                            assert(!std::isnan(r1 + r2) && "nan for polar case 2");
                                        }
                                        polar_times_begin = t1_times_idx + 1;
                                        assert(polar_times_begin >= 0 && "wrong idx calculation for case 2");
                                    }
                                }
                                else
                                {
                                    int extreme_polar_bin = static_cast<int>((extreme_polar + PI_HALF) / THETA_BIN_SIZE);
                                    double asymptotic_line = fabs(atan(los.z / sqrt(los_xy_norm2)));
                                    // std::cerr << "asymptotic " << asymptotic_line << '\n';
                                    if (entry_polar < exit_polar)
                                    {
                                        // std::cerr << "polar case 5\n"; // should be bug-free
                                        int t1_times_idx = extreme_polar_bin - entry_polar_bin + 2;
                                        int t2_times_idx = t1_times_idx + 1;

                                        double theta = 0;

                                        if (entry_time < t_extreme_polar && t_extreme_polar < exit_time)
                                        {
                                            case_n = 5;
                                            theta = extreme_polar_bin * THETA_BIN_SIZE - PI_HALF;
                                            if (fabs(theta - extreme_polar) < 2 * EPSILON)
                                            {
                                                // this root may be out of range due to floating-point error
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
                                                // std::cerr << theta << ' ' << r1 - r2 << ' ' << r1 + r2 << '\n';
                                            }
                                        }
                                        else
                                        {
                                            case_n = 6;
                                            theta = exit_polar_bin * THETA_BIN_SIZE - PI_HALF;
                                        }

                                        polar_times_end = t2_times_idx;

                                        for (; theta > std::max(entry_polar, asymptotic_line); theta -= THETA_BIN_SIZE)
                                        {
                                            auto [r1, r2] = calculate_roots(tan(theta));
                                            polar_times[t1_times_idx--] = r1 - r2;
                                            // std::cerr << "loop2 " << tan(theta) << ' ' << r1 - r2 << '\n';
                                        }
                                        for (; theta > std::max(entry_polar, EPSILON); theta -= THETA_BIN_SIZE)
                                        {
                                            auto [r1, r2] = calculate_roots(tan(theta));
                                            polar_times[t1_times_idx--] = r1 + r2;
                                            // std::cerr << "loop3 " << tan(theta) << ' ' << r1 + r2 << '\n';
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
                                            // std::cerr << "loop4 " << tan(theta) << ' ' << r1 - r2 << '\n';
                                            // assert(!std::isnan(r1 - r2) && "nan for polar case 5");
                                        }

                                        polar_times_begin = t1_times_idx + 1;
                                        // if (polar_times_begin < 0)
                                        // std::cerr << polar_times_begin << std::endl;
                                        assert(polar_times_begin >= 0 && "wrong index for case 5");
                                    }
                                    else
                                    {
                                        // std::cerr << "polar case 6\n";
                                        case_n = 7;
                                        int t1_times_idx = extreme_polar_bin - entry_polar_bin + 2;
                                        int t2_times_idx = t1_times_idx + 1;
                                        double theta = 0;

                                        // the equations have two valid solutions in this range
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
                                            case_n = 8;
                                            theta = entry_polar_bin * THETA_BIN_SIZE - PI_HALF;
                                        }

                                        polar_times_begin = t1_times_idx + 1;

                                        for (; theta > std::max(exit_polar, asymptotic_line); theta -= THETA_BIN_SIZE)
                                        {
                                            auto [r1, r2] = calculate_roots(tan(theta));
                                            polar_times[t2_times_idx++] = r1 + r2;
                                        }

                                        for (; theta > std::max(exit_polar, EPSILON); theta -= THETA_BIN_SIZE)
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
                                            assert(!std::isnan(r1 + r2) && "nan for polar case 6");
                                        }
                                        polar_times_end = t2_times_idx;
                                    }
                                }

                                while (polar_times_begin < polar_times_end && polar_times[polar_times_begin] <= entry_time)
                                    ++polar_times_begin;

                                while (polar_times_begin < polar_times_end && polar_times[polar_times_end - 1] >= exit_time)
                                    --polar_times_end;
                            }
                            else if constexpr (algorithm == 1)
                            {
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
                                std::sort(polar_times.begin() + polar_times_begin, polar_times.begin() + polar_times_end);
                            }

                            // azimuthal bin crossings
                            // std::cerr << "wrap " << wrap << " " << phi_bin_begin << ' ' << phi_bin_end << '\n';
                            /*
                            It is known that the function t(phi) is always increasing or decreasing in the domain

                            So only need to check whether t'(0) > 0

                            t'(0) = nx * ly - lx * ny

                            */

                            bool increase = nearest_point.x * los.y > los.x * nearest_point.y;
                            // std::cerr << los << ' ' << nearest_point << ' ' << increase << '\n';

                            azimuthal_times.fill(0);
                            int azimuthal_times_begin = 0;
                            int azimuthal_times_end = 0;

                            if constexpr (algorithm == 0)
                            {
                                if (!wrap)
                                {
                                    if (increase)
                                    {
                                        int times_idx = 0;
                                        double phi = (phi_bin_begin + 1) * PHI_BIN_SIZE;
                                        if (phi < entry_phi && entry_phi - phi < EPSILON)
                                        {
                                            azimuthal_times[times_idx++] = (nearest_point.y - nearest_point.x * tan(entry_phi)) / (los.x * tan(entry_phi) - los.y);
                                            phi += PHI_BIN_SIZE;
                                        }
                                        for (; phi < exit_phi; phi += PHI_BIN_SIZE)
                                        {
                                            double tan_phi = tan(phi);
                                            double time = (nearest_point.y - nearest_point.x * tan_phi) / (los.x * tan_phi - los.y);
                                            assert(!std::isnan(time) && "azimuthal time is nan");
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
                                        double phi = (phi_bin_begin + 1) * PHI_BIN_SIZE;
                                        if (phi < entry_phi && entry_phi - phi < EPSILON)
                                        {
                                            azimuthal_times[times_idx--] = (nearest_point.y - nearest_point.x * tan(entry_phi)) / (los.x * tan(entry_phi) - los.y);
                                            phi += PHI_BIN_SIZE;
                                        }
                                        for (; phi < exit_phi; phi += PHI_BIN_SIZE)
                                        {
                                            double tan_phi = tan(phi);
                                            double time = (nearest_point.y - nearest_point.x * tan_phi) / (los.x * tan_phi - los.y);
                                            assert(!std::isnan(time) && "azimuthal time is nan");
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
                                        double phi = (phi_bin_end + 1) * PHI_BIN_SIZE;
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
                                            double tan_phi = tan(phi);
                                            double time = (nearest_point.y - nearest_point.x * tan_phi) / (los.x * tan_phi - los.y);
                                            assert(!std::isnan(time) && "azimuthal time 1 is nan");
                                            azimuthal_times[times_idx++] = time;
                                        }
                                        if (fabs(phi - PI_TWO) < EPSILON)
                                            azimuthal_times[times_idx++] = nearest_point.y / (-los.y);

                                        phi = PHI_BIN_SIZE;
                                        for (; phi < entry_phi; phi += PHI_BIN_SIZE)
                                        {
                                            double tan_phi = tan(phi);
                                            double time = (nearest_point.y - nearest_point.x * tan_phi) / (los.x * tan_phi - los.y);
                                            assert(!std::isnan(time) && "azimuthal time 2 is nan");
                                            azimuthal_times[times_idx++] = time;
                                        }
                                        if (fabs(entry_phi - phi) < EPSILON)
                                        {
                                            double time = (nearest_point.y - nearest_point.x * tan(entry_phi)) / (los.x * tan(entry_phi) - los.y);
                                            if (time < exit_time && exit_time - time > EPSILON)
                                                azimuthal_times[times_idx++] = (nearest_point.y - nearest_point.x * tan(entry_phi)) / (los.x * tan(entry_phi) - los.y);
                                        }

                                        azimuthal_times_end = times_idx;
                                    }
                                    else
                                    {
                                        azimuthal_times_end = phi_bin_begin + N_PHI_BINS - phi_bin_end + 2;
                                        int times_idx = azimuthal_times_end - 1;
                                        double phi = (phi_bin_end + 1) * PHI_BIN_SIZE;
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
                                            double tan_phi = tan(phi);
                                            double time = (nearest_point.y - nearest_point.x * tan_phi) / (los.x * tan_phi - los.y);
                                            assert(!std::isnan(time) && "azimuthal time 1 is nan");
                                            azimuthal_times[times_idx--] = time;
                                        }
                                        if (fabs(phi - PI_TWO) < EPSILON)
                                            azimuthal_times[times_idx--] = nearest_point.y / (-los.y);

                                        phi = PHI_BIN_SIZE;
                                        for (; phi < entry_phi; phi += PHI_BIN_SIZE)
                                        {
                                            double tan_phi = tan(phi);
                                            double time = (nearest_point.y - nearest_point.x * tan_phi) / (los.x * tan_phi - los.y);
                                            assert(!std::isnan(time) && "azimuthal time 2 is nan");
                                            azimuthal_times[times_idx--] = time;
                                        }
                                        if (fabs(entry_phi - phi) < EPSILON)
                                            azimuthal_times[times_idx--] = (nearest_point.y - nearest_point.x * tan(entry_phi)) / (los.x * tan(entry_phi) - los.y);
                                        azimuthal_times_begin = times_idx + 1;
                                    }
                                }

                                assert(std::is_sorted(azimuthal_times.begin() + azimuthal_times_begin, azimuthal_times.begin() + azimuthal_times_end) && "azimuthal times not sorted");

                                while (azimuthal_times_begin < azimuthal_times_end && azimuthal_times[azimuthal_times_begin] <= entry_time)
                                    ++azimuthal_times_begin;

                                while (azimuthal_times_begin < azimuthal_times_end && azimuthal_times[azimuthal_times_end - 1] >= exit_time)
                                    --azimuthal_times_end;
                            }
                            else if constexpr (algorithm == 1)
                            {
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
                                std::sort(azimuthal_times.begin() + azimuthal_times_begin, azimuthal_times.begin() + azimuthal_times_end);
                            }

                            // std::sort(times.begin(), times.begin() + times_idx);
                            auto [times, times_size] = three_way_merge(radius_times, polar_times, azimuthal_times,
                                                                       radius_times_begin, radius_times_end,
                                                                       polar_times_begin, polar_times_end,
                                                                       azimuthal_times_begin, azimuthal_times_end, entry_time, exit_time);

                            assert(times_size == std::max(radius_times_end - radius_times_begin, 0) +
                                                     std::max(polar_times_end - polar_times_begin, 0) +
                                                     std::max(azimuthal_times_end - azimuthal_times_begin, 0) + 2);

                            auto debug_print = [&]
                            {
                                std::cerr << std::setprecision(12);
                                std::cerr << "polar case " << case_n << '\n';
                                std::cerr << bin_of_min_rad << '\n';
                                std::cerr << wrap << ' ' << increase << '\n';
                                std::cerr << entry_time << ' ' << exit_time << '\n';
                                std::cerr << entry_polar << ' ' << exit_polar << '\n';
                                std::cerr << t_extreme_polar << ' ' << extreme_polar << '\n';
                                std::cerr << radius_times_begin << ' ' << radius_times_end << ' ' << polar_times_begin << ' ' << polar_times_end << ' ' << azimuthal_times_begin << ' ' << azimuthal_times_end << '\n';
                                std::cerr << "radius times :\n";
                                for (int i = radius_times_begin; i < radius_times_end; ++i)
                                {
                                    std::cerr << std::to_string(radius_times[i]) << ' ';
                                }
                                std::cerr << '\n';
                                std::cerr << "polar times :\n";
                                for (int i = polar_times_begin; i < polar_times_end; ++i)
                                {

                                    std::cerr << std::to_string(polar_times[i]) << ' ';
                                }
                                std::cerr << '\n';
                                std::cerr << "azimuthal times :\n";
                                for (int i = azimuthal_times_begin; i < azimuthal_times_end; ++i)
                                {

                                    std::cerr << std::to_string(azimuthal_times[i]) << ' ';
                                }
                                std::cerr << '\n';
                                std::cerr << "all times :\n";
                                for (int i = 0; i < times_size; ++i)
                                {
                                    std::cerr << times[i] << ' ';
                                }
                                std::cerr << std::endl;
                            };

                            // if (!std::is_sorted(azimuthal_times.begin() + azimuthal_times_begin, azimuthal_times.begin() + azimuthal_times_end))
                            // {
                            //     debug_print();
                            //     return -1;
                            // }

                            // if (std::any_of(times.begin(), times.begin() + times_size, [](auto t)
                            //                 { return std::isnan(t); }))
                            // {
                            //     debug_print();
                            //     return -1;
                            // }
                            if (!std::is_sorted(times.begin(), times.begin() + times_size))
                            {
                                debug_print();
                                return -1;
                            }
                            // assert(std::is_sorted(times.begin(), times.begin() + times_size));

                            auto spacecraft_time_iter = std::lower_bound(times.begin(), times.begin() + times_size, spacecraft_time);
                            // assert(spacecraft_time_iter == times.begin());

                            // std::cerr << "spacecraft pos " << spacecraft_time_iter - times.begin() << '\n';
                            // I don't put the spacecraft time in the times array, so I don't need this line

                            // if (!obs_in_bin)
                            //     ++spacecraft_time_iter;
                            //  std::cerr << "times_idx " << times_idx << '\n';

                            // calculate the matrix elements
                            // only loop through time after spacecraft_time
                            // std::cerr << "times size " << times_size << '\n';
                            // std::vector<std::tuple<int, float, float, float>> arr_indices;
                            // bool repeated = false;

                            for (auto it = spacecraft_time_iter + 1; it < times.begin() + times_size; ++it)
                            {
                                // assert(entry_time < *it && *it < exit_time);
                                auto time = 0.5 * (*it + *(it - 1));
                                auto arc_length = *it - *(it - 1);
                                auto v = nearest_point + los.scale(time);
                                auto r = v.norm();
                                int r_idx = floor((r - R_MIN) / RAD_BIN_SIZE);
                                if (r_idx == N_RAD_BINS)
                                    --r_idx;
                                assert(r_idx != N_RAD_BINS);

                                // if (r < R_MIN)
                                //     continue;
                                if (r_idx < 0)
                                {
                                    // std::cerr << "r_idx < 0\n";
                                    r_idx = 0;
                                }
                                int theta_idx = floor((atan(v.z / sqrt(v.x * v.x + v.y * v.y)) + PI_HALF) / THETA_BIN_SIZE);
                                // if (theta_idx == N_THETA_BINS / 2)
                                //     ++cnt2;
                                if (theta_idx == N_THETA_BINS)
                                    --theta_idx;
                                assert(theta_idx != N_THETA_BINS);
                                auto phi = atan2(v.y, v.x);
                                if (phi < 0)
                                {
                                    phi += PI_TWO;
                                }
                                int phi_idx = floor(phi / PHI_BIN_SIZE);
                                if (phi_idx == N_PHI_BINS)
                                    --phi_idx;
                                assert(phi_idx != N_PHI_BINS);

                                // buckets[phi_idx]++;

                                int arr_idx = phi_idx * N_RAD_BINS * N_THETA_BINS + theta_idx * N_RAD_BINS + r_idx;
                                // arr_indices.emplace_back(arr_idx, (r - R_MIN) / RAD_BIN_SIZE, (atan(v.z / sqrt(v.x * v.x + v.y * v.y)) + PI_HALF) / THETA_BIN_SIZE, phi / PHI_BIN_SIZE);
                                // bin_count[arr_idx]++;
                                // if (bin_count[arr_idx] > 100)
                                //     continue;
                                // bin_to_pixels[arr_idx].emplace_back(i + k, j + l);

                                // Thomson scattering
                                if constexpr (high_resolution)
                                {
                                    A_row_index_and_value.emplace_back(arr_idx, Operators<OperatorClass::Thompson, float>{}(impact2, r) * arc_length);
                                }
                                else
                                {
                                    // if (A_row[arr_idx] != 0)
                                    // {
                                    //     debug_print();
                                    //     std::cerr << A_row[arr_idx] << '\n';
                                    //     std::cerr << "current time " << *it << ' ' << *(it - 1) << '\n';
                                    //     std::cerr << (r - R_MIN) / RAD_BIN_SIZE << " " << (atan(v.z / sqrt(v.x * v.x + v.y * v.y)) + PI_HALF) / THETA_BIN_SIZE << " " << 
                                    //     phi / PHI_BIN_SIZE << '\n';
                                    //     repeated = true;
                                    // }
                                    A_row[arr_idx] += Operators<OperatorClass::Thompson, float>{}(impact2, r) * arc_length;
                                }
                            }
                            // if (repeated)
                            // {
                            //     for (auto [arr_idx, r_idx, theta_idx, phi_idx] : arr_indices)
                            //     {
                            //         std::cerr << arr_idx << " " << r_idx << " " << theta_idx << " " << phi_idx << '\n';
                            //     }
                            //     std::cerr << '\n';
                            // }

                            // std::cout << '\n';
                            ++count;
                            y += pB_val;
                        }
                        else
                        {
                            ++instr_lim;
                        }
                    }
                }

                // timer.stop("calculation time");
                if (n_los != 0) // If not all bin is empty, build the matrix
                {
                    // timer.start();
                    ++row_cnt;
                    all_y.push_back(y / n_los);
                    // if (fc == 0 && i < 100 && j < 100)
                    //     std::cerr << i << ' ' << j << ' ' << y << ' ' << n_los << '\n';
                    all_y_idx.push_back((i / BINNING_FACTOR) * ceil((IMAGE_SIZE - mod_n) / BINNING_FACTOR) + floor(j / BINNING_FACTOR));
                    if constexpr (high_resolution)
                    {
                        std::sort(A_row_index_and_value.begin(), A_row_index_and_value.end());
                        A.add_dense_row(A_row_index_and_value, n_los);
                    }
                    else
                    {
                        A.add_row(A_row, n_los);
                    }

                    // timer.stop("saving time");
                    /*phi_buckets_file << i << ' ' << j;
                    for (int n : buckets)
                    {
                        phi_buckets_file << ' ' << n;
                    }
                    phi_buckets_file << '\n';

                    max_phi_buckets = std::max(max_phi_buckets, *std::max_element(buckets.begin(), buckets.end()));*/
                }
            }
        }
        std::cerr << instr_lim << " not in instr limit\n";
        block_idx.push_back(all_y.size());
        std::cerr << "current y count " << all_y.size() << '\n';
        // cleaning
        free(fits_image);
        free(fits_header);

        // std::cout << "bin only passed by one pixel\n";

        // if (fc == 0)
        // {

        //     std::fstream bin_to_pixels_file;
        //     bin_to_pixels_file.open("bin_to_pixels", std::fstream::out | std::fstream::trunc);

        //     for (auto &p : bin_to_pixels)
        //     {
        //         for (auto [x, y] : p)
        //             bin_to_pixels_file << std::to_string(x * IMAGE_SIZE + y) << ' ';
        //         bin_to_pixels_file << '\n';
        //     }
        //     bin_to_pixels_file.close();
        // }

        // phi_buckets_file.close();
        // bin_pixel_file.close();

        // std::cout << '\n';
        timer.stop("computation time per image");
        A.print_status();
    }

    // std::cout << "max phi buckets " << max_phi_buckets << '\n';
    // write output
    write_info(output_dir, "info", std::string(config_dir), std::string(fits_dir));
    write_vector(std::string(output_dir) + "y_data", "y", all_y);

    float y_norm = 0;
    for (float f : all_y)
        y_norm += f * f;
    std::cerr << "norm of y_all " << y_norm << '\n';
    // std::cerr << "number of los that pass thourgh the equatorial plane " << cnt << '\n';
    // std::cerr << "number of los that pass thourgh the theta = 0 bin " << cnt2 << '\n';

    write_vector(std::string(output_dir) + "y_idx", "y index", all_y_idx);
    write_vector(std::string(output_dir) + "block_idx", "block index", block_idx);

    A.save(output_dir);
    A.print_status();
}
