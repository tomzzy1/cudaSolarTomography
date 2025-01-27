#pragma once
#include <charconv>
#include <cassert>
#include <cstring>
#include <fstream>
#include <fitsfile.h>
#include "constants.hpp"
#include "build_params.hpp"
#include "type.hpp"
#include "timer.hpp"

#if __cplusplus > 201703L
#include <chrono>
#else
#include <ctime>
#endif

namespace cudaSolarTomography
{
#ifndef __CUDACC__

#define __device__
#define __host__

#endif

    template <typename T>
    struct ImageParameters
    {
        int n_files;
        int file_idx;
        T center_x;
        T center_y;
        T pixel_size;
        T dist_to_sun;
        T roll_offset;
        T b_scale;
        T b_zero;
        Image_t *pb_vector;
        Vector3d<T> solar_pole;
        Vector3d<T> sun_to_obs_vec;
        Rotation<T> r23;
        long long ref_time;
        long long cur_time;

        __device__ __host__ void print()
        {
            printf("%d %f %f %f %f\n", file_idx, center_x, center_y, pixel_size, dist_to_sun);
        }
    };

    struct GridParameters
    {
        GridParameters() = default;
        GridParameters(
            int n_rad_bins, int n_theta_bins,
            int n_phi_bins, double r_max, double r_min) : n_rad_bins(n_rad_bins),
                                                          n_theta_bins(n_theta_bins), n_phi_bins(n_phi_bins),
                                                          r_max(r_max), r_min(r_min),
                                                          n_bins(n_rad_bins * n_theta_bins * n_phi_bins),
                                                          AROW_SIZE(n_rad_bins + n_theta_bins + n_phi_bins),
                                                          r_diff(r_max - r_min),
                                                          r_max2(r_max * r_max),
                                                          r_min2(r_min * r_min),
                                                          rad_bin_size(r_diff / n_rad_bins),
                                                          theta_bin_size(std::numbers::pi / n_theta_bins),
                                                          phi_bin_size(PI_TWO / n_phi_bins)
        {
        }
        int n_rad_bins;
        int n_theta_bins;
        int n_phi_bins;
        int n_bins;
        int AROW_SIZE;

        double r_max;
        double r_min;
        double r_diff;
        double r_max2;
        double r_min2;

        double rad_bin_size;
        double theta_bin_size;
        double phi_bin_size;
    };

    struct InstrParameters
    {
        InstrParameters() = default;
        InstrParameters(double instr_r_max, double instr_r_min,
                        int image_size, double pixel_size,
                        int binning_factor, double scale_factor) : instr_r_max(instr_r_max), instr_r_min(instr_r_min),
                                                                   image_size(image_size), pixel_size(pixel_size),
                                                                   binning_factor(binning_factor), scale_factor(scale_factor),
                                                                   bin_size(binning_factor * binning_factor),
                                                                   row_size(image_size / binning_factor),
                                                                   y_size(row_size * row_size)
        {
        }
        double instr_r_max;
        double instr_r_min;

        int image_size;
        double pixel_size;

        int binning_factor;
        double scale_factor;

        int bin_size;
        int row_size;
        int y_size;
    };

#if defined(LASCO_C2)
    constexpr std::string_view orbit_dir = "../data/orbit/";
#endif
#if defined(COR) || defined(LASCO_C2)
    std::pair<Vector3d<double>, double> get_orbit(char *fits_header)
    {

        /* the STEREO .fts files give the Sun-Spacecraft vector in
         *  Heliocentric Aries Ecliptic Coordinates.  This differs
         *  from GCI in the origin point and choice of Z-axis (ecliptic
         *  N, vs. equatorial N (celestial pole).  Therefore these coords.
         *  need to rotated about the x-axis.
         */

        Vector3d<double> observatory;

        double carrington_longitude = 0;

        hgetr8(fits_header, "HAEX_OBS", &observatory.x);
        hgetr8(fits_header, "HAEY_OBS", &observatory.y);
        hgetr8(fits_header, "HAEZ_OBS", &observatory.z);
        hgetr8(fits_header, "CRLN_OBS", &carrington_longitude);
#ifdef LASCO_C2
        carrington_longitude = carrington_longitude > 180 ? carrington_longitude - 360 : carrington_longitude;
#endif
        std::cerr << "carrington longitude in degree " << carrington_longitude << '\n';
        carrington_longitude *= DEG_TO_RAD; // convert the unit of carrington longitude

        /* the J2000.0 angle between the Ecliptic and mean Equatorial planes
         *  is 23d26m21.4119s - From Allen's Astrophysical Quantities, 4th ed. (2000)  */

        Rotation rx(Axis::x, J20000_ANGLE);

        auto sun_to_obs_vec = rx.rotate(observatory.scale(0.001 / SUN_RADIUS_KM)); // convert the unit to km then to radius_of_sun

        return {sun_to_obs_vec, carrington_longitude};
    }

#elif defined(LASCO_C2)

    std::pair<Vector3d<double>, double> get_orbit(char *fits_header, std::string &fits_filename, const Vector3d<double> &solar_pole)
    {

        /* the STEREO .fts files give the Sun-Spacecraft vector in
         *  Heliocentric Aries Ecliptic Coordinates.  This differs
         *  from GCI in the origin point and choice of Z-axis (ecliptic
         *  N, vs. equatorial N (celestial pole).  Therefore these coords.
         *  need to rotated about the x-axis.
         */

        std::cerr << "get orbit\n";
        std::string fits_time(hgetc(fits_header, "TIME_OBS"));

        std::cerr << fits_time << '\n';

        // fetch hour and minute

        int hour = 0;
        int minute = 0;
        std::from_chars(fits_time.data(), fits_time.data() + 2, hour);
        std::from_chars(fits_time.data() + 3, fits_time.data() + 5, minute);

        double fraction_of_day = hour / 24.0 + minute / (24 * 60.0);

        // fetch date

        std::string fits_date(hgetc(fits_header, "DATE_OBS"));

        std::cerr << fits_date << '\n';

        auto fits_year = fits_date.substr(0, 4);
        auto fits_month = fits_date.substr(5, 2);
        auto fits_day = fits_date.substr(8, 2);

        std::string date;
        date.reserve(14);

        // format yyyy-mm-dd-hhhh
        date = fits_year + '-' + fits_month + '-' + fits_day;
        // std::to_chars(date.data() + 11, date.data() + 15, std::rint(fraction_of_day * 10000), std::chars_format::fixed, 4);
        // work for old GCC 8.5.0
        std::stringstream stream;
        // stream << std::fixed << std::setprecision(4) << std::rint(fraction_of_day * 10000);

        date += '.';
        // date += stream.str();
        date += std::to_string(static_cast<int>(std::rint(fraction_of_day * 10000)));

        std::cerr << date << '\n';

        auto iter = std::lower_bound(ephemeris_dates.begin(), ephemeris_dates.end(), date);

        std::cerr << iter - ephemeris_dates.begin() << '\n';

        // jd is Julian Date
        auto getjd = [](std::string_view date)
        {
            return fd2jd(const_cast<char *>(std::string(date).c_str()));
        };

        double jd = getjd(date);

        std::cerr << "jd " << jd << '\n';

        double diff = 0;

        if (iter == ephemeris_dates.end())
        {
            diff = jd - getjd(*(iter - 1));
            if (std::abs(diff) >= 60.0)
            {
                std::cerr << "get_orbit: date is not in range\n";
                exit(-1);
            }
        }
        else
        {
            double jd1 = getjd(*iter);
            double jd2 = getjd(*(iter - 1));
            // std::cerr << *iter << " " << *(iter - 1) << '\n';
            // std::cerr << "jd1 " << jd1 << " jd2 " << jd2 << '\n';
            diff = jd1 - jd < jd - jd2 ? jd1 - jd : jd2 - jd;
        }

        // is this correct name?
        double ecliptic_longitude = 360.0 * diff / 27.2753;

        std::cerr << "ecliptic longitude " << ecliptic_longitude << '\n';

        int line_no = rint(fraction_of_day * 144.0);
        if (line_no == 144)
            --line_no;
        std::cerr << "line number " << line_no << '\n';

        // find local orbit file
        constexpr int VERSION_MAX = 10;
        int version = 1;
        FILE *orbit_file = nullptr;
        std::string orbit_prefix("SO_OR_PRE_");
        std::string orbit_file_name;
        while (!orbit_file && version < VERSION_MAX)
        {
            orbit_file_name = orbit_prefix + fits_year + fits_month + fits_day + "_V0" + std::to_string(version++) + ".DAT";
            orbit_file = fopen((std::string(orbit_dir) + orbit_file_name).c_str(), "r");
        }
        if (VERSION_MAX == version)
        {
            std::cerr << "failed to find the orbit file\n";
            std::terminate();
        }
        if (orbit_file)
        {
            std::cerr << "orbit file " << orbit_file_name << " opened\n";
        }
        // fseek(orbit_file, 512 * line_no, SEEK_SET);
        std::array<char, 512> buffer;
        for (int i = 0; i <= line_no; ++i)
        {
            if (!fgets(buffer.data(), 512, orbit_file))
            {
                std::cerr << "error reading orbit files\n";
                std::exit(-1);
            }
        }
        // fgets(buffer.data(), 512, orbit_file);
        fclose(orbit_file);
        // std::cerr << "line " << std::string(buffer.data(), 512) << '\n';
        // std::cerr << "line read\n";
        //
        Vector3d<double> earth_to_sun_vec;
        Vector3d<double> earth_to_soho_vec;
        sscanf(buffer.data() + 305, "%lf %lf %lf", &earth_to_sun_vec.x, &earth_to_sun_vec.y, &earth_to_sun_vec.z);
        sscanf(buffer.data() + 44, "%lf %lf %lf", &earth_to_soho_vec.x, &earth_to_soho_vec.y, &earth_to_soho_vec.z);
        // std::cerr << "earth to sun " << earth_to_sun_vec << " earth to soho " << earth_to_soho_vec << '\n';

        auto sun_to_obs_vec = earth_to_soho_vec - earth_to_sun_vec;

        // calculate angle between earth and soho, projected onto equ. plane
        // e = solar_pole * (solar_pole .* earth_sun)  - earth_sun
        // o = sun_obs - solar_pole * (solar_pole .* sun_obs)
        // Then get the direction vector of e and o
        auto e = solar_pole.scale(solar_pole.dot(earth_to_sun_vec)) - earth_to_sun_vec;
        e = e.scale(1 / e.norm());
        // std::cerr << "e " << e << '\n';
        auto o = sun_to_obs_vec - solar_pole.scale(solar_pole.dot(sun_to_obs_vec));
        o = o.scale(1 / o.norm());
        // std::cerr << "o " << o << '\n';

        auto c = e.cross(o);
        // std::cerr << "c " << c << '\n';

        // double carrington_longitude = ecliptic_longitude + asin(solar_pole.dot(c)) * RAD_TO_DEG;
        double carrington_longitude = ecliptic_longitude * DEG_TO_RAD + asin(solar_pole.dot(c));
        std::cerr << "sun-observer vector " << sun_to_obs_vec << " carrington longitude " << carrington_longitude << '\n';
        return {sun_to_obs_vec.scale(1 / SUN_RADIUS_KM), carrington_longitude};
    }

#endif

    template <typename T, size_t s, size_t s1, size_t s2, size_t s3>
    __device__ void three_way_merge(std::array<T, s> &res, const std::array<T, s1> &arr1, const std::array<T, s2> &arr2, const std::array<T, s3> &arr3, int begin1, int end1, int begin2,
                                    int end2, int begin3, int end3, real entry_time, real exit_time)
    {
        int i1 = begin1;
        int i2 = begin2;
        int i3 = begin3;
        int i = 0;
        res[i++] = entry_time;
        while (i1 < end1 && i2 < end2 && i3 < end3)
        {
            if (arr1[i1] < arr2[i2])
            {
                if (arr1[i1] < arr3[i3])
                {
                    res[i++] = arr1[i1++];
                }
                else
                {
                    res[i++] = arr3[i3++];
                }
            }
            else
            {
                if (arr2[i2] < arr3[i3])
                {
                    res[i++] = arr2[i2++];
                }
                else
                {
                    res[i++] = arr3[i3++];
                }
            }
        }

        if (i1 >= end1) // disable warning
        {
            while (i2 < end2 && i3 < end3)
            {
                if (arr2[i2] < arr3[i3])
                {
                    res[i++] = arr2[i2++];
                }
                else
                {
                    res[i++] = arr3[i3++];
                }
            }
            while (i2 < end2)
            {
                res[i++] = arr2[i2++];
            }
            while (i3 < end3)
            {
                res[i++] = arr3[i3++];
            }
        }
        else if (i2 >= end2)
        {
            while (i1 < end1 && i3 < end3)
            {
                if (arr1[i1] < arr3[i3])
                {
                    res[i++] = arr1[i1++];
                }
                else
                {
                    res[i++] = arr3[i3++];
                }
            }
            while (i1 < end1)
            {
                res[i++] = arr1[i1++];
            }
            while (i3 < end3)
            {
                res[i++] = arr3[i3++];
            }
        }
        else if (i3 >= end3)
        {
            while (i2 < end2 && i1 < end1)
            {
                if (arr2[i2] < arr1[i1])
                {
                    res[i++] = arr2[i2++];
                }
                else
                {
                    res[i++] = arr1[i1++];
                }
            }
            while (i2 < end2)
            {
                res[i++] = arr2[i2++];
            }
            while (i1 < end1)
            {
                res[i++] = arr1[i1++];
            }
        }
        // in some corner cases, the azimuthal crossings will compute a time close to the entry time or exit time (might be slightly larger or smaller),
        // so don't add exit tiem and entry time in these cases
        // assert(i >= 1 && "failed to put the entry time into the result");
        res[i++] = exit_time;
    }

    template <typename T>
    __device__ void three_way_merge(T *res, const T *arr1, const T *arr2, const T *arr3, int begin1, int end1, int begin2,
                                    int end2, int begin3, int end3, real entry_time, real exit_time)
    {
        int i1 = begin1;
        int i2 = begin2;
        int i3 = begin3;
        int i = 0;
        res[i++] = entry_time;
        while (i1 < end1 && i2 < end2 && i3 < end3)
        {
            if (arr1[i1] < arr2[i2])
            {
                if (arr1[i1] < arr3[i3])
                {
                    res[i++] = arr1[i1++];
                }
                else
                {
                    res[i++] = arr3[i3++];
                }
            }
            else
            {
                if (arr2[i2] < arr3[i3])
                {
                    res[i++] = arr2[i2++];
                }
                else
                {
                    res[i++] = arr3[i3++];
                }
            }
        }

        if (i1 >= end1) // disable warning
        {
            while (i2 < end2 && i3 < end3)
            {
                if (arr2[i2] < arr3[i3])
                {
                    res[i++] = arr2[i2++];
                }
                else
                {
                    res[i++] = arr3[i3++];
                }
            }
            while (i2 < end2)
            {
                res[i++] = arr2[i2++];
            }
            while (i3 < end3)
            {
                res[i++] = arr3[i3++];
            }
        }
        else if (i2 >= end2)
        {
            while (i1 < end1 && i3 < end3)
            {
                if (arr1[i1] < arr3[i3])
                {
                    res[i++] = arr1[i1++];
                }
                else
                {
                    res[i++] = arr3[i3++];
                }
            }
            while (i1 < end1)
            {
                res[i++] = arr1[i1++];
            }
            while (i3 < end3)
            {
                res[i++] = arr3[i3++];
            }
        }
        else if (i3 >= end3)
        {
            while (i2 < end2 && i1 < end1)
            {
                if (arr2[i2] < arr1[i1])
                {
                    res[i++] = arr2[i2++];
                }
                else
                {
                    res[i++] = arr1[i1++];
                }
            }
            while (i2 < end2)
            {
                res[i++] = arr2[i2++];
            }
            while (i1 < end1)
            {
                res[i++] = arr1[i1++];
            }
        }
        // in some corner cases, the azimuthal crossings will compute a time close to the entry time or exit time (might be slightly larger or smaller),
        // so don't add exit tiem and entry time in these cases
        // assert(i >= 1 && "failed to put the entry time into the result");
        res[i++] = exit_time;
    }

    template <typename T, size_t s1, size_t s2, size_t s3>
    auto three_way_merge(const std::array<T, s1> &arr1, const std::array<T, s2> &arr2, const std::array<T, s3> &arr3, int begin1, int end1, int begin2,
                         int end2, int begin3, int end3, T entry_time, T exit_time)
    {
        std::array<T, s1 + s2 + s3 + 3> res;
        // three_way_merge(res, arr1, arr2, arr3, begin1, end1, begin2, end2, begin3, end3, entry_time, exit_time);
        int i1 = begin1;
        int i2 = begin2;
        int i3 = begin3;
        int i = 0;
        res[i++] = entry_time;
        while (i1 < end1 && i2 < end2 && i3 < end3)
        {
            if (arr1[i1] < arr2[i2])
            {
                if (arr1[i1] < arr3[i3])
                {
                    res[i++] = arr1[i1++];
                }
                else
                {
                    res[i++] = arr3[i3++];
                }
            }
            else
            {
                if (arr2[i2] < arr3[i3])
                {
                    res[i++] = arr2[i2++];
                }
                else
                {
                    res[i++] = arr3[i3++];
                }
            }
        }

        if (i1 >= end1) // disable warning
        {
            while (i2 < end2 && i3 < end3)
            {
                if (arr2[i2] < arr3[i3])
                {
                    res[i++] = arr2[i2++];
                }
                else
                {
                    res[i++] = arr3[i3++];
                }
            }
            while (i2 < end2)
            {
                res[i++] = arr2[i2++];
            }
            while (i3 < end3)
            {
                res[i++] = arr3[i3++];
            }
        }
        else if (i2 >= end2)
        {
            while (i1 < end1 && i3 < end3)
            {
                if (arr1[i1] < arr3[i3])
                {
                    res[i++] = arr1[i1++];
                }
                else
                {
                    res[i++] = arr3[i3++];
                }
            }
            while (i1 < end1)
            {
                res[i++] = arr1[i1++];
            }
            while (i3 < end3)
            {
                res[i++] = arr3[i3++];
            }
        }
        else if (i3 >= end3)
        {
            while (i2 < end2 && i1 < end1)
            {
                if (arr2[i2] < arr1[i1])
                {
                    res[i++] = arr2[i2++];
                }
                else
                {
                    res[i++] = arr1[i1++];
                }
            }
            while (i2 < end2)
            {
                res[i++] = arr2[i2++];
            }
            while (i1 < end1)
            {
                res[i++] = arr1[i1++];
            }
        }
        assert(i1 == end1 && i2 == end2 && i3 == end3);
        res[i++] = exit_time;
        return std::pair{res, i};
        // int times_size = std::max(end1 - begin1, 0) +
        //               std::max(end2 - begin2, 0) +
        //               std::max(end3 - begin3, 0) + 2;
        // return std::pair{res, times_size};
    }

    long long get_time_in_seconds(char *fits_header)
    {
        std::string fits_time(hgetc(fits_header, "TIME_OBS"));

        std::cerr << fits_time << '\n';

        // fetch hour and minute

        int hour = 0;
        int minute = 0;
        std::from_chars(fits_time.data(), fits_time.data() + 2, hour);
        std::from_chars(fits_time.data() + 3, fits_time.data() + 5, minute);

        // fetch date

        std::string fits_date(hgetc(fits_header, "DATE_OBS"));

        std::cerr << fits_date << '\n';

        int year = 0;
        int month = 0;
        int day = 0;

        std::from_chars(fits_date.data(), fits_date.data() + 4, year);
        std::from_chars(fits_date.data() + 5, fits_date.data() + 7, month);
        std::from_chars(fits_date.data() + 8, fits_date.data() + 10, day);

#if __cplusplus > 201703L
        using namespace std::chrono;
        std::chrono::sys_days date = std::chrono::year(year) / month / day;
        auto time_point = sys_time<seconds>(date) + hours(hour) + minutes(minute);
        return static_cast<long long>(system_clock::to_time_t(time_point));
#else
        std::tm time;
        std::memset(&time, 0, sizeof(std::tm));
        time.tm_year = year - 1900;
        time.tm_mon = month - 1;
        time.tm_mday = day;
        time.tm_hour = hour;
        time.tm_min = minute;

        std::time_t time_point = std::mktime(&time);

        return static_cast<long long>(time_point);

#endif
    }

    void compute_parameters_from_files(const std::vector<std::string> &sub_matrix_filenames, std::vector<ImageParameters<Params_t>> &params, std::string_view ref_filename = "")
    {
        Timer timer;

        int n_files = sub_matrix_filenames.size();
        // unit solar pole (1, delta_pole, alpha_pole) (in spherical coordinate)
        const Vector3d<double> solar_pole = {
            cos(DELTA_POLE) * cos(ALPHA_POLE),
            cos(DELTA_POLE) * sin(ALPHA_POLE),
            sin(DELTA_POLE)};

        // all precompuatations in CPU

        timer.start();

        long long ref_time = -1;

        for (int fc = 0; fc < n_files; ++fc)
        {
            timer.start();

            std::fstream fits_file;

            std::cerr << "current file " << sub_matrix_filenames[fc] << " " << fc + 1 << " of " << n_files << " files\n";

            std::string fits_filename = std::string(fits_dir) + sub_matrix_filenames[fc];
            if (fits_file.open(fits_filename); !fits_file.is_open())
            {
                std::cerr << "Failed to open the FITS file " << fits_filename << "\n";
                std::exit(-1);
            }
            fits_file.close();

            // build the sub matrix here

            char *fits_header = nullptr;

            int max_header_size = 0; // maximum number of bytes in FITS header
            int header_size = 0;     // number of bytes in FITS header

            if (fits_header = fitsrhead(const_cast<char *>(fits_filename.c_str()), &max_header_size, &header_size); !fits_header)
            {
                std::cerr << "failed to read the FITS header " << sub_matrix_filenames[fc] << '\n';
                free(fits_header);
                std::exit(-1);
            }

            double roll_offset = 0;
            hgetr8(fits_header, "CROTA", &roll_offset);
            std::cerr << "roll_offset " << roll_offset << '\n';

            // get the arcsec/pixel conversion factor from the FITS file
            double pixel_x_size = 0;
            double pixel_y_size = 0;
            double b_scale = 1;
            double b_zero = 0;

            hgetr8(fits_header, "CDELT1", &pixel_x_size);
            hgetr8(fits_header, "CDELT2", &pixel_y_size);
            hgetr8(fits_header, "BSCALE", &b_scale);
            hgetr8(fits_header, "BZERO", &b_zero);

            std::cerr << "b_scale " << b_scale << " b_zero " << b_zero << '\n';

            if (fabs(pixel_x_size - pixel_y_size) > 0.0001)
            {
                std::cerr << "the width and length of pixel don't match\n";
                std::exit(-1);
            }

            double pixel_size = pixel_x_size;
            timer.stop("read FITS files");
// get the orbit here
#if defined(COR) || defined(LASCO_C2)
            auto [sun_to_obs_vec, carrington_longitude] = get_orbit(fits_header);
#elif defined(LASCO_C2) // deprecated
            auto [sun_to_obs_vec, carrington_longitude] = get_orbit(fits_header, fits_filename, solar_pole);
#endif

            double dist_to_sun = sun_to_obs_vec.norm();
            std::cerr << "dist_to_sun " << dist_to_sun << '\n';

            double center_x = 0;
            double center_y = 0;
            hgetr8(fits_header, "CRPIX1", &center_x);
            hgetr8(fits_header, "CRPIX2", &center_y);

            // change the index from 1-based to 0-based
            --center_x;
            --center_y;

            std::cerr << pixel_size << '\n';
            std::cerr << "center_x " << center_x << " center_y " << center_y << '\n';

            timer.start();

            // calculate the arcsec from the center to each pixel

            // zero the y component of sun_to_obs_vec
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

            // std::cerr << "solar pole " << solar_pole2 << "\nsun_to_obs_vec " << sun_to_obs_vec2 << '\n';
            // std::cerr << "bscale " << b_scale << " bzero " << b_zero << '\n';

            ry = Rotation(Axis::y, atan2(solar_pole2.x, solar_pole2.z));
            rz = Rotation(Axis::z, carrington_longitude);
            Rotation r23 = rz.compose(ry);
            sun_to_obs_vec = r23.rotate(sun_to_obs_vec2);

            std::cerr << "solar pole 3 " << r23.rotate(solar_pole2) << '\n'
                      << "sun_to_obs_vec3 " << r23.rotate(sun_to_obs_vec2) << '\n';

            std::cerr << "Rotation matrix r23\n"
                      << r23 << '\n';

            long long cur_time = -1;

#if defined(LASCO_C2)

            cur_time = get_time_in_seconds(fits_header);

            if (!ref_filename.empty() && sub_matrix_filenames[fc] == ref_filename)
            {
                ref_time = cur_time;
            }
            else if (fc == n_files / 2)
                ref_time = cur_time;
#endif

            params[fc] = {
                n_files,
                fc, // file_idx
                static_cast<Params_t>(center_x),
                static_cast<Params_t>(center_y),
                static_cast<Params_t>(pixel_size),
                static_cast<Params_t>(dist_to_sun),
                static_cast<Params_t>(roll_offset),
                static_cast<Params_t>(b_scale),
                static_cast<Params_t>(b_zero),
                nullptr, // reinterpret_cast<Image_t *>(fits_image),
                solar_pole.to<Params_t>(),
                sun_to_obs_vec.to<Params_t>(),
                r23.to<Params_t>(),
                0, // ref_time
                cur_time};
            timer.stop("Roation matrix computation");
            // cleaning
            free(fits_header);
        }

        // set ref_time
        for (int fc = 0; fc < n_files; ++fc)
        {
            params[fc].ref_time = ref_time;
        }

        timer.stop("Parameter Computation");
        // return params;
    }

    void fetch_pb_vector_from_files(const std::vector<std::string> &sub_matrix_filenames, std::vector<ImageParameters<Params_t>> &params)
    {
        Timer timer;

        int n_files = sub_matrix_filenames.size();
        // all precompuatations in CPU

        timer.start();

        for (int fc = 0; fc < n_files; ++fc)
        {
            std::fstream fits_file;

            // std::cerr << "current file " << sub_matrix_filenames[fc] << " " << fc + 1 << " of " << n_files << " files\n";

            std::string fits_filename = std::string(fits_dir) + sub_matrix_filenames[fc];
            if (fits_file.open(fits_filename); !fits_file.is_open())
            {
                std::cerr << "Failed to open the FITS file " << fits_filename << "\n";
                std::exit(-1);
            }
            fits_file.close();

            // build the sub matrix here

            char *fits_image = nullptr;
            char *fits_header = nullptr;

            int max_header_size = 0; // maximum number of bytes in FITS header
            int header_size = 0;     // number of bytes in FITS header

            if (fits_header = fitsrhead(const_cast<char *>(fits_filename.c_str()), &max_header_size, &header_size); fits_header)
            {
                if (fits_image = fitsrimage(const_cast<char *>(fits_filename.c_str()), header_size, fits_header); !fits_image)
                {
                    std::cerr << "failed to read the FITS iamge " << sub_matrix_filenames[fc] << '\n';
                    free(fits_image);
                    std::exit(-1);
                }
            }
            else
            {
                std::cerr << "failed to read the FITS header " << sub_matrix_filenames[fc] << '\n';
                free(fits_header);
                std::exit(-1);
            }

            params[fc].pb_vector = reinterpret_cast<Image_t *>(fits_image);
            // cleaning
            free(fits_header);
        }

        timer.stop("Read pb vector from files");
    }

    std::vector<ImageParameters<Params_t>> get_all_parameters_from_files(const std::vector<std::string> &sub_matrix_filenames)
    {
        int n_files = sub_matrix_filenames.size();
        std::vector<ImageParameters<Params_t>> params(n_files);
        compute_parameters_from_files(sub_matrix_filenames, params);
        fetch_pb_vector_from_files(sub_matrix_filenames, params);
        return params;
    }

    std::vector<ImageParameters<Params_t>> get_all_parameters_from_files_with_virtual_viewpoints(const std::vector<std::string> &sub_matrix_filenames, int n_viewpoints, double degree = -1)
    {
        int n_files = sub_matrix_filenames.size();
        std::vector<ImageParameters<Params_t>> params(n_viewpoints);

        // unit solar pole (1, delta_pole, alpha_pole) (in spherical coordinate)
        const Vector3d<double> solar_pole = {
            cos(DELTA_POLE) * cos(ALPHA_POLE),
            cos(DELTA_POLE) * sin(ALPHA_POLE),
            sin(DELTA_POLE)};

        // all precompuatations in CPU

        long long ref_time = -1;

        Vector3d<double> obs_begin;
        Vector3d<double> obs_end;
        double carrington_begin;
        double carrington_end;

        for (int fc: std::array<int, 2>{0, n_files - 1})
        {
            std::fstream fits_file;

            std::cerr << "current file " << sub_matrix_filenames[fc] << " " << fc + 1 << " of " << n_files << " files\n";

            std::string fits_filename = std::string(fits_dir) + sub_matrix_filenames[fc];
            if (fits_file.open(fits_filename); !fits_file.is_open())
            {
                std::cerr << "Failed to open the FITS file " << fits_filename << "\n";
                std::exit(-1);
            }
            fits_file.close();

            // build the sub matrix here

            char *fits_header = nullptr;

            int max_header_size = 0; // maximum number of bytes in FITS header
            int header_size = 0;     // number of bytes in FITS header

            if (fits_header = fitsrhead(const_cast<char *>(fits_filename.c_str()), &max_header_size, &header_size); !fits_header)
            {
                std::cerr << "failed to read the FITS header " << sub_matrix_filenames[fc] << '\n';
                free(fits_header);
                std::exit(-1);
            }

// get the orbit here
#if defined(COR) || defined(LASCO_C2)
            auto [sun_to_obs_vec, carrington_longitude] = get_orbit(fits_header);
#elif defined(LASCO_C2) // deprecated
            auto [sun_to_obs_vec, carrington_longitude] = get_orbit(fits_header, fits_filename, solar_pole);
#endif

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

            // std::cerr << "solar pole " << solar_pole2 << "\nsun_to_obs_vec " << sun_to_obs_vec2 << '\n';
            // std::cerr << "bscale " << b_scale << " bzero " << b_zero << '\n';

            ry = Rotation(Axis::y, atan2(solar_pole2.x, solar_pole2.z));
            rz = Rotation(Axis::z, carrington_longitude);
            Rotation r23 = rz.compose(ry);
            sun_to_obs_vec = r23.rotate(sun_to_obs_vec2);

            if (fc == 0)
            {
                obs_begin = sun_to_obs_vec;
                carrington_begin = carrington_longitude;
            }
            else
            {
                obs_end = sun_to_obs_vec;
                carrington_end = carrington_longitude;
            }
        }

        std::vector<Vector3d<double>> virtual_viewpoints(n_viewpoints);
        // virtual_viewpoints = obs_begin;
        
        // calculation of virtual viewpoints
        // obs_begin = Rotation(Axis::z, carrington_begin).rotate(obs_begin);
        // obs_end = Rotation(Axis::z, carrington_end).rotate(obs_end);   

        std::cerr << "carrington begin " << carrington_begin << " carrington end " << carrington_end << '\n';
        std::cerr << "observer begin " << obs_begin << " observer end " << obs_end << '\n';

        // first scale obs_end to the same length as obs_begin
        auto r = obs_begin.norm();

        // normalize
        obs_begin = obs_begin.scale(1 / obs_begin.norm());
        obs_end = obs_end.scale(1 / obs_end.norm());

        std::cerr << "After normalize: observer begin " << obs_begin << " observer end " << obs_end << '\n';
        std::cerr << "angle between begin and end " << std::acos(obs_begin.dot(obs_end)) << '\n';
        
        // calculated the projected vector and normalize
        auto tang_vec = obs_end - obs_begin.scale(obs_end.dot(obs_begin));
        tang_vec = tang_vec.scale(1 / tang_vec.norm());

        std::cerr << "tangent vector " << tang_vec << '\n';
        std::cerr << "tangent vector * obs_begin " << tang_vec.dot(obs_begin) << '\n';

        // g(t) = r (obs_begin * cos(t) + tang_vec * sin(t))
        // g * obs_begin = r * obs_begin * obs_begin * cos(t) + 0
        // g * tang_vec = r * sin(t)
        double theta = std::acos(obs_end.dot(obs_begin));
        double theta2 = std::asin(obs_end.dot(tang_vec));
        std::cerr << obs_begin.dot(obs_begin) << '\n';
        std::cerr << (obs_begin.scale(std::cos(theta)) + tang_vec.scale(std::sin(theta))) << '\n';
        std::cerr << (obs_begin.scale(std::cos(theta2)) + tang_vec.scale(std::sin(theta2))) << '\n';
        std::cerr << obs_end << '\n';
        std::cout << "theta calculated in two ways " << theta << ' ' << theta2 << '\n';
        for (int i = 0; i < n_viewpoints; ++i)
        {
            double t = (degree < 0 ? theta : degree * DEG_TO_RAD) * i / (n_viewpoints - 1);
            virtual_viewpoints[i] = (obs_begin.scale(std::cos(t)) + tang_vec.scale(std::sin(t))).scale(r);
        }

        for (int v = 0; v < n_viewpoints; ++v)
        {
            std::fstream fits_file;

            std::cerr << "current viewpoints " << v << " \n";

            std::string fits_filename = std::string(fits_dir) + sub_matrix_filenames[0];
            if (fits_file.open(fits_filename); !fits_file.is_open())
            {
                std::cerr << "Failed to open the FITS file " << fits_filename << "\n";
                std::exit(-1);
            }
            fits_file.close();

            // build the sub matrix here

            char *fits_header = nullptr;

            int max_header_size = 0; // maximum number of bytes in FITS header
            int header_size = 0;     // number of bytes in FITS header

            if (fits_header = fitsrhead(const_cast<char *>(fits_filename.c_str()), &max_header_size, &header_size); !fits_header)
            {
                std::cerr << "failed to read the FITS header " << sub_matrix_filenames[0] << '\n';
                free(fits_header);
                std::exit(-1);
            }

            double roll_offset = 0;
            hgetr8(fits_header, "CROTA", &roll_offset);
            std::cerr << "roll_offset " << roll_offset << '\n';

            // get the arcsec/pixel conversion factor from the FITS file
            double pixel_x_size = 0;
            double pixel_y_size = 0;
            double b_scale = 1;
            double b_zero = 0;

            hgetr8(fits_header, "CDELT1", &pixel_x_size);
            hgetr8(fits_header, "CDELT2", &pixel_y_size);
            hgetr8(fits_header, "BSCALE", &b_scale);
            hgetr8(fits_header, "BZERO", &b_zero);

            std::cerr << "b_scale " << b_scale << " b_zero " << b_zero << '\n';

            if (fabs(pixel_x_size - pixel_y_size) > 0.0001)
            {
                std::cerr << "the width and length of pixel don't match\n";
                std::exit(-1);
            }

            double pixel_size = pixel_x_size;
            double carrington_longitude = 0;
            auto& sun_to_obs_vec = virtual_viewpoints[v];

            double dist_to_sun = sun_to_obs_vec.norm();
            std::cerr << "dist_to_sun " << dist_to_sun << '\n';

            double center_x = 0;
            double center_y = 0;
            hgetr8(fits_header, "CRPIX1", &center_x);
            hgetr8(fits_header, "CRPIX2", &center_y);

            // change the index from 1-based to 0-based
            --center_x;
            --center_y;

            std::cerr << pixel_size << '\n';
            std::cerr << "center_x " << center_x << " center_y " << center_y << '\n';

            // calculate the arcsec from the center to each pixel

            // zero the y component of sun_to_obs_vec
            // Rotation rz(Axis::z, -atan2(sun_to_obs_vec.y, sun_to_obs_vec.x));
            // auto sun_to_obs_vec1 = rz.rotate(sun_to_obs_vec);

            // // zero the z componenet of rz * sun_to_obs_vec
            // Rotation ry(Axis::y, -atan2(sun_to_obs_vec1.z, sun_to_obs_vec1.x));
            // Rotation ryz = ry.compose(rz);
            // // zero the y component of solor_pole
            // auto solar_pole1 = ryz.rotate(solar_pole);
            // Rotation rx(Axis::x, atan2(solar_pole1.y, solar_pole1.z));
            // Rotation r12 = rx.compose(ryz);
            // auto sun_to_obs_vec2 = r12.rotate(sun_to_obs_vec);
            // auto solar_pole2 = r12.rotate(solar_pole);

            // std::cerr << "solar pole " << solar_pole2 << "\nsun_to_obs_vec " << sun_to_obs_vec2 << '\n';
            // std::cerr << "bscale " << b_scale << " bzero " << b_zero << '\n';

            // ry = Rotation(Axis::y, atan2(solar_pole2.x, solar_pole2.z));
            // rz = Rotation(Axis::z, carrington_longitude);
            
            Rotation rz(Axis::z, -atan2(sun_to_obs_vec.y, sun_to_obs_vec.x));
            Rotation rz_inv(Axis::z, atan2(sun_to_obs_vec.y, sun_to_obs_vec.x));
            auto sun_to_obs_vec1 = rz.rotate(sun_to_obs_vec);
            Rotation ry(Axis::y, -atan2(sun_to_obs_vec1.z, sun_to_obs_vec1.x));
            Rotation ry_inv(Axis::y, atan2(sun_to_obs_vec1.z, sun_to_obs_vec1.x));
            Rotation ryz = ry.compose(rz);
            std::cerr << "final sun to the obs vector " << sun_to_obs_vec << '\n';
            std::cerr << "sun to the obs vector rotated " << ryz.rotate(sun_to_obs_vec) << '\n';
            Rotation r23 = rz_inv.compose(ry_inv);
            std::cerr << "sun to the obs vector rotated twice " << r23.rotate(ryz.rotate(sun_to_obs_vec)) << '\n';
            // Rotation r23 = rz.compose(ry);
            // sun_to_obs_vec = r23.rotate(sun_to_obs_vec2);

            // std::cerr << "solar pole 3 " << r23.rotate(solar_pole2) << '\n'
            //           << "sun_to_obs_vec3 " << r23.rotate(sun_to_obs_vec2) << '\n';

            std::cerr << "Rotation matrix r23\n"
                      << r23 << '\n';

            long long cur_time = -1;

            params[v] = {
                n_viewpoints,
                v, // file_idx
                static_cast<Params_t>(center_x),
                static_cast<Params_t>(center_y),
                static_cast<Params_t>(pixel_size),
                static_cast<Params_t>(dist_to_sun),
                static_cast<Params_t>(roll_offset),
                static_cast<Params_t>(b_scale),
                static_cast<Params_t>(b_zero),
                nullptr, // reinterpret_cast<Image_t *>(fits_image),
                solar_pole.to<Params_t>(),
                sun_to_obs_vec.to<Params_t>(),
                r23.to<Params_t>(),
                0, // ref_time
                cur_time};
            // cleaning
            free(fits_header);
        }
        // return params;
        // fetch_pb_vector_from_files(sub_matrix_filenames, params);

        return params;
    }

}
