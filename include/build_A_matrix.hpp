#include "vector_maths.hpp"
#include <cassert>
#include "build_params.hpp"
#include <chrono>
#include <charconv>

namespace cudaSolarTomography
{
#if defined(LASCO_C2)
    constexpr std::string_view orbit_dir = "../data/orbit/";
#endif

#if defined(COR)
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

        carrington_longitude *= DEG_TO_RAD; // convert the unit of carlington longitude

        /* the J2000.0 angle between the Ecliptic and mean Equatorial planes
         *  is 23d26m21.4119s - From Allen's Astrophysical Quantities, 4th ed. (2000)  */

        Rotation rx(Axis::x, J20000_ANGLE);

        auto sun_to_obs_vec = rx.rotate(observatory.scale(0.001 / SUN_RADIUS_KM)); // convert the unit to km then to radisu_of_sun

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
            fgets(buffer.data(), 512, orbit_file);
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
        std::cerr << "earth to sun " << earth_to_sun_vec << " earth to soho " << earth_to_soho_vec << '\n';

        auto sun_to_obs_vec = earth_to_soho_vec - earth_to_sun_vec;

        // calculate angle between earth and soho, projected onto equ. plane
        // e = solar_pole * (solar_pole .* earth_sun) - earth_sun
        // o = sun_obs - solar_pole * (solar_pole .* sun_obs)
        // Then get the direction vector of e and o
        auto e = solar_pole.scale(solar_pole.dot(earth_to_sun_vec)) - earth_to_sun_vec;
        e = e.scale(1 / e.norm());
        std::cerr << "e " << e << '\n';
        auto o = sun_to_obs_vec - solar_pole.scale(solar_pole.dot(sun_to_obs_vec));
        o = o.scale(1 / o.norm());
        std::cerr << "o " << o << '\n';

        auto c = e.cross(o);
        std::cerr << "c " << c << '\n';
        std::cerr << "ecliptic longitude " << ecliptic_longitude << '\n';
        double carrington_longitude = ecliptic_longitude * DEG_TO_RAD + asin(solar_pole.dot(c));
        return {sun_to_obs_vec.scale(1 / SUN_RADIUS_KM), carrington_longitude};
    }

#endif

    template <typename T, size_t s1, size_t s2, size_t s3>
    auto three_way_merge(const std::array<T, s1> &arr1, const std::array<T, s2> &arr2, const std::array<T, s3> &arr3, int begin1, int end1, int begin2,
                         int end2, int begin3, int end3, T entry_time, T exit_time)
    {
        std::array<T, s1 + s2 + s3 + 3> res;
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
    }

}
