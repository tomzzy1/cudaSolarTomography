#include <fstream>
#include <iostream>
#include <string>

#include <map>
#include <algorithm>
#include <span>

#include <argparse/argparse.hpp>

#include "coordinates.hpp"
#include "HollowSphere.hpp"
#include "scatter.hpp"

#include "constants.hpp"

using namespace gte;
using namespace cudaSolarTomography;


constexpr double DIFF_EPSILON = 1e-9;


int main(int argc, char *argv[]) {
    argparse::ArgumentParser program("los");
    program.add_description("Calculate one row of the solar tomography matrix given a FITS file and the (i,j) pixel.");

    program.add_argument("fts_fname")
        .help("FITS filename");

    program.add_argument("i")
        .help("pixel row")
        .scan<'f', double>();

    program.add_argument("j")
        .help("pixel column")
        .scan<'f', double>();

    program.add_argument("-b", "--binning-factor")
        .help("binning factor: if one argument is provided, the same binning factor is applied to rows and columns and if the arguments are provided, the first and second arguments specify the row and column binning factors, respectively")
        .nargs(1, 2)
        .default_value(std::vector<size_t>{1, 1})
        .scan<'i', size_t>();

    auto &builda_or_wcs = program.add_mutually_exclusive_group(true);

    builda_or_wcs.add_argument("-a", "--builda")
        .help("use builda approach")
        .flag();

    builda_or_wcs.add_argument("-w", "--wcs")
        .help("use world coordinate system")
        .flag();

    program.add_epilog("Either the builda (approximate) or wcs (full world coordinate system) option must be specified. In the case of wcs, the apparent Carrington longitude is determined from the CRLN_OBS and HGLN_OBS FITS header entries.");

    try {
        program.parse_args(argc, argv);
    }
    catch (const std::exception& err) {
        std::cerr << err.what() << std::endl;
        std::cerr << program;
        return 1;
    }

    size_t binning_j;
    size_t binning_i;

    if (program.get<std::vector<size_t>>("binning-factor")[1] == 0) {
        binning_i = binning_j = program.get<std::vector<size_t>>("binning-factor")[0];
    }
    else {
        binning_i = program.get<std::vector<size_t>>("binning-factor")[0];
        binning_j = program.get<std::vector<size_t>>("binning-factor")[1];
    }

    std::string fts_fname = program.get<std::string>("fts_fname");

    sun_obs_frame frame;
    struct fits_header_info info;
    WCS* wcs_ptr = nullptr;

    if (program["--builda"] == true) {
        get_fits_header_info(fts_fname.c_str(), &info);
    }
    else if (program["--wcs"] == true) {
        wcs_ptr = new WCS(fts_fname);
    }
    else {
        assert(false);
    }

    size_t i0 = program.get<double>("i");
    size_t i1 = i0 + binning_i;
    size_t j0 = program.get<double>("j");
    size_t j1 = j0 + binning_j;

    HollowSphere hollow_sphere(N_RAD_BINS, N_THETA_BINS, N_PHI_BINS, R_MIN, R_MAX);

    std::map<size_t, double> row_map;

    for (size_t i = i0; i < i1; i++) {
        for (size_t j = j0; j < j1; j++) {
            if (program["--builda"] == true) {
                frame = get_builda_coordinates(info, i, j);
            }
            else if (program["--wcs"] == true) {
                frame = wcs_ptr->get_frame(i, j, Unit::Rsun);
            }
            else {
                assert(false);
            }

            double impact = Length(frame.nrpt);
            double impact2 = impact * impact;

            if (impact < R_MIN) {
                std::cerr << "line of sight (" << i << ", " << j << ") has impact parameter "
                          << impact << " [Rsun] < R_MIN=" << R_MIN << " [Rsun]\n";
                continue;
            }
            else if (impact > R_MAX) {
                std::cerr << "line of sight (" << i << ", " << j << ") has impact parameter "
                          << impact << " [Rsun] > R_MAX=" << R_MAX << " [Rsun]\n";
                continue;
            }

            Ray3<double> obs_ray(frame.sun_obs, frame.unit);

            std::vector<double> intersect_times = hollow_sphere.intersect(obs_ray);
            std::sort(intersect_times.begin(), intersect_times.end());

            if (intersect_times.size() >= 2) {
                for (size_t k = 0; k < intersect_times.size() - 1; k++) {
                    double segment_length = intersect_times[k+1] - intersect_times[k];
                    if (std::abs(segment_length) < DIFF_EPSILON) {
                        continue;
                    }
                    double t_mid = intersect_times[k] + segment_length / 2;
                    Vector3<double> xyz_mid = frame.sun_obs + t_mid * frame.unit;
                    Vector3<size_t> ijk = hollow_sphere.xyz2ijk(xyz_mid);

                    double r_mid = Length(xyz_mid);

                    size_t I = hollow_sphere.ijk2I(ijk);
                    double v = thomson_scatter(r_mid, impact2) * segment_length * SUN_RADIUS_CM;

                    auto [it, success] = row_map.insert({I, v});
                    if (!success) {
                        it->second += v;
                    }
                }
            }
        }
    }

    // cleanup
    delete wcs_ptr;

    std::cout << row_map.size() << '\n';
    std::cout << std::setprecision(100);
    for (auto v_j : row_map) {
        std::cout << v_j.first << '\t' << v_j.second << '\n';
    }

    return 0;
}
