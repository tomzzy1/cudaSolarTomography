#include <string>
#include <cassert>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <sstream>

#include <argparse/argparse.hpp>
#include <indicators/cursor_control.hpp>
#include <indicators/progress_bar.hpp>

#include "FITS_Image.hpp"
#include "MAT_File.hpp"
#include "coordinates.hpp"
#include "HollowSphere.hpp"
#include "scatter.hpp"
#include "Config.hpp"


constexpr double DIFF_EPSILON = 1e-9;

constexpr size_t BUFFER_LENGTH = 4096;

// typedef float VALUE_T;
typedef double VALUE_T;


// TODO: change things like int IMAGE_SIZE to size_t IMAGE_SIZE in constants.hpp


static void
output_fts_fnames(MAT_File& mat_file, const std::vector<std::filesystem::path>& fts_fnames) {
    char buffer[BUFFER_LENGTH];
    std::ostringstream ss;

    size_t dims[2];
    dims[0] = 1;

    size_t count = 0;
    for (auto fts_fname : fts_fnames) {
        std::ostringstream ss;
        // std::setw(3) sets max number of FITS files to 100!
        ss << "fts_fname_" << std::setw(3) << std::setfill('0') << count;

        const std::string& path_str = std::filesystem::canonical(fts_fname).string();
        assert(path_str.size() < PATH_MAX);

        path_str.copy(buffer, path_str.size());
        dims[1] = path_str.size();

        matvar_t *matvar = Mat_VarCreate(ss.str().c_str(), MAT_C_CHAR, MAT_T_UTF8, 2, dims, buffer, 0);
        assert(matvar);
        assert(Mat_VarWriteAppend(mat_file.matfp, matvar, MAT_COMPRESSION_ZLIB, 2) == 0);
        Mat_VarFree(matvar);

        count++;
    }
}


static std::vector<std::string>
parse_fts_list(const std::string& fts_list_fname) {
    std::vector<std::string> fts_fnames;
    std::ifstream file(fts_list_fname);
    assert(file.is_open());
    std::string line;
    std::getline(file, line);
    std::stringstream sstream(line);
    size_t n_fts;
    sstream >> n_fts;
    for (size_t i = 0; i < n_fts; i++) {
        std::getline(file, line);
        fts_fnames.push_back(line);
    }
    file.close();
    return fts_fnames;
}


int main(int argc, char* argv[]) {
    using namespace indicators;

    argparse::ArgumentParser program("fts2mat");
    program.add_description("Process a set of FITS image files to generate the corresponding observation matrix for use in tomography reconstruction.");

    program.add_argument("mat_fname")
        .help("Output MATLAB filename.");

    program.add_argument("config_json")
        .help("JSON configuration file.");

    auto& fts_fnames_or_ftslist_fname = program.add_mutually_exclusive_group(true);

    fts_fnames_or_ftslist_fname.add_argument("-f", "--fts-fname")
        .nargs(argparse::nargs_pattern::any)
        .append()
        .default_value(std::vector<const std::string>{})
        .help("FITS filename.");

    fts_fnames_or_ftslist_fname.add_argument("-l", "--fts-list-fname")
        .default_value(std::string{})
        .help("FITS filename list file (line 1: # of FITS records, and each subsequent line is a FITS filename).");

    program.add_argument("-p", "--path")
        .default_value(std::string("."))
        .help("Base path for the FITS files.");

    program.add_argument("-A", "--A-key")
        .default_value(std::string("A"))
        .help("Key associated with sparse observation matrix stored in the output MATLAB file.");

    auto& bin_or_oversample = program.add_mutually_exclusive_group(false);

    bin_or_oversample.add_argument("-b", "--binning-factor")
        .help("Binning factor: if one argument is provided, the same binning factor is applied to rows and columns and if two arguments are provided, the first and second arguments specify the row and column binning factors, respectively.")
        .nargs(1, 2)
        .default_value(std::vector<size_t>{1, 1})
        .scan<'i', size_t>();

    bin_or_oversample.add_argument("-o", "--oversample-factor")
        .help("Oversample factor: if one argument is provided, the same oversample factor is applied to rows and columns and if two arguments are provided, the first and second arguments specify the row and column oversample factors, respectively.")
        .nargs(1, 2)
        .default_value(std::vector<size_t>{1, 1})
        .scan<'i', size_t>();

    auto& builda_or_wcs = program.add_mutually_exclusive_group(true);

    builda_or_wcs.add_argument("-a", "--builda")
        .help("Use the builda sun-observer vector calculation approach.")
        .flag();

    builda_or_wcs.add_argument("-w", "--wcs")
        .help("Use world coordinate system (wcslib) sun-observer vector calculation approach.")
        .flag();

    program.add_epilog("Either the -f (--fts-fname) or -l (--fts-list-fname) option must be specified.\n\nEither the builda (approximate) or wcs (full world coordinate system) option must be specified. In the case of wcs, the apparent Carrington longitude is determined from the CRLN_OBS and HGLN_OBS FITS header entries.");

    try {
        program.parse_args(argc, argv);
    }
    catch (const std::exception& err) {
        std::cerr << err.what() << std::endl;
        std::cerr << program;
        return 1;
    }

    Config config = {};

    std::string config_json = program.get<std::string>("config_json");
    parse_json_config(config, config_json);

    std::filesystem::path fts_path(program.get<std::string>("path"));

    size_t binning_factor_j;
    size_t binning_factor_i;

    if (program.get<std::vector<size_t>>("binning-factor")[1] == 0) {
        binning_factor_i = binning_factor_j = program.get<std::vector<size_t>>("binning-factor")[0];
    }
    else {
        binning_factor_i = program.get<std::vector<size_t>>("binning-factor")[0];
        binning_factor_j = program.get<std::vector<size_t>>("binning-factor")[1];
    }

    size_t oversample_factor_j;
    size_t oversample_factor_i;

    if (program.get<std::vector<size_t>>("oversample-factor")[1] == 0) {
        oversample_factor_i = oversample_factor_j = program.get<std::vector<size_t>>("oversample-factor")[0];
    }
    else {
        oversample_factor_i = program.get<std::vector<size_t>>("oversample-factor")[0];
        oversample_factor_j = program.get<std::vector<size_t>>("oversample-factor")[1];
    }

    MAT_File mat_file(program.get<std::string>("mat_fname").c_str());

    std::vector<std::string> fts_fnames;
    if (program.get<std::string>("fts-list-fname").empty()) {
        fts_fnames = program.get<std::vector<std::string>>("fts-fname");
    }
    else {
        fts_fnames = parse_fts_list(program.get<std::string>("fts-list-fname"));
    }

    std::stringstream csr_data_key_ss;
    csr_data_key_ss << program.get<std::string>("A-key") << "_csr_data";
    std::string csr_data_key(csr_data_key_ss.str());

    std::stringstream csr_indices_key_ss;
    csr_indices_key_ss << program.get<std::string>("A-key") << "_csr_indices";
    std::string csr_indices_key(csr_indices_key_ss.str());

    std::stringstream csr_indptr_key_ss;
    csr_indptr_key_ss << program.get<std::string>("A-key") << "_csr_indptr";
    std::string csr_indptr_key(csr_indptr_key_ss.str());

    std::stringstream block_indptr_key_ss;
    block_indptr_key_ss << program.get<std::string>("A-key") << "_block_indptr";
    std::string block_indptr_key(block_indptr_key_ss.str());

    std::stringstream shape_key_ss;
    shape_key_ss << program.get<std::string>("A-key") << "_csr_shape";
    std::string shape_key(shape_key_ss.str());

    std::vector<size_t> block_indptr;
    block_indptr.reserve(fts_fnames.size() + 1);
    block_indptr.push_back(0);

    std::vector<size_t> csr_indptr;
    csr_indptr.push_back(0);

    sun_obs_frame frame;
    struct fits_header_info info;
    std::unique_ptr<WCS> wcs_ptr;


    HollowSphere hollow_sphere(config.n_rad_bins,
                               config.n_theta_bins,
                               config.n_phi_bins,
                               config.r_min,
                               config.r_max);

    std::map<size_t, VALUE_T> row_map;

    std::vector<VALUE_T> block_data;
    std::vector<size_t> block_indices;

    std::vector<VALUE_T> block_y;
    std::vector<size_t> block_y_idx;

    bool first_fts_fname_it = true;
    std::array<size_t, 2> naxes {};

    show_console_cursor(false);

    ProgressBar bar{
        option::BarWidth{50},
        option::Start{"["},
        option::Fill{"="},
        option::Lead{">"},
        option::Remainder{" "},
        option::End{"]"},
        option::ShowElapsedTime{true},
        option::ShowRemainingTime{true}
    };

    std::vector<std::filesystem::path> fts_fullpaths;

    // If this fails, increase setw in ouput_fts_fnames
    assert(fts_fnames.size() < 100);

    // The below would be better accomplished with enumerate:
    // https://godbolt.org/z/yS2AkG

    size_t count = 0;
    for (const std::string& fts_fname : fts_fnames) {
        fts_fullpaths.push_back(std::filesystem::canonical(fts_path / std::filesystem::path(fts_fname)));

        bar.set_option(option::PostfixText{fts_fullpaths.back().filename()});
        bar.set_progress(count++ / double(fts_fnames.size()) * 100);

        std::unique_ptr<FITS_Image> fits_image = load_FITS_image(fts_fullpaths.back());

        if (first_fts_fname_it) {
            naxes = fits_image->naxes;
            first_fts_fname_it = false;
        }
        else {
            assert(naxes == fits_image->naxes);
        }

        if (program["--builda"] == true) {
            get_fits_header_info(fts_fullpaths.back().c_str(), &info);
        }
        else if (program["--wcs"] == true) {
            wcs_ptr = std::unique_ptr<WCS>(new WCS(fts_fullpaths.back()));
        }
        else {
            assert(false);
        }

        block_data.clear();
        block_indices.clear();

        block_y.clear();
        block_y_idx.clear();

        // disregard the last horizontal bin if it is not full
        const size_t mod_i = fits_image->naxes[1] % binning_factor_i == 0 ? 0 : binning_factor_i;
        for (size_t i = 0; i < fits_image->naxes[1] - mod_i; i += binning_factor_i) {
            // disregard the last vertical bin if it is not full
            const size_t mod_j = fits_image->naxes[0] % binning_factor_j == 0 ? 0 : binning_factor_j;
            for (size_t j = 0; j < fits_image->naxes[0] - mod_j; j += binning_factor_j){
                double y_row = 0;
                std::map<size_t, double> row_map;
                size_t n_los = 0;

                for (size_t k = 0; k < binning_factor_i; k++) {
                    size_t row = i + k;
                    for (size_t y_k = 0; y_k < oversample_factor_i; y_k++) {
                        double y = row - 0.5 + 1 / (2 * oversample_factor_i) + y_k / oversample_factor_i;
                        for (size_t l = 0; l < binning_factor_j; l++) {
                            size_t col = j + l;
                            size_t I_index = row * (fits_image->naxes[0]) + col;

                            for (size_t x_l = 0; x_l < oversample_factor_j; x_l++) {
                                double x = col - 0.5 + 1 / (2 * oversample_factor_j) + x_l / oversample_factor_j;

                                if (program["--builda"] == true) {
                                    frame = get_builda_coordinates(info, y, x);
                                }
                                else if (program["--wcs"] == true) {
                                    frame = wcs_ptr->get_frame(y, x, Unit::Rsun);
                                }
                                else {
                                    assert(false);
                                }

                                double impact = Length(frame.nrpt);
                                double impact2 = impact * impact;

                                if ((impact < config.r_min) || (impact > config.r_max)) {
                                    continue;
                                }

                                gte::Ray3<double> obs_ray(frame.sun_obs, frame.unit);

                                std::vector<double> intersect_times = hollow_sphere.intersect(obs_ray);
                                std::sort(intersect_times.begin(), intersect_times.end());

                                if (intersect_times.size() >= 2) {
                                    y_row += (*fits_image)[I_index] * SCALE_FACTOR;
                                    n_los++;
                                    for (size_t time_idx = 0; time_idx < intersect_times.size() - 1; time_idx++) {
                                        double segment_length = intersect_times[time_idx + 1] - intersect_times[time_idx];
                                        if (std::abs(segment_length) < DIFF_EPSILON) {
                                            continue;
                                        }
                                        double t_mid = intersect_times[time_idx] + segment_length / 2;
                                        gte::Vector3<double> xyz_mid = frame.sun_obs + t_mid * frame.unit;
                                        gte::Vector3<size_t> ijk = hollow_sphere.xyz2ijk(xyz_mid);

                                        double r_mid = Length(xyz_mid);

                                        size_t I = hollow_sphere.ijk2I(ijk);
                                        double v = thomson_scatter(r_mid, impact2) * segment_length * SUN_RADIUS_CM;

                                        auto [it, success] = row_map.insert({I, v});
                                        if (!success) {
                                            it->second += v;
                                        }
                                    } // for time_idx (LOS times of intersection with voxel grid boundaries)
                                } // if (intersect_times.size() >= 2) (if LOS intersects a voxel)
                            } // for x_l (0, ..., oversample factor - 1)
                        } // for l (0, ..., binning factor - 1)
                    }  // for y_k (0, ..., oversample factor - 1)
                } // for k (0, ..., binning_factor - 1)

                if (n_los > 0) {
                    for (auto v_j : row_map) {
                        block_data.push_back(v_j.second / n_los);
                        block_indices.push_back(v_j.first);
                    }

                    size_t I_binned_index = (i / binning_factor_i) * ceil((fits_image->naxes[0] - mod_j) / binning_factor_j) + floor(j / binning_factor_j);

                    block_y.push_back(y_row / n_los);
                    block_y_idx.push_back(I_binned_index);

                    csr_indptr.push_back(csr_indptr.back() + row_map.size());
                }
            } // for j (FITS image column incremented by binning factor)
        } // for i (FITS image row incremented by binning factor)

        // The code will crash here if block_data.size() == 0
        mat_file.append(csr_data_key.c_str(), block_data.data(), block_data.size());
        mat_file.append(csr_indices_key.c_str(), block_indices.data(), block_indices.size());

        mat_file.append("y", block_y.data(), block_y.size());
        mat_file.append("y_idx", block_y_idx.data(), block_y_idx.size());

        block_indptr.push_back(block_indptr.back() + block_y.size());
    } // for fts_fnames (FITS file)
    bar.set_progress(100);
    show_console_cursor(true);

    mat_file.append(csr_indptr_key.c_str(), csr_indptr.data(), csr_indptr.size());
    mat_file.append(block_indptr_key.c_str(), block_indptr.data(), block_indptr.size());

    double _R_MAX = config.r_max;
    double _R_MIN = config.r_min;

    mat_file.append("R_MAX", &_R_MAX, 1);
    mat_file.append("R_MIN", &_R_MIN, 1);

    size_t _N_RAD_BINS = config.n_rad_bins;
    size_t _N_THETA_BINS = config.n_theta_bins;
    size_t _N_PHI_BINS = config.n_phi_bins;
    size_t _N_BINS = config.n_rad_bins * config.n_theta_bins * config.n_phi_bins;

    mat_file.append("N_RAD_BINS", &_N_RAD_BINS, 1);
    mat_file.append("N_THETA_BINS", &_N_THETA_BINS, 1);
    mat_file.append("N_PHI_BINS", &_N_PHI_BINS, 1);
    mat_file.append("N_BINS", &_N_BINS, 1);

    double _INSTR_R_MAX = config.instr_r_max;
    double _INSTR_R_MIN = config.instr_r_min;

    mat_file.append("INSTR_R_MAX", &_INSTR_R_MAX, 1);
    mat_file.append("INSTR_R_MIN", &_INSTR_R_MIN, 1);

    assert(binning_factor_i == binning_factor_j);
    size_t _BINNING_FACTOR = binning_factor_i;
    mat_file.append("BINNING_FACTOR", &_BINNING_FACTOR, 1);

    assert(oversample_factor_i == oversample_factor_j);
    size_t _OVERSAMPLE_FACTOR = oversample_factor_i;
    mat_file.append("OVERSAMPLE_FACTOR", &_OVERSAMPLE_FACTOR, 1);

    assert(naxes[0] == naxes[1]);
    size_t IMAGE_SIZE = naxes[0];

    mat_file.append("IMAGE_SIZE", &IMAGE_SIZE, 1);

    size_t shape[2] = {block_indptr.back(), _N_BINS};
    mat_file.append(shape_key.c_str(), shape, 2);

    output_fts_fnames(mat_file, fts_fullpaths);

    return 0;
}
