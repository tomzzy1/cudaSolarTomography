#pragma once

#include <string>

#include "constants.hpp"

using namespace cudaSolarTomography;


struct Config {
    // Number of radial, theta (latitude), and phi (longitude) bins.
    size_t n_rad_bins;
    size_t n_theta_bins;
    size_t n_phi_bins;

    // Inner and output hollow sphere radii [Rsun].
    double r_min;
    double r_max;

    // Inner and outer data radial mask
    double instr_r_min;
    double instr_r_max;
};


struct Config& parse_json_config(struct Config& config, const std::string& json_fname);


const Config DEFAULT_CONFIG = {
    N_RAD_BINS,
    N_THETA_BINS,
    N_PHI_BINS,
    R_MIN,
    R_MAX,
    INSTR_R_MIN,
    INSTR_R_MAX
};
