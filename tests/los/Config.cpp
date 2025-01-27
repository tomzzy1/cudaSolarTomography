#include <fstream>
#include <nlohmann/json.hpp>

#include "Config.hpp"


using namespace cudaSolarTomography;

using json = nlohmann::json;


struct Config& parse_json_config(struct Config& config, const std::string& json_fname) {
    std::ifstream f(json_fname);
    json data = json::parse(f);

    config.n_rad_bins = data["N_RAD_BINS"];
    config.n_theta_bins = data["N_THETA_BINS"];
    config.n_phi_bins = data["N_PHI_BINS"];

    config.r_min = data["R_MIN"];
    config.r_max = data["R_MAX"];

    config.instr_r_min = data["INSTR_R_MIN"];
    config.instr_r_max = data["INSTR_R_MAX"];

    return config;
}
