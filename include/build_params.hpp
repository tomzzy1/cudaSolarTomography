#pragma once

namespace cudaSolarTomography
{
    
// #define COR
#define LASCO_C2
#if defined(COR)
    constexpr double INSTR_R_MAX = 4.2; // unit is Rsun
    constexpr double INSTR_R_MIN = 1.6; // unit is Rsun

    constexpr int IMAGE_SIZE = 1024;
    constexpr double PIXEL_SIZE = 7.5 * 1024 / IMAGE_SIZE;
    constexpr int BINNING_FACTOR = 1;
    constexpr double SCALE_FACTOR = 1e10;

    constexpr int BIN_SCALE_FACTOR = 1;

    constexpr int N_RAD_BINS = 20 * BIN_SCALE_FACTOR;
    constexpr int N_THETA_BINS = 30 * BIN_SCALE_FACTOR;
    constexpr int N_PHI_BINS = 2 * N_THETA_BINS;
    // constexpr int N_RAD_BINS = 40;
    // constexpr int N_THETA_BINS = 30;
    // constexpr int N_PHI_BINS = 60;

    // outer and inner radius of the computation ball (So it's a hollow sphere)
    constexpr double R_MAX = 3.5; // unit is Rsun
    constexpr double R_MIN = 1.6; // unit is Rsun

    constexpr std::string_view config_dir = "../config/20080201-20080214_cor1a.conf";
    constexpr std::string_view fits_dir = "../data/cor1a1/";

    using Image_t = float;

#elif defined(LASCO_C2)
    constexpr double INSTR_R_MAX = 6.3; // unit is Rsun
    constexpr double INSTR_R_MIN = 2.1; // unit is Rsun
    // constexpr double INSTR_R_MAX = 6.2; // unit is Rsun
    // constexpr double INSTR_R_MIN = 2.5; // unit is Rsun

    constexpr int IMAGE_SIZE = 512;
    constexpr double PIXEL_SIZE = 23.8 * 512 / IMAGE_SIZE;
    constexpr int BINNING_FACTOR = 1;
    constexpr double SCALE_FACTOR = 1e10;

    constexpr int BIN_SCALE_FACTOR = 1;

    // constexpr int N_RAD_BINS = 20 * BIN_SCALE_FACTOR;
    // constexpr int N_THETA_BINS = 30 * BIN_SCALE_FACTOR;
    // constexpr int N_PHI_BINS = 2 * N_THETA_BINS;

    // constexpr int N_RAD_BINS = 50;
    // constexpr int N_THETA_BINS = 75;
    // constexpr int N_PHI_BINS = 150;

    // constexpr int N_RAD_BINS = 38;
    // constexpr int N_THETA_BINS = 72;
    // constexpr int N_PHI_BINS = 150;

    // constexpr int N_RAD_BINS = 76;
    // constexpr int N_THETA_BINS = 143;
    // constexpr int N_PHI_BINS = 300;

    // constexpr int N_RAD_BINS = 30;
    // constexpr int N_THETA_BINS = 101;
    // constexpr int N_PHI_BINS = 129;

    // constexpr int N_RAD_BINS = 30;
    // constexpr int N_THETA_BINS = 72;
    // constexpr int N_PHI_BINS = 150;

    // constexpr int N_RAD_BINS = 30;
    // constexpr int N_THETA_BINS = 143;
    // constexpr int N_PHI_BINS = 300;

    // constexpr int N_RAD_BINS = 76;
    // constexpr int N_THETA_BINS = 72;
    // constexpr int N_PHI_BINS = 150;

    // constexpr int N_RAD_BINS = 30;
    // constexpr int N_THETA_BINS = 36;
    // constexpr int N_PHI_BINS = 75;

    // constexpr int N_RAD_BINS = 45;
    // constexpr int N_THETA_BINS = 180;
    // constexpr int N_PHI_BINS = 360;

    // constexpr int N_RAD_BINS = 45;
    // constexpr int N_THETA_BINS = 360;
    // constexpr int N_PHI_BINS = 720;

    constexpr int N_RAD_BINS = 30;
    constexpr int N_THETA_BINS = 75;
    constexpr int N_PHI_BINS = 150;

    // constexpr int N_RAD_BINS = 45;
    // constexpr int N_THETA_BINS = 90;
    // constexpr int N_PHI_BINS = 180;

    // constexpr int N_RAD_BINS = 45;
    // constexpr int N_THETA_BINS = 120;
    // constexpr int N_PHI_BINS = 240;

    // outer and inner radius of the computation ball (So it's a hollow sphere)
    constexpr double R_MAX = 6.5; // unit is Rsun
    constexpr double R_MIN = 2.0; // unit is Rsun

    constexpr std::string_view config_dir = "../config/lasco_c2_2023.conf";
    constexpr std::string_view fits_dir = "../data/lasco_c2_2023/";
    // constexpr std::string_view config_dir = "../config/lasco_c2_2023_partial.conf";
    // constexpr std::string_view fits_dir = "../data/lasco_c2_2023_partial/";
    // constexpr std::string_view config_dir = "../config/lasco_c2.conf";
    // constexpr std::string_view fits_dir = "../data/lasco_c2/";
    // constexpr std::string_view config_dir = "../config/lasco_c2_partial.conf";
    // constexpr std::string_view fits_dir = "../data/lasco_c2_partial/";
    // constexpr std::string_view config_dir = "../config/lasco_c2_2007_partial.conf";
    // constexpr std::string_view fits_dir = "../data/lasco_c2_2007_partial/";
    // constexpr std::string_view config_dir = "../config/my_lasco_c2.conf";
    // constexpr std::string_view fits_dir = "../data/my_lasco_c2/";
    // constexpr std::string_view config_dir = "../data/temp.conf";
    // constexpr std::string_view fits_dir = "../data/temp/";

    using Image_t = double;

#endif

}
