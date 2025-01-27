#include "scatter.hpp"

#include <cmath>

#include "constants.hpp"

using namespace cudaSolarTomography;


double
thomson_scatter(double r, double impact2) {
    double ratio = impact2 / r / r;
    double s_gamma = 1.0 / r;
    double s_gamma2 = s_gamma * s_gamma;
    double c_gamma = sqrt(1 - s_gamma2);
    double c_gamma2 = c_gamma * c_gamma;
    double vdhA = c_gamma * s_gamma2;
    double vdhB = 3 * s_gamma2 - 1 + (c_gamma2 / s_gamma) * (4 - 3 * c_gamma2) * log((1 + s_gamma) / c_gamma);
    vdhB /= 8;
    return ((1 - Q_LIMB) * vdhA + Q_LIMB * vdhB) / (1 - Q_LIMB / 3) * ratio * THOMPSON_CONSTANT;
}
