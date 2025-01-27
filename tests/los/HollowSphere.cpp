#include <cstddef>

#include "Mathematics/IntrRay3Sphere3.h"
#include "Mathematics/IntrRay3Plane3.h"
#include "Mathematics/IntrRay3Cone3.h"
#include "Mathematics/DistLineRay.h"

#include "HollowSphere.hpp"
#include "coordinates.hpp"
#include "constants.hpp"

using namespace gte;
using namespace cudaSolarTomography;


constexpr double LINE_EPSILON = 1e-9;


HollowSphere::HollowSphere(size_t nRadBins, size_t nThetaBins, size_t nPhiBins, double rMin, double rMax)
        :
        n_rad_bins(nRadBins),
        n_theta_bins(nThetaBins),
        n_phi_bins(nPhiBins),
        r_min(rMin),
        r_max(rMax),
        rad_bin_size((rMax - rMin) / nRadBins),
        theta_bin_size(std::numbers::pi / nThetaBins),
        phi_bin_size(PI_TWO / nPhiBins),
        theta_0_bin_edge({0, 0, 0}, {0, 0, 1}),
        theta_90_bin_edge({0, 0, 0}, {0, 0, 1})
{
    radial_bin_edges.reserve(n_rad_bins + 1);
    for (size_t i = 0; i < n_rad_bins + 1; i++) {
        radial_bin_edges.push_back(Sphere3<double>({0, 0, 0}, radEdge(i)));
    }

    phi_bin_edges.reserve(n_phi_bins);
    for (size_t i = 0; i < n_phi_bins; i++) {
        phi_bin_edges.push_back(Plane3<double>({-sin(i * phi_bin_size), cos(i * phi_bin_size), 0},
                                               {0, 0, 0}));
    }

    Ray3<double> z_positive({0, 0, 0}, {0, 0, 1});
    Ray3<double> z_negative({0, 0, 0}, {0, 0, -1});
    theta_bin_edges.reserve(n_theta_bins - 2);
    for (size_t i = 0; i < size_t(n_theta_bins/2) - 1; i++) {
        double angle = (i+1) * theta_bin_size;
        theta_bin_edges.push_back(Cone3<double>(z_positive, PI_HALF - angle));
        theta_bin_edges.push_back(Cone3<double>(z_negative, PI_HALF - angle));
    }
}


std::vector<double> HollowSphere::intersect(const Ray3<double>& ray3) {
    std::vector<double> intersect_times;
    intersect_times.reserve(2 * std::max(n_rad_bins, std::max(n_theta_bins, n_phi_bins)));

    FIQuery<double, Ray3<double>, Sphere3<double>> r_max_query;
    FIQuery<double, Ray3<double>, Sphere3<double>>::Result r_max_result = r_max_query(ray3, radial_bin_edges.back());

    if (r_max_result.intersect && r_max_result.numIntersections == 2) {
        double t_enter = r_max_result.parameter[0];
        double t_exit = r_max_result.parameter[1];

        for (const auto& radial_bin_edge : radial_bin_edges) {
            FIQuery<double, Ray3<double>, Sphere3<double>> query;
            FIQuery<double, Ray3<double>, Sphere3<double>>::Result result = query(ray3, radial_bin_edge);
            if (result.intersect) {
                for (size_t i = 0; i < result.numIntersections; i++) {
                    double t = result.parameter[i];
                    if ((t >= t_enter) && (t <= t_exit)) {
                        intersect_times.push_back(result.parameter[i]);
                    }
                }
            }
        }

        for (const auto& phi_bin_edge : phi_bin_edges) {
            FIQuery<double, Ray3<double>, Plane3<double>> query;
            FIQuery<double, Ray3<double>, Plane3<double>>::Result result = query(ray3, phi_bin_edge);
            if (result.intersect) {
                double t = result.parameter;
                if ((t >= t_enter) && (t <= t_exit)) {
                    intersect_times.push_back(result.parameter);
                }
            }
        }

        for (const auto& theta_bin_edge : theta_bin_edges) {
            FIQuery<double, Ray3<double>, Cone3<double>> query;
            FIQuery<double, Ray3<double>, Cone3<double>>::Result result = query(ray3, theta_bin_edge);
            switch (result.type) {
            case result.isEmpty:
            case result.isPoint:
                break;
            case result.isSegment: {
                double t1 = result.t[0].x[0] + result.t[0].x[1] * sqrt(result.t[0].d);
                if ((t1 >= t_enter) && (t1 <= t_exit)) {
                    intersect_times.push_back(t1);
                }
                double t2 = result.t[1].x[0] + result.t[1].x[1] * sqrt(result.t[1].d);
                if ((t2 >= t_enter) && (t2 <= t_exit)) {
                    intersect_times.push_back(t2);
                }
                break;
            }
            case result.isRayPositive: {
                double t = result.t[0].x[0] + result.t[0].x[1] * sqrt(result.t[0].d);
                if ((t >= t_enter) && (t <= t_exit)) {
                    intersect_times.push_back(t);
                }
                break;
            }
            case result.isRayNegative: {
                double t = result.t[1].x[0] + result.t[1].x[1] * sqrt(result.t[1].d);
                if ((t >= t_enter) && (t <= t_exit)) {
                    intersect_times.push_back(t);
                }
                break;
            }
            default:
                assert(false);
            }
        }

        FIQuery<double, Ray3<double>, Plane3<double>> query_theta_0_bin_edge;
        FIQuery<double, Ray3<double>, Plane3<double>>::Result result_theta_0_bin_edge = query_theta_0_bin_edge(ray3, theta_0_bin_edge);
        if (result_theta_0_bin_edge.intersect) {
            double t = result_theta_0_bin_edge.parameter;
            if ((t >= t_enter) && (t <= t_exit)) {
                intersect_times.push_back(result_theta_0_bin_edge.parameter);
            }
        }

        DCPLine3Ray3<double> query_theta_90_bin_edge;
        DCPLine3Ray3<double>::Result result_theta_90_bin_edge = query_theta_90_bin_edge(theta_90_bin_edge, ray3);
        // CHECK THAT THIS ACTUALLY WORKS
        if (result_theta_90_bin_edge.distance < LINE_EPSILON) {
            double t = result_theta_90_bin_edge.parameter[1];
            if ((t >= t_enter) && (t <= t_exit)) {
                intersect_times.push_back(t);
            }
        }
    }

    return intersect_times;
}

double HollowSphere::radCenter(size_t i) const {
    assert(i >= 0 && i < n_rad_bins);
    return r_min + rad_bin_size/2 + i*rad_bin_size;
}

double HollowSphere::radEdge(size_t i) const {
    assert(i >= 0 && i <= n_rad_bins);
    return r_min + i*rad_bin_size;
}

double HollowSphere::thetaCenter(size_t i) const {
    assert(i >= 0 && i < n_theta_bins);
    return -PI_HALF + theta_bin_size/2 + i*theta_bin_size;
}

double HollowSphere::thetaEdge(size_t i) const {
    assert(i >= 0 && i <= n_theta_bins);
    return -PI_HALF + i*theta_bin_size;
}

double HollowSphere::phiCenter(size_t i) const {
    assert(i >= 0 && i < n_phi_bins);
    return phi_bin_size/2 + i*phi_bin_size;
}

double HollowSphere::phiEdge(size_t i) const {
    assert(i >= 0 && i <= n_phi_bins);
    return i*phi_bin_size;
}

Vector3<double> HollowSphere::xyz2rtp(const Vector3<double>& xyz) const {
    double rad = Length(xyz);
    double theta = std::numbers::pi - std::acos(xyz[2] / rad);
    double phi = wrap_angle(std::atan2(xyz[1], xyz[0]));
    Vector3<double> rtp;
    rtp[0] = rad;
    rtp[1] = theta;
    rtp[2] = phi;
    return rtp;
}

Vector3<size_t> HollowSphere::rtp2ijk(const Vector3<double>& rtp) const {
    double rad = rtp[0];
    assert(rad >= r_min && rad <= r_max);
    double theta = rtp[1];
    assert(theta >= -std::numbers::pi && theta <= std::numbers::pi);
    double phi = wrap_angle(rtp[2]);
    Vector3<size_t> ijk;
    ijk[0] = size_t((rad - r_min) / rad_bin_size);
    ijk[1] = size_t(theta / theta_bin_size);
    ijk[2] = size_t(phi / phi_bin_size);
    return ijk;
}

Vector3<size_t> HollowSphere::xyz2ijk(const Vector3<double>& xyz) const {
    return rtp2ijk(xyz2rtp(xyz));
}

size_t HollowSphere::ijk2I(const Vector3<size_t>& ijk) const {
    assert(0 <= ijk[0] && ijk[0] < n_rad_bins);
    assert(0 <= ijk[1] && ijk[1] < n_theta_bins);
    assert(0 <= ijk[2] && ijk[2] < n_phi_bins);
    return ijk[0] + ijk[1] * n_rad_bins + ijk[2] * n_rad_bins * n_theta_bins;
}
