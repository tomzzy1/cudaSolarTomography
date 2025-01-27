#pragma once

#include <vector>
#include <cassert>
#include <numbers>

#include "Mathematics/Hypersphere.h"
#include "Mathematics/Hyperplane.h"
#include "Mathematics/Line.h"
#include "Mathematics/Ray.h"
#include "Mathematics/Cone.h"


class HollowSphere {
public:
    // Construct a hollow sphere grid with the specified number of
    // bins and inner and outer radii [Rsun]. The implementation makes
    // relies heavily on the Geometric Tools Engine
    // (https://www.geometrictools.com).
    HollowSphere(size_t nRadBins, size_t nThetaBins, size_t nPhiBins, double rMin, double rMax);

    // Number of radial, theta (latitude), and phi (longitude) bins.
    const size_t n_rad_bins;
    const size_t n_theta_bins;
    const size_t n_phi_bins;

    // Inner and outer hollow sphere radii [Rsun].
    const double r_min;
    const double r_max;

    // Size of a radial [Rsun], theta [rad], and phi [rad] bins.
    const double rad_bin_size;
    const double theta_bin_size;
    const double phi_bin_size;

    // Return the center of the ith radial [Rsun], theta [rad], or phi
    // [rad] bin.
    double radCenter(size_t i) const;
    double thetaCenter(size_t i) const;
    double phiCenter(size_t i) const;

    // Return the ith radial [Rsun], theta [rad], or phi [rad] bin
    // edge.
    double radEdge(size_t i) const;
    double thetaEdge(size_t i) const;
    double phiEdge(size_t i) const;

    // Return the points of bin edge intersections along the
    // line-of-site specified by ray. The points are sorted and each
    // is the distance [Rsun] from the observer along the
    // line-of-site, i.e., at the vector coordinate sun_to_obs +
    // distance[i] * unit.
    std::vector<double> intersect(const gte::Ray3<double>& ray3);

    // Convert Cartesian vector xyz (each component has unit [Rsun])
    // to rtp, i.e., radius [Rsun], theta [rad], and phi [rad].
    gte::Vector3<double> xyz2rtp(const gte::Vector3<double>& xyz) const;

    // Convert rtp, i.e., radius [Rsun], theta [rad], and phi [rad] to
    // the ijk bin location, i.e., i=radial bin, j=theta bin, and
    // k=phi bin.
    gte::Vector3<size_t> rtp2ijk(const gte::Vector3<double>& rtp) const;

    // Convert xyz (each component has unit [Rsun]) to the ijk bin
    // location, i.e., i=radial bin, j=theta bin, and k=phi bin.
    gte::Vector3<size_t> xyz2ijk(const gte::Vector3<double>& xyz) const;

    // Convert the ijk bin location, i.e., i=radial bin, j=theta bin,
    // and k=phi bin, to the corresponding flat or linear index (in
    // FORTRAN order), i.e., see numpy.ravel_multi_index.
    size_t ijk2I(const gte::Vector3<size_t>& ijk) const;

private:
    // Vector of spheres at each radial bin edge.
    std::vector<gte::Sphere3<double>> radial_bin_edges;

    // Vector of planes at each longitude bin edge.
    std::vector<gte::Plane3<double>> phi_bin_edges;

    // Vector of cones at each latitude bin edge.
    std::vector<gte::Cone3<double>> theta_bin_edges;

    // Plane at latitude 0.
    gte::Plane3<double> theta_0_bin_edge;

    // Line passing through the origin and the north/south poles of
    // the hollow sphere which corresponds to latitude +- 90 [deg].
    gte::Line3<double> theta_90_bin_edge;
};
