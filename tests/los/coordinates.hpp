#pragma once

#include "Mathematics/Vector3.h"


// Coordinate vector units.
// Meters: [m]
// Rsun:   solar radii
enum class Unit {Meters, Rsun};


struct sun_obs_frame {
    gte::Vector3<double> sun_obs;
    gte::Vector3<double> unit;
    gte::Vector3<double> nrpt;
    Unit distance_units;
};


////////////////////////////////////////////////////////////////////////////////
// utility functions

double wrap_angle(double x, bool radians=true);

gte::Vector3<double>
sph2cart(const gte::Vector3<double> r_theta_phi, bool radians=true);

std::string
frame2str(const sun_obs_frame& frame);


////////////////////////////////////////////////////////////////////////////////
// builda support

struct fits_header_info {
    double HAEX_OBS;
    double HAEY_OBS;
    double HAEZ_OBS;

    double CRLN_OBS;

    double CROTA;

    double CDELT1;
    double CDELT2;

    double CRPIX1;
    double CRPIX2;
};

bool
get_fits_header_info(const char* fts_fname, struct fits_header_info* info);

std::pair<gte::Vector3<double>, double>
get_orbit(const struct fits_header_info& info);

sun_obs_frame
get_builda_coordinates(const struct fits_header_info& info,
                       double i, double j);


////////////////////////////////////////////////////////////////////////////////
// WCS support

class WCS {
public:
    // Construct a wcslib
    // (https://www.atnf.csiro.au/people/mcalabre/WCS/) FITS "World
    // Coordinate System" (WCS) wrapper. The input is the FITS file
    // stored at fts_fname. If zero_based is true, then pixel
    // coordinates uses C convention and the top left pixel is (0, 0).
    // Otherwise, use the FORTRAN, i.e., one-based, convention and the
    // top left pixel is (1, 1).
    WCS(const std::string& fts_fname, bool zero_based=true);

    // Cleanup of wcs_ptr which contains some dynamically allocated
    // material.
    ~WCS();

    // The order is AXIS1 (x) then AXIS2 (y).
    // pixcrd[0]=column and pixcrd[1]=row pixel coordinates [pixels]
    // world[0]=Tx and world[1]=Ty helioprojective Cartesian [deg]
    std::vector<double> pixcrd;
    std::vector<double> world;

    // Convert N pixel coordinates to world coordinates. The input is
    // pixcrd = {x_0, y_0, x_1, y_2, ..., x_{N-1}, y_{N-1}} and the
    // output is world = {Tx_0, Ty_0, ..., Tx_{N-1}, Ty_{N-1}}.
    bool pixel_to_world(size_t N);

    // Convert a single x (column) and y (row) pixel coordinate to
    // world coordinates, i.e., the output vector will have two
    // elements {Tx, Ty}.
    std::vector<double>& pixel_to_world(double x, double y);

    // Return apparent Carrington longitude [rad].
    double L0(void) const;

    // Convert a helioprojective Cartesian coordinate vector [m] to
    // heliographic Cartesian coordinates [m].
    gte::Vector3<double>
    hpc_to_hgc(const gte::Vector3<double>& x_hpc) const;

    // Return the coordinate frame describing the line-of-sight
    // associated with pixel i (row) and j (column), i.e., the Sun to
    // observer vector, unit vector of the line-of-site, and vector to
    // the point nearest to the Sun center at the origin along the
    // line of site. The coordinate frame is heliographic Carrington.
    // Note that if x = (near point vector) - (sun to observer) then
    // unit = x / |x|.
    sun_obs_frame
    get_frame(double i, double j, Unit unit=Unit::Meters);

private:
    // The coordinate transformation parameters (see the wcslib
    // documentation).
    struct wcsprm* wcs_ptr;

    // Number of coordinate representations found (see the wcslib
    // documentation).
    int nwcs;

    size_t NAXIS1;   // The number of columns in the image [pixels]
    size_t NAXIS2;   // The number of rows in the image [pixels]

    double HGLN_OBS;  // Stonyhurst heliographic longitude of the observer [deg]
    double CRLN_OBS;  // Carrington heliographic longitude of the observer [deg]

    gte::Vector3<double> observer_heq;  // observer heliocentric Earth equatorial [m]
    gte::Vector3<double> observer_hgc;  // observer heliogrpahic Carrington [m]

    /* (copied from Sunpy helioprojective Cartesian documentation)

     Tx (aka “theta_x”) is the angle relative to the plane containing
     the Sun-observer line and the Sun’s rotation axis, with positive
     values in the direction of the Sun’s west limb.

     Ty (aka “theta_y”) is the angle relative to the Sun’s equatorial
     plane, with positive values in the direction of the Sun’s north
     pole.p
     */
    double center_Tx;  // Helioprojective Cartesian Tx [deg] of center pixel
    double center_Ty;  // Helioprojective Cartesian Ty [deg] of center pixel

    // Unit vector from observer to center pixel
    gte::Vector3<double> center_unit_hpc;

    // Distance to from observer to center of sun [m]
    double distance_to_sun_center;

    // Apparent Carrington longitude of the Sun-disk center as seen
    // from Earth [rad].
    double _L0;

    // From documentation for wcslib function wcsp2s.
    //
    // imgcrd:    Array of intermediate world coordinates.
    // phi,theta: Longitude and latitude in the native coordinate system of the projection [deg].
    // stat:      Status return value for each coordinate:
    //            - 0:  Success.
    //            - 1+: A bit mask indicating invalid pixel coordinate element(s).
    std::vector<double> imgcrd;
    std::vector<double> phi;
    std::vector<double> theta;
    std::vector<int> stat;
};


// Convert a helioprojective Cartesian coordinate (HPC) vector to
// heliocentric Cartesian coordinates (HCC). The distnace_to_sun_center must
// in the same units.
gte::Vector3<double> hpc_to_hcc(const gte::Vector3<double>& x_hpc,
                                double distance_to_sun_center);

// Convert a heliocentric Cartesian coordinate (HCC) vector to
// heliocentric Earth equatorial (HEQ), aka, heliographic Stonyhurst,
// coordinates. The observer_heq vector is the HEQ frame Sun to
// observer coordinate vector.
gte::Vector3<double> hcc_to_heq(const gte::Vector3<double>& x_hcc,
                                const gte::Vector3<double>& observer_heq);

// Convert a heliocentric Earth equatorial (HEQ), aka, heliographic
// Stonyhurst, coordinate vector to heliographic Carrington
// coordinates. The parameter L0 is the apparent Carrington longitude
// [rad].
gte::Vector3<double> heq_to_hgc(const gte::Vector3<double>& x_heq, double L0);
