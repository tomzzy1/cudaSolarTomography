#include <string>
#include <cassert>
#include <sstream>

#include "fitsio.h"
#include "wcslib/wcslib.h"

#include "Mathematics/Rotation.h"
#include "Mathematics/Matrix3x3.h"

#include "coordinates.hpp"
#include "constants.hpp"

using namespace gte;
using namespace cudaSolarTomography;


////////////////////////////////////////////////////////////////////////////////
// utility functions

double wrap_angle(double x, bool radians) {
    double y = std::fmod(x, radians ? PI_TWO : 360);
    if (y < 0)
        y += radians ? PI_TWO : 360;
    return y;
}


Vector3<double>
sph2cart(const Vector3<double> r_theta_phi, bool radians) {
    double cos_theta = cos(r_theta_phi[1] * (radians ? 1 : DEG_TO_RAD));
    double sin_theta = sin(r_theta_phi[1] * (radians ? 1 : DEG_TO_RAD));

    double cos_phi = cos(r_theta_phi[2] * (radians ? 1 : DEG_TO_RAD));
    double sin_phi = sin(r_theta_phi[2] * (radians ? 1 : DEG_TO_RAD));

    return {
        r_theta_phi[0] * cos_phi * cos_theta,
        r_theta_phi[0] * sin_phi * cos_theta,
        r_theta_phi[0] * sin_theta
    };
}


std::string
frame2str(const sun_obs_frame& frame) {
    std::ostringstream ss;
    if (frame.distance_units == Unit::Rsun) {
        ss << "Sun to observer vector (Rsun)\n";
        ss << "=============================\n";
    }
    else if (frame.distance_units == Unit::Meters) {
        ss << "Sun to observer vector (m)\n";
        ss << "==========================\n";
    }
    else {
        assert(false);
    }
    ss << "[0]: " << frame.sun_obs[0] << '\n';
    ss << "[1]: " << frame.sun_obs[1] << '\n';
    ss << "[2]: " << frame.sun_obs[2] << '\n';
    ss << '\n';
    ss << "LOS unit vector\n";
    ss << "===============\n";
    ss << "[0]: " << frame.unit[0] << '\n';
    ss << "[1]: " << frame.unit[1] << '\n';
    ss << "[2]: " << frame.unit[2] << '\n';
    ss << '\n';
    if (frame.distance_units == Unit::Rsun) {
        ss << "LOS near point vector (Rsun)\n";
        ss << "============================\n";
    }
    else if (frame.distance_units == Unit::Meters) {
        ss << "LOS near point vector (m)\n";
        ss << "=========================\n";
    }
    else {
        assert(false);
    }
    ss << "[0]: " << frame.nrpt[0] << '\n';
    ss << "[1]: " << frame.nrpt[1] << '\n';
    ss << "[2]: " << frame.nrpt[2] << '\n';
    return ss.str();
}


////////////////////////////////////////////////////////////////////////////////
// builda support

std::pair<Vector3<double>, double>
get_orbit(const struct fits_header_info& info)
{

    /* the STEREO .fts files give the Sun-Spacecraft vector in
     *  Heliocentric Aries Ecliptic Coordinates.  This differs
     *  from GCI in the origin point and choice of Z-axis (ecliptic
     *  N, vs. equatorial N (celestial pole).  Therefore these coords.
     *  need to rotated about the x-axis.
     */

    Vector3<double> observatory = {info.HAEX_OBS, info.HAEY_OBS, info.HAEZ_OBS};

    double carrington_longitude = info.CRLN_OBS;

    carrington_longitude *= DEG_TO_RAD; // convert the unit of carlington longitude

    /* the J2000.0 angle between the Ecliptic and mean Equatorial planes
     *  is 23d26m21.4119s - From Allen's Astrophysical Quantities, 4th ed. (2000)  */

    Matrix3x3<double> Rx = Rotation<3, double>(AxisAngle<3, double>({1, 0, 0}, -J20000_ANGLE));

    Vector3<double> sun_to_observer_vector = Rx * observatory / SUN_RADIUS_M;

    return {sun_to_observer_vector, carrington_longitude};
}


bool
get_fits_header_info(const char* fts_fname, struct fits_header_info* info) {
    fitsfile *fptr;
    char *header = NULL;
    int nkeyrec, nreject, nkeyids;
    int status = 0;
    struct fitskeyid keyids[11];
    struct fitskey *keys;

    // borrowed from wcslib tfitshdr.c
    if (fits_open_file(&fptr, fts_fname, READONLY, &status)) {
        fits_report_error(stderr, status);
        fits_close_file(fptr, &status);
        return false;
    }

    if (fits_hdr2str(fptr, 0, NULL, 0, &header, &nkeyrec, &status)) {
        fits_report_error(stderr, status);
        fits_close_file(fptr, &status);
        return false;
    }
    fits_close_file(fptr, &status);

    // Number remaining.
    nkeyrec = strlen(header) / 80;

    nkeyids = 9;
    strcpy(keyids[0].name, "HAEX_OBS");
    strcpy(keyids[1].name, "HAEY_OBS");
    strcpy(keyids[2].name, "HAEZ_OBS");
    strcpy(keyids[3].name, "CRLN_OBS");
    strcpy(keyids[4].name, "CROTA");
    strcpy(keyids[5].name, "CDELT1");
    strcpy(keyids[6].name, "CDELT2");
    // DOCUMENT INDEX 1/0 ISSUE!
    strcpy(keyids[7].name, "CRPIX1");
    strcpy(keyids[8].name, "CRPIX2");

    if ((status = fitshdr(header, nkeyrec, nkeyids, keyids, &nreject, &keys))) {
        fprintf(stderr, "fitskey ERROR %d: %s.\n", status, fitshdr_errmsg[status]);
        wcsdealloc(keys);
        return false;
    }
    fits_free_memory(header, &status);

    for (size_t i = 0; i < nkeyids; i++) {
        assert(keyids[i].count == 1);
        assert(keys[keyids[i].idx[0]].type == 5);
    }

    info->HAEX_OBS = keys[keyids[0].idx[0]].keyvalue.f;
    info->HAEY_OBS = keys[keyids[1].idx[0]].keyvalue.f;
    info->HAEZ_OBS = keys[keyids[2].idx[0]].keyvalue.f;
    info->CRLN_OBS = keys[keyids[3].idx[0]].keyvalue.f;
    info->CROTA = keys[keyids[4].idx[0]].keyvalue.f;
    info->CDELT1 = keys[keyids[5].idx[0]].keyvalue.f;
    info->CDELT2 = keys[keyids[6].idx[0]].keyvalue.f;
    info->CRPIX1 = keys[keyids[7].idx[0]].keyvalue.f;
    info->CRPIX2 = keys[keyids[8].idx[0]].keyvalue.f;

    wcsdealloc(keys);

    return true;
}


sun_obs_frame
get_builda_coordinates(const struct fits_header_info& info,
                       double i, double j) {
    auto[sun_ob1, carrington_longtiude] = get_orbit(info);

    // DOCUMENT INDEX 1/0 ISSUE!
    double x_image_j = PIXEL_SIZE * (j - info.CRPIX1 + 1);
    double y_image_i = PIXEL_SIZE * (i - info.CRPIX2 + 1);

    double rho = hypot(x_image_j, y_image_i) * ARCSEC_TO_RAD;
    double eta = atan2(-x_image_j, y_image_i) + info.CROTA * DEG_TO_RAD;

    double z_angle = -atan2(sun_ob1[1], sun_ob1[0]);

    Matrix3x3<double> Rz = Rotation<3, double>(AxisAngle<3, double>({0, 0, 1}, z_angle));

    Vector3<double> sob = DoTransform(Rz, sun_ob1);

    double y_angle = atan2(sob[2], sob[0]);

    Matrix3x3<double> Ry = Rotation<3, double>(AxisAngle<3, double>({0, 1, 0}, y_angle));

    // Could instead do this with EulerAngles, but I haven't yet deciphered how to do it
    Matrix3x3<double> Rzy = DoTransform(Ry, Rz);

    Vector3<double> SPOL1 = {
        cos(DELTA_POLE) * cos(ALPHA_POLE),
        cos(DELTA_POLE) * sin(ALPHA_POLE),
        sin(DELTA_POLE)
    };

    Vector3<double> r3tmp = DoTransform(Rzy, SPOL1);

    double x_angle = atan2(r3tmp[1], r3tmp[2]);

    Matrix3x3<double> Rx_tmp = Rotation<3, double>(AxisAngle<3, double>({1, 0, 0}, x_angle));
    // Could instead do this with EulerAngles, but I haven't yet deciphered how to do it
    Matrix3x3<double> R12 = DoTransform(Rx_tmp, Rzy);

    Vector3<double> spol2 = DoTransform(R12, SPOL1);

    double p_angle = -atan2(spol2[0], spol2[2]);

    Matrix3x3<double> R23a = Rotation<3, double>(AxisAngle<3, double>({0, 1, 0}, p_angle));
    Matrix3x3<double> R23b = Rotation<3, double>(AxisAngle<3, double>({0, 0, 1}, carrington_longtiude));

    // Could instead do this with EulerAngles, but I haven't yet deciphered how to do it
    Matrix3x3<double> R23 = DoTransform(R23b, R23a);

    Matrix3x3<double> Rx = Rotation<3, double>(AxisAngle<3, double>({1, 0, 0}, eta));

    double s_rho = sin(rho);
    double c_rho = cos(rho);

    // Could instead do this with EulerAngles, but I haven't yet deciphered how to do it
    Vector3<double> unit = DoTransform(R23, DoTransform(Rx, Vector3<double>{-c_rho, 0, s_rho}));
    Vector3<double> sun_ob3 = DoTransform(R23, DoTransform(R12, sun_ob1));
    Vector3<double> nrpt = sun_ob3 - Dot(sun_ob3, unit) * unit;

    return {sun_ob3, unit, nrpt};
}


////////////////////////////////////////////////////////////////////////////////
// WCS support

Vector3<double> hpc_to_hcc(const Vector3<double>& x_hpc,
                           double distance_to_sun_center) {
    return {x_hpc[1], x_hpc[2], distance_to_sun_center - x_hpc[0]};

}

Vector3<double> hcc_to_heq(const Vector3<double>& x_hcc,
                           const Vector3<double>& observer_heq) {
    double lon = atan2(observer_heq[1], observer_heq[0]);
    double lat = asin(observer_heq[2] / Length(observer_heq));

    Vector3<double> x = {x_hcc[2], x_hcc[0], x_hcc[1]};
    Matrix3x3<double> T = Rotation<3, double>(EulerAngles<double>(0, 1, 2, 0, -lat, lon));

    return DoTransform(T, x);
}

Vector3<double> heq_to_hgc(const Vector3<double>& x_heq, double L0) {
    Matrix3x3<double> Rz = Rotation<3, double>(AxisAngle<3, double>({0, 0, 1}, L0));
    return DoTransform(Rz, x_heq);
}


WCS::WCS(const std::string& fts_fname, bool zero_based) {
    fitsfile *fptr;
    char *header = nullptr;
    int nkeyrec, nreject;
    int status = 0;
    size_t i;
    int nkeyids = 0;
    struct fitskeyid keyids[10];
    struct fitskey *keys = nullptr;

    // Borrowed from wcslib C/test/twcshdr.c

    // Open the FITS test file and read the primary header.
    fits_open_file(&fptr, fts_fname.c_str(), READONLY, &status);
    if ((status = fits_hdr2str(fptr, 1, NULL, 0, &header, &nkeyrec,
                               &status))) {
        fits_report_error(stderr, status);
        // Raise exception? Cleanup?
        assert(false);
    }

    // Parse the primary header of the FITS file.
    if ((status = wcspih(header, nkeyrec, WCSHDR_all, 2, &nreject, &nwcs, &wcs_ptr))) {
        fprintf(stderr, "wcspih ERROR %d: %s.\n", status,wcshdr_errmsg[status]);
    }

    // Read coordinate arrays from the binary table extension.
    if ((status = fits_read_wcstab(fptr, wcs_ptr->nwtb, (wtbarr *) wcs_ptr->wtb,
                                   &status))) {
        fits_report_error(stderr, status);
        assert(false);}

    // Translate non-standard WCS keyvalues.
    stat.reserve(NWCSFIX);
    if ((status = wcsfix(NWCSFIX, 0, wcs_ptr, stat.data()))) {
        for (i = 0; i < NWCSFIX; i++) {
            if (stat[i] > 0) {
                fprintf(stderr, "wcsfix ERROR %d: %s.\n", status,
                        wcsfix_errmsg[stat[i]]);
            }
        }
        // Raise exception? Cleanup?
        assert(false);
    }

    // Initialize the wcsprm struct, also taking control of memory allocated by
    // fits_read_wcstab().
    if ((status = wcsset(wcs_ptr))) {
        fprintf(stderr, "wcsset ERROR %d: %s.\n", status, wcs_errmsg[status]);
        assert(false);
    }


    // DOCUMENT!
    if (zero_based) {
        wcs_ptr->crpix[0] -= 1;
        wcs_ptr->crpix[1] -= 1;
    }

    // Get Carrington frame information from the FITS file header.
    // These keywords can be found in COR1 FITS records, but
    // adjustments are likely necessary for other cases.
    nkeyids = 7;
    strcpy(keyids[0].name, "NAXIS1");
    strcpy(keyids[1].name, "NAXIS2");
    strcpy(keyids[2].name, "HGLN_OBS");
    strcpy(keyids[3].name, "CRLN_OBS");
    strcpy(keyids[4].name, "HEQX_OBS");
    strcpy(keyids[5].name, "HEQY_OBS");
    strcpy(keyids[6].name, "HEQZ_OBS");
    strcpy(keyids[7].name, "HGCX_OBS");
    strcpy(keyids[8].name, "HGCY_OBS");
    strcpy(keyids[9].name, "HGCZ_OBS");

    if ((status = fitshdr(header, nkeyrec, nkeyids, keyids, &nreject, &keys))) {
        fprintf(stderr, "fitskey ERROR %d: %s.\n", status, fitshdr_errmsg[status]);
        // Raise exception? Cleanup?
        assert(false);
    }

    for (size_t i = 0; i < 2; i++) {
        assert(keyids[i].count == 1);
        assert(keys[keyids[i].idx[0]].type == 2);
    }

    for (size_t i = 2; i < 7; i++) {
        assert(keyids[i].count == 1);
        assert(keys[keyids[i].idx[0]].type == 5);
    }

    NAXIS1 = keys[keyids[0].idx[0]].keyvalue.i;
    NAXIS2 = keys[keyids[1].idx[0]].keyvalue.i;

    HGLN_OBS = keys[keyids[2].idx[0]].keyvalue.f;
    CRLN_OBS = keys[keyids[3].idx[0]].keyvalue.f;

    double HEQX_OBS = keys[keyids[4].idx[0]].keyvalue.f;
    double HEQY_OBS = keys[keyids[5].idx[0]].keyvalue.f;
    double HEQZ_OBS = keys[keyids[6].idx[0]].keyvalue.f;
    observer_heq = Vector3<double>{HEQX_OBS, HEQY_OBS, HEQZ_OBS};

    _L0 = wrap_angle((CRLN_OBS - HGLN_OBS) * DEG_TO_RAD);

    const std::vector<double>& world = pixel_to_world((NAXIS1 - 1) / 2., (NAXIS2 - 1) / 2.);
    center_Tx = world[0];
    center_Ty = world[1];

    center_unit_hpc = sph2cart({1, center_Ty, center_Tx}, false);

    //Vector3<double> observer_hgc;
    if ((keyids[7].count == 1) && (keyids[8].count == 1) && (keyids[9].count == 1)) {
        double HGCX_OBS = keys[keyids[7].idx[0]].keyvalue.f;
        double HGCY_OBS = keys[keyids[8].idx[0]].keyvalue.f;
        double HGCZ_OBS = keys[keyids[9].idx[0]].keyvalue.f;
        observer_hgc = Vector3<double>{HGCX_OBS, HGCY_OBS, HGCZ_OBS};
    }
    else {
        observer_hgc = heq_to_hgc(observer_heq, L0());
    }

    distance_to_sun_center = Length(observer_hgc);

    // Finished with the FITS file.
    fits_close_file(fptr, &status);
    assert(status == 0);
    if (header) {
        fits_free_memory(header, &status);
        assert(status == 0);
    }
    // cleanup
    if (keys) {
        wcsdealloc(keys);
    }
}


WCS::~WCS() {
    assert(wcsvfree(&nwcs, &wcs_ptr) == 0);
}


bool
WCS::pixel_to_world(size_t N) {
    assert(pixcrd.size() >= 2*N);
    imgcrd.reserve(2*N);
    phi.reserve(N);
    theta.reserve(N);
    world.reserve(2*N);
    stat.reserve(N);
    return wcsp2s(wcs_ptr, N, 2, pixcrd.data(), imgcrd.data(), phi.data(), theta.data(), world.data(), stat.data()) == 0;
}

std::vector<double>&
WCS::pixel_to_world(double x, double y) {
    pixcrd.clear();
    pixcrd.reserve(2);
    pixcrd.push_back(x);
    pixcrd.push_back(y);
    world.resize(2);
    assert(pixel_to_world(1));
    return world;
}

double
WCS::L0(void) const {
    return _L0;
}


Vector3<double>
WCS::hpc_to_hgc(const gte::Vector3<double>& x_hpc) const {
    return heq_to_hgc(hcc_to_heq(hpc_to_hcc(x_hpc, distance_to_sun_center),
                                 observer_heq),
                      L0());
}


// ALSO NEED A VECTORIZED VERSION
sun_obs_frame
WCS::get_frame(double i, double j, Unit unit) {
    const std::vector<double>& world = pixel_to_world(j, i);

    const Vector3<double>& pixel_ij_unit_hpc = sph2cart({1, world[1], world[0]}, false);

    double abs_cos_theta = Dot(center_unit_hpc, pixel_ij_unit_hpc);
    Vector3<double> nrpt_hpc = sph2cart({abs_cos_theta * distance_to_sun_center, world[1], world[0]}, false);
    Vector3<double> nrpt_hgc = hpc_to_hgc(nrpt_hpc);

    Vector3<double> unit_hgc = nrpt_hgc - observer_hgc;
    unit_hgc /= Length(unit_hgc);

    if (Length(nrpt_hgc) < SUN_RADIUS_M) {
        // nearest point vector is on solar surface
        double D = distance_to_sun_center;
        double R = SUN_RADIUS_M;
        double a0 = D*D - R*R;
        double a1 = Dot(unit_hgc, observer_hgc);
        double discriminant = a1*a1 - a0;
        assert(discriminant > 0);
        nrpt_hgc = observer_hgc - (a1 + sqrt(discriminant)) * unit_hgc;
    }

    if (unit == Unit::Meters) {
        return {observer_hgc, unit_hgc, nrpt_hgc};
    }
    else if (unit == Unit::Rsun) {
        return {
            observer_hgc / SUN_RADIUS_M,
            unit_hgc,
            nrpt_hgc / SUN_RADIUS_M};
    }
    else {
        assert(false);
    }
}
