#pragma once
#include <array>
#include <string_view>
#include "build_params.hpp"

#if __cplusplus >= 202002L
#include <numbers>
#else
namespace std::numbers
{
    inline constexpr double pi = M_PI;
}
#endif

namespace cudaSolarTomography
{
    constexpr double SUN_RADIUS_KM = 6.957e5; // the unit is km.
    // constexpr double SUN_RADIUS = 6.957e5; // (https://docs.sunpy.org/en/stable/reference/sun.html) the nominal solar radius in km.
    constexpr double SUN_RADIUS_M = SUN_RADIUS_KM * 1e3;  // the unit is m.
    constexpr double SUN_RADIUS_CM = SUN_RADIUS_KM * 1e5; // the unit is cm.
    constexpr double DEG_TO_RAD = std::numbers::pi / 180.;
    constexpr double RAD_TO_DEG = 180. / std::numbers::pi;
    constexpr double PI_HALF = std::numbers::pi / 2.;
    constexpr double PI_TWO = std::numbers::pi * 2.;

    // However, for comparing, first use this pole
    // constexpr double ALPHA_POLE = 286.11 * DEG_TO_RAD; // J2000 solar pole coordinates
    constexpr double ALPHA_POLE = 286.13 * DEG_TO_RAD; // (https://docs.sunpy.org/en/stable/reference/sun.html) right ascension (RA) of the solar north pole (epoch J2000.0) in radians
    // constexpr double DELTA_POLE = 63.85 * DEG_TO_RAD;
    constexpr double DELTA_POLE = 63.87 * DEG_TO_RAD; // (https://docs.sunpy.org/en/stable/reference/sun.htm) declination of the solar north pole (epoch J2000.0) in radians
    constexpr double ARCSEC_TO_RAD = DEG_TO_RAD / 3600.;
    /* the J2000.0 angle between the Ecliptic and mean Equatorial planes
     * is 23d26m21.4119s - From Allen's Astrophysical Quantities, 4th ed. (2000), page 13 */
    constexpr double J20000_ANGLE = 0.40909262920459; // unit is radians
    // the unit of INSTR_RMAX/MIN is of the sun radius
    constexpr double Q_LIMB = 0.63;
    // constexpr double Q_LIMB = 0.33;  // value in Kramar et al., change to this at some point
    /* There seem to be different definitions of the Thomson cross section. The THOMPSON_CONSTANT agrees with the following paper:
     *
     * Kramar, M., Jones, S., Davila, J. et al. On the Tomographic
     * Reconstruction of the 3D Electron Density for the Solar Corona
     * from STEREO COR1 Data. Sol Phys 259, 109–121 (2009).
     * https://doi.org/10.1007/s11207-009-9401-2
     *
     * In Kramar just below (2),
     *
     * \sigma = 7.95 * 10−26 cm^2.
     *
     * Another reference is Billings, A guide to the solar corona, 1966
     * (https://www.sciencedirect.com/book/9781483231006/a-guide-to-the-solar-corona).
     * In chapter 6 equation (2):
     *
     * \sigma = 7.95 * 10^-26 per steradian
     *
     * Note that \sigma is equal to the square of the classical electron radius. Note the following:
     * scipy.constants.physical_constants['classical electron radius'][0]**2 = 7.940787682024162e-30 m^2
     * where scipy.constants.physical_constants['classical electron radius'] = (2.8179403262e-15, 'm', 1.3e-24)
     * which agrees with 1/(4*np.pi*scipy.constants.epsilon_0) * scipy.constants.e**2/(scipy.constants.m_e * scipy.constants.c**2) = 2.8179403262049284e-15
     *
     * The builda code uses THOMPSON_CONSTANT = 1.2497e-15 with a comment that this is equal to (3/16)*(1e10 * Thompson X-section).
     *
     * The Thomson cross section is given by
     * scipy.constants.physical_constants['Thomson cross section'] = (6.6524587321e-29, 'm^2', 6e-38)
     * which agrees with 8 * np.pi / 3 * scipy.constants.physical_constants['classical electron radius'][0]**2 = 6.652458732150248e-29
     *
     * So, the builda THOMSON_CONSTANT \approx scipy.constants.physical_constants['Thomson cross section'][0] * 3/16 * 1e4 * 1e10
     * where 1e4 converts from m^2 to cm^2.
     */
    constexpr double THOMPSON_CONSTANT = 1.2497e-15;             // (3 / 16) * (1e10 * Thompson X-section)
    constexpr double THOMSON_X_SECTION = 6.6524587321e-29 * 1e4; // cm^2  = 8 * pi / 3 * r_e^2 where r_e is the classical electron radius and \sigma = r_e^2 in Billings 1966
    constexpr double SCATTER_FACTOR = 3 * (THOMSON_X_SECTION * SCALE_FACTOR) / 16;
    constexpr float OUTLIER_MAX = 500;

    constexpr int BIN_SIZE = BINNING_FACTOR * BINNING_FACTOR;
    constexpr int ROW_SIZE = IMAGE_SIZE / BINNING_FACTOR;
    constexpr int Y_SIZE = ROW_SIZE * ROW_SIZE;

    constexpr int N_BINS = N_RAD_BINS * N_THETA_BINS * N_PHI_BINS; // the number of bins in spherical coordinate

    // estimated number for possible elements in a row, if crashed, should be larger
    constexpr int AROW_SIZE = N_RAD_BINS + N_PHI_BINS + N_THETA_BINS;

    constexpr double R_DIFF = R_MAX - R_MIN;
    constexpr double R_MAX2 = R_MAX * R_MAX;
    constexpr double R_MIN2 = R_MIN * R_MIN;
    constexpr double RAD_BIN_SIZE = (R_MAX - R_MIN) / N_RAD_BINS;
    constexpr double THETA_BIN_SIZE = std::numbers::pi / N_THETA_BINS;
    constexpr double PHI_BIN_SIZE = PI_TWO / N_PHI_BINS;
    constexpr double RAD_BIN_SIZE_HALF = RAD_BIN_SIZE / 2;
    constexpr double THETA_BIN_SIZE_HALF = THETA_BIN_SIZE / 2;
    constexpr double PHI_BIN_SIZE_HALF = PHI_BIN_SIZE / 2;

    constexpr double delta = 0.0;
    constexpr double EPSILON = 1e-6;

    constexpr std::array<std::string_view, 380> ephemeris_dates = {
        "1996-01-17.06",
        "1996-02-13.40",
        "1996-03-11.73",
        "1996-04-08.03",
        "1996-05-05.28",
        "1996-06-01.50",
        "1996-06-28.70",
        "1996-07-25.90",
        "1996-08-22.13",
        "1996-09-18.39",
        "1996-10-15.67",
        "1996-11-11.97",
        "1996-12-09.28",
        "1997-01-05.61",
        "1997-02-01.95",
        "1997-03-01.29",
        "1997-03-28.60",
        "1997-04-24.87",
        "1997-05-22.10",
        "1997-06-18.30",
        "1997-07-15.50",
        "1997-08-11.72",
        "1997-09-07.97",
        "1997-10-05.24",
        "1997-11-01.53",
        "1997-11-28.84",
        "1997-12-26.17",
        "1998-01-22.50",
        "1998-02-18.84",
        "1998-03-18.17",
        "1998-04-14.46",
        "1998-05-11.70",
        "1998-06-07.91",
        "1998-07-05.11",
        "1998-08-01.32",
        "1998-08-28.55",
        "1998-09-24.82",
        "1998-10-22.10",
        "1998-11-18.41",
        "1998-12-15.72",
        "1999-01-12.05",
        "1999-02-08.40",
        "1999-03-07.73",
        "1999-04-04.04",
        "1999-05-01.30",
        "1999-05-28.52",
        "1999-06-24.72",
        "1999-07-21.92",
        "1999-08-18.14",
        "1999-09-14.40",
        "1999-10-11.67",
        "1999-11-07.97",
        "1999-12-05.28",
        "2000-01-01.61",
        "2000-01-28.95",
        "2000-02-25.29",
        "2000-03-23.61",
        "2000-04-19.89",
        "2000-05-17.12",
        "2000-06-13.33",
        "2000-07-10.52",
        "2000-08-06.74",
        "2000-09-02.98",
        "2000-09-30.25",
        "2000-10-27.54",
        "2000-11-23.84",
        "2000-12-21.16",
        "2001-01-17.50",
        "2001-02-13.84",
        "2001-03-13.17",
        "2001-04-09.47",
        "2001-05-06.72",
        "2001-06-02.93",
        "2001-06-30.13",
        "2001-07-27.34",
        "2001-08-23.57",
        "2001-09-19.83",
        "2001-10-17.11",
        "2001-11-13.41",
        "2001-12-10.72",
        "2002-01-07.05",
        "2002-02-03.39",
        "2002-03-02.73",
        "2002-03-30.04",
        "2002-04-26.31",
        "2002-05-23.54",
        "2002-06-19.74",
        "2002-07-16.94",
        "2002-08-13.16",
        "2002-09-09.41",
        "2002-10-06.68",
        "2002-11-02.98",
        "2002-11-30.28",
        "2002-12-27.61",
        "2003-01-23.95",
        "2003-02-20.29",
        "2003-03-19.61",
        "2003-04-15.90",
        "2003-05-13.14",
        "2003-06-09.35",
        "2003-07-06.55",
        "2003-08-02.76",
        "2003-08-29.99",
        "2003-09-26.26",
        "2003-10-23.54",
        "2003-11-19.85",
        "2003-12-17.16",
        "2004-01-13.50",
        "2004-02-09.84",
        "2004-03-08.17",
        "2004-04-04.48",
        "2004-05-01.74",
        "2004-05-28.96",
        "2004-06-25.15",
        "2004-07-22.36",
        "2004-08-18.58",
        "2004-09-14.84",
        "2004-10-12.11",
        "2004-11-08.41",
        "2004-12-05.72",
        "2005-01-02.05",
        "2005-01-29.39",
        "2005-02-25.73",
        "2005-03-25.05",
        "2005-04-21.33",
        "2005-05-18.56",
        "2005-06-14.76",
        "2005-07-11.96",
        "2005-08-08.18",
        "2005-09-04.42",
        "2005-10-01.69",
        "2005-10-28.98",
        "2005-11-25.29",
        "2005-12-22.61",
        "2006-01-18.94",
        "2006-02-15.28",
        "2006-03-14.61",
        "2006-04-10.91",
        "2006-05-08.16",
        "2006-06-04.37",
        "2006-07-01.57",
        "2006-07-28.78",
        "2006-08-25.01",
        "2006-09-21.27",
        "2006-10-18.55",
        "2006-11-14.85",
        "2006-12-12.17",
        "2007-01-08.50",
        "2007-02-04.84",
        "2007-03-04.17",
        "2007-03-31.48",
        "2007-04-27.75",
        "2007-05-24.98",
        "2007-06-21.18",
        "2007-07-18.38",
        "2007-08-14.60",
        "2007-09-10.85",
        "2007-10-08.12",
        "2007-11-04.42",
        "2007-12-01.73",
        "2007-12-29.05",
        "2008-01-25.39",
        "2008-02-21.73",
        "2008-03-20.05",
        "2008-04-16.34",
        "2008-05-13.58",
        "2008-06-09.79",
        "2008-07-06.98",
        "2008-08-03.19",
        "2008-08-30.43",
        "2008-09-26.70",
        "2008-10-23.98",
        "2008-11-20.29",
        "2008-12-17.61",
        "2009-01-13.94",
        "2009-02-10.28",
        "2009-03-09.62",
        "2009-04-05.92",
        "2009-05-03.17",
        "2009-05-30.39",
        "2009-06-26.59",
        "2009-07-23.79",
        "2009-08-20.02",
        "2009-09-16.28",
        "2009-10-13.56",
        "2009-11-09.85",
        "2009-12-07.17",
        "2010-01-03.49",
        "2010-01-30.83",
        "2010-02-27.17",
        "2010-03-26.49",
        "2010-04-22.76",
        "2010-05-19.00",
        "2010-06-16.20",
        "2010-07-13.40",
        "2010-08-09.61",
        "2010-09-05.86",
        "2010-10-03.13",
        "2010-10-30.42",
        "2010-11-26.73",
        "2010-12-24.05",
        "2011-01-20.39",
        "2011-02-16.73",
        "2011-03-16.06",
        "2011-04-12.35",
        "2011-05-09.60",
        "2011-06-05.81",
        "2011-07-03.01",
        "2011-07-30.21",
        "2011-08-26.44",
        "2011-09-22.71",
        "2011-10-19.99",
        "2011-11-16.29",
        "2011-12-13.61",
        "2012-01-09.94",
        "2012-02-06.28",
        "2012-03-04.62",
        "2012-03-31.92",
        "2012-04-28.19",
        "2012-05-25.41",
        "2012-06-21.61",
        "2012-07-18.82",
        "2012-08-15.04",
        "2012-09-11.29",
        "2012-10-08.56",
        "2012-11-04.86",
        "2012-12-02.17",
        "2012-12-29.49",
        "2013-01-25.83",
        "2013-02-22.17",
        "2013-03-21.49",
        "2013-04-17.78",
        "2013-05-15.02",
        "2013-06-11.22",
        "2013-07-08.42",
        "2013-08-04.63",
        "2013-08-31.87",
        "2013-09-28.14",
        "2013-10-25.43",
        "2013-11-21.73",
        "2013-12-19.05",
        "2014-01-15.38",
        "2014-02-11.73",
        "2014-03-11.06",
        "2014-04-07.36",
        "2014-05-04.61",
        "2014-05-31.83",
        "2014-06-28.03",
        "2014-07-25.23",
        "2014-08-21.46",
        "2014-09-17.72",
        "2014-10-14.00",
        "2014-11-11.30",
        "2014-12-08.61",
        "2015-01-04.94",
        "2015-02-01.28",
        "2015-02-28.62",
        "2015-03-27.93",
        "2015-04-24.20",
        "2015-05-21.43",
        "2015-06-17.64",
        "2015-07-14.84",
        "2015-08-11.05",
        "2015-09-07.30",
        "2015-10-04.57",
        "2015-10-31.86",
        "2015-11-28.17",
        "2015-12-25.49",
        "2016-01-21.83",
        "2016-02-18.17",
        "2016-03-16.50",
        "2016-04-12.79",
        "2016-05-10.03",
        "2016-06-06.25",
        "2016-07-03.44",
        "2016-07-30.65",
        "2016-08-26.88",
        "2016-09-23.15",
        "2016-10-20.43",
        "2016-11-16.73",
        "2016-12-14.05",
        "2017-01-10.38",
        "2017-02-06.72",
        "2017-03-06.06",
        "2017-04-02.36",
        "2017-04-29.63",
        "2017-05-26.85",
        "2017-06-23.05",
        "2017-07-20.25",
        "2017-08-16.47",
        "2017-09-12.73",
        "2017-10-10.00",
        "2017-11-06.30",
        "2017-12-03.61",
        "2017-12-30.94",
        "2018-01-27.27",
        "2018-02-23.61",
        "2018-03-22.93",
        "2018-04-19.22",
        "2018-05-16.45",
        "2018-06-12.66",
        "2018-07-09.86",
        "2018-08-06.07",
        "2018-09-02.31",
        "2018-09-29.58",
        "2018-10-26.87",
        "2018-11-23.17",
        "2018-12-20.49",
        "2019-01-16.83",
        "2019-02-13.17",
        "2019-03-12.50",
        "2019-04-08.80",
        "2019-05-06.05",
        "2019-06-02.27",
        "2019-06-29.47",
        "2019-07-26.67",
        "2019-08-22.90",
        "2019-09-19.15",
        "2019-10-16.44",
        "2019-11-12.74",
        "2019-12-10.05",
        "2020-01-06.38",
        "2020-02-02.72",
        "2020-03-01.06",
        "2020-03-28.37",
        "2020-04-24.64",
        "2020-05-21.87",
        "2020-06-18.07",
        "2020-07-15.27",
        "2020-08-11.49",
        "2020-09-07.74",
        "2020-10-05.01",
        "2020-11-01.30",
        "2020-11-28.61",
        "2020-12-25.93",
        "2021-01-22.27",
        "2021-02-18.61",
        "2021-03-17.94",
        "2021-04-14.23",
        "2021-05-11.47",
        "2021-06-07.68",
        "2021-07-04.88",
        "2021-08-01.09",
        "2021-08-28.32",
        "2021-09-24.59",
        "2021-10-21.87",
        "2021-11-18.17",
        "2021-12-15.49",
        "2022-01-11.82",
        "2022-02-08.17",
        "2022-03-07.50",
        "2022-04-03.81",
        "2022-05-01.07",
        "2022-05-28.29",
        "2022-06-24.49",
        "2022-07-21.69",
        "2022-08-17.91",
        "2022-09-14.17",
        "2022-10-11.44",
        "2022-11-07.74",
        "2022-12-05.05",
        "2023-01-01.38",
        "2023-01-28.72",
        "2023-02-25.06",
        "2023-03-24.38",
        "2023-04-20.66",
        "2023-05-17.89",
        "2023-06-14.10",
        "2023-07-11.30",
        "2023-08-07.51",
        "2023-09-03.75",
        "2023-10-01.02",
        "2023-10-28.31",
        "2023-11-24.61",
        "2023-12-21.93",
        "2024-01-18.27",
        "2024-02-14.61",
        "2024-03-12.94",
        "2024-04-09.24",
        "2024-05-06.49"};
}
