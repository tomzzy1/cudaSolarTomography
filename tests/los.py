from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import scipy as sp
import astropy.io.fits


# the J2000.0 angle between the Ecliptic and mean Equatorial planes
#is 23d26m21.4119s - From Allen's Astrophysical Quantities, 4th ed. (2000), page 13
J20000_ANGLE = 0.40909262920459
J20000_rot = sp.spatial.transform.Rotation.from_euler('x', J20000_ANGLE)

# Document me (or get from sunpy)
SUN_RADIUS = 6.957e5  # the unit is km.

# Document me (or get from sunpy)
ALPHA_POLE = np.radians(286.13)
DELTA_POLE = np.radians(63.87)


@dataclass
class Header:
    # observatory position (units?)
    obs_x: float
    obs_y: float
    obs_z: float

    carrington_lon: float  # in radians

    center_x: float
    center_y: float
    roll_offset: float     # in radians

    delta_x: float
    delta_y: float


def get_fts_header_info(fts_fname):
    """
    """
    header = {}
    with astropy.io.fits.open(fts_fname) as fts:
        assert len(fts) == 1
        fts = fts[0]
        return Header(fts.header['HAEX_OBS'],
                      fts.header['HAEY_OBS'],
                      fts.header['HAEZ_OBS'],
                      np.radians(fts.header['CRLN_OBS']),
                      fts.header['CRPIX1'] - 1,
                      fts.header['CRPIX2'] - 1,
                      np.radians(fts.header['CROTA']),
                      fts.header['CDELT1'],
                      fts.header['CDELT2'])


def get_sun_to_obs_vector(header):
    """
    """
    observatory = np.array([header.obs_x, header.obs_y, header.obs_z])

    sun_to_observation_vector = J20000_rot.apply(observatory)
    return sun_to_observation_vector * 1e-3 / SUN_RADIUS


class LOS(ABC):
    # Document me
    SPOL1 = np.array([np.cos(DELTA_POLE) * np.cos(ALPHA_POLE),
                      np.cos(DELTA_POLE) * np.sin(ALPHA_POLE),
                      np.sin(DELTA_POLE)])

    @property
    @abstractmethod
    def IMAGE_SIZE(self):
        """Image size in pixels."""
        pass

    @property
    @abstractmethod
    def PIXEL_SIZE(self):
        """Pixel size in arcseconds."""
        pass

    def __init__(self, header, i, j):
        # DOCUMENT EACH STEP OF WHAT APPEARS BELOW!!!
        self.sun_to_observation_vector = get_sun_to_obs_vector(header)
        self.center_x = header.center_x
        self.center_y = header.center_y
        self.roll_offset = header.roll_offset
        self.carrington_lon = header.carrington_lon

        self.dist_to_sun = np.linalg.norm(self.sun_to_observation_vector)

        self.i = i
        self.j = j
        self.x_image_j = self.PIXEL_SIZE * (self.j - self.center_x)
        self.y_image_i = self.PIXEL_SIZE * (self.i - self.center_y)

        self.rho = np.radians(np.sqrt(self.x_image_j**2 + self.y_image_i**2) / 3600)
        self.eta = np.arctan2(-self.x_image_j, self.y_image_i) + self.roll_offset

        z_angle = -np.arctan2(self.sun_to_observation_vector[1],
                              self.sun_to_observation_vector[0])

        Rz = sp.spatial.transform.Rotation.from_euler('z', z_angle)

        sob = Rz.apply(self.sun_to_observation_vector)

        y_angle = np.arctan2(sob[2], sob[0])

        Rzy = sp.spatial.transform.Rotation.from_euler('zy', [z_angle, y_angle])

        r3tmp = Rzy.apply(self.SPOL1)

        x_angle = np.arctan2(r3tmp[1], r3tmp[2])

        self.R12 = sp.spatial.transform.Rotation.from_euler('zyx', [z_angle, y_angle, x_angle])

        spol2 = self.R12.apply(self.SPOL1)

        p_angle = -np.arctan2(spol2[0], spol2[2])

        self.R23 = sp.spatial.transform.Rotation.from_euler('yz', [p_angle, self.carrington_lon])

        Rx = sp.spatial.transform.Rotation.from_euler('x', self.eta)

        r3 = Rx.apply(np.array([np.sin(self.rho)**2,
                                0,
                                np.sin(self.rho) * np.cos(self.rho)]) * self.dist_to_sun)

        g1 = np.array([-np.cos(self.rho), 0, np.sin(self.rho)])
        unit1 = Rx.apply(g1)

        self.sun_ob1 = self.sun_to_observation_vector
        self.sun_ob2 = self.R12.apply(self.sun_ob1)
        self.sun_ob3 = self.R23.apply(self.sun_ob2)
        self.nrpt = self.R23.apply(r3)
        self.unit = self.R23.apply(unit1)
        self.impact = np.linalg.norm(self.nrpt)

    def __call__(self, t):
        return self.nrpt + t * self.unit


class COR1LOS(LOS):
    IMAGE_SIZE = 1024
    PIXEL_SIZE = 7.5 * 1024 / IMAGE_SIZE


if __name__ == '__main__':
    """
    current file: 20080202_000500_1P4c1A.fts, 2 of 14 files
    Computed dist: 207.8832066 Rsun
    Computed dsun: 207.8832066 Rsun
    Header's dsun: nan Rsun

                Sun_ob1: [-186.955, 83.5583, 35.7967]
          Rz(a) Sun_ob1: [204.778, 0, 35.7967]
    Ry(b) Rz(a) Sun_ob1: [207.883, 0, 0]
            R12 Sun_ob1: [207.883, 1.77636e-14, 0]
                  spol2: [-0.125491, 0, 0.992095]
                  spol3: [0, -1.38778e-17, 1]
    Polar angle: -0.125823 radians = -7.20913 deg
         Header's Observed Latitude = 2.54639e-313 deg
    Carrington longitude: -1.36965329069 radians =  -78.475353 deg
    COMPUTED sun_ob1:        [-186.9546297, 83.55829561, 35.79672143]
    HEADER'S J2000 sun_obs:  [nan, nan, nan]
          Computed sun_ob3:  [41.2045428, -202.0817988, -26.08754157]

    Sub-Spacecraft Latitude  computed as ATAN(sun_ob3[2]/Sqrt{sun_ob3[0]^2+sun_ob3[1]^2)}: -7.209130758 deg
    Sub-Spacecraft Longitude computed as ATAN{sun_ob3[1]/sun_ob3[0]}:                       281.524647 deg

    ENTERING BUILDROW: rho1 = 0.0168341867155, eta1 = -0.610189590928
    entry theta = 53.9077 deg, exit theta = 54.4206 deg
    entry phi = 357.961 deg, exit phi = 1.54169 deg, ==> wrap = 1
    entry distance t1 = -0.0659061, exit distance t2 = 0.0659061,
    nrpt = 2.048087028 -0.009284771606 2.837412501
    unit = -0.188384648 0.9721860336 0.1391601253
    point of entry 2.060502734, -0.07335780716, 2.828240993
    point of exit  2.035671321, 0.05478826395, 2.846584009
    entry/exit bins (binbin) = 19 19 23 24 59 0
    RADIAL Bins: binrmin= 19, bin crossings:
    Theta Bins: bin crossings: (5, -54 deg, -0.0440552)
    PHI Bins: bin crossings: (59,0.00955041)
    times: -0.0659061 -0.0440552 0.00955041 0.0659061
    Voxel Numbers: (19,23,59, 35879)(19,24,59, 35899)(19,24,0, 499)
    EXITING BUILDROW (ontarget = 1)
    ENTERING BUILDROW: rho1 = 0.0168542381043, eta1 = -0.61199036373
    EXITING BUILDROW (ontarget = 0)
    ENTERING BUILDROW: rho1 = 0.0168743440178, eta1 = -0.613786851077
    EXITING BUILDROW (ontarget = 0)
    ENTERING BUILDROW: rho1 = 0.0168945042612, eta1 = -0.615579056694
    EXITING BUILDROW (ontarget = 0)
    ENTERING BUILDROW: rho1 = 0.0168645492404, eta1 = -0.609002244266
    EXITING BUILDROW (ontarget = 0)
    ENTERING BUILDROW: rho1 = 0.0168845645721, eta1 = -0.610801189542
    EXITING BUILDROW (ontarget = 0)
    ENTERING BUILDROW: rho1 = 0.0169046344161, eta1 = -0.612595869058
    EXITING BUILDROW (ontarget = 0)
    ENTERING BUILDROW: rho1 = 0.0169247585786, eta1 = -0.614386286449
    EXITING BUILDROW (ontarget = 0)
    ENTERING BUILDROW: rho1 = 0.0168949354554, eta1 = -0.607819166923
    EXITING BUILDROW (ontarget = 0)
    ENTERING BUILDROW: rho1 = 0.0169149148312, eta1 = -0.60961628113
    EXITING BUILDROW (ontarget = 0)
    ENTERING BUILDROW: rho1 = 0.0169349487068, eta1 = -0.611409149179
    EXITING BUILDROW (ontarget = 0)
    ENTERING BUILDROW: rho1 = 0.0169550368889, eta1 = -0.613197774617
    EXITING BUILDROW (ontarget = 0)
    ENTERING BUILDROW: rho1 = 0.0169253452328, eta1 = -0.606640339206
    EXITING BUILDROW (ontarget = 0)
    ENTERING BUILDROW: rho1 = 0.016945288754, eta1 = -0.60843561889
    EXITING BUILDROW (ontarget = 0)
    ENTERING BUILDROW: rho1 = 0.0169652867618, eta1 = -0.610226671927
    EXITING BUILDROW (ontarget = 0)
    ENTERING BUILDROW: rho1 = 0.0169853390638, eta1 = -0.612013501774
    EXITING BUILDROW (ontarget = 0)
    """

    """
    ipdb> A1[32207, :].tocsr().indices
    array([  499, 35879, 35899], dtype=int32)

    ipdb> A2[32207, :].tocsr().indices
    array([  499, 35879], dtype=int32)
    """
    fts_fname = '../data/cor1a1/20080202_000500_1P4c1A.fts'

    i = 225 * 4. # 900
    j = 192 * 4  # 768

    header = get_fts_header_info(fts_fname)
    los = COR1LOS(header, i, j)

    RMAX = 3.5
    abstrmax = np.sqrt(RMAX**2 - los.impact**2)
    t1 = -abstrmax
    t2 = abstrmax

    print(f'sun_ob1: {los.sun_ob1}')
    print(f'sun_ob2: {los.sun_ob2}')
    print(f'sun_ob3: {los.sun_ob3}')
    print(f'sun_ob3[0]: {los.sun_ob3[0]:.100g}')
    print(f'sun_ob3[1]: {los.sun_ob3[1]:.100g}')
    print(f'sun_ob3[2]: {los.sun_ob3[2]:.100g}')
    print(f'sun_obs3: {los.sun_ob3 * 1e3 * SUN_RADIUS}')
    print(f'nrpt: {los.nrpt}')
    print(f'nrpt: {los.nrpt * 1e3 * SUN_RADIUS}')
    print(f'unit: {los.unit}')
    print(f'unit[0]: {los.unit[0]:.100g}')
    print(f'unit[1]: {los.unit[1]:.100g}')
    print(f'unit[2]: {los.unit[2]:.100g}')
    print(f'entry distance: {t1}')
    print(f'exit distance: {t2}')
    print(f'point of entry: {los(t1)}')
    print(f'point of exit:  {los(t2)}')
