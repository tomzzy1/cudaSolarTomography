from dataclasses import dataclass

import numpy as np


@dataclass
class HollowSphere:
    R_MIN: float
    R_MAX: float
    N_RAD_BINS: int
    N_THETA_BINS: int
    N_PHI_BINS: int

    def __init__(self, m):
        self.R_MIN = m['R_MIN']
        self.R_MAX = m['R_MAX']
        self.N_RAD_BINS = m['N_RAD_BINS']
        self.N_THETA_BINS = m['N_THETA_BINS']
        self.N_PHI_BINS = m['N_PHI_BINS']
        self.__post_init__()


    def __post_init__(self):
        self.rad_centers = np.linspace(self.R_MIN + (self.R_MAX-self.R_MIN)/self.N_RAD_BINS/2,
                                       self.R_MAX - (self.R_MAX-self.R_MIN)/self.N_RAD_BINS/2,
                                       self.N_RAD_BINS)
        self.lat_centers = np.linspace(-90 + 90/self.N_THETA_BINS, 90 - 90/self.N_THETA_BINS, self.N_THETA_BINS)
        self.lon_centers = np.linspace(360/self.N_PHI_BINS/2, 360 - 360/self.N_PHI_BINS/2, self.N_PHI_BINS)

        self.rad_edges = np.linspace(self.R_MIN, self.R_MAX, self.N_RAD_BINS+1)
        self.lat_edges = np.linspace(-90, 90, self.N_THETA_BINS+1)
        self.lon_edges = np.linspace(0, 360, self.N_PHI_BINS+1)

        self.shape = (self.N_RAD_BINS, self.N_THETA_BINS, self.N_PHI_BINS)

        self.RAD_BIN_SIZE = (self.R_MAX - self.R_MIN) / self.N_RAD_BINS
        self.THETA_BIN_SIZE = 180 / self.N_THETA_BINS
        self.PHI_BIN_SIZE = 360 / self.N_PHI_BINS


    @property
    def theta_centers(self):
        return self.lat_centers

    @property
    def theta_edges(self):
        return self.lat_edges

    @property
    def phi_centers(self):
        return self.lon_centers

    @property
    def phi_edges(self):
        return self.lon_edges

    def rtp2ijk(self, r, t, p, degrees=True):
        """
        Return the hollow sphere bin (i, j, k) that contains the
        spherical point (r, t, p).

        i: radial bin
        j: theta bin
        k: phi bin
        """
        if not degrees:
            t = np.degrees(t)
            p = np.degrees(p)
        assert r >= self.R_MIN and r <= self.R_MAX
        assert t >= -90 and t <= 90
        p %= 360
        i = int((r - self.R_MIN) // self.RAD_BIN_SIZE)
        j = int(t // self.THETA_BIN_SIZE)
        k = int(p // self.PHI_BIN_SIZE)
        return i, j, k

    def xyz2rtp(self, x, y, z, degrees=True):
        """
        Cartesian to spherical conversion.

        Note that theta in [-pi/2, pi/2] so theta here is (pi/2 -
        theta) in
        https://en.wikipedia.org/wiki/Spherical_coordinate_system
        """
        r = np.linalg.norm([x, y, z])
        t = np.pi - np.arccos(z / r)
        p = np.arctan2(y, x) % (2 * np.pi)
        if degrees:
            t = np.degrees(t)
            p = np.degrees(p)
        return r, t, p

    def xyz2ijk(self, x, y, z):
        """
        Return the hollow sphere bin (i, j, k) that contains the
        Cartesian point (x, y, z).

        i: radial bin
        j: theta bin
        k: phi bin
        """
        return self.rtp2ijk(*self.xyz2rtp(x, y, z))

    def rtp2xyz(self, r, t, p, degrees=True):
        """
        Spherical to Cartesian conversion.

        Note that theta in [-pi/2, pi/2] so theta here is (pi/2 -
        theta) in
        https://en.wikipedia.org/wiki/Spherical_coordinate_system
        """
        if degrees:
            t = np.radians(t)
            p = np.radians(p)
        x = r * np.sin(t) * np.cos(p)
        y = r * np.sin(t) * np.sin(p)
        z = -r * np.cos(t)
        return x, y, z

    def ijk2I(self, i, j, k):
        """
        Convert from (i, j, k) index to linear index.
        """
        return np.ravel_multi_index((i, j, k), self.shape, order='F')

    def I2ijk(self, I):
        """
        Convert from linear index to (i, j, k).
        """
        return np.unravel_index(I, self.shape, order='F')

    def get_centers(self):
        """
        Return a list of tuples. The ith tuple is the center radius, latitude, and longitude of the ith voxel. UPDATE
        """
        i, j, k = self.I2ijk(range(np.prod(self.shape)))
        rad_centers = np.array([self.rad_centers[i_l] for i_l in i])
        theta_centers = np.array([self.lat_centers[j_l] for j_l in j])
        phi_centers = np.array([self.lon_centers[k_l] for k_l in k])
        return [x.reshape(self.shape) for x in [rad_centers, theta_centers, phi_centers]]



if __name__ == '__main__':
    m = {'R_MIN': 1.6,
         'R_MAX': 3.5,
         'N_RAD_BINS': 20,
         'N_THETA_BINS': 30,
         'N_PHI_BINS': 60}

    grid = HollowSphere(m)

    r = np.exp(1)
    theta = 35.655
    phi = 208.9

    print('rtp', r, theta, phi)

    x, y, z = grid.rtp2xyz(r, theta, phi)
    print('xyz', x, y, z)

    print('rtp', grid.xyz2rtp(x, y, z))

    i, j, k = grid.rtp2ijk(r, theta, phi)
    print(f'{grid.rad_edges[i]} <= {r} <= {grid.rad_edges[i+1]}? {grid.rad_edges[i] <= r <= grid.rad_edges[i+1]}')
    print(f'{grid.theta_edges[j]} <= {theta} <= {grid.theta_edges[j+1]}? {grid.theta_edges[j] <= theta <= grid.theta_edges[j+1]}')
    print(f'{grid.phi_edges[k]} <= {phi} <= {grid.phi_edges[k+1]}? {grid.phi_edges[k] <= phi <= grid.phi_edges[k+1]}')

    I = grid.ijk2I(i, j, k)
    print(I, i + j*grid.N_RAD_BINS + k*grid.N_RAD_BINS*grid.N_THETA_BINS)
    k_prime = I // (grid.N_RAD_BINS*grid.N_THETA_BINS)
    j_prime = (I - k_prime * grid.N_RAD_BINS*grid.N_THETA_BINS) // grid.N_RAD_BINS
    i_prime = I - k_prime * grid.N_RAD_BINS*grid.N_THETA_BINS - j_prime*grid.N_RAD_BINS
    print((i, j, k), (i_prime, j_prime, k_prime))
