from py_cuda_solartomography import GridParameters
from py_cuda_solartomography import InstrParameters
from py_cuda_solartomography import build_A_matrix_with_mapping_to_y
from py_cuda_solartomography import get_simulation_x

import numpy as np
import os
from pathlib import Path
import scipy

# projection_dir = '../data/projection/lasco_2023/no_noise/'
projection_dir = '../data/projection/lasco_2008/Gaussian/'
sim_dir = '../data/mhd_2008/'

# n_rad_bins = 30
# n_theta_bins = 75
# n_phi_bins = 150
n_rad_bins = 30
n_theta_bins = 101
n_phi_bins = 129

shape = (n_rad_bins, n_theta_bins, n_phi_bins)

# Gaussian 1 (std = /50)
# Gaussian 2 (std = /500)
# Gaussian 3 (std = /200)
# noise_t = 'no noise'
noise_t = 'Gaussian'

grid_params = GridParameters(
    n_rad_bins, n_theta_bins, n_phi_bins, 6.5, 2.0
)  # n_rad_bins, n_polar_bins, n_phi_bins, r_max, r_min
instr_params = InstrParameters(
    6.3, 2.1, 512, 23.8, 1, 1e10
)  # instr_r_max, instr_r_min, image_size, pixel_size, binning_factor, scale_factor

def make_config(path):
    file_list = list(filter(lambda f: ".fts" in f, os.listdir(path)))
    file_list.sort()
    return file_list

filenames = make_config(Path("../data/lasco_c2_2023_partial"))
row_idx, col_idx, val, y_mapping = build_A_matrix_with_mapping_to_y(
       filenames, grid_params, instr_params
    )

A = scipy.sparse.csr_array(
    (val, col_idx, row_idx),
    shape=(len(row_idx) - 1, n_rad_bins * n_theta_bins * n_phi_bins),
)

x_sim = get_simulation_x(sim_dir, grid_params)

y_zip = A @ x_sim

n_rows = 0
for i, filename in enumerate(filenames):
    pb_vector = []
    for j in range(instr_params.y_size):
        row_n = i * instr_params.y_size + j
        row_size = y_mapping[row_n + 1] - y_mapping[row_n]
        if row_size <= 0:
            pb_vector.append(0)
        else:
            y = y_zip[n_rows]
            n_rows += 1
            y /= instr_params.scale_factor
            pb_vector.append(y)
    assert(len(pb_vector) == instr_params.image_size * instr_params.image_size)
    pb_vector = np.array(pb_vector, dtype=np.float64)
    if noise_t == 'Gaussian':
        noise = np.random.normal(0, (np.max(pb_vector) - np.min(pb_vector)) / 500, pb_vector.shape)
        pb_vector += noise
        # pb_vector.tofile(Path(projection_dir) / Path('Gaussian') / Path(filename))
        pb_vector.tofile(Path(projection_dir) / Path(filename))
    else:
        pb_vector.tofile(Path(projection_dir) / Path(filename))


