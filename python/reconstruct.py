#!/usr/bin/env python3

from py_cuda_solartomography import GridParameters
from py_cuda_solartomography import InstrParameters
from py_cuda_solartomography import build_and_reconstruct_with_projection
from py_cuda_solartomography import get_simulation_x
from py_cuda_solartomography import build_A_matrix_with_projection
from py_cuda_solartomography import reconstruct

from dgrad2 import hlaplac

import scipy
import scipy.fft as fft
import numpy as np
import os
from pathlib import Path
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import sys

def make_config(path):
    file_list = list(filter(lambda f: ".fts" in f, os.listdir(path)))
    file_list.sort()
    return file_list

def main(argv=None):
    if argv is None:
        argv = sys.argv

    parser = ArgumentParser('Reconstruct for solar tomography',
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--lambda_tik', '-l', type=float, default=2e-5, help='regularization parameter')

    args = parser.parse_args(argv[1:])

    # projection_dir = '../data/projection/lasco_2023/'
    # projection_dir = '../data/projection/lasco_2023/Gaussian/'
    projection_dir = '../data/projection/lasco_2023/no_noise/'
    sim_dir = '../data/mhd/'

    n_rad_bins = 30
    n_theta_bins = 75
    n_phi_bins = 150

    shape = (n_rad_bins, n_theta_bins, n_phi_bins)
    shape_inv = (n_phi_bins, n_theta_bins, n_rad_bins)

    grid_params = GridParameters(
        n_rad_bins, n_theta_bins, n_phi_bins, 6.5, 2.0
    )  # n_rad_bins, n_polar_bins, n_phi_bins, r_max, r_min
    instr_params = InstrParameters(
        6.3, 2.1, 512, 23.8, 1, 1e10
    )  # instr_r_max, instr_r_min, image_size, pixel_size, binning_factor, scale_factor


    D = hlaplac(n_rad_bins, n_theta_bins, n_phi_bins)
    x_sim = get_simulation_x(sim_dir, grid_params)
    filenames = make_config('../data/lasco_c2_2023_partial/')

    # lambda_list = [1e-5, 5e-6, 2e-6, 1e-6, 5e-7, 1e-7]
    row_idx, col_idx, val, y = build_A_matrix_with_projection(
            make_config(Path("../data/lasco_c2_2023_partial")), projection_dir, grid_params, instr_params
        )
    x = reconstruct(grid_params, row_idx, col_idx, val, y, D.indptr,
        D.indices,
        D.data.astype(np.float32),
        args.lambda_tik)

    x.tofile("x_result")

if __name__ == '__main__':
    sys.exit(main())