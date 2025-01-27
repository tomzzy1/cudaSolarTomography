from py_cuda_solartomography import GridParameters
from py_cuda_solartomography import InstrParameters
from py_cuda_solartomography import build_A_matrix
from py_cuda_solartomography import normalized_input
from py_cuda_solartomography import A_transpose

# from py_cuda_solartomography import build_and_reconstruct
# from py_cuda_solartomography import reconstruct
from dgrad2 import hlaplac

import scipy
import numpy as np
import os
from pathlib import Path

import time

n_rad_bins = 20
n_theta_bins = 30
n_phi_bins = 60
# n_rad_bins = 30
# n_theta_bins = 75
# n_phi_bins = 150

# n_rad_bins = 10
# n_theta_bins = 10
# n_phi_bins = 20


def make_config(path):
    file_list = list(filter(lambda f: ".fts" in f, os.listdir(path)))
    file_list.sort()
    return file_list


def py_build_A():

    grid_params = GridParameters(
        n_rad_bins, n_theta_bins, n_phi_bins, 6.5, 2.0
    )  # n_rad_bins, n_polar_bins, n_phi_bins, r_max, r_min
    instr_params = InstrParameters(
        6.3, 2.1, 512, 23.8, 1, 1e10
    )  # instr_r_max, instr_r_min, image_size, pixel_size, binning_factor, scale_factor
    row_idx, col_idx, val, y = build_A_matrix(
        make_config(Path("../data/lasco_c2_partial")), grid_params, instr_params
    )
    print(val)
    assert len(y) + 1 == len(row_idx)
    A = scipy.sparse.csr_array(
        (val, col_idx, row_idx),
        shape=(len(row_idx) - 1, n_rad_bins * n_theta_bins * n_phi_bins),
    )
    print(A)
    print(y)
    return A, y


def py_build_and_reconstruct():
    # n_rad_bins = 30
    # n_theta_bins = 75
    # n_phi_bins = 150
    grid_params = GridParameters(
        n_rad_bins, n_theta_bins, n_phi_bins, 6.5, 2.0
    )  # n_rad_bins, n_polar_bins, n_phi_bins, r_max, r_min
    instr_params = InstrParameters(
        6.3, 2.1, 512, 23.8, 1, 1e10
    )  # instr_r_max, instr_r_min, image_size, pixel_size, binning_factor, scale_factor
    D = hlaplac(n_rad_bins, n_theta_bins, n_phi_bins)
    x = build_and_reconstruct(
        make_config(Path("../data/lasco_c2_partial")),
        grid_params,
        instr_params,
        D.indptr,
        D.indices,
        D.data.astype(np.float32),
        2e-5,
    )
    print(x)
    return x


def py_reconstruct():
    # n_rad_bins = 30
    # n_theta_bins = 75
    # n_phi_bins = 150
    grid_params = GridParameters(
        n_rad_bins, n_theta_bins, n_phi_bins, 6.5, 2.0
    )  # n_rad_bins, n_polar_bins, n_phi_bins, r_max, r_min
    instr_params = InstrParameters(
        6.3, 2.1, 512, 23.8, 1, 1e10
    )  # instr_r_max, instr_r_min, image_size, pixel_size, binning_factor, scale_factor
    D = hlaplac(n_rad_bins, n_theta_bins, n_phi_bins)
    A, y = py_build_A()
    x = reconstruct(
        grid_params,
        A.indptr,
        A.indices,
        A.data.astype(np.float32),
        y.astype(np.float32),
        D.indptr,
        D.indices,
        D.data.astype(np.float32),
        2e-5,
    )
    print(x)
    return x


def py_normalized_input(A, y):
    grid_params = GridParameters(n_rad_bins, n_theta_bins, n_phi_bins, 6.5, 2.0)
    x = normalized_input(
        grid_params,
        A.indptr,
        A.indices,
        A.data.astype(np.float32),
        y.astype(np.float32),
    )
    print(x)
    return x

def py_transpose(A):
    grid_params = GridParameters(n_rad_bins, n_theta_bins, n_phi_bins, 6.5, 2.0)
    return A_transpose(
    grid_params, A.indptr, A.indices, A.data.astype(np.float32)
)

start = time.time()
A, y = py_build_A()
print(f"build A {time.time() - start}s")

start = time.time()
A_T = A.transpose().tocsr()
print(type(A_T))
print(A.shape, A_T.shape)
# row_idx_T, col_idx_T, val_T = py_transpose(A)
# print(A_T.data, val_T)
# print(A_T.indices, col_idx_T)
# print(A_T.indices.shape, col_idx_T.shape)
# print(A_T.indptr, row_idx_T)
# print(A_T.indptr.shape, row_idx_T.shape)

print(A_T.shape, A.shape)
A_T_A = A_T @ A
print(A_T_A.shape)
trace = A_T_A.diagonal().sum()
print(trace)
print((A_T @ y) * (1 / trace))
print(f"numpy normalize {time.time() - start}s")
start = time.time()
py_normalized_input(A, y)
print(f"normalize {time.time() - start}s")
# py_build_and_reconstruct()
# py_reconstruct()
