#!/usr/bin/env python3
import scipy.interpolate
import scipy.ndimage
from read_mhd import read_mhd
from read_mhd import interp
from py_cuda_solartomography import GridParameters
from py_cuda_solartomography import InstrParameters
from py_cuda_solartomography import build_A_matrix
from py_cuda_solartomography import build_A_matrix_with_projection
from py_cuda_solartomography import reconstruct
from py_cuda_solartomography import build_A_matrix_with_mapping_to_y
from py_cuda_solartomography import build_A_matrix_with_virtual_viewpoints

from dgrad2 import hlaplac
from dgrad2 import r3

import scipy
import scipy.fft as fft
import numpy as np
import os
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import time
import h5py
import cv2
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

def tikhonov(x, y, A, D, lambda_tik):
    """
    Return ||*y* - *A*@*x*||^2 + ||*lambda_tik* * *D*@*x*||^2.
    """
    cost = 0
    # data term
    z = y - A @ x
    data_term = np.dot(z, z)
    cost += data_term
    # regularization term
    z_D = D @ x
    reg_term = lambda_tik**2 * np.dot(z_D, z_D)
    cost += reg_term
    print(f'||y - Ax||^2 = {data_term}, ||{lambda_tik}*D x||^2 = {reg_term}, cost = {cost}')
    # logger.info(f'||y - Ax||^2 = {data_term}, ||{lambda_tik}*D x||^2 = {reg_term}, cost = {cost}')
    return cost


def tikhonov_grad(x, y, A, D, lambda_tik):
    """
    Return the gradient ot ||*y* - *A*@*x*||^2 + ||*lambda_tik* * *D*@*x*||^2.
    """
    grad = np.zeros_like(x)
    # data term
    grad += 2 * (A.T @ (A @ x))
    grad -= 2 * (y @ A)
    # regularization term
    w = D @ x
    grad += 2 * lambda_tik**2 * (D.T @ w)
    return grad

def comp(fn):
    l = fn.split('.')[0].split('-')
    month = int(l[1])
    day = int(l[2])
    hour = int(l[3])
    return (month, day, hour)

def make_config(path):
    file_list = list(filter(lambda f: ".fts" in f, os.listdir(path)))
    file_list.sort(key=comp)
    return file_list

def make_projection(path):
    file_list = os.listdir(path)
    file_list.sort(key=lambda s: int(s.split('_')[0]))
    return file_list

def relative_error(x1, x2):
    error = 0
    for i in range(len(x1)):
        error += abs(x2[i] - x1[i]) / x1[i]
    return error / len(x1)

def RMS_std_error(x1, x2, shape):
    x1_3d = np.reshape(x1, shape, order='F')
    x2_3d = np.reshape(x2, shape, order='F')
    error = 0.0
    # for i in range(n_rad_bins):
    for i in range(shape[0]):
        std = np.std(x1_3d[i])
        rad_error = np.linalg.norm(x1_3d[i] - x2_3d[i]) / np.sqrt(shape[1] * shape[2]) / std
        # print(i, rad_error)
        error += rad_error
        # print(i, std, error)
    return error / shape[0]

def RMS_avg_error(x1, x2, shape):
    x1_3d = np.reshape(x1, shape, order='F')
    x2_3d = np.reshape(x2, shape, order='F')
    error = 0.0
    # for i in range(n_rad_bins):
    for i in range(shape[0]):
        avg = np.average(x1_3d[i])
        rad_error = np.linalg.norm(x1_3d[i] - x2_3d[i]) / np.sqrt(shape[1] * shape[2]) / avg
        # print(i, rad_error)
        error += rad_error
        # print(i, std, error)
    return error / shape[0]

def RMSP_error(x1, x2):
    return np.linalg.norm((x1 - x2) / x1)

def main(argv=None):
    if argv is None:
        argv = sys.argv

    parser = ArgumentParser('Experiments about resolution',
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('prog', type=int, help='program index')
    args = parser.parse_args(argv[1:])

    prog = args.prog
    n_rad_bins = [10 + 5 * i for i in range(8)]
    n_phi_bins = [20, 30, 50, 60, 75, 100, 150, 300]
    print(n_rad_bins)

    if prog == 0:
        for i in n_rad_bins:
        #     for k in n_phi_bins:
        # for i in [150]:
            for k in [600]:
                j = k // 2
                print(i, j, k)
                # read_mhd('../data/mhd_2023/', i, j, k)
                read_mhd('../data/mhd/', i, j, k)
    elif prog == 1:
        # used to control whether the projections used is Gaussian 
        filtered = False
        # for i in n_rad_bins[6:]:
        #     for k in n_phi_bins:
        # for i in [45]:
        #     for k in [300]:
        # for e in [0]:
        #     for d in range(1, 7):
        # for i in [150]:
        #     for k in [300]:
        # for i in [200]:
        #     for k in [600]:
        # for i in n_rad_bins:
        #     # for k in [300]:
        #     for k in [20]:
        for i in [30]:
            for k in [150]:
                # i = 60 // d
                # j = 180 // d
                # k = 300 // d 
                j = k // 2
                print(i, j, k)
                if filtered:
                    projection_dir = Path(f'../data/mhd_projections_no_noise_gaussian_filtered/{i}_{j}_{k}')
                else:
                    # projection_dir = Path(f'../data/mhd_projections_no_noise/{i}_{j}_{k}')
                    projection_dir = Path(f'../data/mhd_projections_no_noise/{i}_{j}_{k}')
                    # projection_dir = Path(f'../data/mhd_full_projections/{i}_{j}_{k}')
                    # projection_dir = Path(f'../data/mhd_full_projections_3/{i}_{j}_{k}')
                    # projection_dir = Path(f'../data/mhd_full_projections_4/{i}_{j}_{k}')
                    # projection_dir = Path(f'../data/mhd_full_projections_gaussian2/{i}_{j}_{k}')
                projection_dir.mkdir(parents=True, exist_ok=True)
                sim_dir = 'mhd_resolutions/'

                shape = (i, j, k)

                # noise_t = 'Gaussian'
                # noise_t = 'Gaussian2'
                noise_t = 'no noise'

                grid_params = GridParameters(
                    i, j, k, 6.5, 2.0
                )  # n_rad_bins, n_theta_bins, n_phi_bins, r_max, r_min
                # instr_params = InstrParameters(
                #     6.3, 2.1, 512, 23.8, 1, 1e10
                # )  # instr_r_max, instr_r_min, image_size, pixel_size, binning_factor, scale_factor
                instr_params = InstrParameters(
                    6.3, 2.1, 512, 23.8, 1, 1e10
                )  # instr_r_max, instr_r_min, image_size, pixel_size, binning_factor, scale_factor

                filenames = make_config(Path("../data/lasco_c2_2023_partial"))
                # filenames = make_config(Path("../data/lasco_c2_2023"))
                row_idx, col_idx, val, y_mapping = build_A_matrix_with_mapping_to_y(
                    filenames, grid_params, instr_params
                    )
                
                print(row_idx)
                assert(not np.any(np.isnan(row_idx)))
                assert(not np.any(np.isnan(col_idx)))
                assert(not np.any(np.isnan(val)))

                A = scipy.sparse.csr_array(
                    (val, col_idx, row_idx),
                    shape=(len(row_idx) - 1, i * j * k),
                )

                # print(A)
                x_sim = np.fromfile(Path("mhd_resolutions") / f"x_corhel_{i}_{j}_{k}", dtype=np.float32)
                # x_sim = np.fromfile(Path("mhd_resolutions_3") / f"x_corhel_{i}_{j}_{k}", dtype=np.float32)
                # x_sim = np.fromfile(Path("mhd_resolutions_4") / f"x_corhel_{i}_{j}_{k}", dtype=np.float32)
                assert(not np.any(np.isnan(x_sim)))
                # print('x_sim read')

                y_zip = A @ x_sim

                n_rows = 0
                for row, filename in enumerate(filenames):
                    print(f'row {row} filename {filename}')
                    pb_vector = []
                    for col in range(instr_params.y_size):
                        row_n = row * instr_params.y_size + col
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
                    if filtered:
                        image = np.reshape(pb_vector, (512, 512))
                        log_image = np.log(image)
                        mask = log_image == -np.inf
                        # fill the invalid region to for later filtering
                        for i in range(image.shape[0]):
                            for j in range(image.shape[1]):
                                if image[i][j] == -np.inf:
                                    avg = 0
                                    cnt = 0
                                    for ii in range(max(0, i - 1), min(image.shape[0], i + 2)):
                                        for jj in range(max(0, j - 1), min(image.shape[1], j + 2)):
                                            if image[ii][jj] != -np.inf:
                                                avg += image[ii][jj]
                                                cnt += 1
                                    if cnt > 0:
                                        image[i][j] = avg / cnt
                        # convert to polar coordinate and filter
                        log_polar_image = cv2.linearPolar(log_image, (512 // 2, 512 // 2), 512 // 2, cv2.WARP_FILL_OUTLIERS)
                        log_smoothed_polar_image = scipy.ndimage.gaussian_filter(log_polar_image, sigma=3, mode='wrap')
                        # log_smoothed_polar_image = scipy.ndimage.gaussian_filter1d(log_polar_image, sigma=3, axis=1, mode='nearest')
                        log_smoothed_image = cv2.linearPolar(log_smoothed_polar_image, (512 // 2, 512 // 2), 512 // 2, cv2.WARP_INVERSE_MAP | cv2.WARP_FILL_OUTLIERS)
                        log_smoothed_image[mask] = -np.inf
                        log_smoothed_image[log_smoothed_image == 0] = -np.inf

                        smoothed_image = np.exp(log_smoothed_image)
                        pb_vector = smoothed_image.flatten()
                    if noise_t == 'Gaussian':
                        print(np.max(pb_vector), np.min(pb_vector), np.average(pb_vector), np.std(pb_vector))
                        # noise = np.random.normal(0, (np.max(pb_vector) - np.min(pb_vector)) / 1000, pb_vector.shape)
                        # noise = np.random.normal(0, np.average(pb_vector) / 100, pb_vector.shape)
                        # pb_vector += noise
                        # pb_vector.tofile(Path(projection_dir) / Path('Gaussian') / Path(filename))
                        pb_vector.tofile(projection_dir / Path(filename))
                    elif noise_t == 'Gaussian2':
                        noise = np.random.normal(0, np.std(pb_vector) / 100, pb_vector.shape)
                        pb_vector = np.maximum(np.zeros(pb_vector.shape), pb_vector + noise)
                        pb_vector.tofile(Path(projection_dir) / Path(filename))
                    else:
                        pb_vector.tofile(projection_dir / Path(filename))
    elif prog == 2:
        #for l in [i * 1e-6 for i in range(1, 11)]:
        # for l in [1e-6]:
        # for l in [1e-8, 5e-8, 1e-7, 5e-7]:
        for l in [0]:
            # with open(f'error_vs_resolution_RMS_{l}.txt', 'w') as f:
            # with open(f'error_vs_resolution_double.txt', 'w') as f:
            with open(f'error_vs_resolution_float_self_30_150_300_lambda{l}.txt', 'w') as f:
                # for i in n_rad_bins:
                #    for k in n_phi_bins:
                # for i in [10]:
                #     for k in [20]:
                for i in [30]:
                    # for k in [150]:
                    for k in [300]:
                # for e in [0]:
                #     for d in range(1, 7):
                #         i = 60 // d
                #         j = 180 // d
                #         k = 300 // d 
                        j = k // 2
                        shape = (i, j, k)
                        print(shape)
                        # projection_dir = f'../data/mhd_full_projections_4/60_180_300/'
                        projection_dir = f'../data/mhd_projections_no_noise/{i}_{j}_{k}/'
                        # projection_dir = f'../data/mhd_projections_no_noise/45_150_300/'
                        # projection_dir = f'../data/mhd_projections/{i}_{j}_{k}/'
                        # projection_dir = f'../data/mhd_projections/45_150_300/'
                        # projection_dir = f'../data/mhd_projections_no_noise_gaussian_filtered/45_150_300/'
                        # projection_dir = f'../data/mhd_projections_no_noise/avg/'
                        # projection_dir = f'../data/mhd_projections_no_noise/{i}_150_300/'
                        # projection_dir = f'../data/mhd_full_projections_4/{i}_300_600/'

                        t0 = time.time()
                        grid_params = GridParameters(
                            i, j, k, 6.5, 2.0
                        )  # n_rad_bins, n_polar_bins, n_phi_bins, r_max, r_min
                        instr_params = InstrParameters(
                            6.3, 2.1, 512, 23.8, 1, 1e10
                        )  # instr_r_max, instr_r_min, image_size, pixel_size, binning_factor, scale_factor


                        D = hlaplac(i, j, k)
                        row_idx, col_idx, val, y = build_A_matrix_with_projection(
                            make_config(Path("../data/lasco_c2_2023_partial")), projection_dir, grid_params, instr_params)
                        
                        # A = scipy.sparse.csr_array(
                        #     (val, col_idx, row_idx),
                        #     shape=(len(row_idx) - 1, i * j * k),
                        # )
                        
                        # assert(not np.any(np.isnan(row_idx)))
                        # assert(not np.any(np.isnan(col_idx)))
                        # assert(not np.any(np.isnan(val)))
                        
                        x = reconstruct(grid_params, row_idx, col_idx, val, y, D.indptr,
                            D.indices,
                            D.data.astype(np.float32),
                            l)
                        # x0 = np.zeros(i * j * k)
                        # res = scipy.optimize.minimize(tikhonov, x0, (y, A, D, 1e-6), jac=tikhonov_grad, bounds=None, method='L-BFGS-B')
                        # x = res.x
                        assert(not np.any(np.isnan(x)))
                        
                        t1 = time.time()
                        
                        np.array(x).tofile(f"x_{i}_{j}_{k}_{l}")
                        # np.array(x).tofile(f"x_{i}_{j}_{k}")
                        # x = np.fromfile(f"x_{i}_{j}_{k}", dtype=np.float32)
                        
                        sim_dir = 'mhd_resolutions'
                        # x_sim = np.fromfile(Path("mhd_resolutions_4") / f"x_corhel_{i}_{j}_{k}", dtype=np.float32)
                        x_sim = np.fromfile(Path("mhd_resolutions") / f"x_corhel_{i}_{j}_{k}", dtype=np.float32)
                        # x_sim = np.fromfile(Path("mhd_resolutions_3") / f"x_corhel_{i}_{j}_{k}", dtype=np.float32)
                        # if k != 300:
                        #     x_sim = np.fromfile(Path(f"mhd_resolutions_{i}_{150}_{300}") / f"x_corhel_{i}_{j}_{k}", dtype=np.float32)
                        # else:
                        #     x_sim = np.fromfile(Path("mhd_resolutions") / f"x_corhel_{i}_{j}_{k}", dtype=np.float32)
                        
                        assert(not np.any(np.isnan(x_sim)))
                        assert(np.count_nonzero(x_sim) == len(x_sim))

                        x_sim = np.reshape(x_sim, shape, order='F')[1:-max((i // 10 + 1), 1)].flatten().copy()
                        x = np.reshape(x, shape, order='F')[1:-max((i // 10 + 1), 1)].flatten().copy()
                        error = relative_error(x_sim, x)
                        # error = RMS_std_error(x_sim, x, (i - (max((i // 10 + 1), 1)) - 1, j, k))
                        # error = RMS_avg_error(x_sim, x, shape)
                        print(f'{i} {j} {k} {error} {t1 - t0}s {len(val)}\n')
                        f.write(f'{i} {j} {k} {error} {t1 - t0}s {len(val)}\n')
                        # print(f'{i} {j} {k} {error} {t1 - t0}s {0}\n')
                        # f.write(f'{i} {j} {k} {error} {t1 - t0}s {0}\n')

    elif prog == 3 or prog == 4:
        # i = 30
        # k = 150
        # j = k // 2
        # i = 20
        # k = 100
        # j = k // 2
        # i = 10
        # k = 50
        # j = k // 2
        # for i in n_rad_bins:
        for i in [45]:
            for k in n_phi_bins:
                j = k // 2
                if prog == 3:
                    print(i, j, k)
                    projection_dir = Path(f'../data/mhd_full_projections/{i}_{j}_{k}')
                    projection_dir.mkdir(parents=True, exist_ok=True)

                    shape = (i, j, k)

                    noise_t = 'Gaussian'

                    grid_params = GridParameters(
                        i, j, k, 6.5, 2.0
                    )  # n_rad_bins, n_theta_bins, n_phi_bins, r_max, r_min
                    instr_params = InstrParameters(
                        6.3, 2.1, 512, 23.8, 1, 1e10
                    )  # instr_r_max, instr_r_min, image_size, pixel_size, binning_factor, scale_factor
                    viewpoints = make_config(Path("../data/lasco_c2_2023"))
                    row_idx, col_idx, val, y_mapping = build_A_matrix_with_mapping_to_y(
                        viewpoints, grid_params, instr_params
                        )

                    A = scipy.sparse.csr_array(
                        (val, col_idx, row_idx),
                        shape=(len(row_idx) - 1, i * j * k),
                    )

                    x_sim = np.fromfile(Path("mhd_resolutions") / f"x_corhel_{i}_{j}_{k}", dtype=np.float32)

                    y_zip = A @ x_sim

                    n_rows = 0
                    for row, filename in enumerate(viewpoints):
                        pb_vector = []
                        for col in range(instr_params.y_size):
                            row_n = row * instr_params.y_size + col
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
                            pb_vector.tofile(projection_dir / Path(filename))
                        else:
                            pb_vector.tofile(projection_dir / Path(filename))

                elif prog == 4:
                    with open(f'error_vs_viewpoints_{i}_{j}_{k}.txt', 'w') as f:
                        full_viewpoints = make_config(Path("../data/lasco_c2_2023"))
                        dates_list = []
                        basic_list = []
                        for vp in full_viewpoints:
                            dates = vp.split('.')[0].split('-')
                            if dates[3] == '9':
                                basic_list.append(vp)
                            else:
                                dates_list.append(dates)
                        viewpoints_list = [basic_list]
                        size = len(basic_list)
                        cur_day = dates_list[0][2]
                        while size < len(full_viewpoints):
                            new_list = viewpoints_list[-1].copy()
                            for _ in range(3):
                                for date in dates_list:
                                    if cur_day != date[2]:
                                        new_list.append('-'.join(date) + '.fts')
                                        cur_day = date[2]
                                        dates_list.remove(date)
                                        break
                            viewpoints_list.append(new_list)
                            size += 3
                        print([len(vp) for vp in viewpoints_list])
                        for viewpoints in viewpoints_list:

                            t0 = time.time()
                            grid_params = GridParameters(
                                i, j, k, 6.5, 2.0
                            )  # n_rad_bins, n_polar_bins, n_phi_bins, r_max, r_min
                            instr_params = InstrParameters(
                                6.3, 2.1, 512, 23.8, 1, 1e10
                            )  # instr_r_max, instr_r_min, image_size, pixel_size, binning_factor, scale_factor

                            print(f'Projection dir ../data/mhd_full_projections/{i}_{j}_{k}/')
                            D = hlaplac(i, j, k)
                            row_idx, col_idx, val, y = build_A_matrix_with_projection(
                                viewpoints, f'../data/mhd_full_projections/{i}_{j}_{k}/', grid_params, instr_params)
                            
                            x = reconstruct(grid_params, row_idx, col_idx, val, y, D.indptr,
                                D.indices,
                                D.data.astype(np.float32),
                                7e-6)
                            
                            t1 = time.time()
                        
                            np.array(x).tofile(f"x_{i}_{j}_{k}_{len(viewpoints)}viewpoints")
                            
                            sim_dir = 'mhd_resolutions'
                            x_sim = np.fromfile(Path("mhd_resolutions") / f"x_corhel_{i}_{j}_{k}", dtype=np.float32)
                            f.write(f'{i} {j} {k} {relative_error(x_sim, x)} {t1 - t0}s {len(viewpoints)}\n')

    elif prog == 5 or prog == 6:
        i = 30
        # j = 75
        # k = 150
        for k in n_phi_bins[:-2]:
        # for k in [300]:
        # for k in [150, 300]:
            j = k // 2
            print(i, j, k)
            if prog == 5:
                for n_viewpoints in range(12, 63, 3):
                # for n_viewpoints in range(48, 63, 3):
                # for n_viewpoints in range(12, 47, 3):
                # n_viewpoints = 15
                    projection_dir = Path(f'../data/mhd_projections_virtual_noise/{i}_{j}_{k}/{n_viewpoints}vps')
                    projection_dir.mkdir(parents=True, exist_ok=True)

                    shape = (i, j, k)

                    noise_t = 'Gaussian'
                    # noise_t = 'no noise'

                    grid_params = GridParameters(
                        i, j, k, 6.5, 2.0
                    )  # n_rad_bins, n_theta_bins, n_phi_bins, r_max, r_min
                    instr_params = InstrParameters(
                        6.3, 2.1, 512, 23.8, 1, 1e10
                    )  # instr_r_max, instr_r_min, image_size, pixel_size, binning_factor, scale_factor
                    viewpoints = make_config(Path("../data/lasco_c2_2023"))
                    row_idx, col_idx, val, y_mapping = build_A_matrix_with_virtual_viewpoints(
                        viewpoints, n_viewpoints, grid_params, instr_params
                        )

                    A = scipy.sparse.csr_array(
                        (val, col_idx, row_idx),
                        shape=(len(row_idx) - 1, i * j * k),
                    )

                    # print(val, col_idx, row_idx)
                    # print(np.count_nonzero(col_idx[col_idx < 0]), np.count_nonzero(col_idx[col_idx >= i * j * k]))
                    # print(np.argmax(col_idx < 0), np.argmax(col_idx >= i * j * k))
                    # print(np.max(col_idx), np.min(col_idx))

                    x_sim = np.fromfile(Path("mhd_resolutions") / f"x_corhel_{i}_{j}_{k}", dtype=np.float32)

                    y_zip = A @ x_sim

                    n_rows = 0
                    for row in range(n_viewpoints):
                        pb_vector = []
                        for col in range(instr_params.y_size):
                            row_n = row * instr_params.y_size + col
                            row_size = y_mapping[row_n + 1] - y_mapping[row_n]
                            if row_size <= 0:
                                pb_vector.append(0)
                            else:
                                y = y_zip[n_rows]
                                n_rows += 1
                                y /= instr_params.scale_factor
                                pb_vector.append(y)
                        # print(len(pb_vector), instr_params.image_size * instr_params.image_size)
                        assert(len(pb_vector) == instr_params.image_size * instr_params.image_size)
                        pb_vector = np.array(pb_vector, dtype=np.float64)
                        if noise_t == 'Gaussian':
                            # noise = np.random.normal(0, (np.max(pb_vector) - np.min(pb_vector)) / 500, pb_vector.shape)
                            # pb_vector += noise
                            noise = np.random.normal(0, np.std(pb_vector) / 500, pb_vector.shape)
                            pb_vector += noise
                            pb_vector = np.maximum(np.zeros(pb_vector.shape), pb_vector)
                            # pb_vector.tofile(Path(projection_dir) / Path('Gaussian') / Path(filename))
                            pb_vector.tofile(projection_dir / Path(f'{row}_in_{n_viewpoints}vps'))
                        else:
                            pb_vector.tofile(projection_dir / Path(f'{row}_in_{n_viewpoints}vps'))
            elif prog == 6:
                # with open(f'error_vs_virtual_viewpoints_{i}_{j}_{k}_noise.txt', 'w') as f:
                # for l in [1e-8, 1e-6]:
                for l in [5e-8, 1e-7]:
                    with open(f'error_vs_virtual_viewpoints_{i}_{j}_{k}_lambda{l}.txt', 'w') as f:
                        for n_viewpoints in range(12, 63, 3):
                        # for n_viewpoints in range(12, 47, 3):
                        # for n_viewpoints in range(48, 63, 3):
                        # n_viewpoints = 15
                            # projection_dir = Path(f'../data/mhd_projections_virtual_noise/{i}_{j}_{k}')
                            projection_dir = Path(f'../data/mhd_projections_virtual/{i}_{j}_{k}')
                            projection_dir.mkdir(parents=True, exist_ok=True)

                            shape = (i, j, k)

                            t0 = time.time()
                            grid_params = GridParameters(
                                i, j, k, 6.5, 2.0
                            )  # n_rad_bins, n_theta_bins, n_phi_bins, r_max, r_min
                            instr_params = InstrParameters(
                                6.3, 2.1, 512, 23.8, 1, 1e10
                            )  # instr_r_max, instr_r_min, image_size, pixel_size, binning_factor, scale_factor
                            viewpoints = make_config(Path("../data/lasco_c2_2023"))
                            # virtual_viewpoints = make_projection(f'../data/mhd_projections_virtual/{i}_{j}_{k}/{n_viewpoints}vps')
                            # virtual_viewpoints = make_projection(f'../data/mhd_projections_virtual/{30}_{j}_{k}/{n_viewpoints}vps')'
                            virtual_viewpoints = make_projection(f'../data/mhd_projections_virtual/{30}_{150}_{300}/{n_viewpoints}vps')
                            print('viewpoints: ', viewpoints)
                            print('virtual_viewpoints: ', virtual_viewpoints)
                            # row_idx, col_idx, val, y = build_A_matrix_with_projection(
                            #     viewpoints, f'../data/mhd_projections_virtual/{30}_{j}_{k}/{n_viewpoints}vps/' , grid_params, instr_params, virtual_viewpoints, n_viewpoints
                            #     )
                            row_idx, col_idx, val, y = build_A_matrix_with_projection(
                                viewpoints, f'../data/mhd_projections_virtual/{30}_{150}_{300}/{n_viewpoints}vps/' , grid_params, instr_params, virtual_viewpoints, n_viewpoints
                                )
                            
                            D = hlaplac(i, j, k)
                            
                            x = reconstruct(grid_params, row_idx, col_idx, val, y, D.indptr,
                                        D.indices,
                                        D.data.astype(np.float32),
                                        l)
                                                        
                            t1 = time.time()
                            np.array(x).tofile(f"x_{i}_{j}_{k}_{n_viewpoints}viewpoints_lambda{l}")

                            # print(val, col_idx, row_idx)
                            # print(np.count_nonzero(col_idx[col_idx < 0]), np.count_nonzero(col_idx[col_idx >= i * j * k]))
                            # print(np.argmax(col_idx < 0), np.argmax(col_idx >= i * j * k))
                            # print(np.max(col_idx), np.min(col_idx))

                            x_sim = np.fromfile(Path("mhd_resolutions") / f"x_corhel_{i}_{j}_{k}", dtype=np.float32)

                            x_sim = np.reshape(x_sim, shape, order='F')[1:-max((i // 10 + 1), 1)].flatten().copy()
                            x = np.reshape(x, shape, order='F')[1:-max((i // 10 + 1), 1)].flatten().copy()
                            error = relative_error(x_sim, x)
                            # error = RMS_std_error(x_sim, x, shape)
                            # error = RMS_avg_error(x_sim, x, shape)
                            print(f'{i} {j} {k} {error} {t1 - t0}s {n_viewpoints}\n')
                            f.write(f'{i} {j} {k} {error} {t1 - t0}s {n_viewpoints}\n')


    elif prog == 7:
        # density = read_mhd('../data/mhd_2023/', 45, 150, 300)
        # density_3d = np.reshape(density, (300, 150, 45))
        # for i in n_rad_bins:
        #     for k in n_phi_bins:
        #         j = k // 2
        #         print(i, j, k)
        #         interp(density_3d, i, j, k)
        mhd_path = Path('../data/mhd_2023/')
        for fname in os.listdir(mhd_path):
            if not fname.split(".")[-1] == "h5":
                continue
            print(mhd_path / fname)
            f = h5py.File(mhd_path / fname, "r+")
            phi = f["fakeDim0"][:]
            polar = f["fakeDim1"][:]
            rad = f["fakeDim2"][:]
            N_phi = f["fakeDim0"].shape[0]
            N_polar = f["fakeDim1"].shape[0]
            rad_polys = []
            rad_polys2 = []
            for i in range(N_phi):
                print(f"rad_interps {i}")
                for j in range(N_polar):
                    rad_polys.append(scipy.interpolate.Akima1DInterpolator(np.reciprocal(rad)[::-1], f["Data-Set-2"][i][j][:][::-1]))
                    rad_polys2.append(scipy.interpolate.Akima1DInterpolator(rad, f["Data-Set-2"][i][j][:]))
                    # rad_polys.append(np.poly1d(np.polyfit(
                    #     rad, f["Data-Set-2"][i][j][:], 6
                    # )))
                    # rad_polys.append(scipy.interpolate.interp1d(
                    #     rad, f["Data-Set-2"][i][j][:], kind='cubic'
                    # ))
                    # rad_polys2.append(scipy.interpolate.interp1d(
                    #     rad, f["Data-Set-2"][i][j][:], kind='linear'
                    # ))
                    # print(rad_polys)
            # for i in n_rad_bins:
            #     for k in n_phi_bins:
            # for alpha in [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7]:
            #     i = int(10 * alpha)
            #     j = int(20 * alpha)
            #     k = int(40 * alpha)
            for alpha in [0]:
                i = 200
                j = 300
                k = 600
                # j = k // 2
                print(i, j, k)
                my_rad = np.linspace(2, 6.5, i + 1)[1:]
                density1 = np.zeros((N_phi, N_polar, i))
                for ii in range(N_phi):
                    for jj in range(N_polar):
                        density1[ii, jj] = rad_polys[ii * N_polar + jj](np.reciprocal(my_rad)[::-1])[::-1]
                        # if abs(alpha - 5) < 0.1:
                        #     print(density1[ii, jj], rad_polys2[ii * N_polar + jj](my_rad))
                        # density1[ii, jj] = rad_polys[ii * N_polar + jj](my_rad)
                        # print(rad_polys2[ii * N_polar + jj](my_rad), density1[ii, jj])
                        # print(rad ,my_rad)
                # density1 = np.reshape(density1.flatten(order='F'), (i, N_polar, N_phi), order='C') 
                density1 = density1.transpose((2, 1, 0))
                density2 = np.zeros((i, j, k))
                my_polar = np.linspace(0, np.pi, j + 1)[1:][::-1]
                my_phi = np.linspace(0, 2 * np.pi, k + 1)[1:]
                my_polar_grid, my_phi_grid = np.meshgrid(my_polar, my_phi)
                my_grid = np.vstack([my_polar_grid.ravel(), my_phi_grid.ravel()]).T
                for ii in range(i):
                    reg_interp = scipy.interpolate.RegularGridInterpolator((polar, phi), density1[ii])
                    density2[ii] = np.reshape(reg_interp(my_grid), (j, k), order='F')
                density = density2.flatten(order='F') * 1e8
                assert(not np.any(np.isnan(density)))
                density.astype(np.float32).tofile(Path("mhd_resolutions_3") / f"x_corhel_{i}_{j}_{k}")

            
    elif prog == 8:
        # density = 
        density = read_mhd('../data/mhd_2023/', 60, 120, 240)
        # density = read_mhd('../data/mhd_2023/', 60, 180, 300)
        # density = np.fromfile('mhd_resolutions_4/x_corhel_60_180_300', dtype=np.float32)
        assert(not np.any(np.isnan(density)))
        # density = np.reshape(density, (60, 180, 300), order='F')
        density = np.reshape(density, (60, 120, 240), order='F')
        for d in range(1, 7):
            # i = 60 // d
            # j = 180 // d
            # k = 300 // d 
            i = 60 // d
            j = 120 // d
            k = 240 // d 
            print(i, j, k)
            new_density = np.zeros((i, j, k))
            for ii in range(i):
                for jj in range(j):
                    for kk in range(k):
                        # print(density[ii * d: d * (ii + 1), jj * d: d * (jj + 1) , kk * d: d * (kk + 1)])
                        new_density[ii, jj, kk] = np.average(density[ii * d: d * (ii + 1), jj * d: d * (jj + 1) , kk * d: d * (kk + 1)])
            assert(not np.any(np.isnan(new_density)))
            new_density.flatten(order='F').astype(np.float32).tofile(Path("mhd_resolutions_4") / f"x_corhel_{i}_{j}_{k}")   
    elif prog == 8:
        file_map = dict()
        projection_dir = Path('../data/mhd_projections_no_noise/')
        for folder in os.listdir(projection_dir):
            print(folder)
            if folder[:2].isnumeric():
                for fn in os.listdir(projection_dir / folder):
                    if fn not in file_map:
                        file_map[fn] = [np.fromfile(projection_dir / folder / fn)]
                    else:
                        file_map[fn].append(np.fromfile(projection_dir / folder / fn))
        dir = projection_dir / 'avg'
        dir.mkdir(parents=True, exist_ok=True)
        for fn, files in file_map.items():
            new_file = np.average(np.vstack(files), axis=0)
            print(new_file.shape)
            new_file.tofile(dir / fn)

    elif prog == 9 or prog == 10:
        i = 30
        # j = 75
        # k = 150
        # for k in n_phi_bins[:-1]:
        # for n_viewpoints in range(12, 47, 3):
        # for n_viewpoints in range(48, 76, 3):
        # for degree in [130.0, 150.0]:
        # for degree in [90.0, 100.0, 110.0]:
        # for degree in [80.0]:
        # for degree in [160.0]:
        for degree in [80.0, 100.0, 120.0, 140.0, 160.0, 180.0, 200.0]:
            for k in [150, 300]:
            # for k in [300]:
                j = k // 2
                print(i, j, k)
                if prog == 9:
                    for n_viewpoints in range(12, 63, 3):
                    # for n_viewpoints in range(12, 47, 3):
                    # for n_viewpoints in range(48, 72, 3):
                    # for degree in [120.0, 130.0, 140.0, 150.0, 180.0, 200.0]:
                    # for degree in [90.0, 100.0, 110.0]:
                    # for degree in [80.0, 100.0, 120.0, 140.0, 180.0, 200.0]:
                    # n_viewpoints = 15
                        projection_dir = Path(f'../data/mhd_projections_virtual_noise2/{i}_{j}_{k}_{degree}degree/{n_viewpoints}vps')
                        projection_dir.mkdir(parents=True, exist_ok=True)

                        shape = (i, j, k)

                        noise_t = 'Gaussian'
                        # noise_t = 'no noise'

                        grid_params = GridParameters(
                            i, j, k, 6.5, 2.0
                        )  # n_rad_bins, n_theta_bins, n_phi_bins, r_max, r_min
                        instr_params = InstrParameters(
                            6.3, 2.1, 512, 23.8, 1, 1e10
                        )  # instr_r_max, instr_r_min, image_size, pixel_size, binning_factor, scale_factor
                        viewpoints = make_config(Path("../data/lasco_c2_2023"))
                        row_idx, col_idx, val, y_mapping = build_A_matrix_with_virtual_viewpoints(
                            viewpoints, n_viewpoints, degree, grid_params, instr_params
                            )
                        # print('build A return to python')

                        A = scipy.sparse.csr_array(
                            (val, col_idx, row_idx),
                            shape=(len(row_idx) - 1, i * j * k),
                        )

                        # print(val, col_idx, row_idx)
                        # print(np.count_nonzero(col_idx[col_idx < 0]), np.count_nonzero(col_idx[col_idx >= i * j * k]))
                        # print(np.argmax(col_idx < 0), np.argmax(col_idx >= i * j * k))
                        # print(np.max(col_idx), np.min(col_idx))

                        x_sim = np.fromfile(Path("mhd_resolutions") / f"x_corhel_{i}_{j}_{k}", dtype=np.float32)

                        y_zip = A @ x_sim

                        n_rows = 0
                        for row in range(n_viewpoints):
                            pb_vector = []
                            for col in range(instr_params.y_size):
                                row_n = row * instr_params.y_size + col
                                row_size = y_mapping[row_n + 1] - y_mapping[row_n]
                                if row_size <= 0:
                                    pb_vector.append(0)
                                else:
                                    y = y_zip[n_rows]
                                    n_rows += 1
                                    y /= instr_params.scale_factor
                                    pb_vector.append(y)
                            # noise: std / 500
                            # noise2: std / 250
                            # print(len(pb_vector), instr_params.image_size * instr_params.image_size)
                            assert(len(pb_vector) == instr_params.image_size * instr_params.image_size)
                            pb_vector = np.array(pb_vector, dtype=np.float64)
                            if noise_t == 'Gaussian':
                                # noise = np.random.normal(0, (np.max(pb_vector) - np.min(pb_vector)) / 500, pb_vector.shape)
                                # noise = np.random.normal(0, np.std(pb_vector) / 500, pb_vector.shape)
                                noise = np.random.normal(0, np.std(pb_vector) / 250, pb_vector.shape)
                                pb_vector += noise
                                pb_vector = np.maximum(np.zeros(pb_vector.shape), pb_vector)
                                # pb_vector.tofile(Path(projection_dir) / Path('Gaussian') / Path(filename))
                                pb_vector.tofile(projection_dir / Path(f'{row}_in_{n_viewpoints}vps'))
                            else:
                                pb_vector.tofile(projection_dir / Path(f'{row}_in_{n_viewpoints}vps'))
                elif prog == 10:
                    with open(f'error_vs_virtual_viewpoints_{i}_{j}_{k}_{degree}degree_noise2.txt', 'w') as f:
                        # for degree in [120.0, 130.0, 140.0, 150.0, 180.0, 200.0]:
                        # for degree in [90.0, 100.0, 110.0]:
                        for n_viewpoints in range(12, 63, 3):
                        # for n_viewpoints in range(12, 47, 3):
                        # for n_viewpoints in range(48, 63, 3):
                        # n_viewpoints = 15
                            shape = (i, j, k)

                            t0 = time.time()
                            grid_params = GridParameters(
                                i, j, k, 6.5, 2.0
                            )  # n_rad_bins, n_theta_bins, n_phi_bins, r_max, r_min
                            instr_params = InstrParameters(
                                6.3, 2.1, 512, 23.8, 1, 1e10
                            )  # instr_r_max, instr_r_min, image_size, pixel_size, binning_factor, scale_factor
                            viewpoints = make_config(Path("../data/lasco_c2_2023"))
                            # virtual_viewpoints = make_projection(f'../data/mhd_projections_virtual/{i}_{j}_{k}/{n_viewpoints}vps')
                            # virtual_viewpoints = make_projection(f'../data/mhd_projections_virtual/{30}_{j}_{k}/{n_viewpoints}vps')'
                            virtual_viewpoints = make_projection(f'../data/mhd_projections_virtual_noise2/{30}_{150}_{300}_{degree}degree/{n_viewpoints}vps')
                            print('viewpoints: ', viewpoints)
                            print('virtual_viewpoints: ', virtual_viewpoints)
                            # row_idx, col_idx, val, y = build_A_matrix_with_projection(
                            #     viewpoints, f'../data/mhd_projections_virtual/{30}_{j}_{k}/{n_viewpoints}vps/' , grid_params, instr_params, virtual_viewpoints, n_viewpoints
                            #     )
                            row_idx, col_idx, val, y = build_A_matrix_with_projection(
                                viewpoints, f'../data/mhd_projections_virtual_noise2/{30}_{150}_{300}_{degree}degree/{n_viewpoints}vps/' , grid_params, instr_params, virtual_viewpoints, n_viewpoints, degree
                                )
                            
                            D = hlaplac(i, j, k)
                            
                            x = reconstruct(grid_params, row_idx, col_idx, val, y, D.indptr,
                                        D.indices,
                                        D.data.astype(np.float32),
                                        1e-6)
                            
                            t1 = time.time()

                            # print(val, col_idx, row_idx)
                            # print(np.count_nonzero(col_idx[col_idx < 0]), np.count_nonzero(col_idx[col_idx >= i * j * k]))
                            # print(np.argmax(col_idx < 0), np.argmax(col_idx >= i * j * k))
                            # print(np.max(col_idx), np.min(col_idx))

                            x_sim = np.fromfile(Path("mhd_resolutions") / f"x_corhel_{i}_{j}_{k}", dtype=np.float32)

                            x_sim = np.reshape(x_sim, shape, order='F')[1:-max((i // 10 + 1), 1)].flatten().copy()
                            x = np.reshape(x, shape, order='F')[1:-max((i // 10 + 1), 1)].flatten().copy()
                            error = relative_error(x_sim, x)
                            # error = RMS_std_error(x_sim, x, shape)
                            # error = RMS_avg_error(x_sim, x, shape)
                            print(f'{i} {j} {k} {error} {t1 - t0}s {n_viewpoints}\n')
                            f.write(f'{i} {j} {k} {error} {t1 - t0}s {n_viewpoints}\n')
    elif prog == 11:
        i = 30
        k = 150
        j = k // 2
        print(i, j, k)
        n_viewpoints = 12
        degree = 80.0

        projection_dir = Path(f'../data/mhd_projections_virtual/{i}_{j}_{k}')
        projection_dir.mkdir(parents=True, exist_ok=True)

        shape = (i, j, k)

        t0 = time.time()
        grid_params = GridParameters(
            i, j, k, 6.5, 2.0
        )  # n_rad_bins, n_theta_bins, n_phi_bins, r_max, r_min
        instr_params = InstrParameters(
            6.3, 2.1, 512, 23.8, 1, 1e10
        )  # instr_r_max, instr_r_min, image_size, pixel_size, binning_factor, scale_factor
        viewpoints = make_config(Path("../data/lasco_c2_2023"))
        virtual_viewpoints = make_projection(f'../data/mhd_projections_virtual/{30}_{150}_{300}_{degree}degree/{n_viewpoints}vps')
        print('viewpoints: ', viewpoints)
        print('virtual_viewpoints: ', virtual_viewpoints)
        row_idx, col_idx, val, y = build_A_matrix_with_projection(
            viewpoints, f'../data/mhd_projections_virtual/{30}_{150}_{300}_{degree}degree/{n_viewpoints}vps/' , grid_params, instr_params, virtual_viewpoints, n_viewpoints, degree
            )
        
        D = hlaplac(i, j, k)
        
        x = reconstruct(grid_params, row_idx, col_idx, val, y, D.indptr,
                    D.indices,
                    D.data.astype(np.float32),
                    1e-6)
        
        t1 = time.time()

        # print(val, col_idx, row_idx)
        # print(np.count_nonzero(col_idx[col_idx < 0]), np.count_nonzero(col_idx[col_idx >= i * j * k]))
        # print(np.argmax(col_idx < 0), np.argmax(col_idx >= i * j * k))
        # print(np.max(col_idx), np.min(col_idx))

        x_sim = np.fromfile(Path("mhd_resolutions") / f"x_corhel_{i}_{j}_{k}", dtype=np.float32)
        errors = np.abs(x_sim - x) / x_sim
        errors.tofile(f'error_vs_virtual_viewpoints_{i}_{j}_{k}_{degree}degree_{n_viewpoints}vps_errors')
        x_sim.tofile(f'x_error_vs_virtual_viewpoints_{i}_{j}_{k}_{degree}degree_{n_viewpoints}vps')

        x_sim = np.reshape(x_sim, shape, order='F')[1:-max((i // 10 + 1), 1)].flatten().copy()
        x = np.reshape(x, shape, order='F')[1:-max((i // 10 + 1), 1)].flatten().copy()
        error = relative_error(x_sim, x)

        print(f'{i} {j} {k} {error} {t1 - t0}s {n_viewpoints}\n')

    elif prog == 12:
        filenames = make_config(Path("../data/lasco_c2_2023"))
        validate = [filenames[i] for i in range(0, len(filenames), 3)]
        train = list(set(filenames) - set(validate))
        print(train, validate)
        with open("optimal_lambda_vs_resolution_vs.txt", 'w') as f_opt:
            for i in n_rad_bins:
                for k in n_phi_bins:
                    j = k // 2
                    with open(f"lambda_vs_error_{i}_{j}_{k}_vs.txt", 'w') as f:
                        print(i, j, k)

                        shape = (i, j, k)
                        grid_params = GridParameters(
                            i, j, k, 6.5, 2.0
                        )  # n_rad_bins, n_theta_bins, n_phi_bins, r_max, r_min
                        # instr_params = InstrParameters(
                        #     6.3, 2.1, 512, 23.8, 1, 1e10
                        # )  # instr_r_max, instr_r_min, image_size, pixel_size, binning_factor, scale_factor
                        instr_params = InstrParameters(
                            6.3, 2.1, 512, 23.8, 1, 1e10
                        )  # instr_r_max, instr_r_min, image_size, pixel_size, binning_factor, scale_factor


                        # filenames = make_config(Path("../data/lasco_c2_2023"))
                        row_idx, col_idx, val, y = build_A_matrix(
                            train, grid_params, instr_params
                            )
                        
                        D = hlaplac(i, j, k)

                        row_idx2, col_idx2, val2, y2 = build_A_matrix(
                            validate, grid_params, instr_params
                            )
                        
                        y2_norm = np.linalg.norm(y2)
                        
                        A = scipy.sparse.csr_array(
                            (val2, col_idx2, row_idx2),
                            shape=(len(row_idx2) - 1, i * j * k),
                        )
                        
                        min_error = 10000000.0
                        best_lambda = -1.0
                        for l in [5e-6 + i * 5e-7 for i in range(100)]:
                            x = reconstruct(grid_params, row_idx, col_idx, val, y, D.indptr,
                                    D.indices,
                                    D.data.astype(np.float32),
                                    l)
                            y_proj = A @ x
                            # error = relative_error(y_proj, y2)
                            error = 1 - np.dot(y_proj, y2) / np.linalg.norm(y_proj) / y2_norm
                            if error < min_error:
                                min_error = error
                                best_lambda = l
                            f.write(f'{l} {error}\n')
                            print(f'{i} {j} {k} {l} {error}')
                        f_opt.write(f'{i} {j} {k} {best_lambda} {min_error}\n')
                        print(f'{i} {j} {k} {best_lambda} {min_error}')
    elif prog == 13:
        for i in n_rad_bins:
            for k in n_phi_bins[:-1]:
                j = k // 2
                print(i, j, k)
                # read_mhd('../data/mhd_2023/', i, j, k)
                # interp(i, 150, 300, np.reshape(np.fromfile(f'mhd_resolutions/x_corhel_{i}_{150}_{300}', dtype=np.float32), (300, 150, i)), i, j, k)
                interp(i, 150, 300, np.fromfile(f'mhd_resolutions/x_corhel_{i}_{150}_{300}', dtype=np.float32), i, j, k)


if __name__ == '__main__':
    sys.exit(main())