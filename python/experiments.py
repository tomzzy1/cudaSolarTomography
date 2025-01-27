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
import numpy as np
import os
from pathlib import Path
import sys
import time
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

def synthesize_images(filenames, instr_params, y_mapping, y_zip, noise_t, noise_level, projection_dir):
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
        if noise_t == 'Gaussian':
            noise = np.random.normal(0, np.std(pb_vector) / noise_level, pb_vector.shape)
            pb_vector = np.maximum(np.zeros(pb_vector.shape), pb_vector + noise)
            pb_vector.tofile(Path(projection_dir) / Path(filename))
        else:
            pb_vector.tofile(projection_dir / Path(filename))


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
        '''
        Prog 0: Generate interpolated MHD volumes
        The read_mhd function reads the elctron density at the Path
        The function will read all files in this path with extension .h5
        Note that the files downloaded from MHD websites have the format of .hdf, 
        It needs to be convereted to the .h5 format
        The converter can be found in https://www.hdfeos.org/software/h4toh5.php
        The read_mhd will interpolate the original volumes to resolution (i, j, k)
        and store the result in mhd_resolutions/x_corhel_{i}_{j}_{k}
        should be used with interp = 0
        '''
        for i in n_rad_bins:
            for k in [300]: # or for k in n_phi_bins
                j = k // 2
                print(i, j, k)
                read_mhd('../data/mhd/', i, j, k)

    elif prog == 1:
        '''
        Prog 1: Interpolate from the interpolated MHD volumes as the ground truth
        should be used with interp = 1
        '''
        for i in n_rad_bins:
            for k in n_phi_bins[:-1]:
                j = k // 2
                print(i, j, k)
                interp(i, 150, 300, np.fromfile(f'mhd_resolutions/x_corhel_{i}_{150}_{300}', dtype=np.float32), i, j, k)
    elif prog == 2:
        '''
        Prog 2: Synthesize images
        '''
        for i in n_rad_bins:
            for k in [300]: # or for k in n_phi_bins
                j = k // 2
                print(i, j, k)

                # noise_t = 'Gaussian'
                noise_t = 'no noise'
                noise_level = 100

                # set the directory for storing synthetic images here
                if noise_t == 'no noise':
                    projection_dir = Path(f'../data/synthetic_images_no_noise/{i}_{j}_{k}')
                elif noise_t == 'Gaussian': # add Gaussian Noise to images
                    projection_dir = Path(f'../data/synthetic_images_Gaussian_{noise_level}/{i}_{j}_{k}')
                projection_dir.mkdir(parents=True, exist_ok=True)

                shape = (i, j, k)

                grid_params = GridParameters(
                    i, j, k, 6.5, 2.0
                )  # n_rad_bins, n_theta_bins, n_phi_bins, r_max, r_min

                instr_params = InstrParameters(
                    6.3, 2.1, 512, 23.8, 1, 1e10
                )  # instr_r_max, instr_r_min, image_size, pixel_size, binning_factor, scale_factor

                filenames = make_config(Path("../data/lasco_c2_2023_partial"))
                # filenames = make_config(Path("../data/lasco_c2_2023"))
                row_ptr, col_idx, val, y_mapping = build_A_matrix_with_mapping_to_y(
                    filenames, grid_params, instr_params
                    ) # y_mapping provides the mapping from the y vector to the image pixels
                
                print(row_ptr)
                assert(not np.any(np.isnan(row_ptr)))
                assert(not np.any(np.isnan(col_idx)))
                assert(not np.any(np.isnan(val)))

                A = scipy.sparse.csr_array(
                    (val, col_idx, row_ptr),
                    shape=(len(row_ptr) - 1, i * j * k),
                )

                # print(A)
                x_sim = np.fromfile(Path("mhd_resolutions") / f"x_corhel_{i}_{j}_{k}", dtype=np.float32)
                assert(not np.any(np.isnan(x_sim)))

                y_zip = A @ x_sim

                synthesize_images(filenames, instr_params, y_mapping, y_zip, noise_t, noise_level, projection_dir)

    elif prog == 3:
        '''
        Prog 3: Experiments of resolutions
        '''
        # noise_t = 'Gaussian'
        noise_t = 'no noise'
        noise_level = 100
        '''
        interp = 0: use the volumes interpolated again from the original volumes as the ground truth
        interp = 1: use the volumes interpolated from the interpolated volumes as the ground truth
        '''
        interp_t = 0
        for l in [1e-6]:
            with open(f'error_vs_resolution_lambda{l}_interp{interp_t}.txt', 'w') as f:
            # with open(f'error_vs_resolution_double_lambda{l}_interp{interp_t}.txt', 'w') as f:
                for i in n_rad_bins:
                    for k in n_phi_bins:
                        j = k // 2
                        shape = (i, j, k)
                        print(shape)

                        # This directory should match the directory in prog 2
                        if noise_t == 'no noise':
                            # projection_dir = Path(f'../data/synthetic_images_no_noise/{i}_{j}_{k}')
                            projection_dir = Path(f'../data/synthetic_images_no_noise/{i}_{150}_{300}') # use highest angular resolutions for images
                        elif noise_t == 'Gaussian': # add Gaussian Noise to images
                            projection_dir = Path(f'../data/synthetic_images_Gaussian_{noise_level}/{i}_{150}_{300}')

                        t0 = time.time()
                        grid_params = GridParameters(
                            i, j, k, 6.5, 2.0
                        )  # n_rad_bins, n_polar_bins, n_phi_bins, r_max, r_min
                        instr_params = InstrParameters(
                            6.3, 2.1, 512, 23.8, 1, 1e10
                        )  # instr_r_max, instr_r_min, image_size, pixel_size, binning_factor, scale_factor


                        D = hlaplac(i, j, k) # generate regularizaiton matrix
                        row_ptr, col_idx, val, y = build_A_matrix_with_projection(
                            make_config(Path("../data/lasco_c2_2023_partial")), str(projection_dir) + '/', grid_params, instr_params)
                                                
                        x = reconstruct(grid_params, row_ptr, col_idx, val, y, D.indptr,
                            D.indices,
                            D.data.astype(np.float32),
                            l)
                        
                        # x0 = np.zeros(i * j * k)
                        # res = scipy.optimize.minimize(tikhonov, x0, (y, A, D, 1e-6), jac=tikhonov_grad, bounds=None, method='L-BFGS-B')
                        # x = res.x
                        assert(not np.any(np.isnan(x)))
                        
                        t1 = time.time()
                        
                        # store reconstructed volumes
                        np.array(x).tofile(f"x_{i}_{j}_{k}_{l}")

                        if interp_t == 0:
                            x_sim = np.fromfile(Path("mhd_resolutions") / f"x_corhel_{i}_{j}_{k}", dtype=np.float32)
                        elif interp_t == 1:
                            if k != 300:
                                x_sim = np.fromfile(Path(f"mhd_resolutions_{i}_{150}_{300}") / f"x_corhel_{i}_{j}_{k}", dtype=np.float32)
                            else:
                                x_sim = np.fromfile(Path("mhd_resolutions") / f"x_corhel_{i}_{j}_{k}", dtype=np.float32)
                        
                        assert(not np.any(np.isnan(x_sim)))
                        assert(np.count_nonzero(x_sim) == len(x_sim))

                        # remove the first and last several radial slices
                        x_sim = np.reshape(x_sim, shape, order='F')[1:-max((i // 10 + 1), 1)].flatten().copy()
                        x = np.reshape(x, shape, order='F')[1:-max((i // 10 + 1), 1)].flatten().copy()
                        error = relative_error(x_sim, x)
                        # error = RMS_std_error(x_sim, x, (i - (max((i // 10 + 1), 1)) - 1, j, k))
                        # error = RMS_avg_error(x_sim, x, shape)
                        print(f'{i} {j} {k} {error} {t1 - t0}s {len(val)}\n')
                        f.write(f'{i} {j} {k} {error} {t1 - t0}s {len(val)}\n')
                        # print(f'{i} {j} {k} {error} {t1 - t0}s {0}\n')
                        # f.write(f'{i} {j} {k} {error} {t1 - t0}s {0}\n')
    elif prog == 4 or prog == 5:

        '''
        Prog 4:
        Generate synthetic images with virtual viewpoints (the view angle is determined by the 
        first and last real viewpoints)
        This program only store the synthesized images, not included the FITS header, the headers 
        still need to be read from the original FITS files.

        Prog 5:
        Reconstruct with synthetic images generated by Prog 4
        '''
        # noise_t = 'Gaussian'
        noise_t = 'no noise'
        noise_level = 100
        if prog == 4:
            for i in [30]:
                for k in [300]: # for k in n_phi_bins:
                    j = k // 2
                    print(i, j, k)
                    # number of virtual viewpoints desired
                    for n_viewpoints in range(12, 63, 3):
                        if noise_t == 'no noise':
                            # projection_dir = Path(f'../data/synthetic_images_no_noise/{i}_{j}_{k}')
                            projection_dir = Path(f'../data/synthetic_images_virtual_no_noise/{i}_{j}_{k}/{n_viewpoints}vps') # use highest angular resolutions for images
                        elif noise_t == 'Gaussian': # add Gaussian Noise to images
                            projection_dir = Path(f'../data/synthetic_images_virtual_Gaussian_{noise_level}/{i}_{j}_{k}/{n_viewpoints}vps')
                        projection_dir.mkdir(parents=True, exist_ok=True)

                        shape = (i, j, k)

                        grid_params = GridParameters(
                            i, j, k, 6.5, 2.0
                        )  # n_rad_bins, n_theta_bins, n_phi_bins, r_max, r_min
                        instr_params = InstrParameters(
                            6.3, 2.1, 512, 23.8, 1, 1e10
                        )  # instr_r_max, instr_r_min, image_size, pixel_size, binning_factor, scale_factor

                        '''
                        The FITS files should be downloaded by fetch_lasco_c2.py 
                        or the FITS must be named in the format "yyyy-mm-dd-hh.fts"
                        
                        The fits_dir in build_params.hpp should also point to the folder of viewpoints
                        '''
                        viewpoints = make_config(Path("../data/lasco_c2_2023"))
                        row_ptr, col_idx, val, y_mapping = build_A_matrix_with_virtual_viewpoints(
                            viewpoints, n_viewpoints, grid_params, instr_params
                            )

                        A = scipy.sparse.csr_array(
                            (val, col_idx, row_ptr),
                            shape=(len(row_ptr) - 1, i * j * k),
                        )

                        x_sim = np.fromfile(Path("mhd_resolutions") / f"x_corhel_{i}_{j}_{k}", dtype=np.float32)

                        y_zip = A @ x_sim

                        filenames = [f'{row}_in_{n_viewpoints}vps' for row in range(n_viewpoints)]

                        synthesize_images(filenames, instr_params, y_mapping, y_zip, noise_t, noise_level, projection_dir)
        elif prog == 5:
            interp_t = 0
            for i in [30]:
                for k in n_phi_bins:
                    j = k // 2
                    for l in [1e-6]:
                        with open(f'error_vs_virtual_viewpoints_{i}_{j}_{k}_lambda{l}_interp{interp_t}.txt', 'w') as f:
                            for n_viewpoints in range(12, 63, 3):
                                if noise_t == 'no noise':
                                    # projection_dir = Path(f'../data/synthetic_images_no_noise/{i}_{150}_{300}')
                                    projection_dir = Path(f'../data/synthetic_images_virtual_no_noise/{i}_{150}_{300}/{n_viewpoints}vps') # use highest angular resolutions for images
                                elif noise_t == 'Gaussian': # add Gaussian Noise to images
                                    projection_dir = Path(f'../data/synthetic_images_virtual_Gaussian_{noise_level}/{i}_{150}_{300}/{n_viewpoints}vps')
                                # projection_dir.mkdir(parents=True, exist_ok=True)

                                shape = (i, j, k)

                                t0 = time.time()
                                grid_params = GridParameters(
                                    i, j, k, 6.5, 2.0
                                )  # n_rad_bins, n_theta_bins, n_phi_bins, r_max, r_min

                                instr_params = InstrParameters(
                                    6.3, 2.1, 512, 23.8, 1, 1e10
                                )  # instr_r_max, instr_r_min, image_size, pixel_size, binning_factor, scale_factor

                                viewpoints = make_config(Path("../data/lasco_c2_2023")) # real viewpoints for FITS headers
                                virtual_viewpoints = make_projection(projection_dir) # synthesied viewpoints

                                print('viewpoints: ', viewpoints)
                                print('virtual_viewpoints: ', virtual_viewpoints)
                                # row_ptr, col_idx, val, y = build_A_matrix_with_projection(
                                #     viewpoints, f'../data/mhd_projections_virtual/{30}_{j}_{k}/{n_viewpoints}vps/' , grid_params, instr_params, virtual_viewpoints, n_viewpoints
                                #     )
                                row_ptr, col_idx, val, y = build_A_matrix_with_projection(
                                    viewpoints, str(projection_dir) + '/' , grid_params, instr_params, virtual_viewpoints, n_viewpoints
                                    )
                                
                                D = hlaplac(i, j, k)
                                
                                x = reconstruct(grid_params, row_ptr, col_idx, val, y, D.indptr,
                                            D.indices,
                                            D.data.astype(np.float32),
                                            l)
                                                            
                                t1 = time.time()
                                np.array(x).tofile(f"x_{i}_{j}_{k}_{n_viewpoints}viewpoints_lambda{l}")

                                if interp_t == 0:
                                    x_sim = np.fromfile(Path("mhd_resolutions") / f"x_corhel_{i}_{j}_{k}", dtype=np.float32)
                                elif interp_t == 1:
                                    if k != 300:
                                        x_sim = np.fromfile(Path(f"mhd_resolutions_{i}_{150}_{300}") / f"x_corhel_{i}_{j}_{k}", dtype=np.float32)
                                    else:
                                        x_sim = np.fromfile(Path("mhd_resolutions") / f"x_corhel_{i}_{j}_{k}", dtype=np.float32)

                                x_sim = np.reshape(x_sim, shape, order='F')[1:-max((i // 10 + 1), 1)].flatten().copy()
                                x = np.reshape(x, shape, order='F')[1:-max((i // 10 + 1), 1)].flatten().copy()
                                error = relative_error(x_sim, x)
                                # error = RMS_std_error(x_sim, x, shape)
                                # error = RMS_avg_error(x_sim, x, shape)
                                print(f'{i} {j} {k} {error} {t1 - t0}s {n_viewpoints}\n')
                                f.write(f'{i} {j} {k} {error} {t1 - t0}s {n_viewpoints}\n')
            
    elif prog == 6 or prog == 7:
        '''
        Prog 6:
        Generate synthetic images with virtual viewpoints (the view angle can be arbitrary)
        This program only store the synthesized images, not included the FITS header, the headers 
        still need to be read from the original FITS files.

        Prog 7:
        Reconstruct with synthetic images generated by Prog 6
        '''
        i = 30
        # noise_t = 'Gaussian'
        noise_t = 'no noise'
        noise_level = 100
        interp_t = 0
        if prog == 6:
            for degree in [80.0, 100.0, 120.0, 140.0, 160.0, 180.0, 200.0]:
                for k in [300]:
                    j = k // 2
                    print(i, j, k)
                    for n_viewpoints in range(12, 63, 3):

                        if noise_t == 'no noise':
                            # projection_dir = Path(f'../data/synthetic_images_no_noise/{i}_{j}_{k}')
                            projection_dir = Path(f'../data/synthetic_images_virtual_no_noise/{i}_{j}_{k}_{degree}degree/{n_viewpoints}vps')
                        elif noise_t == 'Gaussian': # add Gaussian Noise to images
                            projection_dir = Path(f'../data/synthetic_images_virtual_Gaussian_{noise_level}/{i}_{j}_{k}_{degree}degree/{n_viewpoints}vps')
                        projection_dir.mkdir(parents=True, exist_ok=True)

                        shape = (i, j, k)

                        grid_params = GridParameters(
                            i, j, k, 6.5, 2.0
                        )  # n_rad_bins, n_theta_bins, n_phi_bins, r_max, r_min
                        instr_params = InstrParameters(
                            6.3, 2.1, 512, 23.8, 1, 1e10
                        )  # instr_r_max, instr_r_min, image_size, pixel_size, binning_factor, scale_factor

                        viewpoints = make_config(Path("../data/lasco_c2_2023"))
                        row_ptr, col_idx, val, y_mapping = build_A_matrix_with_virtual_viewpoints(
                            viewpoints, n_viewpoints, degree, grid_params, instr_params
                            )

                        A = scipy.sparse.csr_array(
                            (val, col_idx, row_ptr),
                            shape=(len(row_ptr) - 1, i * j * k),
                        )

                        x_sim = np.fromfile(Path("mhd_resolutions") / f"x_corhel_{i}_{j}_{k}", dtype=np.float32)

                        y_zip = A @ x_sim

                        filenames = [f'{row}_in_{n_viewpoints}vps' for row in range(n_viewpoints)]

                        synthesize_images(filenames, instr_params, y_mapping, y_zip, noise_t, noise_level, projection_dir)
            
        elif prog == 7:
            for degree in [80.0, 100.0, 120.0, 140.0, 160.0, 180.0, 200.0]:
                for k in [150, 300]:
                    j = k // 2
                    print(i, j, k)
                    for l in [1e-6]:
                        with open(f'error_vs_virtual_viewpoints_{i}_{j}_{k}_{degree}degree_lambda{l}_interp{interp_t}.txt', 'w') as f:
                            for n_viewpoints in range(12, 63, 3):

                                if noise_t == 'no noise':
                                    # projection_dir = Path(f'../data/synthetic_images_no_noise/{i}_{150}_{300}')
                                    projection_dir = Path(f'../data/synthetic_images_virtual_no_noise/{i}_{150}_{300}_{degree}degree/{n_viewpoints}vps')
                                elif noise_t == 'Gaussian': # add Gaussian Noise to images
                                    projection_dir = Path(f'../data/synthetic_images_virtual_Gaussian_{noise_level}/{i}_{150}_{300}_{degree}degree/{n_viewpoints}vps')

                                shape = (i, j, k)

                                t0 = time.time()
                                grid_params = GridParameters(
                                    i, j, k, 6.5, 2.0
                                )  # n_rad_bins, n_theta_bins, n_phi_bins, r_max, r_min
                                instr_params = InstrParameters(
                                    6.3, 2.1, 512, 23.8, 1, 1e10
                                )  # instr_r_max, instr_r_min, image_size, pixel_size, binning_factor, scale_factor

                                viewpoints = make_config(Path("../data/lasco_c2_2023"))
                                virtual_viewpoints = make_projection(projection_dir)
                                print('viewpoints: ', viewpoints)
                                print('virtual_viewpoints: ', virtual_viewpoints)

                                row_ptr, col_idx, val, y = build_A_matrix_with_projection(
                                    viewpoints, str(projection_dir) + '/' , grid_params, instr_params, virtual_viewpoints, n_viewpoints, degree
                                    )
                                
                                D = hlaplac(i, j, k)
                                
                                x = reconstruct(grid_params, row_ptr, col_idx, val, y, D.indptr,
                                            D.indices,
                                            D.data.astype(np.float32),
                                            l)
                                
                                t1 = time.time()

                                if interp_t == 0:
                                    x_sim = np.fromfile(Path("mhd_resolutions") / f"x_corhel_{i}_{j}_{k}", dtype=np.float32)
                                elif interp_t == 1:
                                    if k != 300:
                                        x_sim = np.fromfile(Path(f"mhd_resolutions_{i}_{150}_{300}") / f"x_corhel_{i}_{j}_{k}", dtype=np.float32)
                                    else:
                                        x_sim = np.fromfile(Path("mhd_resolutions") / f"x_corhel_{i}_{j}_{k}", dtype=np.float32)

                                x_sim = np.reshape(x_sim, shape, order='F')[1:-max((i // 10 + 1), 1)].flatten().copy()
                                x = np.reshape(x, shape, order='F')[1:-max((i // 10 + 1), 1)].flatten().copy()
                                error = relative_error(x_sim, x)
                                # error = RMS_std_error(x_sim, x, shape)
                                # error = RMS_avg_error(x_sim, x, shape)
                                print(f'{i} {j} {k} {error} {t1 - t0}s {n_viewpoints}\n')
                                f.write(f'{i} {j} {k} {error} {t1 - t0}s {n_viewpoints}\n')

if __name__ == '__main__':
    sys.exit(main())