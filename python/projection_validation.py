import scipy.ndimage
from py_cuda_solartomography import GridParameters
from py_cuda_solartomography import InstrParameters
from py_cuda_solartomography import build_and_reconstruct_with_projection
from py_cuda_solartomography import get_simulation_x
from py_cuda_solartomography import build_A_matrix_with_projection
from py_cuda_solartomography import reconstruct

from dgrad2 import hlaplac
from dgrad2 import r3

import scipy
import scipy.fft as fft
import numpy as np
import os
from pathlib import Path
import sys
import matplotlib.pyplot as plt

def make_config(path):
    file_list = list(filter(lambda f: ".fts" in f, os.listdir(path)))
    file_list.sort()
    return file_list

# projection_dir = '../data/projection/lasco_2023/'
# projection_dir = '../data/projection/lasco_2023/Gaussian/'
# # projection_dir = '../data/projection/lasco_2023/no_noise/'
# sim_dir = '../data/mhd/'
projection_dir = '../data/projection/lasco_2008/Gaussian/'
sim_dir = '../data/mhd_2008/'

# n_rad_bins = 30
# n_theta_bins = 75
# n_phi_bins = 150
n_rad_bins = 30
n_theta_bins = 101
n_phi_bins = 129

shape = (n_rad_bins, n_theta_bins, n_phi_bins)
# shape = (25, n_theta_bins, n_phi_bins)
shape_inv = (n_phi_bins, n_theta_bins, n_rad_bins)
# shape_inv = (n_phi_bins, n_theta_bins, 25)

grid_params = GridParameters(
    n_rad_bins, n_theta_bins, n_phi_bins, 6.5, 2.0
)  # n_rad_bins, n_polar_bins, n_phi_bins, r_max, r_min
instr_params = InstrParameters(
    6.3, 2.1, 512, 23.8, 1, 1e10
)  # instr_r_max, instr_r_min, image_size, pixel_size, binning_factor, scale_factor


D = hlaplac(n_rad_bins, n_theta_bins, n_phi_bins)
# D = r3(n_rad_bins, n_theta_bins, n_phi_bins)
x_sim = get_simulation_x(sim_dir, grid_params)
filenames = make_config('../data/lasco_c2_2023_partial/')

def error_func1(x1, x2):
    return 1 - (x1 @ x2) / np.linalg.norm(x1) / np.linalg.norm(x2)

def error_func2(x1, x2):
    return np.linalg.norm(x1 - x2)

def rmsdiqr_error(x1, x2):
    # Calculate Q1 and Q3
    Q1 = np.percentile(x1, 25)
    Q3 = np.percentile(x1, 75)
    
    # Calculate the IQR
    IQR = Q3 - Q1
        
    # Calculate the RMSD for the data within the IQR
    
    return np.linalg.norm(x2 - x1) / IQR

def RMS_func(x1, x2):
    x1_3d = np.reshape(x1, shape, order='F')
    x2_3d = np.reshape(x2, shape, order='F')
    error = 0.0
    for i in range(n_rad_bins):
        rad_error = np.linalg.norm(x1_3d[i] - x2_3d[i]) / np.average(x1_3d[i])
        error += rad_error
        # print(f'RMS error {rad_error} at rad {i}')
    return error / np.sqrt(n_theta_bins * n_phi_bins) / n_rad_bins

# RMS error 
def error_func3(x1, x2):
    x1_3d = np.reshape(x1, shape, order='F')
    x2_3d = np.reshape(x2, shape, order='F')
    error = 0.0
    # for i in range(n_rad_bins):
    for i in range(shape[0]):
        std = np.std(x1_3d[i])
        rad_error = np.linalg.norm(x1_3d[i] - x2_3d[i]) / np.sqrt(n_theta_bins * n_phi_bins) / std
        # print(i, rad_error)
        error += rad_error
        # print(i, std, error)
    return error

def compute_high_frequency_content(x):
    f_transform = np.fft.fft2(x)
    f_shift = np.fft.fftshift(f_transform)

    # Compute the magnitude spectrum
    magnitude_spectrum = np.abs(f_shift)

    # Define a threshold to count high-frequency components
    threshold = np.percentile(magnitude_spectrum, 80)
    high_freq_count = np.sum(magnitude_spectrum > threshold)
    
    return high_freq_count

def error_func4(x1, x2):
    x1_3d = np.reshape(x1, shape, order='F')
    x2_3d = np.reshape(x2, shape, order='F')
    error = 0.0
    for i in range(n_rad_bins):
        hf1 = compute_high_frequency_content(x1_3d[i])
        hf2 = compute_high_frequency_content(x2_3d[i])
        #print(hf1, hf2)
        error += hf1 - hf2
    return error

# smoothness of raidus slice
# def error_func5(x1, x2):
#     # print(np.max(x1), np.max(x2), np.min(x1), np.min(x2))
#     x1_3d = np.reshape(x1, shape, order='F')
#     x2_3d = np.reshape(x2, shape, order='F')

#     error = 0.0
#     for i in range(n_rad_bins):
#         # # var1 = scipy.ndimage.laplace(x1_3d[i]).var()
#         # # var2 = scipy.ndimage.laplace(x2_3d[i]).var()
#         # # print(var1, var2, (var2 - var1) / var1)
#         # # print(var1 / (np.max(x1) - np.min(x1)), var2 / (np.max(x2) - np.min(x2)))
#         # # print(var1 / (np.max(x1_3d[i]) - np.min(x2_3d[i])), var2 / (np.max(x1_3d[i]) - np.min(x2_3d[i])))
#         # # error += abs(var1 - var2)
#         # var = scipy.ndimage.laplace(x1_3d[i]).var()
#         # var_diff = scipy.ndimage.laplace(x1_3d[i] - x2_3d[i]).var()
#         # # print(var_diff / var)
#         # error += var_diff / var
#         std = scipy.ndimage.laplace(x1_3d[i]).std()
#         std_diff = scipy.ndimage.laplace(x1_3d[i] - x2_3d[i]).std()
#         error += std_diff / std
#     return error / n_rad_bins

# new smoonthness error function
def error_func5(x1, x2):
    # print(np.max(x1), np.max(x2), np.min(x1), np.min(x2))
    x1_3d = np.reshape(x1, shape, order='F')
    x2_3d = np.reshape(x2, shape, order='F')
    error = 0.0
    for i in range(x1_3d.shape[0]):
        sob1_theta = scipy.ndimage.sobel(x1_3d[i], axis=0, mode='wrap')
        sob1_phi = scipy.ndimage.sobel(x1_3d[i], axis=1, mode='wrap')
        sob2_theta = scipy.ndimage.sobel(x2_3d[i], axis=0, mode='wrap')
        sob2_phi = scipy.ndimage.sobel(x2_3d[i], axis=1, mode='wrap')
        sob1 = np.sqrt(sob1_theta ** 2 + sob1_phi ** 2)
        sob2 = np.sqrt(sob2_theta ** 2 + sob2_phi ** 2)
        #rad_error = (1 + np.corrcoef(sob1.flatten(), sob2.flatten())[0, 1]) / 2
        rad_error = 1 - np.corrcoef(sob1.flatten(), sob2.flatten())[0, 1]
        # rad_error = np.exp(np.corrcoef(sob1.flatten(), sob2.flatten())[0, 1])
        # print(f'Sobel error {rad_error} at rad {i}')
        error += rad_error
    return error / n_rad_bins
    # return n_rad_bins / error

# def error_func6(x1, x2):
#     x1_3d = np.reshape(x1, shape_inv, order='C')
#     x2_3d = np.reshape(x2, shape_inv, order='C')
#     error = 0.0
#     for i in range(n_phi_bins):
#         var = scipy.ndimage.laplace(x1_3d[i]).var()
#         var_diff = scipy.ndimage.laplace(x1_3d[i] - x2_3d[i]).var()
#         # print(var_diff / var)
#         error += var_diff / var
#     return error / n_phi_bins


# def error_func6(x1, x2):
#     x1_3d = np.reshape(x1, shape_inv, order='C')
#     x2_3d = np.reshape(x2, shape_inv, order='C')
#     error = 0.0
#     for i in range(n_phi_bins):
#         sob1_theta = scipy.ndimage.sobel(x1_3d[i], axis=0)
#         sob1_r = scipy.ndimage.sobel(x1_3d[i], axis=1)
#         sob2_theta = scipy.ndimage.sobel(x2_3d[i], axis=0)
#         sob2_r = scipy.ndimage.sobel(x2_3d[i], axis=1)
#         sob1 = np.sqrt(sob1_theta ** 2 + sob1_r ** 2)
#         sob2 = np.sqrt(sob2_theta ** 2 + sob2_r ** 2)
#         error += np.corrcoef(sob1.flatten(), sob2.flatten())[0, 1]
#     return error / n_phi_bins

def error_func6(x1, x2):
    x1_3d = np.reshape(x1, shape_inv, order='C')
    x2_3d = np.reshape(x2, shape_inv, order='C')
    print(x1_3d.shape)

    error = 0.0
    for i in range(n_phi_bins):
        for j in range(n_theta_bins):
            sob1 = np.absolute(scipy.ndimage.sobel(x1_3d[i][j])) 
            sob2 = np.absolute(scipy.ndimage.sobel(x2_3d[i][j])) 
            rad_error = 1 - np.corrcoef(sob1.flatten(), sob2.flatten())[0, 1]
            # rad_error = (1 + np.corrcoef(sob1.flatten(), sob2.flatten())[0, 1]) / 2
            # rad_error = np.exp(np.corrcoef(sob1.flatten(), sob2.flatten())[0, 1])
            # print(i, j, rad_error)
            error += rad_error
    # return n_phi_bins * n_theta_bins / error
    return error / (n_phi_bins * n_theta_bins)

def error_func7(x1, x2):
    # r = 5
    # mask = np.ones((n_theta_bins, n_phi_bins))
    # x_center = n_theta_bins // 2
    # y_center = n_phi_bins // 2
    # for i in range(n_theta_bins):
    #     for j in range(n_phi_bins):
    #         if abs(i - x_center) < r / 2 and abs(j - y_center) < r / 2:
    #             mask[i][j] = 0
    x2_min = np.min(np.abs(x2))
    x2[x2 < 0] = x2_min
    # x1 = np.log10(x1)
    # x2 = np.log10(x2)
    x1_3d = np.reshape(x1, shape, order='F')
    x2_3d = np.reshape(x2, shape, order='F')

    error = 0.0
    for i in range(n_rad_bins):
        rad_error = 0
        # x1_fft = fft.fft2(x1_3d[i])
        # x2_fft = fft.fft2(x2_3d[i])

        # x1_fft_shift = fft.fftshift(x1_fft) * mask
        # x2_fft_shift = fft.fftshift(x2_fft) * mask

        # x1_fft_ishift = fft.ifftshift(x1_fft_shift)
        # x2_fft_ishift = fft.ifftshift(x2_fft_shift)

        # x1_new = np.abs(fft.ifft2(x1_fft_ishift))
        # x2_new = np.abs(fft.ifft2(x2_fft_ishift))
        x1_new = x1_3d[i]
        x2_new = x2_3d[i]

        counts, bin_edges = np.histogram(x1_new, bins='auto')
        # print(counts, bin_edges)
        # print(len(counts), len(bin_edges))
        for j in range(n_theta_bins):
            for k in range(n_phi_bins):
                idx1 = np.digitize(x1_new[j][k], bin_edges, right=False) - 1
                if idx1 <= len(counts) * 0.15:
                    idx2 = np.digitize(x2_new[j][k], bin_edges, right=False) - 1
                    rad_error += max(idx2 - idx1, 0)
        # error += rad_error / (n_theta_bins * n_phi_bins) / len(counts)
        error += rad_error / len(counts)
    return error / grid_params.n_bins

# compare the difference of shape in azimuthal slices
def error_func8(x1, x2):
    x2_min = np.min(np.abs(x2))
    x2[x2 < 0] = x2_min
    # x1 = np.log10(x1)
    # x2 = np.log10(x2)
    x1_3d = np.reshape(x1, shape_inv, order='C')
    x2_3d = np.reshape(x2, shape_inv, order='C')

    error = 0.0
    for i in range(n_phi_bins):
        azi_error = 0
        x1_new = x1_3d[i]
        x2_new = x2_3d[i]

        counts, bin_edges = np.histogram(x1_new, bins='auto')
        # print(counts, bin_edges)
        # print(len(counts), len(bin_edges))
        for j in range(n_theta_bins):
            for k in range(n_rad_bins):
                idx1 = np.digitize(x1_new[j][k], bin_edges, right=False) - 1
                if idx1 <= len(counts) * 0.15:
                    idx2 = np.digitize(x2_new[j][k], bin_edges, right=False) - 1
                    azi_error += max(idx2 - idx1, 0)
        # error += azi_error / (n_theta_bins * n_rad_bins) / len(counts)
        error += azi_error / len(counts)
    return error / grid_params.n_bins

def error_func9(x1, x2):
    return error_func7(x1, x2) + 0.5 * error_func8(x1, x2) 

def error_func10(x1, x2):
    # print(np.max(x1), np.max(x2), np.min(x1), np.min(x2))
    x1_3d = np.reshape(x1, shape, order='F')
    x2_3d = np.reshape(x2, shape, order='F')

    error = 0.0
    for i in range(n_rad_bins):
        x1_lap = np.absolute(scipy.ndimage.laplace(x1_3d[i])) 
        # counts, bin_edges = np.histogram(x1_lap, bins='auto')
        # print(x1_lap)
        # print(counts, bin_edges)
        x1_lap_min = np.min(x1_lap)
        x1_lap_max = np.max(x1_lap)
        threshold = x1_lap_min + 0.05 * (x1_lap_max - x1_lap_min)
        x1_min = np.min(x1_3d[i])
        x1_max = np.max(x1_3d[i])
        x1_range = x1_max - x1_min
        # print(x1, x2)
        # counts, bin_edges = np.histogram(x1, bins='auto')
        # rad_error = 0
        for j in range(n_theta_bins):
            for k in range(n_phi_bins):
                if x1_lap[j][k] > threshold:
                    error += abs(x2_3d[i][j][k] - x1_3d[i][j][k]) / x1_range
                    
        # error += rad_error / len(counts)
    return error

# compare the difference of all pixels
# def error_func8(x1, x2):
#     x2_min = np.min(np.abs(x2))
#     x2[x2 < 0] = x2_min
#     x1 = np.log10(x1)
#     x2 = np.log10(x2)
#     x1_3d = np.reshape(x1, shape, order='F')
#     x2_3d = np.reshape(x2, shape, order='F')

#     error = 0.0
#     for i in range(n_rad_bins):
#         rad_error = 0
#         x1_new = x1_3d[i]
#         x2_new = x2_3d[i]

#         counts, bin_edges = np.histogram(x1_new, bins='auto')
#         for j in range(n_theta_bins):
#             for k in range(n_phi_bins):
#                 idx1 = np.digitize(x1_new[j][k], bin_edges, right=False) - 1
#                 idx2 = np.digitize(x2_new[j][k], bin_edges, right=False) - 1
#                 rad_error += abs(idx2 - idx1)
#         error += rad_error / (n_theta_bins * n_phi_bins) / len(counts)
#     return error

row_idx, col_idx, val, y = build_A_matrix_with_projection(
    make_config(Path("../data/lasco_c2_2023_partial")), projection_dir, grid_params, instr_params
)

def error_func11(x1, x2):
    error = 0
    for i in range(len(x1)):
       error += abs(x2[i] - x1[i]) / x1[i]
    # x1_3d = np.reshape(x1, shape, order='F')
    # x2_3d = np.reshape(x2, shape, order='F')
    # for i in range(n_rad_bins):
    #     rad_error = 0
    #     for j1, j2 in zip(x1_3d[i], x2_3d[i]):
    #         rad_error += abs(j1[i] - j2[i]) / j1[i]
    #     print(i, rad_error)
    #     error += rad_error
    # return error / grid_params.n_bins
    return error / len(x1)

def error_func12(x1, x2):
    # print(np.max(x1), np.max(x2), np.min(x1), np.min(x2))
    x1_3d = np.reshape(x1, shape, order='F')
    x2_3d = np.reshape(x2, shape, order='F')

    error = 0.0
    for i in range(n_rad_bins):
        x1_sob = np.absolute(scipy.ndimage.sobel(x1_3d[i])) 
        # counts, bin_edges = np.histogram(x2_sob, bins='auto')
        # print(x2_sob)
        # print(counts, bin_edges)
        x1_sob_min = np.min(x1_sob)
        x1_sob_max = np.max(x1_sob)
        threshold = x1_sob_min + 0.1 * (x1_sob_max - x1_sob_min)
        x1_min = np.min(x1_3d[i])
        x1_max = np.max(x1_3d[i])
        x1_range = x1_max - x1_min
        # print(x1, x2)
        # counts, bin_edges = np.histogram(x1, bins='auto')
        rad_error = 0
        for j in range(n_theta_bins):
            for k in range(n_phi_bins):
                if x1_sob[j][k] > threshold:
                    # rad_error += abs(x2_3d[i][j][k] - x1_3d[i][j][k]) / x1_range
                    rad_error += abs(x2_3d[i][j][k] - x1_3d[i][j][k]) / x1_3d[i][j][k]
        # error += rad_error / len(counts)
        # print(f"rad error {rad_error} at radius {i}")
        error += rad_error
    return error

# def error_func13(x1, x2):
#     # print(np.max(x1), np.max(x2), np.min(x1), np.min(x2))
#     x1_3d = np.reshape(x1, shape, order='F')
#     x2_3d = np.reshape(x2, shape, order='F')

#     error = 0.0
#     for i in range(n_rad_bins):
#         x1_sob = np.absolute(scipy.ndimage.sobel(x1_3d[i])) 
#         x1_sob_min = np.min(x1_sob)
#         x1_sob_max = np.max(x1_sob)
#         threshold = x1_sob_min + 0.1 * (x1_sob_max - x1_sob_min)
#         # print(x1, x2)
#         # counts, bin_edges = np.histogram(x1, bins='auto')
#         # rad_error = 0
#         x2_sob = np.absolute(scipy.ndimage.sobel(x2_3d[i])) 
#         x2_sob_min = np.min(x2_sob)
#         x2_sob_max = np.max(x2_sob)
#         threshold2 = x2_sob_min + 0.1 * (x2_sob_max - x2_sob_min)
#         for j in range(n_theta_bins):
#             for k in range(n_phi_bins):
#                 if (x1_sob[j][k] > threshold and x2_sob[j][k] < threshold2) or (x1_sob[j][k] < threshold and x2_sob[j][k] > threshold2) :
#                     error += 1
#         # error += rad_error / len(counts)
#     return error

def error_func13(x1, x2):
    x1_3d = np.reshape(x1, shape, order='F')
    x2_3d = np.reshape(x2, shape, order='F')

    error = 0.0
    for i in range(n_rad_bins):
        x1_sob = np.absolute(scipy.ndimage.sobel(x1_3d[i])) 
        # counts, bin_edges = np.histogram(x2_sob, bins='auto')
        # print(x2_sob)
        # print(counts, bin_edges)
        x1_sob_min = np.min(x1_sob)
        x1_sob_max = np.max(x1_sob)
        # threshold = x1_sob_min + 0.1 * (x1_sob_max - x1_sob_min)
        x1_sob_range = x1_sob_max - x1_sob_min
        # print(x1, x2)
        # counts, bin_edges = np.histogram(x1, bins='auto')
        rad_error = 0
        for j in range(n_theta_bins):
            for k in range(n_phi_bins):
                rad_error += abs(x2_3d[i][j][k] - x1_3d[i][j][k]) / x1_3d[i][j][k] * (x1_sob[j][k] - x1_sob_min) / x1_sob_range
        # error += rad_error / len(counts)
        # print(f"rad error {rad_error} at radius {i}")
        error += rad_error
    return error

# def error_func14(x1, x2):
#     x1_3d = np.reshape(x1, shape, order='F')
#     x2_3d = np.reshape(x2, shape, order='F')

#     error = 0.0
#     for i in range(n_rad_bins):
#         x1_sob = np.absolute(scipy.ndimage.sobel(x1_3d[i])) 
#         x2_sob = np.absolute(scipy.ndimage.sobel(x2_3d[i]))
#         rad_error = 0
#         for j in range(n_theta_bins):
#             for k in range(n_phi_bins):
#                 rad_error += abs(x2_3d[i][j][k] - x1_3d[i][j][k]) / x1_3d[i][j][k] # * abs(x2_sob[j][k] - x1_sob[j][k]) / x1_sob[j][k]
#         # error += rad_error / len(counts)
#         # print(f"rad error {rad_error} at radius {i}")
#         shape_factor = (1 - np.linalg.norm(x1_sob @ x2_sob.T) / (np.linalg.norm(x1_sob) * np.linalg.norm(x2_sob)))
#         print(f'shape_factor {shape_factor}')
#         error += rad_error * shape_factor
#     return error

def error_func14(x1, x2):
    x1_3d = np.reshape(x1, shape, order='F')
    x2_3d = np.reshape(x2, shape, order='F')

    error = 0.0
    for i in range(n_rad_bins):
        # x1_sob = np.absolute(scipy.ndimage.sobel(x1_3d[i])) 
        # x2_sob = np.absolute(scipy.ndimage.sobel(x2_3d[i]))
        rad_error = 0
        for j in range(n_theta_bins):
            for k in range(n_phi_bins):
                rad_error += abs(x2_3d[i][j][k] - x1_3d[i][j][k]) / x1_3d[i][j][k] # * abs(x2_sob[j][k] - x1_sob[j][k]) / x1_sob[j][k]
        # error += rad_error / len(counts)
        # print(f"rad error {rad_error} at radius {i}")
        # shape_factor = (1 - np.linalg.norm(x1_sob @ x2_sob.T) / (np.linalg.norm(x1_sob) * np.linalg.norm(x2_sob)))
        shape_factor = 1 / np.corrcoef(x1_3d[i].flatten(), x2_3d[i].flatten())[0, 1]
        # print(f'shape_factor {shape_factor}')
        error += rad_error * shape_factor
    return error

# difference is donimated by the raidus-axis
# def error_func14(x1, x2):
#     # print(np.max(x1), np.max(x2), np.min(x1), np.min(x2))
#     x1_3d = np.reshape(x1, shape, order='F')
#     x2_3d = np.reshape(x2, shape, order='F')

#     grad_x = scipy.ndimage.sobel(x1_3d, axis=0)
#     grad_y = scipy.ndimage.sobel(x1_3d, axis=1)
#     grad_z = scipy.ndimage.sobel(x1_3d, axis=2)
#     x1_sob = np.sqrt(grad_x ** 2 + grad_y ** 2 + grad_z ** 2)
#     x1_sob_min = np.min(x1_sob)
#     x1_sob_max = np.max(x1_sob)
#     threshold = x1_sob_min + 0.1 * (x1_sob_max - x1_sob_min)
#     error = 0.0
#     for i in range(n_rad_bins):
#         for j in range(n_theta_bins):
#             for k in range(n_phi_bins):
#                 if x1_sob[i][j][k] > threshold:
#                     error += abs(x2_3d[i][j][k] - x1_3d[i][j][k]) / x1_3d[i][j][k]
#         # error += rad_error / len(counts)
#     return error

prog = 11

if prog == 0:
    lambda_list = [1e-5, 5e-6, 2e-6, 1e-6, 5e-7, 1e-7]
    # lambda_list = [5e-5, 4e-5, 3e-5, 2e-5, 1e-5, 5e-6, 2e-6, 1e-6, 5e-7, 1e-7]
    # lambda_list = [8e-6, 7.2e-6, 2e-5]

    for l in lambda_list:
        # x = build_and_reconstruct_with_projection(filenames, projection_dir, grid_params, instr_params, D.indptr, D.indices,
        #         D.data.astype(np.float32), 2e-6)
        x = reconstruct(grid_params, row_idx, col_idx, val, y, D.indptr,
            D.indices,
            D.data.astype(np.float32),
            l)
        
        print(f'lambda = {l:.2e}')
        print(f'lambda = {l:.2e}', file=sys.stderr)

        print(f'ZDA of x_sim {np.sum(x_sim < 0)}, of x {np.sum(x < 0)}')

        error1 = error_func1(x_sim, x)
        print(f'error func1: cosine similarity {error1}')

        error2 = error_func2(x_sim, x)
        print(f'erorr func2: norm of diff {error2}')

        error3 = error_func3(x_sim, x)
        print(f'erorr func3: radial RMS error {error3}')

        # error4 = error_func4(x_sim, x)
        # print(f'erorr func4: high-frequency componenet {error4}')

        error5 = error_func5(x_sim, x)
        print(f'erorr func5: Radial Slice Smoothness difference {error5}')

        error6 = error_func6(x_sim, x)
        print(f'erorr func6: Radial Segment Continuity difference {error6}')

        # error6 = error_func6(x_sim, x)
        # print(f'erorr func6: Azimuthal Slice Smoothness difference {error6}')

        error7 = error_func7(x_sim, x)
        print(f'erorr func7: Radial shape difference {error7}')

        # error8 = error_func8(x_sim, x)
        # print(f'erorr func8: Azimuthal shape difference {error8}')

        error10 = error_func10(x_sim, x)
        print(f'erorr func10: Radial Shape difference with laplacian {error10}')

        error11 = error_func11(x_sim, x)
        print(f'erorr func11: Each bin difference {error11}')

        error12 = error_func12(x_sim, x)
        print(f'erorr func12: Ralative difference with Sobel {error12}')

        error13 = error_func13(x_sim, x)
        print(f'erorr func13: Relative difference with Sobel Weighting {error13}')

        error14 = error_func14(x_sim, x)
        print(f'erorr func14: Each bin difference weighted with correlation coefficient {error14}')

        # error_5_11 = error_func5(x_sim, x) * error_func11(x_sim, x)
        # print(f'erorr func5 with 11: Measure of both difference and continuity {error_5_11}')

        # error_5_6_11 = error_func5(x_sim, x) * error_func6(x_sim, x) * error_func11(x_sim, x)
        # print(f'erorr func5, 6, 11: Measure of both difference and continuity {error_5_6_11}')

        # error8 = error_func8(x_sim, x)
        # print(f'erorr func8: all pixels difference {error8}')

elif prog == 1:
    l = 5e-5
    while l >= 5e-6:
        x = reconstruct(grid_params, row_idx, col_idx, val, y, D.indptr,
            D.indices,
            D.data.astype(np.float32),
            l)
        
        print(f'lambda = {l:.2e}')
        print(f'lambda = {l:.2e}', file=sys.stderr)
        print(f'ZDA of x_sim {np.sum(x_sim < 0)}, of x {np.sum(x < 0)}')
        # error9 = error_func9(x_sim, x)
        # print(f'erorr func9: shape difference {error9}')
        # error7 = error_func7(x_sim, x)
        # print(f'erorr func7: Radial shape difference {error7}')
        # error8 = error_func8(x_sim, x)
        # print(f'erorr func8: Azimuthal shape difference {error8}')

        # l -= 2e-7
        l -= 5e-7

elif prog == 2:
    l = 7e-6
    while l >= 1e-6:
        x = reconstruct(grid_params, row_idx, col_idx, val, y, D.indptr,
            D.indices,
            D.data.astype(np.float32),
            l)
        
        print(f'lambda = {l:.2e}')
        print(f'lambda = {l:.2e}', file=sys.stderr)
        print(f'ZDA of x_sim {np.sum(x_sim < 0)}, of x {np.sum(x < 0)}')
        error3 = error_func3(x_sim, x)
        print(f'erorr func3: radial RMS error {error3}')

        l -= 5e-7

elif prog == 3:
    # l = 5e-6
    l = 2e-5
    # while l >= 5e-7:
    while l >= 5e-6:
        x = reconstruct(grid_params, row_idx, col_idx, val, y, D.indptr,
            D.indices,
            D.data.astype(np.float32),
            l)
        
        print(f'lambda = {l:.2e}')
        print(f'lambda = {l:.2e}', file=sys.stderr)
        error11 = error_func11(x_sim, x)
        print(f'erorr func11: Each bin difference {error11}')

        # l -= 2e-7
        l -= 1e-7

elif prog == 4:
    l = 5e-6
    while l >= 5e-7:
        x = reconstruct(grid_params, row_idx, col_idx, val, y, D.indptr,
            D.indices,
            D.data.astype(np.float32),
            l)
        
        print(f'lambda = {l:.2e}')
        print(f'lambda = {l:.2e}', file=sys.stderr)
        error7 = error_func7(x_sim, x)
        print(f'erorr func7: Radial shape difference {error7}')

        # l -= 2e-7
        l -= 1e-7

elif prog == 5:
    l = 5e-6
    while l >= 5e-7:
        x = reconstruct(grid_params, row_idx, col_idx, val, y, D.indptr,
            D.indices,
            D.data.astype(np.float32),
            l)
        
        print(f'lambda = {l:.2e}')
        print(f'lambda = {l:.2e}', file=sys.stderr)
        error3 = error_func3(x_sim, x)
        print(f'erorr func3: radial RMS error {error3}')

        # l -= 2e-7
        l -= 1e-7

elif prog == 6:
    # l = 5e-6
    l = 2e-6
    while l >= 5e-7:
        x = reconstruct(grid_params, row_idx, col_idx, val, y, D.indptr,
            D.indices,
            D.data.astype(np.float32),
            l)
        
        print(f'lambda = {l:.2e}')
        print(f'lambda = {l:.2e}', file=sys.stderr)
        error10 = error_func10(x_sim, x)
        print(f'erorr func10: Radial Shape difference with laplacian {error10}')

        # l -= 2e-7
        l -= 1e-7

elif prog == 7:
    l = 5e-6
    # l = 2e-6
    # l = 1e-5
    while l >= 1e-7:
    # while l > 2e-6:
        x = reconstruct(grid_params, row_idx, col_idx, val, y, D.indptr,
            D.indices,
            D.data.astype(np.float32),
            l)
        
        print(f'lambda = {l:.2e}')
        print(f'lambda = {l:.2e}', file=sys.stderr)
        error12 = error_func12(x_sim, x)
        print(f'erorr func12: Radial Shape difference with Sobel {error12}')

        # l -= 2e-7
        l -= 1e-7

elif prog == 8:
    # l = 5e-6
    # l = 2e-6
    l = 2e-5
    # while l >= 1e-7:
    while l > 5e-6:
        x = reconstruct(grid_params, row_idx, col_idx, val, y, D.indptr,
            D.indices,
            D.data.astype(np.float32),
            l)
        
        print(f'lambda = {l:.2e}')
        print(f'lambda = {l:.2e}', file=sys.stderr)
        error11 = error_func11(x_sim, x)
        print(f'erorr func11: Each bin difference {error11}')

        l -= 5e-7
        # l -= 1e-7

elif prog == 9:
    # l = 5e-6
    # l = 2e-6
    l = 2e-5
    # while l >= 1e-7:
    while l > 5e-6:
        x = reconstruct(grid_params, row_idx, col_idx, val, y, D.indptr,
            D.indices,
            D.data.astype(np.float32),
            l)
        
        print(f'lambda = {l:.2e}')
        print(f'lambda = {l:.2e}', file=sys.stderr)
        error11 = error_func11(x_sim, x)
        print(f'erorr func11: Each bin difference {error11}')
        error14 = error_func14(x_sim, x)
        print(f'erorr func14: Each bin difference weighted with correlation coefficient {error14}')

        l -= 2e-7
        # l -= 1e-7

elif prog == 10:
    # l = 5e-6
    # l = 2e-6
    l = 1e-5
    while l >= 1e-6:
    # while l > 5e-6:
        x = reconstruct(grid_params, row_idx, col_idx, val, y, D.indptr,
            D.indices,
            D.data.astype(np.float32),
            l)
        
        print(f'lambda = {l:.2e}')
        print(f'lambda = {l:.2e}', file=sys.stderr)
        error_5_6_11 = error_func5(x_sim, x) * error_func6(x_sim, x) * error_func11(x_sim, x)
        print(f'erorr func5, 6, 11: Measure of both difference and continuity {error_5_6_11}')

        # l -= 2e-7
        l -= 1e-7

elif prog == 11:
    # l = 9e-6
    # l = 2e-6
    # l = 2e-5
    l = 1e-4
    l_list = []
    metric1 = []
    metric2 = []
    metric3 = []
    metric4 = []
    # x_sim = np.reshape(x_sim, (30, 75, 150), order='F')[:25, :, :].flatten(order='F').copy()
    while l >= 1e-6:
    # while l >= 3e-6:
    # while l >= 1e-7:
        x = reconstruct(grid_params, row_idx, col_idx, val, y, D.indptr,
            D.indices,
            D.data.astype(np.float32),
            l)
        
        print(f'lambda = {l:.2e}')
        print(f'lambda = {l:.2e}', file=sys.stderr)
        # x = np.reshape(x, (30, 75, 150), order='F')[:25, :, :].flatten(order='F').copy()
        error1 = error_func11(x_sim, x)
        # error1 = RMS_func(x_sim, x)
        error2 = error_func5(x_sim, x)
        error3 = error_func6(x_sim, x)
        error4 = error_func3(x_sim, x)
        m1 = error1
        m2 = error1 + error2 * 2
        m3 = error1 + error2 * 2 + error3 * 0.5
        # m2 = error1 + 0.25 * error2
        # m3 = error1 + 0.25 * error2 + 0.15 * error3
        # m2 = error1 * error2
        # m3 = error1 * error2 * error3
        # m2 = error1 * error2 * error3
        # m3 = error1 * error2 * (error3 ** 0.2)
        m4 = error4
        print(f"errors {error1} {error2} {error3}")
        print(f"metrics {m1} {m2} {m3}")
        #error2 = error_func5(x_sim, x) * error_func6(x_sim, x) * error_func11(x_sim, x)
        #error3 = error_func5(x_sim, x) * (error_func6(x_sim, x) ** 0.5) * error_func11(x_sim, x)

        l_list.append(l)
        metric1.append(m1)
        metric2.append(m2)
        metric3.append(m3)
        metric4.append(m4)

        # l -= 2e-7
        l -= 5e-7

    plt.figure()
    plt.plot(l_list, metric1)
    plt.plot(l_list, metric2)
    plt.plot(l_list, metric3)
    np.array(metric1).tofile("metric1")
    np.array(metric2).tofile("metric2")
    np.array(metric3).tofile("metric3")
    np.array(metric4).tofile("metric4")
    np.array(l_list).tofile("l_list")
    plt.ylabel(r"$\lambda$")
    plt.xlabel("error")
    plt.savefig("metric.pdf")
    print(f"min lambda for RMS error is {l_list[np.argmin(metric4)]}")
    print(f"min lambda for metric1 is {l_list[np.argmin(metric1)]}")
    print(f"min lambda for metric2 is {l_list[np.argmin(metric2)]}")
    print(f"min lambda for metric3 is {l_list[np.argmin(metric3)]}")

elif prog == 12:
    x = reconstruct(grid_params, row_idx, col_idx, val, y, D.indptr,
        D.indices,
        D.data.astype(np.float32),
        7.2e-6)
        
    error = error_func3(x_sim, x)
    # error = error_func5(x_sim, x) * error_func11(x_sim, x)


    
