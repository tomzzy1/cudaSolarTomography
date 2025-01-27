#!/usr/bin/env python3
import logging
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from pathlib import Path
import os
import sys
import h5py
from scipy.interpolate import griddata
from scipy.interpolate import LinearNDInterpolator
from scipy.interpolate import RegularGridInterpolator
import numpy as np
import itertools
import matplotlib.pyplot as plt


def sphere_to_cart(p):
    x = p[2] * np.sin(p[1]) * np.cos(p[0])
    y = p[2] * np.sin(p[1]) * np.sin(p[0])
    z = p[2] * np.cos(p[1])
    return (x, y, z)


def sphere_to_cart_2d(p):
    x = p[1] * np.cos(p[0])
    y = p[1] * np.sin(p[0])
    return (x, y)

# first interpolate rad with polynomial, than linear on polar and phi
# interpolate from the original MHD volumes to desired resolution
def read_mhd(mhd_path, N_rad, N_theta, N_phi):
    mhd_path = Path(mhd_path)
    for fname in os.listdir(mhd_path):
        if not fname.split(".")[-1] == "h5":
            continue
        print(mhd_path / fname)
        f = h5py.File(mhd_path / fname, "r+")
        phi = f["fakeDim0"][:]
        polar = f["fakeDim1"][:]
        rad = f["fakeDim2"][:]
        rad_grid, polar_grid = np.meshgrid(rad, polar)
        grid = np.column_stack((polar_grid.ravel(), rad_grid.ravel()))
        print("grid", grid, grid.shape)
        points = np.array(list(map(sphere_to_cart_2d, grid)))
        print("points", points)
        my_rad = np.linspace(2, 6.5, N_rad + 1)[1:]
        # mhd use polar from 0, pi
        # solartom use polar from -pi / 2 to pi / 2
        my_polar = np.linspace(0, np.pi, N_theta + 1)[1:][::-1]
        my_rad_grid, my_polar_grid = np.meshgrid(my_rad, my_polar)
        my_grid = np.column_stack((my_polar_grid.ravel(), my_rad_grid.ravel()))
        my_points = np.array(list(map(sphere_to_cart_2d, my_grid)))
        my_densities = []
        # print("density at phi = 0, polar = 0", f['Data-Set-2'][0][0][:])
        for i in range(f["fakeDim0"].shape[0]):
            my_density = griddata(
                points, np.ndarray.flatten(f["Data-Set-2"][i][:][:]), my_points
            )
            my_densities.append(my_density)

        # downsample
        if N_phi != f["fakeDim0"].shape[0]:
            print('downsample')
            # assert f["fakeDim0"].shape[0] % N_phi == 0
            if f["fakeDim0"].shape[0] % N_phi == 0:
                df = f["fakeDim0"].shape[0] // N_phi
                my_densities2 = []
                for i in range(0, f["fakeDim0"].shape[0], df):
                    my_densities2.append(np.average(my_densities[i : i + df], axis=0))
                density = np.concatenate(my_densities2)
                print(len(density))
            else:
                my_phi = np.linspace(0, 2 * np.pi, N_phi + 1)[1:]
                density = np.zeros((N_rad, N_theta, N_phi))
                for i in range(N_theta):
                    for j in range(N_rad):
                        phi_val = np.zeros(f["fakeDim0"].shape[0])
                        for k in range(f["fakeDim0"].shape[0]):
                            phi_val[k] = my_densities[k][i * N_rad + j]
                        phi_val2 = griddata(phi, phi_val, my_phi)
                        # print(len(phi_val2))
                        density[j][i] = phi_val2
                print(density.shape)
                density = density.flatten(order='F')
            print(len(density), N_phi * N_theta * N_rad)
            assert len(density) == N_phi * N_theta * N_rad
        else:
            density = np.concatenate(my_densities)
        density *= 1e6 * 1e2
        print(density.shape, max(density), min(density), density)

        # print(np.average(np.ndarray.flatten(f['Data-Set-2'][:][:][:]) - density))
        # density.tofile(mhd_path / "x_corhel_db")
        density_path = Path("mhd_resolutions")
        density_path.mkdir(parents=True, exist_ok=True)
        density.astype(np.float32).tofile(density_path / f"x_corhel_{N_rad}_{N_theta}_{N_phi}")
        return density
    
# Interpolate from the interpolated volumes to the desired resolution
# The radial resolution must match   
def interp(N_rad_src, N_theta_src, N_phi_src, density_src, N_rad, N_theta, N_phi):
    assert(N_rad_src == N_rad)
    phi_src = np.linspace(0, 2 * np.pi, N_phi_src + 1)[1:]
    polar_src = np.linspace(0, np.pi, N_theta_src + 1)[1:]
    my_phi = np.linspace(0, 2 * np.pi, N_phi + 1)[1:]
    my_polar = np.linspace(0, np.pi, N_theta + 1)[1:]
    polar, phi = np.meshgrid(my_polar, my_phi, indexing='ij')
    
    density_src = np.reshape(density_src, (N_rad_src, N_theta_src, N_phi_src), order='F')
    density = np.zeros((N_rad, N_theta, N_phi))
    for i in range(N_rad_src):
        interp = RegularGridInterpolator((polar_src, phi_src), density_src[i], method='cubic')
        rad_slice = interp((polar, phi))
        assert(not np.any(np.isnan(rad_slice)))
        # print(rad_slice.shape)
        density[i] = rad_slice

    print(density.shape, density)
    density = density.flatten(order='F')
    print(max(density), min(density))
    assert(not np.any(np.isnan(density)))
    # print(np.average(np.ndarray.flatten(f['Data-Set-2'][:][:][:]) - density))
    # density.tofile(mhd_path / "x_corhel_db")
    mhd_dir = Path(f"mhd_resolutions_{N_rad_src}_{N_theta_src}_{N_phi_src}")               
    mhd_dir.mkdir(parents=True, exist_ok=True)
    density.astype(np.float32).tofile(mhd_dir / f"x_corhel_{N_rad}_{N_theta}_{N_phi}")
    return density

# deprecated  
def interp_old2(N_rad_src, N_theta_src, N_phi_src, density_src, N_rad, N_theta, N_phi):
    phi = np.linspace(0, 2 * np.pi, N_phi_src + 1)[1:]
    polar = np.linspace(0, np.pi, N_theta_src + 1)[1:]
    rad = np.linspace(2, 6.5, N_rad_src + 1)[1:]
    rad_grid, polar_grid = np.meshgrid(rad, polar)
    grid = np.column_stack((polar_grid.ravel(), rad_grid.ravel()))
    # print("grid", grid, grid.shape)
    points = np.array(list(map(sphere_to_cart_2d, grid)))
    # print("points", points)
    my_rad = np.linspace(2, 6.5, N_rad + 1)[1:]
    # mhd use polar from 0, pi
    # solartom use polar from -pi / 2 to pi / 2
    my_polar = np.linspace(0, np.pi, N_theta + 1)[1:]
    my_rad_grid, my_polar_grid = np.meshgrid(my_rad, my_polar)
    my_grid = np.column_stack((my_polar_grid.ravel(), my_rad_grid.ravel()))
    my_points = np.array(list(map(sphere_to_cart_2d, my_grid)))
    my_densities = []
    # print("density at phi = 0, polar = 0", f['Data-Set-2'][0][0][:])
    for i in range(N_phi_src):
        my_density = griddata(
            points, np.ndarray.flatten(density_src[i][:][:]), my_points
        )
        assert(not np.any(np.isnan(my_density)))
        my_densities.append(my_density)

    # downsample
    if N_phi != N_phi_src:
        print('downsample')
        # assert f["fakeDim0"].shape[0] % N_phi == 0
        if N_phi_src % N_phi == 0:
            df = N_phi_src // N_phi
            my_densities2 = []
            for i in range(0, N_phi_src, df):
                my_densities2.append(np.average(my_densities[i : i + df], axis=0))
            density = np.concatenate(my_densities2)
            print(len(density))
        else:
            my_phi = np.linspace(0, 2 * np.pi, N_phi + 1)[1:]
            density = np.zeros((N_rad, N_theta, N_phi))
            for i in range(N_theta):
                for j in range(N_rad):
                    phi_val = np.zeros(N_phi_src)
                    for k in range(N_phi_src.shape[0]):
                        phi_val[k] = my_densities[k][i * N_rad + j]
                    phi_val2 = griddata(phi, phi_val, my_phi)
                    # print(len(phi_val2))
                    density[j][i] = phi_val2
            print(density.shape)
            density = density.flatten(order='F')
        print(len(density), N_phi * N_theta * N_rad)
        assert len(density) == N_phi * N_theta * N_rad
    else:
        density = np.concatenate(my_densities)
    assert(not np.any(np.isnan(density)))
    print(density.shape, max(density), min(density), density)

    # print(np.average(np.ndarray.flatten(f['Data-Set-2'][:][:][:]) - density))
    # density.tofile(mhd_path / "x_corhel_db")
    mhd_dir = Path(f"mhd_resolutions_{N_rad_src}_{N_theta_src}_{N_phi_src}")               
    mhd_dir.mkdir(parents=True, exist_ok=True)
    density.astype(np.float32).tofile(mhd_dir / f"x_corhel_{N_rad}_{N_theta}_{N_phi}")

# deprecated 
def interp_old(density, N_rad, N_theta, N_phi):
    phi = np.linspace(0, 2 * np.pi, 301)[1:]
    polar = np.linspace(0, np.pi, 150 + 1)[1:]
    rad = np.linspace(2, 6.5, 45 + 1)[1:]
    rad_grid, polar_grid = np.meshgrid(rad, polar)
    grid = np.column_stack((polar_grid.ravel(), rad_grid.ravel()))
    print("grid", grid, grid.shape)
    points = np.array(list(map(sphere_to_cart_2d, grid)))
    print("points", points)
    my_rad = np.linspace(2, 6.5, N_rad + 1)[1:]
    # mhd use polar from 0, pi
    # solartom use polar from -pi / 2 to pi / 2
    my_polar = np.linspace(0, np.pi, N_theta + 1)[1:]
    my_rad_grid, my_polar_grid = np.meshgrid(my_rad, my_polar)
    my_grid = np.column_stack((my_polar_grid.ravel(), my_rad_grid.ravel()))
    my_points = np.array(list(map(sphere_to_cart_2d, my_grid)))
    my_densities = []
    # print("density at phi = 0, polar = 0", f['Data-Set-2'][0][0][:])
    for i in range(300):
        my_density = griddata(
            points, np.ndarray.flatten(density[i][:][:]), my_points
        )
        my_densities.append(my_density)

    # downsample
    if N_phi != 300:
        print('downsample')
        assert 300 % N_phi == 0
        df = 300 // N_phi
        my_densities2 = []
        for i in range(0, 300, df):
            my_densities2.append(np.average(my_densities[i : i + df], axis=0))
        density = np.concatenate(my_densities2)
        print(len(density))
        assert len(density) == N_phi * N_theta * N_rad
    else:
        density = np.concatenate(my_densities)
    density *= 1e6 * 1e2
    print(density.shape, max(density), min(density), density)

    # print(np.average(np.ndarray.flatten(f['Data-Set-2'][:][:][:]) - density))
    # density.tofile(mhd_path / "x_corhel_db")
    density.astype(np.float32).tofile(Path("mhd_resolutions_2") / f"x_corhel_{N_rad}_{N_theta}_{N_phi}")

# deprecated
def main(argv=None):
    if argv is None:
        argv = sys.argv
    parser = ArgumentParser(
        "parse metadata of pb files", formatter_class=ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("mhd_path", type=Path, help="path to the metadata")
    args = parser.parse_args(argv[1:])
    # Table.read.help('hdf5')
    for fname in os.listdir(args.mhd_path):
        if not fname.split(".")[-1] == "h5":
            continue
        print(args.mhd_path / fname)
        f = h5py.File(args.mhd_path / fname, "r+")
        print(f.keys())
        print(f["Data-Set-2"].shape)
        print(f["Data-Set-2"][:][:][:])
        print(f["fakeDim0"].shape)  # phi
        print(f["fakeDim1"].shape)  # theta
        print(f["fakeDim2"].shape)  # rad
        phi = f["fakeDim0"][:]
        polar = f["fakeDim1"][:]
        rad = f["fakeDim2"][:]
        idx = np.argmax(rad > 2.5)
        # rad += 1
        # print("phi idx", phi)
        # print("polar idx", polar)
        # print("rad idx", rad)
        # N_rad = 30
        # N_theta = 101
        # N_phi = 129
        # N_rad = 76
        # N_theta = 143
        # N_phi = 300
        # N_rad = 30
        # N_theta = 72
        # N_phi = 150
        # N_rad = 30
        # N_theta = 143
        # N_phi = 300
        # N_rad = 76
        # N_theta = 72
        # N_phi = 150
        # N_rad = 30
        # N_theta = 75
        # N_phi = 150
        N_rad = 30
        N_theta = 75
        N_phi = 200
        rad_grid, polar_grid = np.meshgrid(rad, polar)

        print(f'number of radial bin in computation region {np.count_nonzero(rad[rad > 2] - np.count_nonzero(rad[rad < 6.5]))}')
        # print('polar_grid', polar_grid)
        # print('rad_grid', rad_grid)
        grid = np.column_stack((polar_grid.ravel(), rad_grid.ravel()))
        print("grid", grid, grid.shape)
        points = np.array(list(map(sphere_to_cart_2d, grid)))
        print("points", points)
        my_rad = np.linspace(2, 6.5, N_rad + 1)[1:]
        # mhd use polar from 0, pi
        # solartom use polar from -pi / 2 to pi / 2
        my_polar = np.linspace(0, np.pi, N_theta + 1)[1:][::-1]
        my_rad_grid, my_polar_grid = np.meshgrid(my_rad, my_polar)
        my_grid = np.column_stack((my_polar_grid.ravel(), my_rad_grid.ravel()))
        my_points = np.array(list(map(sphere_to_cart_2d, my_grid)))
        my_densities = []
        # print("density at phi = 0, polar = 0", f['Data-Set-2'][0][0][:])
        print(points)
        print(my_points)
        for i in range(f["fakeDim0"].shape[0]):
            print(f"phi bin{i}")
            my_density = griddata(
                points, np.ndarray.flatten(f["Data-Set-2"][i][:][:]), my_points
            )
            my_densities.append(my_density)
            print(
                "density", my_density.shape, np.reshape(my_density, (N_theta, N_rad))[0]
            )
            print(f["Data-Set-2"][i][0][:])

        # downsample
        if N_phi != f["fakeDim0"].shape[0]:
            print('downsample')
            # assert f["fakeDim0"].shape[0] % N_phi == 0
            if f["fakeDim0"].shape[0] % N_phi == 0:
                df = f["fakeDim0"].shape[0] // N_phi
                my_densities2 = []
                for i in range(0, f["fakeDim0"].shape[0], df):
                    my_densities2.append(np.average(my_densities[i : i + df], axis=0))
                density = np.concatenate(my_densities2)
                print(len(density))
            else:
                my_phi = np.linspace(0, 2 * np.pi, N_phi + 1)[1:]
                density = np.zeros((N_rad, N_theta, N_phi))
                for i in range(N_theta):
                    for j in range(N_rad):
                        phi_val = np.zeros(f["fakeDim0"].shape[0])
                        for k in range(f["fakeDim0"].shape[0]):
                            phi_val[k] = my_densities[k][i * N_rad + j]
                        phi_val2 = griddata(phi, phi_val, my_phi)
                        # print(len(phi_val2))
                        density[j][i] = phi_val2
                print(density.shape)
                density = density.flatten(order='F')
            print(len(density), N_phi * N_theta * N_rad)
            assert len(density) == N_phi * N_theta * N_rad
        else:
            density = np.concatenate(my_densities)
        density *= 1e6 * 1e2
        print(density.shape, max(density), min(density), density)

        # print(np.average(np.ndarray.flatten(f['Data-Set-2'][:][:][:]) - density))
        density.tofile(args.mhd_path / "x_corhel_db")
        density.astype(np.float32).tofile(args.mhd_path / "x_corhel")

        # plt.figure()

        # points = np.array(list(map(sphere_to_cart, itertools.product(phi, polar, rad))))
        # print('points', points.shape, points)
        # my_rad = np.linspace(2, 6.5, N_rad + 1)[1:]
        # #print(my_rad)
        # my_polar = np.linspace(0, np.pi, N_theta + 1)[1:]
        # my_phi = np.linspace(0, np.pi * 2, N_phi + 1)[1:]
        # my_grid = np.array(list(map(sphere_to_cart, itertools.product(my_phi, my_polar, my_rad))))
        # print('my_grid', my_grid.shape, my_grid)
        # density = np.ndarray.flatten(f['Data-Set-2'][:][:][:])
        # print(density.shape)
        # my_density = griddata(points, density, my_grid)
        # interp = LinearNDInterpolator(points, density)
        # my_density = interp(my_grid)
        # print('density', my_density.shape, my_density)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sys.exit(main())

