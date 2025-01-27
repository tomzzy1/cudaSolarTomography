#!/usr/bin/env python3

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import sys
import logging
from pathlib import Path

import numpy as np
import scipy as sp
import matplotlib
import matplotlib.pylab as plt

from grid import HollowSphere
from scipy_io_util import loadmat

logger = logging.getLogger('plot_x_slices')


RESOLUTION = 512


def rel_diff(x, y):
    """
    Calculate the relative difference between *x* (truth) and *y*
    only considering values that are not NaN for both *x* and *y*.
    """
    Ix = ~np.isnan(x)
    Iy = ~np.isnan(y)
    I = Ix & Iy
    return np.linalg.norm(x[I] - y[I]) / np.linalg.norm(x[I])


def plot_rad_slices(output_path, x, hollow_sphere, titles=None, vmin=None, vmax=None):
    """
    """
    N_plot = len(x)
    fig, ax = plt.subplots(nrows=N_plot, figsize=(12, 5*N_plot))
    if N_plot == 1:
        ax = [ax]
    im = []
    out_paths = []

    cbformat = matplotlib.ticker.ScalarFormatter()
    cbformat.set_powerlimits((-2,2))

    extent = [hollow_sphere.lon_edges[0], hollow_sphere.lon_edges[-1],
              hollow_sphere.lat_edges[0], hollow_sphere.lat_edges[-1]]

    for n, rad_center_i in enumerate(hollow_sphere.rad_centers):
        for i, x_i in enumerate(x):
            if i == 0:
                vmin_i = min(x_i[n, :, :].flat) if vmin is None else vmin
                vmax_i = max(x_i[n, :, :].flat) if vmax is None else vmax
            if titles is not None:
                title = f'{titles[i]} at radius {rad_center_i:.2f} [Rsun]'
            else:
                title = f'x{i} at radius {rad_center_i:.2f} [Rsun]'
            if i > 0:
                title += f' rel_diff={rel_diff(x[0][n, :, :], x_i[n, :, :]):.3f}'
            if n == 0:
                im.append(ax[i].imshow(x_i[n, :, :], extent=extent, origin='lower', vmin=vmin_i, vmax=vmax_i))
                ax[i].set_xlabel('Longitude [deg]')
                ax[i].set_ylabel('Latitute [deg]')

                cb = fig.colorbar(im[i], ax=ax[i], label='$N_e / $cm$^3$', format=cbformat);
            else:
                im[i].set_data(x_i[n, :, :])
                im[i].set_clim((vmin_i, vmax_i))
            ax[i].set_title(title)

        plot_path = output_path / f'x_rad_slice_{n:03d}.pdf'
        logger.info(f'Saving {plot_path}')

        plt.savefig(plot_path, bbox_inches='tight')
        out_paths.append(plot_path)
    plt.close(fig)
    return out_paths


def plot_theta_slices(output_path, x, hollow_sphere, resolution=RESOLUTION, titles=None, vmin=None, vmax=None):
    """
    """
    N_plot = len(x)
    fig, ax = plt.subplots(ncols=N_plot, figsize=(8*N_plot, 6))
    if N_plot == 1:
        ax = [ax]
    im = []
    out_paths = []

    cbformat = matplotlib.ticker.ScalarFormatter()
    cbformat.set_powerlimits((-2,2))

    extent = (-hollow_sphere.R_MAX, hollow_sphere.R_MAX,
              -hollow_sphere.R_MAX, hollow_sphere.R_MAX)

    NX = resolution
    NY = resolution

    X, Y = np.meshgrid(np.linspace(-hollow_sphere.R_MAX, hollow_sphere.R_MAX, NX),
                       np.linspace(-hollow_sphere.R_MAX, hollow_sphere.R_MAX, NY), indexing='xy')
    R = np.sqrt(X**2 + Y**2)
    THETA = np.rad2deg(np.arctan2(Y, X))
    J = (R < hollow_sphere.R_MIN) | (R > hollow_sphere.R_MAX)
    THETA[THETA < 0] += 360

    for n, lat_center_i in enumerate(hollow_sphere.lat_centers):
        for i, x_i in enumerate(x):
            lat_interp = sp.interpolate.RegularGridInterpolator((hollow_sphere.rad_centers, hollow_sphere.lon_centers),
                                                                x_i[:, n, :],
                                                                method='nearest',
                                                                bounds_error=False,
                                                                fill_value=None)

            Z = lat_interp((R.flat, THETA.flat)).reshape((NY, NX))
            Z[J] = np.nan
            if i == 0:
                vmin_i = min(x_i[:, n, :].flat) if vmin is None else vmin
                vmax_i = max(x_i[:, n, :].flat) if vmax is None else vmax
            if titles is not None:
                title = f'{titles[i]}\nat latitude {lat_center_i:.2f} [deg]'
            else:
                title = f'x{i} \nat latitude {lat_center_i:.2f} [deg]'
            if i > 0:
                title += f' rel_diff={rel_diff(x[0][:, n, :].flat, x_i[:, n, :].flat):.3f}'
            if n == 0:
                im.append(ax[i].imshow(Z, extent=extent, origin='lower', vmin=vmin_i, vmax=vmax_i))
                ax[i].set_xlabel('[Rsun]')
                ax[i].set_ylabel('[Rsun]')
                cb = fig.colorbar(im[i], ax=ax[i], label='$N_e / $cm$^3$', format=cbformat);
            else:
                im[i].set_data(Z)
                im[i].set_clim((vmin_i, vmax_i))
            ax[i].set_title(title)

        plot_path = output_path / f'x_theta_slice_{n:03d}.pdf'
        logger.info(f'Saving {plot_path}')

        plt.savefig(plot_path, bbox_inches='tight')
        out_paths.append(plot_path)
    plt.close(fig)
    return out_paths



def plot_phi_slices(output_path, x, hollow_sphere, resolution=RESOLUTION, titles=None, vmin=None, vmax=None):
    """
    """
    N_plot = len(x)
    fig, ax = plt.subplots(ncols=N_plot, figsize=(6*N_plot, 8))
    if N_plot == 1:
        ax = [ax]
    im = []
    out_paths = []

    cbformat = matplotlib.ticker.ScalarFormatter()
    cbformat.set_powerlimits((-2,2))

    extent = (0, hollow_sphere.R_MAX, -hollow_sphere.R_MAX, hollow_sphere.R_MAX)

    NX = resolution
    NY = 2*NX

    X, Y = np.meshgrid(np.linspace(0, hollow_sphere.R_MAX, NX),
                       np.linspace(-hollow_sphere.R_MAX, hollow_sphere.R_MAX, NY), indexing='xy')
    R = np.sqrt(X**2 + Y**2)
    THETA = np.rad2deg(np.arctan2(Y, X))
    J = (R < hollow_sphere.R_MIN) | (R > hollow_sphere.R_MAX)

    for n, lon_center_i in enumerate(hollow_sphere.lon_centers):
        for i, x_i in enumerate(x):
            lon_interp = sp.interpolate.RegularGridInterpolator((hollow_sphere.rad_centers, hollow_sphere.lat_centers),
                                                                x_i[:, :, n],
                                                                method='nearest',
                                                                bounds_error=False,
                                                                fill_value=None)
            Z = lon_interp((R.flat, THETA.flat)).reshape((NY, NX))
            Z[J] = np.nan
            if i == 0:
                vmin_i = min(x_i[:, :, n].flat) if vmin is None else vmin
                vmax_i = max(x_i[:, :, n].flat) if vmax is None else vmax
            if titles is not None:
                title = f'{titles[i]}\nat longitude {lon_center_i:.2f} [deg]'
            else:
                title = f'x{i}\nat longitude {lon_center_i:.2f} [deg]'
            if i > 0:
                title += f' rel_diff={rel_diff(x[0][:, :, n].flat, x_i[:, :, n].flat):.3f}'
            if n == 0:
                im.append(ax[i].imshow(Z, extent=extent, origin='lower', vmin=vmin_i, vmax=vmax_i))
                ax[i].set_xlabel('[Rsun]')
                ax[i].set_ylabel('[Rsun]')
                cb = fig.colorbar(im[i], ax=ax[i], label='$N_e / $cm$^3$', format=cbformat);
            else:
                im[i].set_data(Z)
                im[i].set_clim((vmin_i, vmax_i))
            ax[i].set_title(title)

        plot_path = output_path / f'x_phi_slice_{n:03d}.pdf'
        logger.info(f'Saving {plot_path}')

        plt.savefig(plot_path, bbox_inches='tight')
        out_paths.append(plot_path)
    plt.close(fig)
    return out_paths


def main(argv=None):
    if argv is None:
        argv = sys.argv

    parser = ArgumentParser('Plot slices of hollow sphere electron density.',
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('output_path', type=Path, help='Location to store plots.')
    parser.add_argument('x_mat_filename', type=Path, nargs='+', help='Path to .mat filename or plain x filename.')
    parser.add_argument('--slices', '-s', type=str, choices=['all', 'rad', 'phi', 'theta'], nargs='+', default=['all'], help='Which slices to plot.')
    parser.add_argument('--key', '-k', type=str, default='x_hat', help='Key for x in the .mat records.')
    parser.add_argument('--resolution', '-r', type=int, default=RESOLUTION, help='Resolution for phi and theta plots.')
    parser.add_argument('--vmin', type=float, default=None, help='Minimum of color scale (determine automatically if not specified).')
    parser.add_argument('--vmax', type=float, default=None, help='Maximum of color scale (determine automatically if not specified).')
    parser.add_argument('--params', type=float, nargs='*', default=None, help='geometry parameters, order is N_RAD_BINS, N_THETA_BINS, N_PHI_BINS, R_MAX, R_MIN')
    args = parser.parse_args(argv[1:])

    x_hat = []
    info = {}
    keys = ['N_BINS', 'N_RAD_BINS', 'N_THETA_BINS', 'N_PHI_BINS', 'R_MAX', 'R_MIN']
    if args.params is not None:
        for i in range(1, 4):
            info[keys[i]] = int(args.params[i - 1])
        for i in range(4, len(keys)):
            info[keys[i]] = args.params[i - 1]
        info['N_BINS'] = info['N_RAD_BINS'] * info['N_THETA_BINS'] * info['N_PHI_BINS']

    for i, x_mat_i in enumerate(args.x_mat_filename):
        logger.info(f'Loading {x_mat_i}')
        isMat = '.mat' in x_mat_i.name
        print(isMat)
        if isMat:
            m = loadmat(x_mat_i, squeeze_me=True)
        else:
            x_hat_i = np.fromfile(x_mat_i, dtype=np.float32)
        if isMat:
            if i == 0:
                for k in keys:
                    info[k] = m[k]
            else:
                for k in keys:
                    assert info[k] == m[k]
        if isMat:
            x_hat_i = m[args.key]
        print(len(x_hat_i), info['N_BINS'])
        assert len(x_hat_i) == info['N_BINS']
        shape = (info['N_RAD_BINS'], info['N_THETA_BINS'], info['N_PHI_BINS'])
        # x_hat_i[x_hat_i < 1] = 1
        x_hat.append(np.reshape(x_hat_i, shape, order='F'))


    hollow_sphere = HollowSphere(info)

    titles = [x.name for x in args.x_mat_filename]

    # x_hat = np.log10(x_hat)

    if 'rad' in args.slices or 'all' in args.slices:
        plot_rad_slices(args.output_path, x_hat, hollow_sphere,
                        titles=titles, vmin=args.vmin, vmax=args.vmax)

    if 'theta' in args.slices or 'all' in args.slices:
        plot_theta_slices(args.output_path, x_hat, hollow_sphere,
                          resolution=args.resolution, titles=titles, vmin=args.vmin, vmax=args.vmax)

    if 'phi' in args.slices or 'all' in args.slices:
        plot_phi_slices(args.output_path, x_hat, hollow_sphere,
                        resolution=args.resolution, titles=titles, vmin=args.vmin, vmax=args.vmax)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    sys.exit(main())
