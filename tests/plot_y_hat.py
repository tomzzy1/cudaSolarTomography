#!/usr/bin/env python3

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import logging
import sys
from pathlib import Path

import numpy as np
import scipy as sp
import matplotlib
import matplotlib.pylab as plt

from scipy_io_util import loadmat


logger = logging.getLogger('plot_y_hat')

# TODO: Add FITS as it is now stored in A_filename.

def main(argv=None):
    if argv is None:
        argv = sys.argv

    parser = ArgumentParser('Create a series of plots comparing measured (y) and synthetic (\\hat{y} = A x) coronagraph images',
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('output_path', type=Path, help='Location to store plots.')
    parser.add_argument('x_hat_filename', type=Path, help='Output .mat filename of reconstruction.')
    parser.add_argument('A_filename', type=Path, help='Path to A matrix file which includes observation matrix and y vector.')
    parser.add_argument('--vmin', type=float, default=0, help='Plot vmin.')
    parser.add_argument('--dpi', type=float, default=300, help='Plot DPI resolution.')

    args = parser.parse_args(argv[1:])

    d_x_hat = scipy.io.loadmat(args.x_hat_filename, squeeze_me=True)
    x_hat = d_x_hat['x_hat']

    d_A = loadmat(args.A_filename, squeeze_me=True)
    y = d_A['y']
    y_idx = d_A['y_idx']
    A = d_A['A']
    block_indptr = d_A['A_block_indptr']

    assert len(y) == len(y_idx)
    assert A.shape[0] == len(y)
    assert A.shape[1] == len(x_hat)
    assert block_indptr[-1] == A.shape[0]

    M = d_A['IMAGE_SIZE'] // d_A['BINNING_FACTOR']

    matplotlib.use('pdf')
    fig, ax = plt.subplots(ncols=2, figsize=(14, 5))

    for block, (i1, i2) in enumerate(zip(block_indptr[:-1], block_indptr[1:])):
        I = slice(i1, i2)
        y_I = y[I]
        y_idx_I = y_idx[I]
        y_hat_I = A[I, :] @ x_hat

        y_vec = np.full(M*M, np.nan)
        y_vec[y_idx_I] = y_I
        y_2D = y_vec.reshape((M, M))

        y_hat_vec = np.full(M*M, np.nan)
        y_hat_vec[y_idx_I] = y_hat_I
        y_hat_2D = y_hat_vec.reshape((M, M))

        vmax = max(y_I)

        if block == 0:
            im0 = ax[0].imshow(y_2D, origin='lower', vmin=args.vmin)
            ax[0].set_title(f'$y$ {block}')
            fig.colorbar(im0, ax=ax[0])
            im1 = ax[1].imshow(y_hat_2D, origin='lower', vmin=args.vmin, vmax=vmax)
            ax[1].set_title(f'$\\hat{{y}}$ {block}')
            fig.colorbar(im1, ax=ax[1]);
        else:
            ax[0].set_title(f'$y$ {block}')
            im0.set_data(y_2D)
            im0.set_clim((args.vmin, vmax))
            ax[1].set_title(f'$\\hat{{y}}$ {block}')
            im1.set_data(y_hat_2D)
            im1.set_clim((args.vmin, vmax))

        plot_path = args.output_path / f'y_{block:03d}.pdf'
        logger.info(f'Saving {plot_path}')

        plt.savefig(plot_path, bbox_inches='tight', dpi=args.dpi)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    sys.exit(main())
