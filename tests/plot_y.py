#!/usr/bin/env python3

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import logging
import sys
from pathlib import Path

import numpy as np
import scipy as sp
import matplotlib
import matplotlib.pylab as plt
import os

from scipy_io_util import loadmat


logger = logging.getLogger('plot_y_hat')

# TODO: Add FITS as it is now stored in A_filename.

def main(argv=None):
    if argv is None:
        argv = sys.argv

    parser = ArgumentParser('Create a series of plots comparing measured (y) and synthetic (\\hat{y} = A x) coronagraph images',
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('output_path', type=Path, help='Location to store plots.')
    parser.add_argument('y_path', type=Path, help='Path to the y vector.')
    parser.add_argument('image_size', type=int, help='height and width of the image')
    parser.add_argument('--vmin', type=float, default=0, help='Plot vmin.')
    parser.add_argument('--dpi', type=float, default=300, help='Plot DPI resolution.')

    args = parser.parse_args(argv[1:])

    M = args.image_size

    matplotlib.use('pdf')

    for y_file in os.listdir(args.y_path):
        y_file_l = y_file.split('.')
        if not y_file_l[-1] == 'fts':
            continue
        y = np.fromfile(args.y_path / y_file)
        y_2D = y.reshape((M, M))
        y_2D = np.log10(y_2D)
        plt.figure()
        plt.imshow(y_2D, origin='lower', vmin=args.vmin)
        plt.title(f'$y$ {y_file}')
        plt.colorbar()

        plot_path = args.output_path / f'y_{y_file_l[0]}.pdf'
        logger.info(f'Saving {plot_path}')
        plt.savefig(plot_path, bbox_inches='tight', dpi=args.dpi)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    sys.exit(main())
