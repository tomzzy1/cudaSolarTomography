#!/usr/bin/env python3

import logging
import sys
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from pathlib import Path

import numpy as np
import scipy as sp
import matplotlib
import matplotlib.pylab as plt

logger = logging.getLogger('plot_bin')

N_RAD_BINS = 20
N_THETA_BINS = 30
N_PHI_BINS = 2 * N_THETA_BINS
N_BINS = N_RAD_BINS * N_THETA_BINS * N_PHI_BINS

def main(argv=None):
    if argv is None:
        argv = sys.argv

    parser = ArgumentParser('Plot the mapping between bin and pixels',
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('A_mat', type=Path, help='A matrix')
    parser.add_argument('--plot_path', '-p', type=Path, help='Path to store plots (no plots generated if not specified).')
    args = parser.parse_args(argv[1:])

    A_mat = sp.io.loadmat(args.A_mat, squeeze_me=True)

    blocks = range(len(A_mat['block_idx']) - 1)

    for i in blocks:
        print(f'Block {i}')

        I = slice(A_mat['block_idx'][i], A_mat['block_idx'][i+1])
        A = A_mat['A'][I]
        
        nonzeros = [0 for _ in range(A.shape[1])]
        print(A.shape)
        for j in range(A.shape[1]):
            nonzeros[j] = A[:, j].nnz

        min_nz = 100000
        min_idx = 0
        for idx, nz in enumerate(nonzeros):
            if nz != 0 and nz < min_nz:
                min_nz = nz 
                min_idx = idx
        
        phi_idx = min_idx // (N_RAD_BINS * N_THETA_BINS)
        min_idx %= (N_RAD_BINS * N_THETA_BINS)
        theta_idx = min_idx // N_RAD_BINS
        min_idx %= N_RAD_BINS
        print(min_nz, max(nonzeros), phi_idx, theta_idx, min_idx)

        plt.figure()
        #plt.bar([i for i in range(400, 600)], nonzeros[400:600])
        plt.bar([i for i in range(0, 36000)], nonzeros[:36000])
        plt.savefig(args.plot_path / "bin_pixel_{}.pdf".format(i))
        plt.close()

        plt.figure()
        plt.hist(nonzeros, bins=30)
        plt.savefig(args.plot_path / "pixel_hist{}.png".format(i))
        plt.close()

        
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    sys.exit(main())
