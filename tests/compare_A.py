#!/usr/bin/env python3

import logging
import sys
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from pathlib import Path

import numpy as np
import scipy as sp
import matplotlib
import matplotlib.pylab as plt

from scipy_io_loadmat import loadmat

logger = logging.getLogger('compare_A')

EPSILON = 1

BINS = 100


def print_stats(x):
    """
    Print summary statistics of values stored in *x*.
    """
    print(f'mean={np.mean(x):.3e}\tmed={np.median(x):.3e}\tstd={np.std(x):.3e}\tmin={min(x):.3e}\tmax={max(x):.3e}\tN={len(x)}')


def main(argv=None):
    if argv is None:
        argv = sys.argv

    parser = ArgumentParser('Compare two builda or build_A solar tomography matrices and report/plot differences.',
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('A_mat1', type=Path, help='A matrix 1 (ground truth).')
    parser.add_argument('A_mat2', type=Path, help='A matrix 2 (comparison).')
    parser.add_argument('--plot_path', '-p', type=Path, help='Path to store plots (no plots generated if not specified).')
    parser.add_argument('--extreme_path', '-e', type=Path, help='Path to store outlier records.')
    parser.add_argument('--block_idx', '-b', type=int, help='Only process the specified (process all if not specified)')
    parser.add_argument('--epsilon', type=float, default=EPSILON, help='Tolerance.')
    parser.add_argument('--bins', type=int, default=BINS, help='Number of bins in histogram.')
    args = parser.parse_args(argv[1:])
    if args.extreme_path is None:
        args.extreme_path = args.plot_path

    A1_mat = loadmat(args.A_mat1, squeeze_me=True)
    A2_mat = loadmat(args.A_mat2, squeeze_me=True)

    for k in ['R_MIN', 'R_MAX', 'N_RAD_BINS', 'N_THETA_BINS', 'N_PHI_BINS',
              'BINNING_FACTOR', 'IMAGE_SIZE']:
        assert A1_mat[k] == A2_mat[k]

    assert len(A1_mat['A_block_indptr']) == len(A2_mat['A_block_indptr'])

    M = A1_mat['IMAGE_SIZE'] // A1_mat['BINNING_FACTOR']

    if args.plot_path is not None:
        fig_y_diff, ax_y_diff = plt.subplots()
        fig_y_missing, ax_y_missing = plt.subplots()
        cmap_missing = matplotlib.colors.ListedColormap(['C0', 'C1'])
        bounds = [0,1,2]
        norm_missing = matplotlib.colors.BoundaryNorm(bounds, cmap_missing.N)
        fmt_missing = matplotlib.ticker.FuncFormatter(lambda x, pos: ['2', '1'][norm_missing(x) - 1])
        fig_A_hist, ax_A_hist = plt.subplots()
        fig_A_diff, ax_A_diff = plt.subplots()
        if args.extreme_path is not None:
            fig_A_extreme, ax_A_extreme = plt.subplots()
    else:
        matplotlib.use('pdf')

    print(f'Comparison report: {args.A_mat1} (truth) vs. {args.A_mat2}')

    if args.block_idx is None:
        blocks = range(len(A1_mat['A_block_indptr']) - 1)
    else:
        blocks = [args.block_idx]

    for i in blocks:
        print(f'Block {i}')

        I1 = slice(A1_mat['A_block_indptr'][i], A1_mat['A_block_indptr'][i+1])
        y1 = A1_mat['y'][I1]
        y1_idx = A1_mat['y_idx'][I1]
        A1 = A1_mat['A'][I1]
        I1_map = {k: v for v, k in enumerate(zip(*np.unravel_index(y1_idx, (M, M))))}

        I2 = slice(A2_mat['A_block_indptr'][i], A2_mat['A_block_indptr'][i+1])
        y2 = A2_mat['y'][I2]
        y2_idx = A2_mat['y_idx'][I2]
        A2 = A2_mat['A'][I2]
        I2_map = {k: v for v, k in enumerate(zip(*np.unravel_index(y2_idx, (M, M))))}

        I_both = set(I1_map) & set(I2_map)
        I1_only = set(I1_map) - set(I2_map)
        I2_only = set(I2_map) - set(I1_map)

        I1_both = [I1_map[x] for x in I_both]
        I2_both = [I2_map[x] for x in I_both]

        I1_1_only = [I1_map[x] for x in I1_only]

        I2_2_only = [I2_map[x] for x in I2_only]

        # Compare y
        rel_diff_y_both = (y1[I1_both] - y2[I2_both]) / y1[I1_both]
        print('y: relative difference stats (values in both vectors)')
        print_stats(rel_diff_y_both)

        if len(I1_only) > 0:
            print(f'y: # in 1 but not 2 {len(I1_only)}')
            print(I1_only)
        if len(I2_only) > 0:
            print(f'y: # in 2 but not 1 {len(I2_only)}')
            print(I2_only)

        # Compare A
        print(len(I1_both), len(I2_both), A1.shape, A2.shape)
        X = (A1[I1_both, :] - A2[I2_both, :]).tocsr()
        Y = A1[I1_both, :].tocsr()

        rel_diff_A_both = dict()
        for idx, I_i in enumerate(I_both):
            X_row = X[[idx], :].todense()
            Y_row = Y[[idx], :].todense()
            rel_diff_A_both[I_i] = np.linalg.norm(X_row) / np.linalg.norm(Y_row)

        print('A: relative difference stats (rows in both matrices)')
        print_stats(list(rel_diff_A_both.values()))

        extreme = {k: v for k, v in rel_diff_A_both.items() if v >= args.epsilon}
        if len(extreme) > 0:
            print(f'{len(extreme)} relative differences larger than {args.epsilon} found')
            if args.extreme_path is not None:
                extreme_mat = args.extreme_path / f'extreme_{i:03d}.mat'
                m = {}
                m['i'], m['j'] = zip(*list(extreme.keys()))
                m['val'] = list(extreme.values())
                m['block_row1'] = [I1_map[x] for x in extreme.keys()]
                m['block_row2'] = [I2_map[x] for x in extreme.keys()]
                m['row1'] = [x + A1_mat['A_block_indptr'][i] for x in m['block_row1']]
                m['row2'] = [x + A2_mat['A_block_indptr'][i] for x in m['block_row2']]
                logger.info(f'Saving {extreme_mat}')
                scipy.io.savemat(extreme_mat, m)

        if args.plot_path is not None:
            # y difference plot
            y_rel_diff = np.full((M, M), np.nan)
            y_rel_diff.flat[np.ravel_multi_index(list(zip(*I_both)), (M, M))] = rel_diff_y_both

            ax_y_diff.set_title(f'Relative difference between y 1 and 2 for block {i}\n1: {args.A_mat1.name} 2: {args.A_mat2.name}')
            ax_y_diff.set_xlabel('j')
            ax_y_diff.set_ylabel('i')

            try:
                y_diff_im.set_data(y_rel_diff)
            except NameError:
                y_diff_im = ax_y_diff.imshow(y_rel_diff, interpolation='nearest', cmap='PiYG')
                fig_y_diff.colorbar(y_diff_im, ax=ax_y_diff)
            lim = max(np.abs(rel_diff_y_both))
            y_diff_im.set_clim(-lim, lim)

            y_rel_diff_fname = args.plot_path / f'y_diff_{i:03d}.pdf'
            logger.info(f'Saving {y_rel_diff_fname}')
            fig_y_diff.savefig(y_rel_diff_fname, bbox_inches='tight')

            if len(I1_only) > 0 or len(I2_only) > 0:
                # y missing values plot
                y_missing = np.full((M, M), np.nan)
                if len(I1_only) > 0:
                    y_missing.flat[np.ravel_multi_index(list(zip(*I1_only)), (M, M))] = 1
                if len(I2_only) > 0:
                    y_missing.flat[np.ravel_multi_index(list(zip(*I2_only)), (M, M))] = 2

                try:
                    y_missing_im.set_data(y_missing)
                except NameError:
                    y_missing_im = ax_y_missing.imshow(y_missing, interpolation='nearest', cmap=cmap_missing, norm=norm_missing)
                    fig_y_missing.colorbar(y_missing_im, ax=ax_y_missing, ticks=[0.5, 1.5], format=fmt_missing)

                ax_y_missing.set_title(f'Values in y found only in 1 and 2 for block {i}\n1: {args.A_mat1.name} 2: {args.A_mat2.name}')
                ax_y_missing.set_xlabel('j')
                ax_y_missing.set_ylabel('i')

                y_missing_fname = args.plot_path / f'y_missing_{i:03d}.pdf'
                logger.info(f'Saving {y_missing_fname}')
                fig_y_missing.savefig(y_missing_fname, bbox_inches='tight')

            # A relative difference histogram plot
            ax_A_hist.clear()
            ax_A_hist.hist(rel_diff_A_both.values(), bins=args.bins)
            ax_A_hist.set_xlabel('Relative difference')
            ax_A_hist.set_ylabel('Number')
            ax_A_hist.set_title(f'Relative difference between 1 and 2 rows for block {i}\n1: {args.A_mat1.name} 2: {args.A_mat2.name}')

            hist_fname = args.plot_path / f'A_hist_{i:03d}.pdf'
            logger.info(f'Saving {hist_fname}')
            fig_A_hist.savefig(hist_fname, bbox_inches='tight')

            # A relative difference image plot
            A_diff = np.full((M, M), np.nan)
            A_diff.flat[np.ravel_multi_index(list(zip(*rel_diff_A_both.keys())), (M, M))] = list(rel_diff_A_both.values())

            ax_A_diff.set_title(f'Relative difference between 1 and 2 rows for block {i}\n1: {args.A_mat1.name} 2: {args.A_mat2.name}')
            ax_A_diff.set_xlabel('j')
            ax_A_diff.set_ylabel('i')
            try:
                A_diff_im.set_data(A_diff)
            except NameError:
                A_diff_im = ax_A_diff.imshow(A_diff, interpolation='nearest', vmin=0, vmax=min(max(rel_diff_A_both.values()), args.epsilon), cmap='viridis_r')
                fig_A_diff.colorbar(A_diff_im, ax=ax_A_diff)
            A_diff_im.set_clim(0, min(max(rel_diff_A_both.values()), args.epsilon))

            A_diff_fname = args.plot_path / f'A_diff_{i:03d}.pdf'
            logger.info(f'Saving {A_diff_fname}')
            fig_A_diff.savefig(A_diff_fname, bbox_inches='tight')

            if len(extreme) > 0:
                # A extreme differences_plot
                A_extreme = np.full((M, M), np.nan)
                A_extreme.flat[np.ravel_multi_index(list(zip(*extreme.keys())), (M, M))] = list(extreme.values())
                try:
                    A_extreme_im.set_data(A_extreme)
                except NameError:
                    A_extreme_im = ax_A_extreme.imshow(A_extreme, interpolation='nearest', cmap='viridis_r')
                    fig_A_extreme.colorbar(A_extreme_im, ax=ax_A_extreme)
                A_extreme_im.set_clim(0, max(extreme.values()))
                ax_A_extreme.set_title(f'Relative differences between 1 and 2 rows $\\geq$ {args.epsilon} for block {i}\n1: {args.A_mat1.name} 2: {args.A_mat2.name}')
                ax_A_extreme.set_xlabel('j')
                ax_A_extreme.set_ylabel('i')

                A_extreme_fname = args.plot_path / f'A_extreme_{i:03d}.pdf'
                logger.info(f'Saving {A_extreme_fname}')
                fig_A_extreme.savefig(A_extreme_fname, bbox_inches='tight')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    sys.exit(main())
