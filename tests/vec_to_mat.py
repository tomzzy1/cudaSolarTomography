#!/usr/bin/env python3

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import sys
import logging
from pathlib import Path

import numpy as np
import scipy as sp

from to_mat import load_vector, parse_buildA_params

logger = logging.getLogger('vec_to_mat')


DTYPE_MAP = {'float32': np.float32,
             'int32':   np.int32}


def main(argv=None):
    if argv is None:
        argv = sys.argv

    parser = ArgumentParser('Convert a binary vector file to .mat format.',
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('mat_filename', type=Path, help='Output .mat filename.')
    parser.add_argument('mat_key', type=str, help='Key to store the vector in the .mat file.')
    parser.add_argument('vec_filename', type=Path, help='Binary vector filename.')
    parser.add_argument('--dtype', type=str, choices=DTYPE_MAP, default='float32', help='The numpy dtype of the data.')
    parser.add_argument('--buildA_params', '-b', type=Path, required=False, default=None, help='If specified, include hollow sphere parameters in the output .mat file.')
    args = parser.parse_args(argv[1:])

    x = load_vector(args.vec_filename, dtype=DTYPE_MAP[args.dtype])
    m = {args.mat_key: x}

    if args.buildA_params is not None:
        m.update(parse_buildA_params(args.buildA_params))

    logger.info(f'Saving to {args.mat_filename}')
    sp.io.savemat(args.mat_filename, m)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    sys.exit(main())
