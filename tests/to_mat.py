#!/usr/bin/env python3

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import logging
import sys
from pathlib import Path

import numpy as np
import scipy as sp

from scipy_io_util import savemat

logger = logging.getLogger('to_mat')


def load_sparse_csr(path, suffix, vdtype=np.float32, idtype=np.int32, N=None):
    """
    Load simple CSR format matrix stored in three files:
    *path*/w{*suffix*} (values), *path*/j{*suffix*} (column indices),
    and path *path*/m{*suffix*} (row index pointers). Values are
    stored as *vdtype* and indices as *idtype*. If given, ensure the
    returned matrix has *N* columns. Return a
    :class:`scipy.sparse.csr_array`.
    """
    w_path = path / f'w{suffix}'
    j_path = path / f'j{suffix}'
    m_path = path / f'm{suffix}'
    logger.info(f'Loading CSR matrix ({w_path}, {j_path}, {m_path})')

    w = np.fromfile(w_path, dtype=vdtype)
    j = np.fromfile(j_path, dtype=idtype)
    m = np.fromfile(m_path, dtype=idtype)

    assert len(w) == len(j)
    assert m[0] == 0 and m[-1] == len(w)

    if N is None:
        logger.info(f'M = {len(m)-1}, N = {max(j)+1}, nnz = {len(w)}')
        A = sp.sparse.csr_array((w, j, m))
    else:
        logger.info(f'M = {len(m)-1}, N = {N}, nnz = {len(w)}')
        A = sp.sparse.csr_array((w, j, m), shape=(len(m)-1, N))
    return A


def load_vector(path, dtype=np.float32):
    """
    Load a simple vector stored at *path*. The values are stored
    as *dtype*. Return a :class:`numpy.array`.
    """
    logger.info(f'Loading vector {path}')
    x = np.fromfile(path, dtype=dtype)
    logger.info(f'N = {len(x)}')
    return x


def load_blocks(path, suffix):
    """
    Return the first line of a builda info file located at
    *path*/info_*suffix* as a :type:`numpy.array`.
    """
    info_path = path / f'info_{suffix}'
    if info_path.exists():
        logger.info(f'Loading info file {info_path}')
        with open(info_path, mode='rb') as fid:
            line = fid.readline().rstrip(b'\n')
        blocks = np.frombuffer(line, dtype=np.int32)
    else:
        block_path = path / f'block_{suffix}'
        x = np.fromfile(block_path, dtype=np.int32)
        assert x[0] == len(x) - 2
        blocks = x[1:]
    return blocks


def parse_build_A_matrix_info(filename):
    """
    Parse C++ code info file *filename* and return a :type:`dict`
    of keys and values.
    """
    info = {}
    logger.info(f'Parsing C++ code {filename}')
    with open(filename) as fid:
        for line in fid:
            k, v = line.rstrip().split(' = ')
            if k in {'R_MIN', 'R_MAX', 'INSTR_R_MIN', 'INSTR_R_MAX'}:
                info[k] = float(v)
            elif k in {'FITS_PATH', 'CONFIG_FILE'}:
                info[k] = str(v).strip('"')
            else:
                info[k] = int(v)
    info['N_BINS'] = info['N_RAD_BINS'] * info['N_THETA_BINS'] * info['N_PHI_BINS']
    return info


def parse_tomroot(filename):
    """
    Parse C code tomroot.h file and return a :type:`str` with the
    path to the "tomography root path."
    """
    with open(filename) as fid:
        for line in fid:
            if line.startswith('#define'):
                _, k, v = line.split(' ')
                assert k == 'TOMROOT'
                return v.lstrip('"').rstrip('"\n')


CONVERT_MAP = {'RMAX':       ('R_MAX', float),
               'RMIN':       ('R_MIN', float),
               'NRAD':       ('N_RAD_BINS', int),
               'NTHETA':     ('N_THETA_BINS', int),
               'NPHI':       ('N_PHI_BINS', int),
               'BINFAC':     ('BINNING_FACTOR', int),
               'IMSIZE':     ('IMAGE_SIZE', int),
               'INSTR_RMIN': ('INSTR_R_MIN', float),
               'INSTR_RMAX': ('INSTR_R_MAX', float),
               'DATADIR':    ('FITS_PATH', str),
               'CONFSTRING': ('CONFIG_FILE', str)}


def parse_buildA_params(filename,
                        convert_map=CONVERT_MAP):
    """
    Parse C code buildA_params.h format file snippet *filename*
    and return a :type:`dict` of keys and values.
    """
    info = {}
    logger.info(f'Parsing C code {filename}')
    with open(filename) as fid:
        for line in fid:
            if line.startswith('#define'):
                try:
                    k, v, _ = line[8:].split(maxsplit=2)
                except ValueError:
                    k, v = line[8:].split(maxsplit=2)
                if k in convert_map:
                    k_prime, convert = convert_map[k]
                    info[k_prime] = convert(v)
    info['N_BINS'] = info['N_RAD_BINS'] * info['N_THETA_BINS'] * info['N_PHI_BINS']
    return info


def parse_buildA_params_and_tomroot(filename,
                                    tomroot_filename,
                                    convert_map=CONVERT_MAP):
    """
    Parse C code buildA_params.h format file snippet *filename*,
    tomroot.h header file *tomroot_filename*, and return a
    :type:`dict` of keys and values.
    """
    tomroot = parse_tomroot(tomroot_filename)
    info = parse_buildA_params(filename, convert_map=convert_map)
    #info['CONFIG_FILE'] = info['CONFIG_FILE'].replace('"', '').replace('TOMROOT', tomroot)
    #info['FITS_PATH'] = info['FITS_PATH'].replace('"', '').replace('TOMROOT', tomroot)
    return info


def main(argv=None):
    if argv is None:
        argv = sys.argv

    parser = ArgumentParser('Convert the output of build_A_matrix or builda to .mat format.',
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('mat_filename', type=Path, help='Output .mat filename.')
    parser.add_argument('matrix_path', type=Path, help='Path to input matrix.')
    parser.add_argument('suffix', type=str, default=None, nargs='?', help='Filename suffix (builda format if specified).')
    parser.add_argument('--exclude', action='store_true', help='Do not include y, y index, block index, and info')
    args = parser.parse_args(argv[1:])

    if args.suffix is None:
        # build_A_matrix (new C++ code)
        val = load_vector(args.matrix_path / '_val')
        col_index = load_vector(args.matrix_path / '_col_index', dtype=np.int32)
        row_index = load_vector(args.matrix_path / '_row_index', dtype=np.int32)
        assert len(val) == len(col_index)
        assert len(val) == row_index[-1]
        if args.exclude:
            A = sp.sparse.csr_array((val, col_index, row_index))
        else:
            y = load_vector(args.matrix_path / 'y_data')
            y_idx = load_vector(args.matrix_path / 'y_idx', dtype=np.int32)
            block_idx = load_vector(args.matrix_path / 'block_idx', dtype=np.int32)
            info = parse_build_A_matrix_info(args.matrix_path / 'info')
            A = sp.sparse.csr_array((val, col_index, row_index), shape=(len(y), info['N_BINS']))
    else:
        # builda (old C code)
        if args.exclude:
            A = load_sparse_csr(args.matrix_path, args.suffix)
        else:
            y = load_vector(args.matrix_path / f'y{args.suffix}')
            y_idx = load_vector(args.matrix_path / f'y_idx{args.suffix}', dtype=np.int32)
            block_idx = load_blocks(args.matrix_path, args.suffix)
            #info = parse_buildA_params_and_tomroot(args.matrix_path / 'buildA_params', args.matrix_path / 'tomroot.h')
            info = parse_buildA_params(args.matrix_path / 'buildA_params')
            info['N_BINS'] = info['N_RAD_BINS'] * info['N_THETA_BINS'] * info['N_PHI_BINS']
            A = load_sparse_csr(args.matrix_path, args.suffix, N=info['N_BINS'])

    if not args.exclude:
        assert len(y) == len(y_idx)
        assert A.shape[0] == len(y)

    m = {'A': A}
    if not args.exclude:
        m['y'] = y
        m['y_idx'] = y_idx
        m['A_block_indptr'] = block_idx
        m.update(info)

    logger.info(f'Saving to {args.mat_filename}')
    savemat(args.mat_filename, m)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    sys.exit(main())
