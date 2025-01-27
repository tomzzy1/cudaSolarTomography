#!/usr/bin/env python3

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import sys
import logging
from pathlib import Path

import numpy as np
import scipy as sp

from scipy_io_util import loadmat

logger = logging.getLogger('reconstruct')


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
    logger.info(f'||y - Ax||^2 = {data_term}, ||{lambda_tik}*D x||^2 = {reg_term}, cost = {cost}')
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


NIT = 0

def callback(intermediate_result):
    """
    Simple :func:`scipy.optimize.minimize` callback function that
    logs the iteration count.
    """
    global NIT
    NIT += 1
    logger.info(f'# iterations = {NIT}')


def main(argv=None):
    if argv is None:
        argv = sys.argv

    parser = ArgumentParser('Compute solar tomography reconstruction x_hat = argmin_{x >= 0} ||y - Ax||^2 + ||\\lambda Dx||^2',
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('x_hat_filename', type=Path, help='Output .mat filename for reconstruction.')
    parser.add_argument('A_filename', type=Path, help='Path to A .mat file which includes observation matrix and y vector.')
    parser.add_argument('D_filename', type=Path, help='Path to D .mat file, i.e., the regularization matrix.')
    parser.add_argument('lambda_tik', type=float, help='Regularization parameter.')
    parser.add_argument('--x0', '-0', type=Path, help='Path to initial guess vector (or use x_0=0 if not specified).')
    parser.add_argument('--key', '-k', type=str, default='x_hat', help='Key to use to store the reconstruction in the .mat file.')
    parser.add_argument('--unbounded', '-u', action='store_true', help='Do not apply bounds constrained (otherwise, use non-negativity constraint).')

    args = parser.parse_args(argv[1:])

    mat = loadmat(args.A_filename, squeeze_me=True)

    A = mat['A']
    y = mat['y']

    D = loadmat(args.D_filename, squeeze_me=True)['D']

    assert A.shape[1] == D.shape[1]
    N = A.shape[1]

    if args.x0 is None:
        x0 = np.zeros(N)
    else:
        x0 = load_vector(args.x0)

    assert len(x0) == N

    if args.unbounded:
        bounds = None
    else:
        bounds = [(0, None) for _ in range(len(x0))]

    minimize_args = (y, A, D, args.lambda_tik)
    print(A.shape, A.nnz)
    NIT = 0
    res = sp.optimize.minimize(tikhonov, x0, minimize_args, jac=tikhonov_grad, bounds=None, callback=callback, method='L-BFGS-B')

    assert res.success
    logger.info(f'status = {res.status}')
    logger.info(res.message)

    ignore_keys = {'A_block_indptr', 'A', 'y', 'y_idx'}

    m = {args.key: res.x}
    for k, v in mat.items():
        if k not in ignore_keys:
            m[k] = v

    m['A_FILENAME'] = str(args.A_filename.absolute())
    m['D_FILENAME'] = str(args.D_filename.absolute())
    m['LAMBDA_TIK'] = args.lambda_tik
    m['x_hat'] = res.x
    m['x0'] =  'None' if args.x0 is None else str(args.x0.absolute())

    logger.info(f'Saving result to {args.x_hat_filename}')
    sp.io.savemat(args.x_hat_filename, m)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    sys.exit(main())
