#!/usr/bin/env python3

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import sys
import logging
from pathlib import Path
from functools import reduce

import h5py
import numpy as np
import scipy as sp

from scipy_io_util import savemat

logger = logging.getLogger('dgrad2')


"""
SolarTom Matlab functionality in derivs_hollowsph.m
"""

def lindex3D(row, col, block, nrows, ncols):
    '''
    Return the "linear index" associated with the 3-D index
    (*row*, *col*, *block*) for a matrix with blocks with *nrows*
    times *ncols* rows.
    '''
    return block * nrows * ncols + col * nrows + row

def scale_factor(scale, j, nrad):
    if not scale:
        return 1
    else:
        return 1 / (1 - j / nrad * 0.8)

def d2theta(nrad, ntheta, nphi, scale=False):
    """
    Return the 2nd theta derivative matrix for a grid with *nrad*
    x *ntheta* x *nphi* bins.
    """
    nbins = nrad * ntheta * nphi
    row_d2theta = np.zeros(3*nbins, dtype=int)
    col_d2theta = np.zeros_like(row_d2theta, dtype=int)
    val_d2theta = np.zeros_like(row_d2theta, dtype=float)
    nd2theta = np.zeros(nbins + 1, dtype=int)

    count = 0
    t_row_count = 0
    for k in range(nphi):
        for i in range(1, ntheta-1):
            for j in range(nrad):
                n = lindex3D(j, i, k, nrad, ntheta)
                row_d2theta[count] = t_row_count
                col_d2theta[count] = n
                val_d2theta[count] = -2 * scale_factor(scale, j, nrad)
                count += 1

                n = lindex3D(j, i-1, k, nrad, ntheta)
                row_d2theta[count] = t_row_count
                col_d2theta[count] = n
                val_d2theta[count] = 1 * scale_factor(scale, j, nrad)
                count += 1

                n = lindex3D(j, i+1, k, nrad, ntheta)
                row_d2theta[count] = t_row_count
                col_d2theta[count] = n
                val_d2theta[count] = 1 * scale_factor(scale, j, nrad)
                count += 1

                nd2theta[t_row_count + 1] = count

                t_row_count += 1
    return sp.sparse.coo_array((val_d2theta[:count],
                               (row_d2theta[:count], col_d2theta[:count])),
                               shape=(t_row_count, nbins))


def d2phi(nrad, ntheta, nphi, scale=False):
    """
    Return the 2nd phi derivative matrix for a grid with *nrad*
    x *ntheta* x *nphi* bins.
    """
    nbins = nrad * ntheta * nphi
    row_d2phi = np.zeros(3*nbins, dtype=int)
    col_d2phi = np.zeros_like(row_d2phi, dtype=int)
    val_d2phi = np.zeros_like(row_d2phi, dtype=float)
    nd2phi = np.zeros(nbins + 1, dtype=int)

    count = 0
    p_row_count = 0
    for k in range(1, nphi - 1):
        for i in range(ntheta):
            for j in range(nrad):
                m = p_row_count

                n = lindex3D(j, i, k-1, nrad, ntheta)
                row_d2phi[count] = m
                col_d2phi[count] = n
                val_d2phi[count] = 1 * scale_factor(scale, j, nrad)
                count += 1

                n = lindex3D(j, i, k, nrad, ntheta)
                row_d2phi[count] = m
                col_d2phi[count] = n
                val_d2phi[count] = -2 * scale_factor(scale, j, nrad)
                count += 1

                n = lindex3D(j, i, k+1, nrad, ntheta)
                row_d2phi[count] = m
                col_d2phi[count] = n
                val_d2phi[count] = 1 * scale_factor(scale, j, nrad)
                count += 1

                nd2phi[p_row_count + 1] = count

                p_row_count += 1

    k = nphi - 1
    for i in range(ntheta):
        for j in range(nrad):
            m = p_row_count

            n = lindex3D(j, i, k-1, nrad, ntheta)
            row_d2phi[count] = m
            col_d2phi[count] = n 
            val_d2phi[count] = 1 * scale_factor(scale, j, nrad)
            count += 1

            n = lindex3D(j, i, k, nrad, ntheta)
            row_d2phi[count] = m
            col_d2phi[count] = n
            val_d2phi[count] = -2 * scale_factor(scale, j, nrad)
            count += 1

            n = lindex3D(j, i, 0, nrad, ntheta)
            row_d2phi[count] = m
            col_d2phi[count] = n
            val_d2phi[count] = 1 * scale_factor(scale, j, nrad)
            count += 1

            nd2phi[p_row_count + 1] = count

            p_row_count += 1

    k = 0
    for i in range(ntheta):
        for j in range(nrad):
            m = p_row_count

            n = lindex3D(j, i, nphi-1, nrad, ntheta)
            row_d2phi[count] = m
            col_d2phi[count] = n
            val_d2phi[count] = 1 * scale_factor(scale, j, nrad)
            count += 1

            n = lindex3D(j, i, k, nrad, ntheta)
            row_d2phi[count] = m
            col_d2phi[count] = n
            val_d2phi[count] = -2 * scale_factor(scale, j, nrad)
            count += 1

            n = lindex3D(j, i, k+1, nrad, ntheta)
            row_d2phi[count] = m
            col_d2phi[count] = n
            val_d2phi[count] = 1 * scale_factor(scale, j, nrad)
            count += 1

            nd2phi[p_row_count + 1] = count

            p_row_count += 1

    return sp.sparse.coo_array((val_d2phi[:count],
                               (row_d2phi[:count], col_d2phi[:count])),
                               shape=(p_row_count, nbins))

def d2r(nrad, ntheta, nphi, scale=False):
    """
    Return the 2nd rad derivative matrix for a grid with *nrad*
    x *ntheta* x *nphi* bins.
    """
    nbins = nrad * ntheta * nphi
    row_d2r = np.zeros(3*nbins, dtype=int)
    col_d2r = np.zeros_like(row_d2r, dtype=int)
    val_d2r = np.zeros_like(row_d2r, dtype=float)
    nd2r = np.zeros(nbins + 1, dtype=int)

    count = 0
    r_row_count = 0
    for k in range(nphi):
        for i in range(ntheta):
            for j in range(1, nrad - 1):
                m = r_row_count

                n = lindex3D(j - 1, i, k, nrad, ntheta)
                row_d2r[count] = m
                col_d2r[count] = n
                val_d2r[count] = 1 * scale_factor(scale, j, nrad)
                count += 1

                n = lindex3D(j, i, k, nrad, ntheta)
                row_d2r[count] = m
                col_d2r[count] = n
                val_d2r[count] = -2 * scale_factor(scale, j, nrad)
                count += 1

                n = lindex3D(j + 1, i, k, nrad, ntheta)
                row_d2r[count] = m
                col_d2r[count] = n
                val_d2r[count] = 1 * scale_factor(scale, j, nrad)
                count += 1

                nd2r[r_row_count + 1] = count

                r_row_count += 1

    return sp.sparse.coo_array((val_d2r[:count],
                               (row_d2r[:count], col_d2r[:count])),
                               shape=(r_row_count, nbins))


def hlaplac(nrad, ntheta, nphi):
    """
    """
    return sp.sparse.vstack((d2theta(nrad, ntheta, nphi),
                             d2phi(nrad, ntheta, nphi))).tocsr()

def r3(nrad, ntheta, nphi):
    return sp.sparse.vstack((d2theta(nrad, ntheta, nphi, True),
                             d2phi(nrad, ntheta, nphi, True),
                             d2r(nrad, ntheta, nphi, True))).tocsr()


"""
Filter approximation to derivative functionality
"""

def zero_pad(x, n):
    """
    Return *x* zero padded to the dimensions specified in *n*.
    """
    assert len(n) == x.ndim
    return np.pad(x,
                  [(0, n_i - s_i) for n_i, s_i in zip(n, x.shape)],
                  'constant')


def differentiator(n, fs=1):
    r"""
    Return linear phase impulse response for a length *n* filter that
    approximates the differential operator. The sampling frequency is
    *fs*.

    The filter length *n* must be even. The remez function returns
    type 3 (for *n* odd) and 4 (for *n* even) linear phase
    filters. However, type 3 linear phase filters are 0 at $\omega =
    0$ and $\omega = \pi$.
    """
    if n % 2 == 1:
        raise ValueError('the filter length n must be even')
    return sp.signal.remez(n,
                           [0, fs / 2],
                           [1],
                           fs=fs,
                           type='differentiator') * fs * 2 * np.pi


# WHY IS format='coo' ?!?

#class Convmtx(sp.sparse.coo_matrix):
class Convmtx(sp.sparse.sparray):
    def __new__(cls, n, H, mode='full'):
        """
        Construct sparse convolution matrix to operate on vector of
        dimension *n* with the kernel *H*. The *mode* parameter can be
        one of:

        - full: standard convolution, i.e., zero-padding at the edges.

        - valid: convolution where only those portions of complete
          overlap, i.e., no zero-padding, are considered.

        - circ: circular convolution, i.e., periodic boundary
          condition at the edges.
        """
        def toeplitz_mapper_full(h):
            if (h == 0).all():
                return sp.sparse.coo_matrix((k[-1], n[-1]))
            else:
                c = h
                r = np.array([c[0]] + [0]*(n[-1]-1))
                return sp.sparse.coo_matrix(sp.linalg.toeplitz(c, r))

        def toeplitz_mapper_valid(h):
            if (h == 0).all():
                return sp.sparse.coo_matrix((k[-1], n[-1]))
            else:
                r = np.zeros(n[-1])
                r[:len(h)] = h
                c = np.zeros(k[-1])
                c[0] = r[0]
                return sp.sparse.coo_matrix(sp.linalg.toeplitz(c, r))

        def toeplitz_mapper_circ(h):
            if (h == 0).all():
                return sp.sparse.coo_matrix((k[-1], n[-1]))
            else:
                c = h
                r = np.zeros(n[-1])
                r[0] = c[0]
                r[1:] = h[:0:-1]
                return sp.sparse.coo_matrix(sp.linalg.toeplitz(c, r))

        def block_mapper_full(n, k, blocks):
            c = [blocks[i] for i in range(k)]
            r = [c[0]] + [None]*(n-1)
            #return sp.sparse.bmat(sp.linalg.toeplitz(c, r).tolist(), format='coo')
            return sp.sparse.bmat(sp.linalg.toeplitz(c, r).tolist())

        def block_mapper_valid(n, k, blocks):
            r = []
            for i in range(n):
                if (n - k - i < 0):
                    r.append(None)
                else:
                    r.append(blocks[n - k - i])

            c = []
            for i in range(n-k, n):
                c.append(blocks[i])

            #return sp.sparse.bmat(sp.linalg.toeplitz(c, r).tolist(), format='coo')
            return sp.sparse.bmat(sp.linalg.toeplitz(c, r).tolist())

        def block_mapper_circ(n, k, blocks):
            c = [blocks[i] for i in range(k)]
            r = []
            r.append(blocks[0])
            r.extend(blocks[:0:-1])
            #return sp.sparse.bmat(sp.linalg.toeplitz(c, r).tolist(), format='coo')
            return sp.sparse.bmat(sp.linalg.toeplitz(c, r).tolist())

        m = H.shape

        if mode == 'full':
            k = tuple(np.array(n) + np.array(m) - 1)
            toeplitz_mapper = toeplitz_mapper_full
            block_mapper = block_mapper_full

            H_zp = zero_pad(H, k)
            c_list = np.split(H_zp.flatten(), np.prod(k[:-1]))
        elif mode == 'valid':
            k = tuple(np.array(n) - np.array(m) + 1)
            toeplitz_mapper = toeplitz_mapper_valid
            block_mapper = block_mapper_valid

            H_zp = zero_pad(H[...,::-1], n)
            c_list = np.split(H_zp.flatten(), np.prod(n[:-1]))
        elif mode == 'circ':
            assert (np.array(m) <= np.array(n)).all()
            k = n
            toeplitz_mapper = toeplitz_mapper_circ
            block_mapper = block_mapper_circ

            H_zp = zero_pad(H, k)
            c_list = np.split(H_zp.flatten(), np.prod(k[:-1]))
        else:
            raise ValueError('Unknown mode {0}'.format(mode))

        blocks = [toeplitz_mapper(x) for x in c_list]

        for n_i, k_i in zip(n[-2::-1], k[-2::-1]):
            if mode == 'full' or mode == 'circ':
                blocks = [block_mapper(n_i, k_i, x) for x in np.split(np.array(blocks), len(blocks)/k_i)]
            elif mode =='valid':
                blocks = [block_mapper(n_i, k_i, x) for x in np.split(np.array(blocks), len(blocks)/n_i)]
            else:
                raise ValueError('Unknown mode {0}'.format(mode))

        return blocks[0]


class SepFilter():
    def __init__(self, n, h_list, mode='full'):
        """
        Construct a separable filter, i.e., the filter that operates on a
        signal with shape *n* where the $i$th component of *h_list* is
        the kernel for the $i$th dimension. The *mode* parameter can be
        one of:

        - full: standard convolution, i.e., zero-padding at the edges.

        - valid: convolution where only those portions of complete
          overlap, i.e., no zero-padding, are considered.

        - circ: circular convolution, i.e., periodic boundary
          condition at the edges.
        """
        self.n = n
        self.ndim = len(n)
        self.h_list = [np.copy(h_i) for h_i in h_list]
        self.m = tuple(map(lambda x: len(x), self.h_list))
        self.mode = mode
        self.k_full = tuple(np.array(self.n) + np.array(self.m) - 1)
        self.k_valid = tuple(np.array(self.n) - np.array(self.m) + 1)
        if self.mode == 'full':
            self.k = self.k_full
        elif self.mode == 'valid':
            self.k = self.k_valid
        elif self.mode == 'circ':
            self.k = self.k_full
        else:
            assert False
        self.H_list = list(map(lambda x: sp.fft.fft(x[1],
                                                    n=self.k_full[x[0]]),
                               enumerate(self.h_list)))

        def reducer(x, y):
            i, y_i = y
            shape = [len(y_i)] + [1]*i
            return np.reshape(y_i, shape) * x

        self.h = reduce(reducer, enumerate(reversed(self.h_list)), 1)
        self.H = reduce(reducer, enumerate(reversed(self.H_list)), 1)


    def operate(self, x):
        """
        Apply the separable filter to the signal vector *x*.
        """
        X = sp.fft.fftn(x, s=self.k_full)
        if np.isrealobj(self.h) and np.isrealobj(x):
            y = np.real(sp.fft.ifftn(self.H * X))
        else:
            y = sp.fft.ifftn(self.H * X)

        if self.mode == 'full' or self.mode == 'circ':
            return y
        elif self.mode == 'valid':
            slice_list = []
            for i in range(self.ndim):
                if self.m[i]-1 == 0:
                    slice_list.append(slice(None, None, None))
                else:
                    slice_list.append(slice(self.m[i]-1, -(self.m[i]-1), None))
            # return y[*slice_list]
            # python 3.10 workaround
            return y[slice_list[:, 0], slice_list[:, 1], slice_list[:, 2]]
        else:
            assert False


    def __matmul__(self, x):
        """
        Apply the separable filter (via the matrix multiplication
        operator) to the signal vector *x*.
        """
        y = self.operate(x)
        return y.reshape(np.prod(y.shape))


    def asmatrix(self):
        """
        Return the sparse matrix representation of the separable filter.
        """
        h_matrix = np.array([1])
        for i in range(self.ndim):
            if self.mode == 'circ':
                h_i = Convmtx([self.k[i]], self.h_list[i], mode=self.mode)
            else:
                h_i = Convmtx([self.n[i]], self.h_list[i], mode=self.mode)
            h_matrix = sp.sparse.kron(h_matrix, h_i)
        return h_matrix


class Gradient2Filter(SepFilter):
    def __init__(self, n, axis, order=3, mode='valid'):
        """
        Construct a second-order gradient operator for signal of dimension
        *n* for dimension *axis*. Use a filter kernel of length
        *order* (must be odd). Use convolution type *mode*.
        """
        # assert that the filter length is odd
        assert(order % 2 == 1)
        self.n = n
        self.ndim = len(self.n)
        self.axis = axis
        if axis < 0 or axis >= self.ndim:
            raise ValueError('0 <= axis (= {0}) < ndim = {1}'.format(axis, self.ndim))

        self.d = differentiator(int(order/2) + 1)
        self.d2 = np.convolve(self.d, self.d)

        self.mode = mode

        h_list = []
        m = []
        for i in reversed(range(self.ndim)):
            if i == axis:
                h_list.append(self.d2)
            else:
                h_list.append(np.array([1]))
            m.append(len(h_list[-1]))
        self.m = m

        if mode == 'circ':
            n_prime = np.array(n) - m + 1
            super(Gradient2Filter, self).__init__(n_prime, h_list, mode=mode)
        else:
            super(Gradient2Filter, self).__init__(n, h_list, mode=mode)


def main(argv=None):
    if argv is None:
        argv = sys.argv

    parser = ArgumentParser('Generate regularization matrix for solar tomography, i.e., 2nd order theta and phi gradient approximations.',
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('mat_filename', type=Path, help='Output .mat filename.')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--mat', '-m', type=Path, default=None, help='Observation matrix file (use N_RAD_BINS, N_THETA_BINS, and N_PHI_BINS keys)')
    group.add_argument('-n', type=int, nargs=3, default=None, help='N_RAD_BIN, N_THETA_BINS, and N_PHI_BINS')
    parser.add_argument('--key', '-k', type=str, default='A', help='Key to use to store the dgrad2 matrix in the .mat file.')
    parser.add_argument('--order', '-o', type=int, nargs='+', default=[3, 3], help='Filter order (use the same filter for the theta and phi matrices if only one argument is specified, i.e., 1 or 2 arguments are accepted)')
    parser.add_argument('--derivs-hollowsph', '-d', action='store_true', help='Use the derivs_hollowsph.m implementation (instead of the optimal derivative filter implementation). The --order option is ignored.')
    parser.add_argument('--derivs-hollowsph-r3', '-r', action='store_true', help='Use the derivs_hollowsph.m implementation with radial derivative. The --order option is ignored.')

    args = parser.parse_args(argv[1:])

    if args.mat is not None:
        with h5py.File(args.mat, 'r') as h5_mat:
            N_RAD_BINS   = int(h5_mat['N_RAD_BINS'][:].squeeze())
            N_THETA_BINS = int(h5_mat['N_THETA_BINS'][:].squeeze())
            N_PHI_BINS   = int(h5_mat['N_PHI_BINS'][:].squeeze())
            logger.info(f'{N_RAD_BINS=} {N_THETA_BINS=} {N_PHI_BINS=}')
            n = (N_RAD_BINS, N_THETA_BINS, N_PHI_BINS)
    elif args.n is not None:
        n = args.n
        N_RAD_BINS, N_THETA_BINS, N_PHI_BINS = n[0], n[1], n[2]
    else:
        assert(False)

    if len(args.order) == 1:
        theta_order = args.order[0]
        phi_order = args.order[0]
    elif len(args.order) == 2:
        theta_order, phi_order = args.order
    else:
        assert(False)

    if args.derivs_hollowsph:
        D = hlaplac(*n)
    elif args.derivs_hollowsph_r3:
        D = r3(*n)
    else:
        d2theta = Gradient2Filter(n[::-1], axis=1, order=theta_order, mode='circ')
        d2phi = Gradient2Filter(n[::-1], axis=2, order=phi_order, mode='circ')
        D = sp.sparse.vstack((d2theta.asmatrix(), d2phi.asmatrix())).tocsr()

    #D = sp.sparse.vstack((d2theta.asmatrix(), d2phi.asmatrix())).tocsc()
    # GPU code use CSR
    #D = sp.sparse.vstack((d2theta.asmatrix(), d2phi.asmatrix())).tocsr()

    # D = hlaplac(*n)
    print(D.data.dtype, D.indices.dtype, D.indptr.dtype)
    if not args.derivs_hollowsph_r3:
        prefix = 'hlaplac'
    else:
        prefix = 'r3'
    D.data.astype(np.float32).tofile(f"w{prefix}_{n[0]}_{n[1]}_{n[2]}")
    D.indices.tofile(f"j{prefix}_{n[0]}_{n[1]}_{n[2]}")
    D.indptr.tofile(f"m{prefix}_{n[0]}_{n[1]}_{n[2]}")
    savemat(args.mat_filename, {'D': D,
                                'N_RAD_BINS':   N_RAD_BINS,
                                'N_THETA_BINS': N_THETA_BINS,
                                'N_PHI_BINS':   N_PHI_BINS})


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    sys.exit(main())
