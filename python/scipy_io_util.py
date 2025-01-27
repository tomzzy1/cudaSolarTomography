import numpy as np
import scipy as sp


def savemat(file_name, mdict, **kwds):
    """
    """
    mdict_ = {}
    for k, v in mdict.items():
        if sp.sparse.issparse(v):
            if isinstance(v, sp.sparse._csr.csr_matrix) or isinstance(v, sp.sparse._csr.csr_array):
                mdict_[k + '_csr_data'] = v.data
                mdict_[k + '_csr_indices'] = v.indices
                mdict_[k + '_csr_indptr'] = v.indptr
                mdict_[k + '_csr_shape'] = v.shape
            else:
                raise NotImplementedError(f'{type(v)}')
        else:
            mdict_[k] = v
    return sp.io.savemat(file_name, mdict_, **kwds)


def loadmat(file_name, **kwds):
    """
    """
    mdict = sp.io.loadmat(file_name, **kwds)
    for csr_data_key in [x for x in mdict.keys() if x.endswith('_csr_data')]:
        prefix = csr_data_key[:-9]
        data = np.squeeze(mdict.pop(prefix + '_csr_data'))
        indices = np.squeeze(mdict.pop(prefix + '_csr_indices'))
        indptr = np.squeeze(mdict.pop(prefix + '_csr_indptr'))
        shape = np.squeeze(mdict.pop(prefix + '_csr_shape'))
        mdict[prefix] = sp.sparse.csr_array((data, indices, indptr), shape=shape)
    return mdict
