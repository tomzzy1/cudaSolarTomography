#!/usr/bin/env python3

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import sys
import logging
from pathlib import Path

import numpy as np
import scipy as sp
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import vtk

from pyvizvtk.viz import Renderer
from pyvizvtk.spherical_grid import spherical_voxel_actor
from pyvizvtk.line import line_actor

from scipy_io_util import loadmat
from grid import HollowSphere
from los import COR1LOS, get_fts_header_info


"""
It would be great to show multiple rows (2 at least) from the same point of fiew:
- Multiple renderers: https://examples.vtk.org/site/Python/Rendering/Model/
- Connected camera: https://discourse.vtk.org/t/synchronising-multiple-cameras/3860
- https://examples.vtk.org/site/Python/Utilities/ShareCamera/ (THIS ONE!!!)
"""



# GENERAL ROUTINE, put elsewhere and use in compare_A (and elsewhere?)
def index_map(mat, block):
    """
    """
    M = mat['IMAGE_SIZE'] // mat['BINNING_FACTOR']
    I = slice(mat['A_block_indptr'][block], mat['A_block_indptr'][block+1])
    y_idx = mat['y_idx'][I]
    return {k: v for v, k in enumerate(zip(*np.unravel_index(y_idx, (M, M))))}


def get_row(mat, block, i, j):
    """
    """
    I_map = index_map(mat, block)
    block_row_index = I_map[(i, j)]
    return mat['A'][[block_row_index + mat['A_block_indptr'][block]], :].tocsr()


def get_fts_filename(mat, block):
    """
    """
    with open(mat['CONFIG_FILE']) as fid:
        n = int(fid.readline())
        fts_filenames = [x.rstrip() for x in fid]
    assert len(fts_filenames) == n
    return Path(mat['FITS_PATH']) / fts_filenames[block]

    return Path(mat[f'fts_fname_{block:03d}'])


def get_los_voxels_actor(row, hollow_sphere, cmap=cm.viridis):
    """
    """
    voxels = []
    norm = Normalize(vmin=min(row.data), vmax=max(row.data))
    I, J, K = hollow_sphere.I2ijk(row.indices)
    for data_i, I_i, J_i, K_i in zip(row.data, I, J, K):
        r1 = hollow_sphere.rad_edges[I_i]
        r2 = hollow_sphere.rad_edges[I_i + 1]

        theta1 = np.radians(hollow_sphere.theta_edges[J_i])
        theta2 = np.radians(hollow_sphere.theta_edges[J_i + 1])

        phi1 = np.radians(hollow_sphere.phi_edges[K_i])
        phi2 = np.radians(hollow_sphere.phi_edges[K_i + 1])

        voxel = spherical_voxel_actor(r1,
                                      r2,
                                      theta1,
                                      theta2,
                                      phi1,
                                      phi2)

        voxel.GetProperty().SetOpacity(1)
        voxel.GetProperty().SetColor(*(cmap(norm(data_i))[:3]))
        voxels.append(voxel)
    return voxels


def get_los_line_actors(mat, header, i, j):
    """
    """
    line_actors = []
    for i_los in range(i * mat['BINNING_FACTOR'], (i + 1) * mat['BINNING_FACTOR']):
        for j_los in range(j * mat['BINNING_FACTOR'], (j + 1) * mat['BINNING_FACTOR']):
            los = COR1LOS(header, i_los, j_los)
            if los.impact > mat['R_MAX']:
                continue
            abstrmax = np.sqrt(mat['R_MAX']**2 - los.impact**2)
            t1 = -abstrmax
            t2 = abstrmax
            entry_xyz = los(t1)
            exit_xyz = los(t2)
            line_actors.append(line_actor(entry_xyz, exit_xyz))
    return line_actors


def main(argv=None):
    if argv is None:
        argv = sys.argv

    parser = ArgumentParser('',
                            formatter_class=ArgumentDefaultsHelpFormatter)
    # parser.add_argument('A_mat', type=Path, help='')
    # parser.add_argument('los_idx', type=int, nargs=3, help='')
    args = parser.parse_args(argv[1:])

    ### WINDOW A
    A_mat, block, i, j = 'mat/A_gold_SolarTom_prelim.mat', 1, 225, 192

    # block, i, j = args.los_idx

    m_A = loadmat(A_mat, squeeze_me=True)
    hollow_sphere = HollowSphere(m_A)

    breakpoint()

    fts_path = get_fts_filename(m_A, block)
    header = get_fts_header_info(fts_path)

    row = get_row(m_A, block, i, j)

    ren = Renderer(position_camera=False, size=(3200, 1600))
    ren.ren.SetViewport(0, 0, 0.5, 1)

    for voxel in get_los_voxels_actor(row, hollow_sphere):
        ren.add_actor(voxel)

    for line in get_los_line_actors(m_A, header, i, j):
        ren.add_actor(line)

    ren.orientation_on()


    ### WINDOW B
    B_mat = 'mat/A_gold_SolarTom_prelim_build_A.mat'

    # m_B = sp.io.loadmat(B_mat, squeeze_me=True)
    m_B = loadmat(B_mat, squeeze_me=True)
    # hollow_sphere = HollowSphere(m_A)

    # fts_path = get_fts_filename(m_A, block)
    # header = get_fts_header_info(fts_path)

    row = get_row(m_B, block, i, j)

    #ren = Renderer(position_camera=False, size=(3200, 1600))
    ren2 = vtk.vtkRenderer()
    ren.ren_win.AddRenderer(ren2)
    ren2.SetViewport(0.5, 0, 1, 1)

    for voxel in get_los_voxels_actor(row, hollow_sphere):
        ren2.AddActor(voxel)

    for line in get_los_line_actors(m_B, header, i, j):
        ren2.AddActor(line)

    ren2.SetBackground((0.3, 0.3, 0.3))
    ren2.SetActiveCamera(ren.camera)

    _om_axes = vtk.vtkAxesActor()
    _om = vtk.vtkOrientationMarkerWidget()
    _om.SetOrientationMarker(_om_axes)
    # Position lower right in the viewport.
    _om.SetViewport((0.8, 0, 1.0, 0.2))
    _om.SetInteractor(ren.iren)
    _om.SetDefaultRenderer(ren2)
    _om.EnabledOn()
    _om.InteractiveOn()

    # ren2.orientation_on()

    ren.camera.SetPosition(20, 20, 0)
    ren.camera.SetFocalPoint(0, 0, 0)
    ren.camera.SetViewUp(0, 0, 1)

    ren.start()



if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    sys.exit(main())
