#!/usr/bin/env python3

import sys
import logging
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from pathlib import Path

import numpy as np
import astropy.units as u
import sunpy.map
from sunpy.coordinates import Helioprojective
from sunpy.coordinates.frames import HeliographicCarrington
import hdf5storage


def main(argv=None):
    if argv is None:
        argv = sys.argv

    parser = ArgumentParser('Compute and store sun to obs and near point vectors for each pixel center (all in heliographic Carrington coordinates).',
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('mat_fname', type=Path)
    parser.add_argument('fts_fname', type=Path)
    args = parser.parse_args(argv[1:])


    fts_map = sunpy.map.Map(args.fts_fname)

    pixel_centers = sunpy.map.all_coordinates_from_map(fts_map)

    center_3d = Helioprojective(0*u.deg, 0*u.deg, 1*u.m,
                                observer=fts_map.observer_coordinate,
                                obstime=fts_map.date)
    pixel_centers_3d = Helioprojective(pixel_centers.Tx, pixel_centers.Ty, 1*u.m,
                                       observer=fts_map.observer_coordinate,
                                       obstime=fts_map.date)
    abs_cos_theta = center_3d.cartesian.dot(pixel_centers_3d.cartesian).to_value()

    nrpt = Helioprojective(pixel_centers.Tx, pixel_centers.Ty,
                           fts_map.observer_coordinate.cartesian.norm() * abs_cos_theta,
                           observer=fts_map.observer_coordinate,
                           obstime=fts_map.date)

    nrpt_hgc = nrpt.transform_to(HeliographicCarrington(observer=fts_map.observer_coordinate))

    obs_hgc = fts_map.observer_coordinate.transform_to(HeliographicCarrington(observer=fts_map.observer_coordinate))

    hdf5storage.savemat(args.mat_fname.as_posix(),
                        {'nrpt_hgc': nrpt_hgc.cartesian.get_xyz().to_value(),
                         'obs_hgc': obs_hgc.cartesian.get_xyz().to_value()})


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    sys.exit(main())
