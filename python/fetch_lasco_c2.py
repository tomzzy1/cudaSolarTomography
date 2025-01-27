#!/usr/bin/env python3

import sys
import logging
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from pathlib import Path
from tempfile import TemporaryDirectory
import dataclasses
from dataclasses import dataclass
from datetime import datetime
from functools import partial
from collections import OrderedDict

import dateutil
import requests
import astropy.io.fits
from astropy.coordinates import HeliocentricMeanEcliptic
from astropy import units as u
from sunpy.map import Map
from sunpy.coordinates.frames import HeliographicCarrington, HeliographicStonyhurst
import numpy as np


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('fetch_lasco_c2')


"""
Also https://sscweb.gsfc.nasa.gov/users_guide/Appendix_C.shtml#:~:text=Geocentric%20Inertial%20(GCI)%20and%20Earth,Greenwich%20meridian%20(0%20longitude).

The Orbit data will describe the position and motion of the spacecraft, and it will be available in several coordinate systems including: geocentric inertial (GCI or ECI or GEI: GeocentricEarthEquatorial) coordinates for the J2000 system; geocentric solar ecliptic (GSE, GeocentricSolarEcliptic); geocentric solar magnetospheric (GSM, GeocentricSolarMagnetospheric) coordinates; and Heliocentric Ecliptic coordinate system (HAE or HEC: HeliocentricMeanEcliptic --- this has the fewest steps to HeliographicCarrington).

"""


INDEX_PATH = Path('../data/lasco_c2/index')

PB_TEMPLATE = 'http://idoc-lasco.ias.u-psud.fr/sitools/datastorage/user/results/kfcorona_sph_new_optimized/C2/Orange/{dt:%Y}/{pB_fname}'

ORBIT_DIR_TEMPLATE = 'https://soho.nascom.nasa.gov/data/ancillary/orbit/predictive/{dt:%Y}'


def __post_init__(self):
    # https://stackoverflow.com/questions/54863458/force-type-conversion-in-python-dataclass-init-method
    for field in dataclasses.fields(self):
        value = getattr(self, field.name)
        if not isinstance(value, field.type):
            setattr(self, field.name, field.type(value))
            # raise ValueError(f'Expected {field.name} to be {field.type}, '
                             # f'got {repr(value)}')


def autoconvert_dataclass(x):
    x.__post_init__ = __post_init__
    y = dataclass(x)
    return y


"""
These are the fields in the SOHO orbit files. Also reference https://soho.nascom.nasa.gov/data/ancillary/

XTENSION= 'BINTABLE'           /Written by IDL:  Sun Dec 16 04:00:58 2007
BITPIX  =                    8 /
NAXIS   =                    2 /Binary table
NAXIS1  =                  260 /Number of bytes per row
NAXIS2  =                  144 /Number of rows
PCOUNT  =                    0 /Random parameter count
GCOUNT  =                    1 /Group count
TFIELDS =                   36 /Number of columns
TFORM1  = '1I      '           /Integer*2 (short integer)
TTYPE1  = 'YEAR    '           /Label for column 1
TFORM2  = '1I      '           /Integer*2 (short integer)
TTYPE2  = 'DAY OF YEAR'        /Label for column 2
TFORM3  = '1J      '           /Integer*4 (long integer)
TTYPE3  = 'ELLAPSED MILLISECONDS OF DAY' /Label for column 3
TFORM4  = '1D      '           /Real*8 (double precision)
TTYPE4  = 'GCI X (KM)'         /Label for column 4
TFORM5  = '1D      '           /Real*8 (double precision)
TTYPE5  = 'GCI Y (KM)'         /Label for column 5
TFORM6  = '1D      '           /Real*8 (double precision)
TTYPE6  = 'GCI Z (KM)'         /Label for column 6
TFORM7  = '1D      '           /Real*8 (double precision)
TTYPE7  = 'GCI VX (KM/S)'      /Label for column 7
TFORM8  = '1D      '           /Real*8 (double precision)
TTYPE8  = 'GCI VY (KM/S)'      /Label for column 8
TFORM9  = '1D      '           /Real*8 (double precision)
TTYPE9  = 'GCI VZ (KM/S)'      /Label for column 9
TFORM10 = '1D      '           /Real*8 (double precision)
TTYPE10 = 'GSE X (KM)'         /Label for column 10
TFORM11 = '1D      '           /Real*8 (double precision)
TTYPE11 = 'GSE Y (KM)'         /Label for column 11
TFORM12 = '1D      '           /Real*8 (double precision)
TTYPE12 = 'GSE Z (KM)'         /Label for column 12
TFORM13 = '1D      '           /Real*8 (double precision)
TTYPE13 = 'GSE VX (KM/S)'      /Label for column 13
TFORM14 = '1D      '           /Real*8 (double precision)
TTYPE14 = 'GSE VY (KM/S)'      /Label for column 14
TFORM15 = '1D      '           /Real*8 (double precision)
TTYPE15 = 'GSE VZ (KM/S)'      /Label for column 15
TFORM16 = '1D      '           /Real*8 (double precision)
TTYPE16 = 'GSM X (KM)'         /Label for column 16
TFORM17 = '1D      '           /Real*8 (double precision)
TTYPE17 = 'GSM Y (KM)'         /Label for column 17
TFORM18 = '1D      '           /Real*8 (double precision)
TTYPE18 = 'GSM Z (KM)'         /Label for column 18
TFORM19 = '1D      '           /Real*8 (double precision)
TTYPE19 = 'GSM VX (KM/S)'      /Label for column 19
TFORM20 = '1D      '           /Real*8 (double precision)
TTYPE20 = 'GSM VY (KM/S)'      /Label for column 20
TFORM21 = '1D      '           /Real*8 (double precision)
TTYPE21 = 'GSM VZ (KM/S)'      /Label for column 21
TFORM22 = '1D      '           /Real*8 (double precision)
TTYPE22 = 'SUN VECTOR X (KM)'  /Label for column 22
TFORM23 = '1D      '           /Real*8 (double precision)
TTYPE23 = 'SUN VECTOR Y (KM)'  /Label for column 23
TFORM24 = '1D      '           /Real*8 (double precision)
TTYPE24 = 'SUN VECTOR Z (KM)'  /Label for column 24
TFORM25 = '1D      '           /Real*8 (double precision)
TTYPE25 = 'HEC X (KM)'         /Label for column 25
TFORM26 = '1D      '           /Real*8 (double precision)
TTYPE26 = 'HEC Y (KM)'         /Label for column 26
TFORM27 = '1D      '           /Real*8 (double precision)
TTYPE27 = 'HEC Z (KM)'         /Label for column 27
TFORM28 = '1D      '           /Real*8 (double precision)
TTYPE28 = 'HEC VX (KM/S)'      /Label for column 28
TFORM29 = '1D      '           /Real*8 (double precision)
TTYPE29 = 'HEC VY (KM/S)'      /Label for column 29
TFORM30 = '1D      '           /Real*8 (double precision)
TTYPE30 = 'HEC VZ (KM/S)'      /Label for column 30
TFORM31 = '1I      '           /Integer*2 (short integer)
TTYPE31 = 'CARRINGTON ROTATION EARTH' /Label for column 31
TFORM32 = '1D      '           /Real*8 (double precision)
TTYPE32 = 'HELIOGRAPHIC LONG. EARTH' /Label for column 32
TFORM33 = '1D      '           /Real*8 (double precision)
TTYPE33 = 'HELIOGRAPHIC LAT. EARTH' /Label for column 33
TFORM34 = '1I      '           /Integer*2 (short integer)
TTYPE34 = 'CARRINGTON ROTATION SOHO' /Label for column 34
TFORM35 = '1D      '           /Real*8 (double precision)
TTYPE35 = 'HELIOGRAPHIC LONG. SOHO' /Label for column 35
TFORM36 = '1D      '           /Real*8 (double precision)
TTYPE36 = 'HELIOGRAPHIC LAT. SOHO' /Label for column 36
END
"""


@autoconvert_dataclass
class DAT:
    date: str
    time: str
    year: int
    doy: int
    ellapsed_ms: int
    gci_x_km: float
    gci_y_km: float
    gci_z_km: float
    gci_vx_kms: float
    gci_vy_kms: float
    gci_vz_kms: float
    gse_x_km: float
    gse_y_km: float
    gse_z_km: float
    gse_vx_kms: float
    gse_vy_kms: float
    gse_vz_kms: float
    gsm_x_km: float
    gsm_y_km: float
    gsm_z_km: float
    gsm_vx_kms: float
    gsm_vy_kms: float
    gsm_vz_kms: float
    gci_sun_vector_x_km: float
    gci_sun_vector_y_km: float
    gci_sun_vector_z_km: float
    hec_x_km: float
    hec_y_km: float
    hec_z_km: float
    hec_vx_kms: float
    hec_vy_kms: float
    hec_vz_kms: float
    cr_earth: int
    heliographic_lon_earth: float
    heliographic_lat_earth: float
    cr_soho: int
    heliographic_lon_soho: float
    heliographic_lat_soho: float



def fetch_file_or_use_cached(cache_path, fname, base_url, **kwds):
    """
    """
    local_fname = (cache_path / fname).absolute()
    if local_fname.exists():
        logger.info(f'Using cached file {local_fname}')
    else:
        remote_fname = base_url.format(**kwds)
        logger.info(f'Fetching {remote_fname} and storing to {local_fname}')
        r = requests.get(remote_fname)
        if r.status_code != 200:
            raise RuntimeError(f'status code = {r.status_code}')
        with open(local_fname, 'wb') as fid:
            fid.write(r.content)
    return local_fname


def find_closest_fts(index, target_dt):
    """
    """
    dts = index.keys()
    closest_dt = min(dts, key=lambda x: (x > target_dt, abs(x - target_dt)))
    return index[closest_dt].name[:-4], closest_dt, closest_dt - target_dt


def find_fts_in_range(index, dt1, dt2):
    """
    """
    return [(path.name[:-4], dt) for dt, path in index.items() if dt1 <= dt <= dt2]


def parse_index(index_path):
    """
    """
    index = {}
    for fname in sorted(index_path.glob('**/*.fts.hdr')):
        date_obs = None
        time_obs = None
        with open(fname) as fid:
            for line in fid:
                if 'DATE_OBS' in line:
                    date_obs = line.split('=')[1].split("'")[1]
                elif 'TIME_OBS' in line:
                    time_obs = line.split('=')[1].split("'")[1]
        assert date_obs is not None and time_obs is not None
        dt = datetime.strptime(date_obs + ' ' + time_obs + '000', '%Y/%m/%d %H:%M:%S.%f')
        index[dt] = fname
    return index


def fetch_pB(cache_path, fts_fname, dt, pB_template=PB_TEMPLATE):
    """
    """
    return fetch_file_or_use_cached(cache_path, fts_fname, pB_template, dt=dt, pB_fname=fts_fname)


def find_dat_file(cache_path, date, local_fname_template='NRL_orbits_{dt:%Y}.txt', orbit_dir_template=ORBIT_DIR_TEMPLATE):
    """
    """
    local_fname = fetch_file_or_use_cached(cache_path, local_fname_template.format(dt=date),
                                           orbit_dir_template, dt=date)
    orbit_dir = orbit_dir_template.format(dt=date)

    dat_files = []
    with open(local_fname) as fid:
        for line in fid:
            if date.strftime('%Y%m%d') in line and '.DAT' in line:
                dat_file = orbit_dir + '/' + line.split('href="')[1].split('"')[0]
                logger.info(f'Found {dat_file}')
                dat_files.append(dat_file)
        if len(dat_files) > 0:
            return max(dat_files, key=lambda x: int(x[-6:-4]))
    raise ValueError(f'{date} not found in {orbit_dir}')


def fetch_dat_file(cache_path, remote_dat_fname):
    """
    """
    return fetch_file_or_use_cached(cache_path, remote_dat_fname.split('/')[-1], remote_dat_fname)


def parse_dat_file(dat_fname):
    """
    """
    with open(dat_fname) as fid:
        orbit_dat = [DAT(*x.split()) for x in fid]
    return orbit_dat


def find_closest_orbit(orbit_dat, target_dt):
    """
    """
    dts = [datetime.strptime(x.date + ' ' + x.time + '000', '%d-%b-%Y %H:%M:%S.%f') for x in orbit_dat]
    i, closest_dt = min(enumerate(dts), key=lambda x: (x[1] > target_dt, abs(x[1] - target_dt)))
    return orbit_dat[i], closest_dt - target_dt


def create_pB_map(pB_fname,
                  orbit,
                  CRVAL1=1.00000, CRVAL2=1.00000,
                  CDELT1=23.799999, CDELT2=23.799999,
                  CTYPE1='HPLN-TAN', CTYPE2='HPLT-TAN',
                  CUNIT1='arcsec  ', CUNIT2='arcsec  ',
                  RSUN_KM=695990,
                  RSUN_ARCSEC=959.63,
                  SCALE=1e-10):
    """
    RSUN is documented here under Important Warning Concerning Units: http://idoc-lasco.ias.u-psud.fr/sitools/client-portal/doc/
    """
    hdul = astropy.io.fits.open(pB_fname)
    assert len(hdul) == 1


    date_obs = datetime.strptime(orbit.date, '%d-%b-%Y').strftime('%Y-%m-%d') + 'T' + orbit.time
    soho_hae = HeliocentricMeanEcliptic(x=orbit.hec_x_km * u.km,
                                        y=orbit.hec_y_km * u.km,
                                        z=orbit.hec_z_km * u.km,
                                        representation_type='cartesian',
                                        obstime=date_obs)

    soho_hgs = soho_hae.transform_to(HeliographicStonyhurst(obstime=date_obs))

    soho_hgc = soho_hae.transform_to(HeliographicCarrington(obstime=date_obs,
                                                            observer='self'))

    # WCS parameters
    hdul[0].header['CRPIX1'] = hdul[0].header['XSUN']
    hdul[0].header['CRPIX2'] = hdul[0].header['YSUN']
    hdul[0].header['CRVAL1'] = CRVAL1
    hdul[0].header['CRVAL2'] = CRVAL2
    hdul[0].header['CDELT1'] = CDELT1
    hdul[0].header['CDELT2'] = CDELT2
    hdul[0].header['CROTA1'] = hdul[0].header['ROLLANGL']
    hdul[0].header['CROTA2'] = hdul[0].header['ROLLANGL']
    hdul[0].header['CROTA']  = hdul[0].header['ROLLANGL']
    hdul[0].header['CTYPE1'] = CTYPE1
    hdul[0].header['CTYPE2'] = CTYPE2
    hdul[0].header['CUNIT1'] = CUNIT1
    hdul[0].header['CUNIT2'] = CUNIT2

    # Position
    hdul[0].header['RSUN_REF'] = RSUN_KM * 1e3
    hdul[0].header['RSUN'] = RSUN_ARCSEC
    hdul[0].header['CRLN_OBS'] = soho_hgc.lon.to('deg').value
    hdul[0].header['CRLT_OBS'] = soho_hgc.lat.to('deg').value
    hdul[0].header['HGLN_OBS'] = soho_hgs.lon.to('deg').value
    hdul[0].header['HGLT_OBS'] = soho_hgs.lat.to('deg').value
    hdul[0].header['DSUN_OBS'] = soho_hgs.radius.to('m').value

    hdul[0].header['GEIX_OBS'] = orbit.gci_x_km * 1e3
    hdul[0].header['GEIY_OBS'] = orbit.gci_y_km * 1e3
    hdul[0].header['GEIZ_OBS'] = orbit.gci_z_km * 1e3
    hdul[0].header['GSEX_OBS'] = orbit.gse_x_km * 1e3
    hdul[0].header['GSEY_OBS'] = orbit.gse_y_km * 1e3
    hdul[0].header['GSEZ_OBS'] = orbit.gse_z_km * 1e3
    hdul[0].header['GSMX_OBS'] = orbit.gsm_x_km * 1e3
    hdul[0].header['GSMY_OBS'] = orbit.gsm_y_km * 1e3
    hdul[0].header['GSMZ_OBS'] = orbit.gsm_z_km * 1e3
    hdul[0].header['HAEX_OBS'] = orbit.hec_x_km * 1e3
    hdul[0].header['HAEY_OBS'] = orbit.hec_y_km * 1e3
    hdul[0].header['HAEZ_OBS'] = orbit.hec_z_km * 1e3

    # Note that HEQ, i.e., heliocentric Earth equatorial, coordinates
    # are the Cartesian form of heliographic Stonyhurst coordinates
    # (HGLN_OBS, HGLT_OBS, DSUN_OBS)
    hdul[0].header['HEQX_OBS'] = soho_hgs.cartesian.x.to('m').value
    hdul[0].header['HEQY_OBS'] = soho_hgs.cartesian.y.to('m').value
    hdul[0].header['HEQZ_OBS'] = soho_hgs.cartesian.z.to('m').value

    hdul[0].header['HGCX_OBS'] = soho_hgc.cartesian.x.to('m').value
    hdul[0].header['HGCY_OBS'] = soho_hgc.cartesian.y.to('m').value
    hdul[0].header['HGCZ_OBS'] = soho_hgc.cartesian.z.to('m').value

    return Map(hdul[0].data * SCALE, hdul[0].header)


def main(argv=None):
    if argv is None:
        argv = sys.argv

    parser = ArgumentParser('Fetch pB FITS data records from LASCO-C2 Legacy Archive and add NRL predictive orbit and WCS parameters to the header.',
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('date_range', type=lambda x: dateutil.parser.parse(x), nargs='+',
                        help='can be one date/time or two dates (and process between the dates in the given range)')
    parser.add_argument('--interval', '-i', type=int, default=24,
                        help='date/time increment period (in hours)')
    parser.add_argument('--output-path', '-o', type=Path, default=Path('../data/lasco_c2').absolute(),
                        help='Path to store SOHO Lasco-C2 FITS files.')
    parser.add_argument('--work-path', '-w', type=Path, default=None,
                        help='Path to store intermediate files. Use a automatically cleaned up temporary directory if not specified.')
    parser.add_argument('--index-path', type=Path, default=INDEX_PATH,
                        help='Path to index of FITS headers (use index_legacy.py to fetch headers).')
    args = parser.parse_args(argv[1:])

    work_path = None
    if args.work_path is None:
        work_path = TemporaryDirectory()
        args.work_path = Path(work_path.name)

    if len(args.date_range) == 1:
        args.date_range.append(args.date_range[0])
    elif len(args.date_range) > 2:
        assert False

    index = parse_index(args.index_path)

    logger.info('Fetching data')
    fts_fnames_and_dts = []
    # Determine which FITS files to process
    if args.interval > 0:
        for date in dateutil.rrule.rrule(dateutil.rrule.HOURLY,
                                         dtstart=args.date_range[0],
                                         until=args.date_range[1],
                                         interval=args.interval):
            logger.info(f'Processing {date}')

            # Find name of pB FITS file closest to specified date
            fts_fname, fts_dt, delta = find_closest_fts(index, date)
            logger.info(f'Nearest pB to {date} is at {fts_dt} (difference of {delta.total_seconds() / 60**2:.1f} hours)')
            fts_fnames_and_dts.append((fts_fname, fts_dt))
    elif args.interval == 0:
        fts_fnames_and_dts = find_fts_in_range(index, args.date_range[0], args.date_range[1])
        logger.info(f'There are {len(fts_fnames_and_dts)} FITS records between {args.date_range[0]} and {args.date_range[1]}')
    else:
        assert False

    pB_fnames_and_orbits = []
    for fts_fname, fts_dt in fts_fnames_and_dts:
            # Download pB FITS file
            pB_fname = fetch_pB(args.work_path, fts_fname, fts_dt)

            # Download corresponding orbit DAT file
            orbit_dat = parse_dat_file(fetch_dat_file(args.work_path,
                                                      find_dat_file(args.work_path, fts_dt)))

            # Find orbit line closest to specified date
            orbit, orbit_delta = find_closest_orbit(orbit_dat, fts_dt)
            logger.info(f'Nearest SOHO orbit to {fts_dt} is a difference of {orbit_delta.total_seconds() / 60} minutes')

            pB_fnames_and_orbits.append((pB_fname, orbit))

    # print(len(pB_fnames_and_orbits), len(fts_fnames_and_dts), pB_fnames_and_orbits, fts_fnames_and_dts)
    logger.info('Processing data')

    # for pB_fname, orbit, in pB_fnames_and_orbits:
    for pB_fname_and_orbit, fts_fname_and_dt in zip(pB_fnames_and_orbits, fts_fnames_and_dts):
        # Update FITS header
        pB_fname, orbit = pB_fname_and_orbit
        fts_fname, fts_dt = fts_fname_and_dt
        pB_map = create_pB_map(pB_fname, orbit)
        # Output FITS file
        # output_fname = (args.output_path / pB_fname.name).absolute()
        fname = str(fts_dt.date()) + '-' + str(fts_dt.hour) + '.fts'
        output_fname = (args.output_path / fname).absolute()
        logger.info(f'Storing FITS record to {output_fname}')
        pB_map.save(output_fname, overwrite=True)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    sys.exit(main())
