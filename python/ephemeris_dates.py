#!/usr/bin/env python3

import math
from datetime import datetime

from sunpy.coordinates.sun import carrington_rotation_number, carrington_rotation_time

SECONDS_PER_DAY = 60 * 60 * 24

N_DECIMAL = 2


if __name__ == '__main__':
    # The first data record in the LASCO legacy data archive
    # (http://idoc-lasco.ias.u-psud.fr/sitools/datastorage/user/results/kfcorona_sph_new_optimized/C2/Orange/1996/)
    # is on 1996/1/28, which is about half way into CR 1905.
    CR1 = 1905
    CR2 = int(math.ceil(carrington_rotation_number()))

    N_CR = CR2 - CR1 + 1
    ephemeris_dates = []
    for cr in range(CR1, CR2 + 1):
        cr_time = carrington_rotation_time(cr).datetime
        cr_date = datetime(cr_time.year, cr_time.month, cr_time.day)
        cr_day_fraction = (cr_time - cr_date).total_seconds() / SECONDS_PER_DAY
        fraction_str = f'{cr_day_fraction:.{N_DECIMAL}f}'[1:]
        ephemeris_dates.append('\t"' + cr_time.strftime('%Y-%m-%d') + fraction_str + '"')


    ephemeris_str = ',\n'.join(ephemeris_dates)
    print(f'constexpr std::array<std::string_view, {N_CR}> ephemeris_dates = {{\n' + ephemeris_str + '};', end='')
