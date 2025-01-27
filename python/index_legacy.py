#!/usr/bin/env python3

import logging
from pathlib import Path

import requests
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('index_legacy')



URL_TEMPLATE = 'http://idoc-lasco.ias.u-psud.fr/sitools/datastorage/user/results/kfcorona_sph_new_optimized/C2/Orange/{year}/'

INDEX_PATH = Path('../data/lasco_c2/index')


if __name__ == '__main__':
    year_begin = 1996
    year_end = 2024

    for year in range(year_begin, year_end):
        logger.info(f'Processing {year}')
        year_url = URL_TEMPLATE.format(year=year)

        r = requests.get(year_url)
        assert r.status_code == 200

        pB_urls = []
        for line in r.text.splitlines():
            if 'pB.fts' in line:
                pB_urls.append(line.split('"')[1])

        for pB_url in tqdm(pB_urls):
            r = requests.get(pB_url, stream=True)
            header = []
            it = r.iter_lines()
            line = next(it)
            while True:
                header.append(line[:80].decode('utf-8'))
                if header[-1].startswith('END'):
                    break
                # This will fail if "END" is not found, i.e., the header is larger than the chunksize specified in iter_lines
                line = line[80:]
            pB_fname = pB_url.split('/')[-1]
            local_header_path = INDEX_PATH / f'{year}'
            local_header_path.mkdir(parents=True, exist_ok=True)
            local_header_fname = local_header_path / (pB_fname + '.hdr')
            with open(local_header_fname, 'w') as fid:
                fid.write('\n'.join(header))
