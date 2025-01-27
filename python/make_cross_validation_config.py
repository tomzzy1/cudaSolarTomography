#!/usr/bin/env python3

import logging
import sys
import os
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from pathlib import Path

logger = logging.getLogger('make_config')

def main(argv=None):
    if argv is None:
        argv = sys.argv

    parser = ArgumentParser('Make config file from the data folder',
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--name', '-n', type=Path, help='name of the config files (w/o extension) (use folder name if not specified).')
    parser.add_argument('path', type=Path, help='Path to FITS data.')
    args = parser.parse_args(argv[1:])

    if args.name is None:
        args.name = args.path.parts[-1]

    with open(f'../config/{args.name}_training.conf', 'w') as f_training:
        with open(f'../config/{args.name}_validation.conf', 'w') as f_validation:
            file_list = os.listdir(args.path)
            file_map = dict()
            for filename in file_list:
                date = filename[:10]
                if date not in file_map:
                    file_map[date] = [filename]
                else:
                    file_map[date].append(filename)

            n_training = 0
            n_validation = 0
            for date, filenames in file_map.items():
                n_validation += 1
                n_training += len(filenames) - 1

            f_training.write(f'{n_training}\n')
            f_validation.write(f'{n_validation}\n')

            for date, filenames in file_map.items():
                f_validation.write(f'{filenames[0]}\n')
                for filename in filenames[1:]:
                    f_training.write(f'{filename}\n')
        
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    sys.exit(main())
