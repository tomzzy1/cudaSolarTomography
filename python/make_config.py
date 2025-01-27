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
    parser.add_argument('--path', '-p', type=Path, help='Path to FITS data.')
    args = parser.parse_args(argv[1:])

    if args.name is None:
        args.name = args.path.parts[-1]

    with open(f'../config/{args.name}.conf', 'w') as f:
        file_list = os.listdir(args.path)
        file_list.sort()
        f.write(f'{len(file_list)}\n')
        for filename in file_list:
            f.write(filename + '\n')

        
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    sys.exit(main())
