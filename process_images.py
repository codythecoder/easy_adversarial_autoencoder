from src.images import process_images
import argparse
from consts import *
from src.misc import all_files
from src.countdown import Remaining
import bz2
import os, time

parser = argparse.ArgumentParser()
parser.add_argument('--input', default='raw_data')
parser.add_argument('--reshaped_folder', default='reshaped_images')
parser.add_argument('--overwrite', action='store_true')
parser.add_argument('--crop', default='fit')
parser.add_argument('--valid_states', default='all')
parser.add_argument('--extract', choices=('replace', 'extract'), default=None)

args = parser.parse_args()

if args.extract:
    files = all_files(args.input)
    countdown = Remaining(len(files))
    for filename in files:
        print (countdown, end='\r')
        countdown.tick()
        if filename.endswith('.bz2'):
            with open(filename[:-4], 'wb') as f, open(filename, 'rb') as g:
                data = g.read()
                f.write(bz2.decompress(data))
            if args.extract == 'replace':
                # print ('removing', filename)
                # print (os.path.isfile(filename))
                # time.sleep(1)
                os.remove(filename)
    print (countdown)

process_images(args.input, args.reshaped_folder, target_shape, overwrite=args.overwrite, crop=args.crop, valid_states=args.valid_states)
