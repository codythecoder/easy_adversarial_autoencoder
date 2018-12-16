from PIL import Image
import argparse
import os
from statistics import mean

parser = argparse.ArgumentParser()
parser.add_argument('image')
parser.add_argument('folder')

args = parser.parse_args()

im = Image.open(args.image)

files = [os.path.join(path, filename) for path, dirs, files in os.walk(args.folder) for filename in files]

image_diffs = {}
for filename in files:
    im_comp = Image.open(filename).convert('RGB')
    pairs = zip(im.getdata(), im_comp.getdata())
    diff = sum(abs(c1-c2) for p1,p2 in pairs for c1,c2 in zip(p1,p2))
    image_diffs[filename] = diff

print (*sorted(image_diffs.items(), key=lambda x:x[1])[:10], sep='\n')
