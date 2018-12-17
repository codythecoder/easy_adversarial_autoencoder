import os
from PIL import Image
from math import ceil
from statistics import mean
import numpy as np
if __name__ == '__main__':
    from misc import *
else:
    from .misc import *

def get_reshaped_folder(out_folder, shape):
    reshaped_folder = '{}/{:0>3},{:0>3}'.format(out_folder, *shape)
    mkdirpath(split_path(reshaped_folder))
    return reshaped_folder

def get_rejected_folder(out_folder, shape):
    rejected_folder = '{}/rejected/{:0>3},{:0>3}'.format(out_folder, *shape)
    mkdirpath(split_path(rejected_folder))
    return rejected_folder

def process_images(folder, out_folder, shape, overwrite=False, crop='fit', valid_states='all'):
    files = all_files(folder)

    reshaped_folder = get_rejected_folder(out_folder, shape)
    rejected_folder = get_rejected_folder(out_folder, shape)

    images = 0
    for i, filename in enumerate(files):
        print ('{} of {} ({} kept)'.format(i, len(files), images), end='\r')

        png_file = filename.rsplit('.')[0] + '.png'
        resized_file = os.path.join(reshaped_folder, png_file)
        rejected_file = os.path.join(rejected_folder, png_file)

        if os.path.isfile(resized_file) and not overwrite:
            images += 1
            continue
        elif os.path.isfile(rejected_file) and not overwrite:
            continue
        else:
            # print (filename)
            im = Image.open(filename)
            if valid_states != 'all' and not valid(im, shape, crop, valid_states):
                im = resize(im, shape, crop)
                mkdirpath(split_path(rejected_file)[:-1])
                im.save(rejected_file)
                continue
            im = resize(im, shape, crop)
            mkdirpath(split_path(resized_file)[:-1])
            im.save(resized_file)
            images += 1

    print ('{0} of {0} ({1} kept)'.format(len(files), images))


def load_images(input_folder, reshaped_folder, shape):
    folder = os.path.join(reshaped_folder, input_folder)
    files = [os.path.join(path, filename) for path, dirs, files in os.walk(folder) for filename in files]

    images = []
    for filename in files:
        im = Image.open(filename)
        im_array = np.asarray(im)/255
        im_array = np.swapaxes(im_array, 0, 1)
        images.append(im_array)

    return images

def resize(im, shape, crop, background='WHITE'):
    # print (repr(im.mode))
    if im.mode == 'RGBA':
        white_im = Image.new("RGBA", im.size, background)
        white_im.paste(im, (0, 0), im)
        im = white_im
    if crop == 'fit':
        old_shape = im.size
        ratio = [old/new for old, new in zip(old_shape, shape)]
        if ratio[0] > ratio[1]:
            new_shape = shape[0], ceil(old_shape[1]/ratio[0])
        else:
            new_shape = ceil(old_shape[0]/ratio[1]), shape[1]
        new_pos = int((shape[0] - new_shape[0])/2), int((shape[1] - new_shape[1])/2)

        im = im.resize(new_shape, Image.BICUBIC)

        new_im = Image.new('RGB', shape, (255,255,255))
        new_im.paste(im, new_pos)
    return new_im

# def valid(im):
#     pix = im.load()
#     w, h = im.size
#     if h / w < target_shape[1] / target_shape[0]:
#         return False
#     if type(pix[0, 0]) in (int, float):
#         return False
#     for j in (0, w-1):
#         for i in range(h):
#             if pix[j, i][:3] != (255, 255, 255):
#                 return False
#     return True

def valid(im, shape, crop, valid_states):
    pix = im.load()
    if type(pix[0, 0]) in (int, float):
        return False
    w, h = im.size
    if crop == 'fit':
        if h / w < shape[1] / shape[0]:
            return False
    if valid_states in ('white_solid', 'white'):
        for j in (0, w-1):
            colors = []
            for i in range(h):
                colors.extend(pix[j, i][:3])
            if mean(colors) <= 252:
                return False
    if valid_states in ('white_solid',):
        colors = []
        for j in range(0, w, 3):
            for i in range(0, h, 3):
                if all(p > 250 for p in pix[j, i][:3]):
                    colors.append(1)
                else:
                    colors.append(0)
        mc = mean(colors)
        # print (mc)
        if mc > 0.775:
            return False
    return True


def save_image(start_time, v, i, name=''):
    im_array = np.swapaxes(v*255, 0, 1)
    im = Image.fromarray(np.clip(im_array, 0, 255).astype('uint8'), mode='RGB')
    if not os.path.isdir('out/{}'.format(start_time)):
        os.mkdir('out/{}'.format(start_time))
    im.save('out/{}/{}{:0>3}.png'.format(start_time, name, i))

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('folder')
    parser.add_argument('shape', nargs=2, type=int)

    args = parser.parse_args()

    load_images(args.folder, args.shape)
