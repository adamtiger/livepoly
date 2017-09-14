'''
This module is for managing the images during
the training.

There are functions for:
 test purposes
 image pre-processing
 pool for storing the images
'''

import numpy as np
from scipy import misc


class Image:

    def __init__(self, orig, segm):
        self.o = orig
        self.s = segm

    def orig(self):
        return self.orig

    def set_orig(self, new_orig):
        self.o = new_orig

    def segm(self):
        return self.segm

    def set_segm(self, new_segm):
        self.s = new_segm

    def shape(self):
        return self.o.shape


def generate_random_image(size=(86, 86, 1)):

    image = np.random.random_integers(0, 256, size=size)

    return image


def read_image(file_name):
    return misc.imread(file_name, mode='RGB')


def convert_grey(source_image): # (height, width, 3 (RGB))

    h = source_image.shape[0]
    w = source_image.shape[1]
    img = np.ndarray((h, w, 1))
    for i in range(h):
        for j in range(w):
            img[i, j] = source_image[i, j, 0] * 299/1000 + source_image[i, j, 1] * 587/1000 + source_image[i, j, 2] * 114/1000

    return img


def crop_out(source_image, position, target_size): # crop a small image around the position

    #  the position will be the left top corner of the rectangle
    img = np.zeros(target_size)
    h = position[0] + target_size[0]
    w = position[1] + target_size[1]

    if h <= source_image.shape[0] and w <= source_image.shape[1]:
        img[:,:,:] = source_image[position[0] : h, position[1] : w, :]
    else:
        img = None

    return img


def check_segmentation(source_image, position):

    if source_image[position[0], position[1], 0] == 255:
        return True

    return False


def pop_img(img):
    misc.imshow(img)

