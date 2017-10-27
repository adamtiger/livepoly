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
import utils as u
import os


class Image:

    def __init__(self, orig, segm):
        self.o = orig
        self.s = segm
        self.__find_segmenting_points()

    def orig(self):
        return self.o

    def set_orig(self, new_orig):
        self.o = new_orig

    def segm(self):
        return self.s

    def set_segm(self, new_segm):
        self.s = new_segm

    def shape(self):
        return self.o.shape

    def get_segm_pts_list(self):
        return self.segm_pts

    def __find_segmenting_points(self):
        
        self.segm_pts = []
        for row in range(int(u.input_size[0]/2), self.s.shape[0] - int(u.input_size[0]/2) - 1):
            for col in range(int(u.input_size[1]/2), self.s.shape[1] - int(u.input_size[1]/2) - 1):
                if self.s[row, col] == 255:
                    self.segm_pts.append((row, col))


def generate_random_image(size=u.input_size):

    image = np.random.random_integers(0, 256, size=size)

    return image


def read_image(file_name):
    return misc.imread(file_name, mode='L')


def write_image(file_name, img):
    misc.imsave(file_name, img)


def convert_grey(source_image, file_name):  # (height, width, 3 (RGB))

    h = source_image.shape[0]
    w = source_image.shape[1]
    img = np.ndarray((h, w))
    for i in range(h):
        for j in range(w):
            img[i, j] = source_image[i, j, 0] * 299/1000 + source_image[i, j, 1] * 587/1000 + source_image[i, j, 2] * 114/1000

    misc.imsave(file_name, img)


def converter(folder, new_folder):

    if not os.path.exists(new_folder):
        os.makedirs(new_folder)

        imgs = os.listdir(folder)
        paths = [folder + "/" + x for x in imgs]
        new_paths = [new_folder + "/" + x for x in imgs]

        for i in range(len(paths)):
            img = misc.imread(paths[i], mode='RGB')
            new_path = new_paths[i]
            convert_grey(img, new_path)


def crop_out(source_image, position, target_size): # crop a small image around the position

    #  the position will be the left top corner of the rectangle
    img = np.zeros(target_size)
    h = position[0] + target_size[0]
    w = position[1] + target_size[1]

    if h <= source_image.shape[0] and w <= source_image.shape[1]:
        img[:,:,0] = source_image[position[0]: h, position[1]: w] # here the size independence is violated !!! (perf.)
    else:
        img = None

    return img/255.0 # normalize the image


# checks whether the 4 pixels next to each other contains
# any segmenting pixel, the 'position' is the left top pixel
def check_segmentation(source_image, position):

    s1 = source_image[position[0], position[1]] == 255
    s2 = source_image[position[0] + 1, position[1]] == 255
    s3 = source_image[position[0], position[1] + 1] == 255
    s4 = source_image[position[0] + 1, position[1] + 1] == 255

    y = np.zeros(u.output_size, dtype=np.float32)

    if s1 or s2 or s3 or s4:
        return y + 1

    return y


def pop_img(img):
    misc.imshow(img)

