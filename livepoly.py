'''
This module is for managing the whole training process for live-polyline.

'''

import os
import pool
import nn

# discover the folder structure and find images
# file name with ending '_orig' is the original version

folder_name = "imgs"

img_names = os.listdir(folder_name)
img_paths = ["imgs/" + x for x in img_names]

images = []

# define a function which reads arbitrary image pair, do not read all the images
for i in range(len(img_paths)):

    orig = None
    segm = None
    if img_paths[i].find("_orig.") is not -1:
        orig =  pool.read_image(img_paths[i])
        segm = pool.read_image(img_paths[i+1])
    else:
        segm = pool.read_image(img_paths[i])
        orig = pool.read_image(img_paths[i+1])

    images.append(pool.Image(orig, segm))




