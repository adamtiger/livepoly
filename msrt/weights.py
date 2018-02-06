'''
This module is responsible to providing the
necessary weight maps under the live-polyline.

Possible weight maps:
    -> luminance
    -> neural
    -> random (Bernoulli-based)
'''


import numpy as np
import math as m
import json


# -------------------------------------------
# Luminance

def luminance(img, min_weight=50, med_weight=150):

    # BGR to LUV then choose luminance
    # Ref.: https://docs.opencv.org/3.1.0/de/d25/imgproc_color_conversions.html
    def rgb2L(r, g, b):

        rgb = np.array([[r / 255.0], [g / 255.0], [b / 255.0]])  # with normalization
        M = np.array([[0.412453, 0.357580, 0.180423], [0.212671, 0.715160, 0.072169], [0.019334, 0.119193, 0.950227]])
        xyz = np.matmul(M, rgb)

        if xyz[1] > 0.008856:
            lum = 116 * m.pow(xyz[1], 1.0/3.0)
        else:
            lum = 903.3 * xyz[1]

        return lum * 255.0/100.0

    # Calculate the weights
    weights = np.zeros((img.shape[0], img.shape[1]))

    for row in range(img.shape[0]):
        for col in range(img.shape[1]):

            temp = rgb2L(img[row, col, 0], img[row, col, 1], img[row, col, 2])
            if temp <= min_weight:
                weights[row, col] = 1.0
            elif temp <= med_weight:
                weights[row, col] = temp / 30.0
            else:
                weights[row, col] = temp

    return weights


# -------------------------------------------
# Neural

def neural(file_name):

    # The neural map is saved as a json file.
    with open(file_name, 'r') as w:
        mtx = json.load(w)
    temp = np.array(mtx)
    weights = 1.0 - temp[42:temp.shape[0]-42, 42:temp.shape[1]-42]
    return weights


# -------------------------------------------
# Random

def bernoulli(piece, ps, pn, epsilon):

    # The input should contain a curve and nothing more

    # Generating a value for a pixel.
    def for_pixel(px):
        rand = np.random.random()
        if piece[px[0], px[1]] == 0:

            if rand < ps:
                weight = epsilon
            else:
                weight = 1.0
        else:

            if rand < pn:
                weight = 1.0
            else:
                weight = epsilon

        return weight

    # Generating random weights for each pixel.
    weights = np.zeros(piece.shape)
    for row in range(piece.shape[0]):
        for col in range(piece.shape[1]):

            weights[row, col] = for_pixel((row, col))

    return weights

