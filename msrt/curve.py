'''
This module implements the curve generator.
Curve generator: If the segmentation curves are
given the generator can choose curve pices with a
given length.
'''

import numpy as np
from msrt import search
from scipy import misc
from matplotlib import pyplot as plt


# -------------------------------------------------
# Constants:

tolerance = 10  # number of pixels as a distance between curves which is tolerated


# -------------------------------------------
# Generates list from segmenting points
# (color value = 0).

def find_segmenting_points(image):
    sgms = []
    for row in range(image.shape[0]):
        for col in range(image.shape[1]):
            if image[row, col] == 0:
                sgms.append((row, col))

    return sgms


# -------------------------------------------
# Function for generating two end points.

def __end_points(image, segm_points, size):
    # First point
    idx = np.random.randint(0, len(segm_points))
    p1 = segm_points[idx]
    p2 = (0, 0)

    # Second point
    bl = (min(p1[0] + size, image.shape[0] - 1), max(p1[1] - size, 0))
    tl = (max(p1[0] - size, 0), max(p1[1] - size, 0))
    br = (min(p1[0] + size, image.shape[0] - 1), min(p1[1] + size, image.shape[1] - 1))
    tr = (max(p1[0] - size, 0), min(p1[1] + size, image.shape[1] - 1))

    found = False
    for r in range(tl[0], bl[0] + 1):
        if image[r, tl[1]] == 0:
            p2 = (r, tl[1])
            found = True

    if not found:
        for c in range(tl[1], tr[1] + 1):
            if image[tl[0], c] == 0:
                p2 = (tl[0], c)
                found = True

    if not found:
        for r in range(tr[0], br[0] + 1):
            if image[r, tr[1]] == 0:
                p2 = (r, tr[1])
                found = True

    if not found:
        for c in range(bl[1], br[1] + 1):
            if image[bl[0], c] == 0:
                p2 = (bl[0], c)
                found = True

    if not found:
        raise AssertionError("No suitable end point was found.")

    return p1, p2


# -------------------------------------------
# Connecting the endpoints. Here the map is
# equivalent with the segmented image where
# the curve points have value 0.

def __connect_end_points(image, pa, pb):

    size = max(abs(pa[0] - pb[0]), abs(pa[1] - pb[1])) + 1

    curve = search.bfs(image, pa, pb, size, fast=True)  # Generate curve with BFS.

    return curve


# -------------------------------------------
# The generator function which gives a segmenting
# curve piece on the image with the given size.

def generate_curve(image, segm_points, size):

    found = False
    while not found:
        pa, pb = __end_points(image, segm_points, size)

        curve = __connect_end_points(image, pa, pb)

        if len(curve) >= size:
            found = True

    return curve[0:size]


# -------------------------------------------
# Compare two curves and decides whether their
# distance is smaller than the tolerance.

def compare_curves(curve1, curve2):

    for p1 in curve1:
        min_distance2 = tolerance * tolerance + 1
        for p2 in curve2:
            min_distance2 = min(min_distance2, (p1[0] - p2[0]) * (p1[0] - p2[0]) + (p1[1] - p2[1]) * (p1[1] - p2[1]))

        if min_distance2 > tolerance * tolerance:
            return False

    return True  # the curves viewed as identical


# -------------------------------------------
# Gives a cirve between the two points.

def get_livepolyline(weight, p0, p1):

    size = max(abs(p0[0] - p1[0]), abs(p0[1] - p1[1]))
    return search.bfs(weight, p0, p1, size)


# -------------------------------------------
# Read two images as a pair (original, segmented).

def image_reader(name_original, name_segmented, crop=False):

    img_o = None
    img_s = None

    if name_original is not None:
        img_o = np.array(misc.imread(name_original, mode='RGB'), dtype=int)

        if crop:
            img_o = img_o[42:img_o.shape[0] - 42, 42:img_o.shape[1] - 42]

    if name_segmented is not None:
        img_s = np.array(misc.imread(name_segmented, mode='L'), dtype=int)

        if crop:
            img_s = img_s[42:img_s.shape[0] - 42, 42:img_s.shape[1] - 42]

        # Invert image (the search algorithm looks for minimum)
        for row in range(img_s.shape[0]):
            for col in range(img_s.shape[1]):
                if img_s[row, col] == 0:
                    img_s[row, col] = 1
                else:
                    img_s[row, col] = 0

    if img_o is not None and img_s is not None:
        if not img_o.shape[0] == img_s.shape[0] or not img_o.shape[1] == img_s.shape[1]:
            raise AssertionError("Wrong image sizes!")

    return img_o, img_s


# -------------------------------------------
# Tests

def show_a_generated_sample():

    name = "example.png"
    img, img_inv = image_reader(name, name)

    size = 400
    segm_points = find_segmenting_points(img_inv)
    curve = generate_curve(img_inv, segm_points, size)

    result = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.float32)
    for row in range(result.shape[0]):
        for col in range(result.shape[1]):
            result[row, col, :] = img[row, col]

    color = np.array([0.5, 0, 0])
    for p in curve:
        result[p[0], p[1], :] = color[:]

    plt.imshow(np.float32(result))
    plt.show()
