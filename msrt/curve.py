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
# the curve points has value 0.

def __connect_end_points(image, pa, pb):
    map = image + 1  # To choose the shortest segmenting path.
    size = max(abs(pa[0] - pb[0]), abs(pa[1] - pb[1]))

    curve = search.bfs(map, pa, pb, size, fast=True)  # Generate curve with BFS.

    return curve


# -------------------------------------------
# The generator function which gives a segmenting
# curve piece on the image with the given size.

def generate_curve(image, segm_points, size):

    pa, pb = __end_points(image, segm_points, size)

    curve = __connect_end_points(image, pa, pb)

    return curve[0:size]


# -------------------------------------------
# Tests

def show_a_generated_sample():
    image = misc.imread("example.png", mode='L')
    map = np.zeros(image.shape, dtype=int)
    # Invert image
    for row in range(image.shape[0]):
        for col in range(image.shape[1]):
            if image[row, col] == 0:
                map[row, col] = 255
            else:
                map[row, col] = 0

    size = 400
    segm_points = find_segmenting_points(map)
    curve = generate_curve(map, segm_points, size)

    result = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.float32)
    for row in range(result.shape[0]):
        for col in range(result.shape[1]):
            result[row, col, :] = image[row, col]/255

    color = np.array([0.5, 0, 0])
    for p in curve:
        result[p[0], p[1], :] = color[:]

    plt.imshow(np.float32(result))
    plt.show()
