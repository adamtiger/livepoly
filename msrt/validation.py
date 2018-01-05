'''
Finding an almost direct connection between the confusion matrix and
the accuracy of the live-polyline should be a huge advantage in order
to learn about the possible outcomes and find the desired neural network
performance.
'''

from msrt.curve import get_livepolyline, compare_curves
from msrt import weights as w
from scipy import misc
import numpy as np
import json as js


def validation_for_a_curve():

    piece = misc.imread("msrt/curve.png", mode='L')
    error = np.zeros((10, 10), dtype=np.float32)

    curve_points = []  # the points are not in order
    for row in range(piece.shape[0]):
        for col in range(piece.shape[1]):
            if piece[row, col] > 0:
                curve_points.append((row, col))

    for ps in range(10):
        for pn in range(10):

            print("ps: " + str(ps) + " pn: " + str(pn))

            for cntr in range(10):
                if cntr % 10 == 0:
                    print(str(cntr) + "/10")
                weight_map = w.bernoulli(piece, (ps + 1) / 10.0, (pn + 1) / 10.0, 0.01)
                cv = get_livepolyline(weight_map, (60, 1), (1, 194))
                if not compare_curves(curve_points, cv):
                    error[ps, pn] += 1.0

            error[ps, pn] /= 10.0

            with open("validation.json", 'w') as j:
                js.dump(error.tolist(), j)

