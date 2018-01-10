'''
The different types of live-poyline has different accuracy.
Here the accuracy means the expected length which is accepted by
the user.

The error rate is measured in two steps:
    1) Calculate the error rate for curves with fixed length.
    2) Calculate the expected length.
'''

from msrt import curve
from msrt import weights as w
import json as js
import numpy as np


# --------------------------------
# Constants
name_o = "example_o.png"
name_s = "example_s.png"
weightjson = "weights.json"


# -------------------------------------------------
# Measuring the error rate on a given image.

def measure_errorrate():

    errors_h = []
    errors_n = []

    img_orig, img_segm = curve.image_reader(name_o, name_s)
    print("Images were read.")

    wh = w.luminance(img_orig)  # weight map for the heuristic
    wn = w.neural(weightjson)  # weight map for the neural
    print("Weights are determined.")

    lengths = [20, 50, 70, 80, 90, 100, 110, 120, 130, 140, 150, 200]
    samples = [200, 200, 200, 100, 100, 100, 100, 100, 100, 50, 50, 50]

    segm_points = curve.find_segmenting_points(img_segm)

    for l, s in zip(lengths, samples):

        print("Current length: " + str(l))

        eh = 0.0  # error rate for the heuristic case
        en = 0.0  # error rate for the neural case
        cntr = 0.0  # counter for the samples

        for _ in range(s):
            curve1 = curve.generate_curve(img_segm, segm_points, l)
            print('curve_1')
            curve2_h = curve.get_livepolyline(wh, curve1[0], curve1[-1])
            print('curve_2h')
            curve2_n = curve.get_livepolyline(wn, curve1[0], curve1[-1])
            print('curve_2n')

            if not curve.compare_curves(curve1, curve2_h):
                eh += 1.0

            if not curve.compare_curves(curve1, curve2_n):
                en += 1.0

            cntr += 1.0

        errors_h.append(eh / cntr)
        errors_n.append(en / cntr)

    with open("errors.json", 'w') as j:
        result = {"heur": errors_h, "neur": errors_n}
        js.dump(result, j)


# -------------------------------------------------
# Measuring the error rate on a given image.
# Acceleration with multiprocessing


def get_data():

    img_orig, img_segm = curve.image_reader(name_o, name_s, crop=True)
    print("Images were read.")

    wh = w.luminance(img_orig)  # weight map for the heuristic
    with open(weightjson, 'w') as j:
        ls = np.ones(img_segm.shape).tolist()
        js.dump(ls, j)
    wn = w.neural(weightjson)  # weight map for the neural
    print("Weights are determined.")

    segm_points = curve.find_segmenting_points(img_segm)

    return wh, wn, img_segm, segm_points


def mp_measure_errorrate(length, sample, wh, wn, img_segm, segm_points):

    eh = 0.0  # error rate for the heuristic case
    en = 0.0  # error rate for the neural case

    for _ in range(sample):
        curve1 = curve.generate_curve(img_segm, segm_points, length)
        curve2_h = curve.get_livepolyline(wh, curve1[0], curve1[-1])
        curve2_n = curve.get_livepolyline(wn, curve1[0], curve1[-1])

        if not curve.compare_curves(curve1, curve2_h):
            eh += 1.0

        if not curve.compare_curves(curve1, curve2_n):
            en += 1.0

    return eh / sample, en / sample
