'''
This module is for measuring the beta on an image
and for calculating the corresponding probabilities.
(Probability: the probability of choosing the segmenting curve
if only two paths exist on the image.)
'''

import numpy as np
import math as m
from scipy.stats import binom
import matplotlib.pyplot as plt
from msrt import curve


# -------------------------------------------------
# The probability that the weight of curve S has bigger weight then curve i.

def psi(ps, pi, ls, li, epsilon):

    ratio = int(m.ceil((ls - li * epsilon)/(1-epsilon)))

    probability = 0.0
    for ns in range(max(0, ratio - li), ls + 1):
        probability += binom.pmf(ns, ls, ps) * (1 - binom.cdf(max(ratio - ns, 0), li, pi))

    return probability


# -------------------------------------------------
# Measuring the distributions of beta in terms of the curve length on a curve.

def measuring_one_curve(curve, lmin):
    beta_mtx = np.zeros((2, len(curve) - lmin + 1), dtype=np.float32)
    for i in range(beta_mtx.shape[1]):
        beta_mtx[0, i] = lmin + i

    for st in range(0, len(curve)):
        en = st + lmin - 1
        while en < len(curve):
            segm_length = en - st + 1

            a = max(abs(curve[st][0] - curve[en][0]), abs(curve[st][1] - curve[en][1]))
            distance = a + 2  # distance means the number of pixels

            beta = float(segm_length)/distance
            idx = segm_length - lmin
            beta_mtx[1, idx] = max(beta, beta_mtx[1, idx])

            en += 1

    return beta_mtx


def measuring_one_curve_simple(curve, lmin):

    beta_max = 0.0
    for st in range(0, len(curve)):
        en = st + lmin - 1
        while en < len(curve):
            segm_length = en - st + 1

            a = max(abs(curve[st][0] - curve[en][0]), abs(curve[st][1] - curve[en][1]))
            distance = a + 2  # distance means the number of pixels

            beta = float(segm_length) / distance
            beta_max = max(beta, beta_max)

            en += 1

    return beta_max


# -------------------------------------------------
# Measuring the distributions of beta in terms of the curve length on real images.

def measure_beta(img, lmin):

    beta_mtx_dict = {}
    lengths = [x for x in range(lmin, 201, 10)]

    # Monte Carlo sampling for measuring the betas for different lengths.
    segm_points = curve.find_segmenting_points(img)
    for l in lengths:
        beta_mtx_dict[l] = []
        for cntr in range(200):
            cv = curve.generate_curve(img, segm_points, l)
            beta_mtx = measuring_one_curve(cv, lmin)
            beta_mtx_dict[l].append(beta_mtx)

    return beta_mtx_dict


# -------------------------------------------------
# Calculating the thresholds.

def thresholds(epsilon, ps, pt, threshold):

    lengths = [x for x in range(15, 301, 5)]
    l_thr = []

    for l in lengths:

        ls = [l] * 50
        lt = [int(l * 10.0 / float(x)) for x in range(11, 51, 1)]

        pst = []
        for t, s in zip(lt, ls):
            temp = psi(ps, pt, s, t, epsilon)
            pst.append(temp)

        # Find the minimum.
        found = False
        idx = 0
        while not found and idx < len(pst):
            if pst[idx] < threshold:
                found = True
            idx += 1

        idx -= 1

        if found:
            l_thr.append(ls[idx] / lt[idx])
        else:
            l_thr.append(None)

    return lengths, l_thr


def thresholds_simple(epsilon, ps, pt):

    threshold = 0.5

    ls = [300] * 50
    lt = [int(300 * 10.0 / float(x)) for x in range(11, 51, 1)]

    pst = []
    for t, s in zip(lt, ls):
        temp = psi(ps, pt, s, t, epsilon)
        #temp = temp / (10.0 * (1.0 - temp) + temp)
        pst.append(temp)

    # Find the minimum.
    found = False
    idx = 0
    while not found and idx < len(pst):
        if pst[idx] < threshold:
            found = True
        idx += 1

    idx -= 1

    return ls[idx] / lt[idx]


# -------------------------------------------------
# Calculating the theoretical error rate.

def theoretical_error(beta_mtx_dict, l_thresholds):

    error = {}

    for k in beta_mtx_dict.keys():

        error[k] = 0.0
        num = 0
        right = 0
        for beta_mtx in beta_mtx_dict[k]:
            num += 1

            # Decide if the curve is chosen right.
            idx = 0
            correct = True
            while idx < beta_mtx.shape[1] and correct:
                idx_c = (int(beta_mtx[0, idx]) - 15) // 5
                beta_max = l_thresholds[idx_c]
                if beta_max is not None and beta_max < beta_mtx[1, idx]:
                    correct = False
                idx += 1

            if correct:
                right += 1

        error[k] = 1.0 - float(right) / num

    return error


# -------------------------------------------------
# Drawing curve for the psi values in terms of the beta.

def psi_curve(l):

    epsilon = 0.01
    ps = 0.95
    pt = 0.2
    lt = [l] * 50
    ls = [int(x / 10 * l) for x in range(11, 51, 1)]

    pst = []
    for t, s in zip(lt, ls):
        temp = psi(ps, pt, s, t, epsilon)
        pst.append(temp)

    plt.plot([x / 10.0 for x in range(11, 51, 1)], pst)
    plt.show()


def thresholds_length():

    epsilon = 0.01
    ps = 0.8
    pt = 0.65
    threshold = 0.95

    l, l_thr = thresholds(epsilon, ps, pt, threshold)

    plt.plot(l, l_thr)
    plt.show()
