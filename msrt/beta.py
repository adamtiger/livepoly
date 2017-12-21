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


# The probability that the weight of curve S has bigger weight then curve i.
def psi(ps, pi, ls, li, epsilon):

    ratio = int(m.ceil((ls - li * epsilon)/(1-epsilon)))

    probability = 0.0
    for ns in range(max(0, ratio - li), ls + 1):
        for ni in range(max(0, ratio - ns), li + 1):
            probability += binom.pmf(ns, ls, ps) * binom.pmf(ni, li, pi)

    return probability


# Measuring the distributions of beta in terms of the curve length on real images.
def measure_beta(img):
    pass


# Drawing curve for the psi values in terms of the beta.
def psi_curve():

    epsilon = 0.01
    ps = 0.8
    pt = 0.65
    l = 100
    lt = [l] * 50
    ls = [int(x / 10 * l) for x in range(11, 51, 1)]

    pst = []
    for t, s in zip(lt, ls):
        temp = psi(ps, pt, s, t, epsilon)
        pst.append(temp)

    plt.plot(ls, pst)
    plt.show()

