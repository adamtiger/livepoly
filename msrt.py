'''
This module responsible for starting the measurements:
    1. Beta measurements. mode = 1
    2. Error rate measuring for neural and heuristic. mode = 2
    3. Validation probabilities for a given curve. mode = 3
'''

import argparse
import multiprocessing as mp
import utils
from msrt import validation as vld
from msrt import errorrate as ert
from msrt import beta
from msrt import curve


parser = argparse.ArgumentParser(description="Measurements of important metrics")

parser.add_argument("--mode", type=int, default=0, metavar='N',
                    help="1: beta, 2: error rate, 3: validation")
parser.add_argument("--trds", type=int, default=4, metavar='N',
                    help="number of threads")

args = parser.parse_args()

# --------------------------------------------
# Constants
vld_f_nm = 'validation.csv'
err_f_nm = 'errors.csv'
theor_err_f_nm = 'th_validation.csv'


# --------------------------------------------
# Helper for starting the processes with pool.

def set_lock(l):
    global lock
    lock = l


def mp_start(title, data, func):
    print(title)

    # Get the data
    inputs = data()

    # This process takes a lot of time
    # multiprocessing is necessary.

    print('Start multiprocessing.')
    l = mp.Lock()
    pool = mp.Pool(processes=args.trds, initializer=set_lock, initargs=(l,))
    pool.map(func, inputs)
    pool.close()
    pool.join()

    print('----- Finished! -----')


# --------------------------------------------
# Functions for the modes

def data_mode1():
    _, piece_img = curve.image_reader(None, vld.img_name)
    piece = curve.find_segmenting_points(piece_img)

    lmin = 30
    threshold = 0.8

    beta_max = beta.measuring_one_curve(piece, lmin)
    beta_max_dict = {'0': [beta_max]}

    inputs = []
    for ps in [x/20.0 for x in range(20, -1, -1)]:
        for pn in [y/20.0 for y in range(20, -1, -1)]:
            inputs.append([beta_max_dict, ps, pn, threshold])

    return inputs


def process_mode1(arg):
    beta_max_dict = arg[0]
    ps = arg[1]
    pn = arg[2]
    threshold = arg[3]

    #beta_max = beta.thresholds_simple(0.01, ps, pn)
    #if beta_max > beta_max_on_curve:
        #error = 0.0
    #else:
        #error = 1.0

    thresolds = beta.thresholds(0.01, ps, pn, threshold)
    error = beta.theoretical_error(beta_max_dict, thresolds)['0']

    lock.acquire()

    utils.csv_append(theor_err_f_nm, [ps, pn, error])

    print('ps: ' + str(ps) + ' pn: ' + str(pn))

    lock.release()


def data_mode2():
    lengths = [20, 50, 70, 80, 90, 100, 110, 120, 130, 140, 150, 200]
    samples = [1000, 500, 200, 200, 200, 100, 100, 100, 100, 100, 100, 50]
    wh, wn, img_segm, segm_points = ert.get_data()

    inputs = []
    for l, s in zip(lengths, samples):
        inputs.append((l, s, wh, wn, img_segm, segm_points))

    return inputs


def process_mode2(arg):
    length = arg[0]
    sample = arg[1]
    wh = arg[2]
    wn = arg[3]
    img_segm = arg[4]
    segm_points = arg[5]

    eh, en = ert.mp_measure_errorrate(length, sample, wh, wn, img_segm, segm_points)

    lock.acquire()

    utils.csv_append(err_f_nm, [length, eh, en])

    print('Length: ' + str(length))

    lock.release()


def data_mode3():
    piece, curve = vld.get_piece_and_sgm_points()
    inputs = []
    for i in range(20):
        for j in range(20):
            ps = i / 20.0
            pn = j / 20.0
            inputs.append((ps, pn, piece, curve))

    return inputs


def process_mode3(arg):
    ps = arg[0]
    pn = arg[1]
    piece = arg[2]
    curve = arg[3]

    err = vld.validation_for_curve(ps, pn, piece, curve)
    data = [ps, pn, err]  # this will be written into csv

    lock.acquire()

    # write into csv
    utils.csv_append(vld_f_nm, data)

    print("ps: " + str(ps) + " pn: " + str(pn))

    lock.release()


def data_mode4():
    _, img = curve.image_reader(None, ert.name_s)
    lmin = 20
    threshold = 0.5

    beta_mtx = beta.measure_beta(img, lmin)
    beta_mtx_dict = {'0': [beta_mtx]}

    inputs = []
    for ps in [x/20.0 for x in range(0, 21)]:
        for pn in [y/20.0 for y in range(0, 21)]:
            inputs.append([beta_mtx_dict, ps, pn, threshold])

    data = ['ps', 'pn']
    for k in beta_mtx_dict.keys():
        data.append(k)

    utils.csv_append(theor_err_f_nm, data)
    print('Data was assembled.')

    return inputs


def process_mode4(arg):
    beta_mtx_dict = arg[0]
    ps = arg[1]
    pn = arg[2]
    threshold = arg[3]

    _, thrs = beta.thresholds(0.01, ps, pn, threshold)

    error = beta.theoretical_error(beta_mtx_dict, thrs)
    data = [ps, pn]
    for k in error.keys():
        data.append(error[k])

    lock.acquire()

    utils.csv_append(theor_err_f_nm, data)

    print('ps: ' + str(ps) + ' pn: ' + str(pn))

    lock.release()


if __name__ == "__main__":

    # MODE 1: Theoretical error rate on a single curve
    if args.mode == 1:

        mp_start('----- MODE 1: THEORETICAL ERROR ON 1 CURVE -----', data_mode1, process_mode1)

    # MODE 2: Error rate measurements
    elif args.mode == 2:

        mp_start('----- MODE 2: VALIDATING PROBS -----', data_mode2, process_mode2)

    # MODE 3: Measured error rate on a single curve. Validation probabilities
    elif args.mode == 3:

        mp_start('----- MODE 3: VALIDATING PROBS -----', data_mode3, process_mode3)

    else:
        print('Wrong mode selection')
