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

def data_mode2():
    lengths = [20, 50, 70, 80, 90, 100, 110, 120, 130, 140, 150, 200]
    samples = [50, 50, 50, 50, 50, 50, 50, 40, 40, 25, 25, 25]
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


if __name__ == "__main__":

    # MODE 1: Beta measurements and theoretical errors
    if args.mode == 1:

        pass

    # MODE 2: Error rate measurements
    elif args.mode == 2:

        mp_start('----- MODE 2: VALIDATING PROBS -----', data_mode2, process_mode2)

    # MODE 3: Validation probabilities
    elif args.mode == 3:

        mp_start('----- MODE 3: VALIDATING PROBS -----', data_mode3, process_mode3)

    else:
        print('Wrong mode selection')
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

def data_mode2():
    lengths = [20, 50, 70, 80, 90, 100, 110, 120, 130, 140, 150, 200]
    samples = [50, 50, 50, 50, 50, 50, 50, 40, 40, 25, 25, 25]
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


if __name__ == "__main__":

    # MODE 1: Beta measurements and theoretical errors
    if args.mode == 1:

        pass

    # MODE 2: Error rate measurements
    elif args.mode == 2:

        mp_start('----- MODE 2: VALIDATING PROBS -----', data_mode2, process_mode2)

    # MODE 3: Validation probabilities
    elif args.mode == 3:

        mp_start('----- MODE 3: VALIDATING PROBS -----', data_mode3, process_mode3)

    else:
        print('Wrong mode selection')
