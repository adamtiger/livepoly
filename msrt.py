'''
This module responsible for starting the measurements:
    1. Beta measurements. mode = 1
    2. Error rate measuring for neural and heuristic. mode = 2
    3. Validation probabilities for a given curve. mode = 3
    4. Theoretical probabilities for a given curve. mode = 4
'''

import argparse
import csv
import multiprocessing as mp
from msrt import validation as vld

parser = argparse.ArgumentParser(description="Measurements of important metrics")

parser.add_argument("--mode", type=int, default=0, metavar='N',
                    help="1: beta, 2: error rate, 3: validation 4: theoretical")
parser.add_argument("--trds", type=int, default=4, metavar='N',
                    help="number of threads")

args = parser.parse_args()

# --------------------------------------------
# Constants
vld_f_nm = 'validation.csv'


# --------------------------------------------
# Helper for starting the processes with pool.

def set_lock(l):
    global lock
    lock = l


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
    with open(vld_f_nm, 'a', newline='\n') as file:
        wrt_obj = csv.writer(file)
        wrt_obj.writerow(data)

    print("ps: " + str(ps) + " pn: " + str(pn))

    lock.release()


if __name__ == "__main__":

    # MODE 1: Beta measurements
    if args.mode == 1:

        pass

    # MODE 2: Error rate measurements
    elif args.mode == 2:

        pass

    # MODE 3: Validation probabilities
    elif args.mode == 3:

        print('----- MODE 3: VALIDATING PROBS -----')

        # Get the data
        inputs = data_mode3()

        # This process takes a lot of time
        # multiprocessing is necessary.

        print('Start multiprocessing.')
        l = mp.Lock()
        pool = mp.Pool(processes=args.trds, initializer=set_lock, initargs=(l,))
        pool.map(process_mode3, inputs)
        pool.close()
        pool.join()

        print('----- Finished! -----')

    # MODE 4: Theoretical probabilities
    elif args.mode == 4:

        pass

    else:
        print('Wrong mode selection')
