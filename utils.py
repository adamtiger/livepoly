'''
This file contains some helper functions and
classes to solve the frequently occuring
common tasks.

'''

import numpy as np
import datetime
import csv

input_size = (86, 86, 1)
output_size = (1, 1, 1)


class TrainData:

    def __init__(self, batch):
        self.data = {}
        self.data['img'] = np.zeros((batch, input_size[0], input_size[1], input_size[2]), dtype=np.float32)
        self.data['sgm'] = np.zeros((batch, output_size[0], output_size[1], output_size[2]), dtype=np.float32)
        self.data['twin'] = np.zeros((batch, output_size[0], output_size[1], output_size[2]), dtype=np.float32)
        self.batch = batch
        self.idx = 0

    def get_x(self):
        return self.data['img']

    def get_ys(self):
        return self.data['sgm']

    def get_yt(self):
        return self.data['twin']

    def append(self, chunk):
        length = chunk.batch
        self.data['img'][self.idx:self.idx + length, :, :, :] = chunk.get_x()[:, :, :, :]
        self.data['sgm'][self.idx:self.idx + length, :, :, :] = chunk.get_ys()[:, :, :, :]
        self.data['twin'][self.idx:self.idx + length, :, :, :] = chunk.get_yt()[:, :, :, :]
        self.idx += length

    def add(self, sample_x, sample_ys, sample_yt):
        self.data['img'][self.idx, :, :, :] = sample_x[:, :, :]
        self.data['sgm'][self.idx, :, :, :] = sample_ys[:, :, :]
        self.data['twin'][self.idx, :, :, :] = sample_yt[:, :, :]
        self.idx += 1

    def clear(self):
        self.idx = 0

    def get_idx(self):
        return self.idx


# -----------------------------------------
# Generating a unique id for file names

def uid():

    unique_id = str(datetime.datetime.today()).replace(' ', '-')
    unique_id = unique_id.replace(':', '')

    return unique_id


# -----------------------------------------
# Appending a new line at the end of the given csv file

def csv_append(f_name, data):

    with open(f_name, 'a', newline='\n') as file:
        wo = csv.writer(file)
        wo.writerow(data)