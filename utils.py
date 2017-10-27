'''
This file contains some helper functions and
classes to solve the frequently occuring
common tasks.

'''

import numpy as np
import datetime

input_size = (86, 86, 1)
output_size = (1, 1, 1)


class TrainData:

    def __init__(self, batch):
        self.data = {}
        self.data['img'] = np.zeros((batch, input_size[0], input_size[1], input_size[2]), dtype=np.float32)
        self.data['sgm'] = np.zeros((batch, output_size[0], output_size[1], output_size[2]), dtype=np.float32)
        self.batch = batch
        self.idx = 0

    def batch_size(self):
        return self.batch

    def get_x(self):
        return self.data['img']

    def get_y(self):
        return self.data['sgm']

    def append(self, chunk):
        length = chunk.batch_size()
        self.data['img'][self.idx:self.idx + length, :, :, :] = chunk.get_x()[:, :, :, :]
        self.data['sgm'][self.idx:self.idx + length, :, :, :] = chunk.get_y()[:, :, :, :]
        self.idx += length

    def add(self, chunk_x, chunk_y):
        self.data['img'][self.idx, :, :, :] = chunk_x[:, :, :]
        self.data['sgm'][self.idx, :, :, :] = chunk_y[:, :, :]
        self.idx += 1

    def clear(self):
        self.idx = 0

    def get_idx(self):
        return self.idx


# generating a unique id for file names
def uid():

    unique_id = str(datetime.datetime.today()).replace(' ', '-')

    return unique_id