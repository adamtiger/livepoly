'''
This module is for creating the network model.
Training and evaluation is provided.
The implementation is in TensorFlow.
It differs from the Keras one because
the first layer will not be trained.

Functions:

create_model() - builds the convolutional network
train_batch(model, batch, epochs)
evaluate(model, test_set)
save_model(model, file_name) # saves the model into a file which is readable for CNTK in CSahrp too
load_model(file_name)
'''

import tensorflow as tf
import numpy as np



input_size = (86, 86, 1)
output_size = (1, 1, 1)


def create_model():

   model = []



   return model


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


def metrics_names(model):
    raise NotImplementedError()


def train_batch(model, data_chunk, batch_size, epochs):
    raise NotImplementedError()


def evaluate(model, test_set, batch_size):
    raise NotImplementedError()


def save_model(model, file_name):
    raise NotImplementedError()


def load_model(model, file_name):
    raise NotImplementedError()