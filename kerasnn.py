'''
This module is for creating the network model.
Training and evaluation is provided.

Functions:

create_model() - builds the convolutional network
train_batch(model, batch, epochs)
evaluate(model, test_set)
save_model(model, file_name) # saves the model into a file which is readable for CNTK in CSahrp too
load_model(file_name)
'''

from keras.models import Sequential
from keras.layers import Conv2D, LocallyConnected2D
from keras.optimizers import Adam
import numpy as np

input_size = (86, 86, 1)
output_size = (1, 1, 1)


def create_model():

    model = Sequential()

    # The input is a numpy array (!) with shape (batch, height, width, channel)
    model.add(LocallyConnected2D(1, (2, 2), strides=(2, 2), padding='VALID', use_bias=False, input_shape=input_size))
    model.add(Conv2D(32, (9, 9), strides=(1, 1), padding='VALID', activation='relu'))
    model.add(Conv2D(32, (9, 9), strides=(1, 1), padding='VALID', activation='relu'))
    model.add(Conv2D(64, (3, 3), strides=(3, 3), padding='VALID', activation='relu'))
    model.add(Conv2D(1, (9, 9), strides=(1, 1), padding='VALID', activation='tanh'))

    opt = Adam(lr=0.0005)
    loss = 'binary_crossentropy'
    mtr = ['accuracy']
    model.compile(loss=loss, optimizer=opt, metrics=mtr)

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
    return model.metrics_names


def train_batch(model, data_chunk, batch_size, epochs):

    model.fit(data_chunk.get_x(), data_chunk.get_y(), batch_size=batch_size, epochs=epochs, verbose=0)


def evaluate(model, test_set, batch_size):

    return model.evaluate(test_set.get_x() , test_set.get_y(), batch_size=batch_size, verbose=0)


def save_model(model, file_name):
    model.save_weights(file_name)


def load_model(model, file_name):
    model.load_weights(file_name)

