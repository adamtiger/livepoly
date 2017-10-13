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

'''
model.add(LocallyConnected2D(1, (2, 2), strides=(2, 2), padding='VALID', use_bias=False, input_shape=input_size))
model.add(Conv2D(32, (9, 9), strides=(1, 1), padding='VALID', activation='relu'))
model.add(Conv2D(32, (9, 9), strides=(1, 1), padding='VALID', activation='relu'))
model.add(Conv2D(64, (3, 3), strides=(3, 3), padding='VALID', activation='relu'))
model.add(Conv2D(1, (9, 9), strides=(1, 1), padding='VALID', activation='tanh'))
'''

def create_model():

    input_variable = tf.placeholder(tf.float32, shape=(None, input_size[0], input_size[1], input_size[2]))
    weights = 0

    def w(k_h, k_w, channels, filters):
        init = tf.truncated_normal([k_h, k_w, channels, filters], stddev=0.1)
        return tf.Variable(init)

    def b(filters):
        init = tf.constant(0.1, shape=[filters])
        return tf.Variable(init)

    def locally_conv_2d(filters, k_size, strides, input_variable, weights):
        return 0

    locally = locally_conv_2d(1, (2, 2), (2, 2), input_variable, weights)
    conv1 = tf.nn.relu(tf.nn.conv2d(locally, w(9, 9, 1, 32), padding='SAME', strides=[1, 1, 1, 1]) + b(32))
    conv2 = tf.nn.relu(tf.nn.conv2d(conv1, w(9, 9, 32, 32), padding='SAME', strides=[1, 1, 1, 1]) + b(32))
    conv3 = tf.nn.relu(tf.nn.conv2d(conv2, w(9, 9, 32, 64), padding='SAME', strides=[1, 3, 3, 1]) + b(64))
    conv4 = tf.nn.tanh(tf.nn.conv2d(conv3, w(9, 9, 64, 1), padding='SAME', strides=[1, 1, 1, 1]) + b(1))

    return conv4


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