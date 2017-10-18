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
The Keras equivalent of the model:
model.add(LocallyConnected2D(1, (2, 2), strides=(2, 2), padding='VALID', use_bias=False, input_shape=input_size))
model.add(Conv2D(32, (9, 9), strides=(1, 1), padding='VALID', activation='relu'))
model.add(Conv2D(32, (9, 9), strides=(1, 1), padding='VALID', activation='relu'))
model.add(Conv2D(64, (3, 3), strides=(3, 3), padding='VALID', activation='relu'))
model.add(Conv2D(1, (9, 9), strides=(1, 1), padding='VALID', activation='tanh'))
'''


def create_model():

    model = {}

    input_variable = tf.placeholder(tf.float32, shape=(None, input_size[0], input_size[1], input_size[2]))

    value = np.ones(shape=(1, input_size[0], input_size[1], input_size[2]), dtype=np.float32)
    value[0, 43, 43, 0] = 10.0
    value[0, 43, 44, 0] = 10.0
    value[0, 44, 43, 0] = 10.0
    value[0, 44, 44, 0] = 10.0
    weights = tf.constant(value, tf.float32, shape=(1, input_size[0], input_size[1], input_size[2]))

    def w(k_h, k_w, channels, filters):
        init = tf.truncated_normal([k_h, k_w, channels, filters], stddev=0.1)
        return tf.Variable(init)

    def b(filters):
        init = tf.constant(0.1, shape=[filters])
        return tf.Variable(init)

    # Special function for the 2x2 local kernels with strides (2, 2)
    # tf.multiply executes an element-wise multiplication
    # convolution with [[1, 1], [1, 1]] kernel does summation.
    def locally_conv_2d(input_variable, weights):

        fixed_weights = tf.constant(1.0, dtype=tf.float32, shape=(2, 2, input_size[2], 1))

        multiply1 = tf.multiply(input_variable, weights)
        locally1 = tf.nn.conv2d(multiply1, fixed_weights, padding='SAME', strides=[1, 2, 2, 1])

        return locally1

    locally = locally_conv_2d(input_variable, weights)
    conv1 = tf.nn.relu(tf.nn.conv2d(locally, w(9, 9, 1, 32), padding='VALID', strides=[1, 1, 1, 1]) + b(32))
    conv2 = tf.nn.relu(tf.nn.conv2d(conv1, w(9, 9, 32, 32), padding='VALID', strides=[1, 1, 1, 1]) + b(32))
    conv3 = tf.nn.relu(tf.nn.conv2d(conv2, w(3, 3, 32, 64), padding='VALID', strides=[1, 3, 3, 1]) + b(64))
    conv4 = tf.nn.sigmoid(tf.nn.conv2d(conv3, w(9, 9, 64, 1), padding='VALID', strides=[1, 1, 1, 1]) + b(1))

    correct = tf.placeholder(tf.int32, shape=(None, output_size[0], output_size[1], output_size[2]))
    loss = tf.losses.mean_squared_error(labels=correct, predictions=conv4)
    train_step = tf.train.AdamOptimizer(1e-3).minimize(loss)

    correct_prediction = tf.equal(tf.argmax(conv4, 1), tf.argmax(correct, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    model['sess'] = sess
    model['forward'] = conv4
    model['train'] = train_step
    model['loss'] = loss
    model['acc'] = accuracy
    model['x'] = input_variable
    model['y'] = correct

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
    return ["train_loss", "train_acc", "test_loss", "test_acc"]


def train_batch(model, data_chunk, batch_size, epochs):
    sum_loss = 0.0
    sum_acc = 0.0
    for epoch in range(epochs):
        result = model['sess'].run([model['train'], model['loss'], model['acc']], feed_dict={model['x']: data_chunk.get_x(), model['y']: data_chunk.get_y()})
        sum_loss += result[1]
        sum_acc += result[2]
    return [sum_loss/epochs, sum_acc/epochs]


def evaluate(model, test_set, batch_size):

    evals = model['sess'].run([model['loss'], model['acc']], feed_dict={model['x']: test_set.get_x(), model['y']: test_set.get_y()})
    return [evals[0], evals[1]]


def predict(model, x, batch_size):
    return model['forward'].run(session=model['sess'], feed_dict={model['x']: x})


def save_model(model, file_name):
    saver = tf.train.Saver()
    saver.save(model['sess'], file_name)


def load_model(model, file_name):
    raise NotImplementedError()