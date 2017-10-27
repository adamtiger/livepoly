'''
This module is for creating the network model.
Training and evaluation is provided.
The implementation is in TensorFlow.
It differs from the Keras one because
the first layer will not be trained.

Functions:

create_model(learning_rate) - builds the convolutional network
train_batch(model, batch, epochs)
evaluate(model, test_set)
save_model(model, file_name) # saves the model into a file which is readable for CNTK in CSharp too
load_model(file_name)
'''

import tensorflow as tf
import numpy as np
import utils as u

'''
The Keras equivalent of the model:
model.add(LocallyConnected2D(1, (2, 2), strides=(2, 2), padding='VALID', use_bias=False, input_shape=input_size))
model.add(Conv2D(32, (9, 9), strides=(1, 1), padding='VALID', activation='relu'))
model.add(Conv2D(32, (9, 9), strides=(1, 1), padding='VALID', activation='relu'))
model.add(Conv2D(64, (3, 3), strides=(3, 3), padding='VALID', activation='relu'))
model.add(Conv2D(1, (9, 9), strides=(1, 1), padding='VALID', activation='tanh'))
'''


# Creating one_hot_encoded array for tensorflow
def one_hot_encoded(one_dim_tensor):
    shape = tf.shape(one_dim_tensor)
    encoded1 = tf.subtract(1.0, one_dim_tensor)
    encoded2 = tf.concat([one_dim_tensor, encoded1], 3)
    encoded = tf.reshape(encoded2, [shape[0], 2])

    return encoded


def create_model(lr, memory):

    model = {}

    input_variable = tf.placeholder(tf.float32, shape=(None, u.input_size[0], u.input_size[1], u.input_size[2]), name='input')

    value = 0.01 * np.ones(shape=(1, u.input_size[0], u.input_size[1], u.input_size[2]), dtype=np.float32)
    value[0, 43, 43, 0] *= 10.0
    value[0, 43, 44, 0] *= 10.0
    value[0, 44, 43, 0] *= 10.0
    value[0, 44, 44, 0] *= 10.0
    weights = tf.constant(value, tf.float32, shape=(1, u.input_size[0], u.input_size[1], u.input_size[2]))

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

        fixed_weights = tf.constant(1.0, dtype=tf.float32, shape=(2, 2, u.input_size[2], 1))

        multiply1 = tf.multiply(input_variable, weights)
        locally1 = tf.nn.conv2d(multiply1, fixed_weights, padding='SAME', strides=[1, 2, 2, 1])

        return locally1

    locally = locally_conv_2d(input_variable, weights)
    conv1 = tf.nn.relu(tf.nn.conv2d(locally, w(9, 9, 1, 32), padding='VALID', strides=[1, 1, 1, 1]) + b(32))
    conv2 = tf.nn.relu(tf.nn.conv2d(conv1, w(9, 9, 32, 32), padding='VALID', strides=[1, 1, 1, 1]) + b(32))
    conv3 = tf.nn.relu(tf.nn.conv2d(conv2, w(3, 3, 32, 64), padding='VALID', strides=[1, 3, 3, 1]) + b(64))
    conv4 = tf.nn.sigmoid(tf.nn.conv2d(conv3, w(9, 9, 64, 1), padding='VALID', strides=[1, 1, 1, 1]) + b(1), name='fwd')
    classes = tf.greater(conv4, 0.5, name='classes')  # returns a tensor with the same size as the input

    correct = tf.placeholder(tf.int32, shape=(None, u.output_size[0], u.output_size[1], u.output_size[2]))
    correct_prediction = tf.equal(classes, tf.greater(correct, 0))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    loss = tf.losses.mean_squared_error(labels=tf.cast(correct, tf.float32), predictions=conv4)
    train_step = tf.train.AdamOptimizer(lr).minimize(loss)
    
    gpu_opt = tf.GPUOptions(per_process_gpu_memory_fraction=memory)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_opt))
    sess.run(tf.global_variables_initializer())

    model['sess'] = sess
    model['forward'] = conv4
    model['train'] = train_step
    model['loss'] = loss
    model['acc'] = accuracy
    model['x'] = input_variable
    model['y'] = correct

    return model


def metrics_names(model):
    return ["train_loss", "train_acc", "test_loss", "test_acc"]


def train_batch(model, data_chunk, epochs):
    sum_loss = 0.0
    sum_acc = 0.0
    for epoch in range(epochs):
        result = model['sess'].run([model['train'], model['loss'], model['acc']], feed_dict={model['x']: data_chunk.get_x(), model['y']: data_chunk.get_y()})
        sum_loss += result[1]
        sum_acc += result[2]
    return [sum_loss/epochs, sum_acc/epochs]


def evaluate(model, test_set):

    evals = model['sess'].run([model['loss'], model['acc']], feed_dict={model['x']: test_set.get_x(), model['y']: test_set.get_y()})
    return [evals[0], evals[1]]


# Returns a numpy array as a result.
def predict(model, x):
    return model['sess'].run(model['forward'], feed_dict={model['x']: x})


def save_model(model, file_name):
    saver = tf.train.Saver()
    saver.save(model['sess'], file_name)


# http://cv-tricks.com/tensorflow-tutorial/save-restore-tensorflow-models-quick-complete-tutorial/
def load_model(file_name):
    sess = tf.Session()
    saver = tf.train.import_meta_graph(file_name + '.meta')
    saver.restore(sess, tf.train.latest_checkpoint('./')) # Find the remaining necessary files.

    # Restoring the input and output tensors.
    graph = tf.get_default_graph()
    fwd_op = graph.get_tensor_by_name('fwd:0')
    x = graph.get_tensor_by_name('input:0')

    model = {"sess": sess, "forward": fwd_op, "x": x}

    return model
