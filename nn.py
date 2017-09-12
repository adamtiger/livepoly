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
from keras import backend as K

if K.backend() is not 'CNTK':
    raise AssertionError("Should be CNTK")

def create_model():

    model = Sequential()

    model.add(LocallyConnected2D(1, (2, 2), strides=(2, 2), padding='VALID', use_bias=False, input_shape=(86, 86, 1)))
    model.add(Conv2D(32, (9, 9), strides=(1, 1), padding='VALID', activation='relu'))
    model.add(Conv2D(32, (9, 9), strides=(1, 1), padding='VALID', activation='relu'))
    model.add(Conv2D(64, (3, 3), strides=(3, 3), padding='VALID', activation='relu'))
    model.add(Conv2D(1, (9, 9), strides=(1, 1), padding='VALID', activation='relu'))

    opt = Adam(lr=0.0005)
    loss = 'binary_crossentropy'
    model.compile(loss=loss, optimizer=opt)

    return model

def train_batch(model, batch, epochs):
    pass

