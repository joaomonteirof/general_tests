from keras.models import Sequential, load_model, Model
from keras.layers import Dense, Dropout, Activation, Flatten, Reshape, Input
from keras.layers.convolutional import Conv2D, Conv2DTranspose, UpSampling2D, ZeroPadding2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.noise import GaussianNoise
from keras import layers
from keras import models
from keras.layers.advanced_activations import LeakyReLU
from keras.utils import np_utils
from keras.callbacks import EarlyStopping, ModelCheckpoint

import numpy as np

model = Sequential()
model.add(Conv2D(10, 5, padding='valid', strides=(1,1), input_shape=(28, 28, 1)))
model.add(MaxPooling2D())
model.add(Activation('relu'))
model.add(Conv2D(20, 5, padding='valid', strides=(1,1)))
model.add(MaxPooling2D())
model.add(Dropout(0.5))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(50))
model.add(Dropout(0.5))
model.add(Activation('relu'))
model.add(Dense(10))
model.add(Activation('softmax'))

model.summary()
