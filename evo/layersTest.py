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



model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28,28,1)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))


model.summary()
