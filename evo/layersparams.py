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

def tensorElementsCount(myTensor):

	tensorShape = myTensor.shape
	accum = 1

	for element in tensorShape:
		accum *= element

	return accum

def countParameters(model):
	#model.summary()
	totalParameters = 0
	layersToPrint = model.layers
	for layer in layersToPrint:
		paramsList = layer.get_weights()

		for params in paramsList:
			try:
				totalParameters += tensorElementsCount(params)
			except AttributeError:
				totalParameters += len(params)

	print(totalParameters)
	return totalParameters

def printParametersShape(model):
	model.summary()
	layersToPrint = model.layers
	for layer in layersToPrint:
		params = layer.get_weights()

		for a in params:
			try:
				print(a.shape)
			except AttributeError:

				print(len(a))


def setLayersParameters(model):

	totalParameters = countParameters(model)

	parametersCopy = np.ones(totalParameters)

	layersList = model.layers

	for layerToUpdate in layersList:

		paramsToUpdate = []
		paramsList = layerToUpdate.get_weights()

		for params in paramsList:

			try:
				shapeWeights = params.shape
				numberOfParameters = tensorElementsCount(params)
				newParameters = parametersCopy[0:numberOfParameters]
				newParameters = newParameters.reshape(shapeWeights)
			
			except AttributeError:
				numberOfParameters = len(params)
				newParameters = parametersCopy[0:numberOfParameters]

			paramsToUpdate.append(newParameters)

			parametersCopy = np.delete(parametersCopy, range(numberOfParameters))

		layerToUpdate.set_weights(paramsToUpdate)
	print 'done!'

encoder = Sequential()
encoder.add(GaussianNoise(0.02, input_shape=(64, 64, 3)))
encoder.add(Conv2D(64, 5, padding='same', strides=(1,1)))
encoder.add(BatchNormalization())
encoder.add(Activation('relu'))
encoder.add(Conv2D(64, 5, padding='same', strides=(1,1)))
encoder.add(BatchNormalization())
encoder.add(Activation('relu'))


encoder.add(Conv2D(64, 5, padding='same', strides=(2,2))) #64-32
encoder.add(BatchNormalization())
encoder.add(Activation('relu'))
encoder.add(GaussianNoise(0.02))

encoder.add(Conv2D(128, 5, padding='same', strides=(1,1)))
encoder.add(BatchNormalization())
encoder.add(Activation('relu'))

encoder.add(Conv2D(128, 5, padding='same', strides=(2,2))) #32-16
encoder.add(BatchNormalization())
encoder.add(Activation('relu'))
encoder.add(GaussianNoise(0.02))
encoder.add(Conv2D(256, 5, padding='same', strides=(1,1)))
encoder.add(BatchNormalization())
encoder.add(Activation('relu'))

encoder.add(Conv2D(256, 5, padding='same', strides=(2,2))) #16-8
encoder.add(BatchNormalization())
encoder.add(Activation('relu'))
encoder.add(GaussianNoise(0.02))
encoder.add(Conv2D(512, 5, padding='same', strides=(1,1)))
encoder.add(BatchNormalization())
encoder.add(Activation('relu'))

encoder.add(Conv2D(512, 5, padding='same', strides=(2,2))) #8-4
encoder.add(BatchNormalization())
encoder.add(Activation('relu'))
encoder.add(GaussianNoise(0.02))


encoder.add(Conv2D(512, 5, padding='same', strides=(1,1))) #4-4
encoder.add(BatchNormalization())
encoder.add(Activation('relu'))

# decoder

decoder = Sequential()

decoder.add(Conv2DTranspose(512, 5, padding='same', strides=(2,2), input_shape=(4, 4, 512))) #4-8
decoder.add(BatchNormalization())
decoder.add(Activation('relu'))
decoder.add(GaussianNoise(0.02))
decoder.add(Conv2D(256, 5, padding='same', strides=(1,1)))
decoder.add(BatchNormalization())
decoder.add(Activation('relu'))
decoder.add(Conv2D(256, 5, padding='same', strides=(1,1)))
decoder.add(BatchNormalization())
decoder.add(Activation('relu'))

decoder.add(Conv2DTranspose(256, 5, padding='same', strides=(2,2))) #8-16
decoder.add(BatchNormalization())
decoder.add(Activation('relu'))
decoder.add(GaussianNoise(0.02))
decoder.add(Conv2D(128, 5, padding='same', strides=(1,1)))
decoder.add(BatchNormalization())
decoder.add(Activation('relu'))
decoder.add(Conv2D(128, 5, padding='same', strides=(1,1)))
decoder.add(BatchNormalization())
decoder.add(Activation('relu'))

decoder.add(Conv2DTranspose(128, 5, padding='same', strides=(2,2))) #16-32
decoder.add(BatchNormalization())
decoder.add(Activation('relu'))
decoder.add(GaussianNoise(0.02))
decoder.add(Conv2DTranspose(3, 5, padding='same', strides=(1,1)))
decoder.add(Activation('sigmoid'))

#merger

i1 = layers.Input(shape=(32, 32, 3))
i2 = layers.Input(shape=(64, 64, 3))
i1p = ZeroPadding2D(padding=(16, 16)) (i1)
o = layers.add([i1p, i2])
merger_model = models.Model([i1, i2], o)

# discriminator

discriminator = Sequential()

discriminator.add(merger_model)

discriminator.add(Conv2D(64, 5, padding='same', strides=(2, 2)))
discriminator.add(Dropout(0.5))
discriminator.add(BatchNormalization())
discriminator.add(LeakyReLU(alpha=0.2))
discriminator.add(Conv2D(128, 5, padding='same', strides=(2, 2)))
discriminator.add(Dropout(0.5))
discriminator.add(BatchNormalization())
discriminator.add(LeakyReLU(alpha=0.2))
discriminator.add(Conv2D(256, 5, padding='same', strides=(2, 2)))
discriminator.add(Dropout(0.5))
discriminator.add(LeakyReLU(alpha=0.2))
discriminator.add(Conv2D(512, 5, padding='same', strides=(2, 2)))
discriminator.add(Dropout(0.5))
discriminator.add(LeakyReLU(alpha=0.2))
discriminator.add(Conv2D(512, 5, padding='same'))
discriminator.add(Dropout(0.5))
discriminator.add(BatchNormalization())
discriminator.add(LeakyReLU(alpha=0.2))
discriminator.add(Flatten())
discriminator.add(Dense(512))
discriminator.add(LeakyReLU(alpha=0.2))
discriminator.add(Dropout(0.5))
discriminator.add(Dense(2))
discriminator.add(Activation('softmax'))

encoder.compile(loss='mean_absolute_error', optimizer='adam')
decoder.compile(loss='mean_absolute_error', optimizer='adam')
discriminator.compile(loss='categorical_crossentropy', optimizer='sgd')


#printParametersShape(encoder)
#countParameters(discriminator)
setLayersParameters(encoder)
