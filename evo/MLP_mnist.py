from __future__ import print_function

import os
import sys
import timeit

#import wget
import cPickle
import gzip

import numpy as np

import theano
import theano.tensor as T

import random

from deap import base
from deap import creator
from deap import tools

theano.config.floatX='float32'

def mnistOutput(softmaxOutput):	#Deve ser 2D com a saida de cada exemplo organizada em cada coluna

	maxPosPerColumn	= softmaxOutput.argmax(axis=0)
				
	return maxPosPerColumn;

def accuracyMeasure(output, target):
	
	a=1.0*(output==target)
	accuracy = np.sum(a)/a.shape[0]
	accuracy = np.float32(accuracy).item()
	return (accuracy,);

def softmax(outputs):
	
	maxOutput = np.max(outputs)
	outputs = outputs-maxOutput
	Exps = np.exp(outputs)
	ExpsSum = sum(Exps)
	normOutputs = Exps/ExpsSum

	return normOutputs;

class HiddenLayer(object):

    def __init__(self, rng, input, n_in, n_out, W=None, b=None, activation=True):

        self.input = input

	if W is None:
		W_values = np.asarray(rng.uniform(low=-np.sqrt(6. / (n_in + n_out)), high=np.sqrt(6. / (n_in + n_out)), size=(n_out, n_in)), dtype=theano.config.floatX)

	if activation:
		W_values *= 4
		
	W = theano.shared(value=W_values, name='W', borrow=True)

	if b is None:
		b_values = np.zeros((n_out,), dtype=theano.config.floatX)
		b = theano.shared(value=b_values, name='b', borrow=True)
	
	self.W = W
	self.b = b

	lin_output = T.dot(self.W, input).T + self.b.T

	if activation:
		self.output=T.nnet.sigmoid(lin_output).T
	else:
		self.output=T.nnet.relu(lin_output).T

	self.params = [self.W, self.b]

class OutputLayer(object):

    def __init__(self, rng, input, n_in, n_out, W=None, b=None, activation=True):

        self.input = input

	if W is None:
		W_values = np.asarray(rng.uniform(low=-np.sqrt(6. / (n_in + n_out)), high=np.sqrt(6. / (n_in + n_out)), size=(n_out, n_in)), dtype=theano.config.floatX)

	if activation:
		W_values *= 4
	
	W = theano.shared(value=W_values, name='W', borrow=True)

	if b is None:
		b_values = np.zeros((n_out,), dtype=theano.config.floatX)
		b = theano.shared(value=b_values, name='b', borrow=True)
	
	self.W = W
	self.b = b

	lin_output = T.dot(self.W, input).T + self.b.T

	self.output=T.nnet.softmax(lin_output).T

	self.params = [self.W, self.b]

class MLP(object):

	def __init__(self, rng, input, targets, n_in, n_hidden, n_out, n_hiddenLayers, activation):

		self.layersList = []

		self.layersList.append(HiddenLayer(rng=rng, input=input, n_in=n_in, n_out=n_hidden, activation=activation))

		if (n_hiddenLayers>1):

			for i in range(n_hiddenLayers-1):
				self.layersList.append(HiddenLayer(rng=rng, input=self.layersList[i].output, n_in=n_hidden, n_out=n_hidden, activation=activation))

		self.layersList.append(OutputLayer(rng=rng, input=self.layersList[-1].output, n_in=n_hidden, n_out=n_out, activation=activation))

		self.L1=0
		self.L2=0

		for layer in self.layersList:		

			self.L1 += abs(layer.W.sum())

		for layer in self.layersList:		

			self.L2 += (layer.W**2).sum()

		self.lossBeforeReg = T.nnet.categorical_crossentropy(self.layersList[-1].output, targets).mean()

		self.params = []
		for parSet in self.layersList:
			self.params+=parSet.params

class HiddenLayerGA(object):

	def __init__(self, rng, inputData, n_in, n_out, W=None, b=None, activation=True):

		self.activation=activation

		if W is None:
			W_values = np.asarray(rng.uniform(low=-np.sqrt(6. / (n_in + n_out)), high=np.sqrt(6. / (n_in + n_out)), size=(n_out, n_in)), dtype=theano.config.floatX)
			W_values *= 10
		else:
			W_values = W

		if b is None:
			b_values = np.zeros((n_out,), dtype=theano.config.floatX)
		else:	
			b_values = b

		self.W = W_values
		self.b = b_values

		self.updateOutput(inputData)

		self.L1 = sum(sum(abs(self.W)))
		self.L2 = sum(sum(self.W**2))

		self.params = [self.W, self.b]	

	def updateOutput(self, input):

		lin_output = (self.W.dot(input)).T + self.b.T

		if self.activation:
			self.output=(1/np.exp(-lin_output)).T
		else:
			self.output=T.nnet.relu(lin_output).T

		return self.output;

class OutputLayerGA(object):

	def __init__(self, rng, inputData, n_in, n_out, W=None, b=None, activation=True):

		self.activation=activation

		if W is None:
			W_values = np.asarray(rng.uniform(low=-np.sqrt(6. / (n_in + n_out)), high=np.sqrt(6. / (n_in + n_out)), size=(n_out, n_in)), dtype=theano.config.floatX)
			W_values *= 10
		else:
			W_values = W

		if b is None:
			b_values = np.zeros((n_out,), dtype=theano.config.floatX)
		else:
			b_values = b
	
		self.W = W_values
		self.b = b_values

		self.L1 = sum(sum(abs(self.W)))
		self.L2 = sum(sum(self.W**2))

		self.updateOutput(inputData)

		self.params = [self.W, self.b]

	def updateOutput(self, input):

		lin_output = (self.W.dot(input)).T + self.b.T

		self.output=softmax(lin_output.T)

		return self.output;

class MLPGA(object):

	def __init__(self, rng, inputData, targets, n_in, n_hidden, n_out, n_hiddenLayers, activation, parameters):

		self.layersList = []
		
		parametersCopy = parameters

		self.layersList.append(HiddenLayerGA(rng, inputData, n_in, n_hidden, parametersCopy[0:(n_in*n_hidden)].reshape(n_hidden, n_in), parametersCopy[(n_in*n_hidden):(n_in*n_hidden+n_hidden)], activation))

		parametersCopy = np.delete(parametersCopy, range(n_in*n_hidden+n_hidden))

		if (n_hiddenLayers>1):

			for i in range(n_hiddenLayers-1):
				self.layersList.append(HiddenLayerGA(rng, self.layersList[i].output, n_hidden, n_hidden, parametersCopy[0:(n_hidden*n_hidden)].reshape(n_hidden,n_hidden), parametersCopy[(n_hidden*n_hidden):(n_hidden*n_hidden+n_hidden)], activation))
				parametersCopy = np.delete(parametersCopy, range(n_hidden*n_hidden+n_hidden))

		self.layersList.append(OutputLayerGA(rng, self.layersList[-1].output, n_hidden, n_out, parametersCopy[0:(n_out*n_hidden)].reshape(n_out,n_hidden), b=parametersCopy[(n_out*n_hidden):(n_out*n_hidden+n_out)], activation=activation))

		self.output = self.layersList[-1].output

		self.lossBeforeReg = T.nnet.categorical_crossentropy(self.layersList[-1].output, targets).mean()

		self.L1Loss = 0
		self.L2Loss = 0

		for layer in self.layersList:
			self.L1Loss+=layer.L1
		for layer in self.layersList:
			self.L2Loss+=layer.L2

		self.cost = self.lossBeforeReg + self.L1Loss + self.L2Loss

	def runForward(self, inputData):

		currentOutput=self.layersList[0].updateOutput(inputData)
		for i in range(1,len(self.layersList)):
			currentOutput = self.layersList[i].updateOutput(currentOutput)

		self.output=currentOutput
		return currentOutput;

class SGDFit():

	def __init__(self, n_inputs, n_hidden, n_outputs, n_hidLayers, activation, learning_rate=0.01, L1_reg=0.00, L2_reg=0.0001, n_epochs=1000, batch_size=100):

		self.learningRate = learning_rate
		self.L1 = L1_reg
		self.L2 = L2_reg
		self.numberOfEpochs = n_epochs
		self.batchSize = batch_size
		self.numberOfInputs = n_inputs
		self.numberOfHiddenUnits = n_hidden
		self.numberOfOutputs = n_outputs
		self.numberOfHiddenLayers = n_hidLayers
		self.input = T.matrix()
		self.targets = T.matrix()
		self.activation = activation
		self.model = MLP(np.random.RandomState(1234), self.input, self.targets, self.numberOfInputs, self.numberOfHiddenUnits, self.numberOfOutputs, self.numberOfHiddenLayers, self.activation)

	def modelFit(self):

		modelOutput = self.model.layersList[-1].output
		modelLoss = self.model.lossBeforeReg + self.L1*self.model.L1 + self.L2*self.model.L2

		gparams = [T.grad(modelLoss, param) for param in self.model.params]

		updates = [(param, param - self.learningRate * gparam) for param, gparam in zip(self.model.params, gparams)]

		cost = theano.function(inputs=[self.input, self.targets], outputs=modelLoss, updates=updates)
		runForward = theano.function(inputs=[self.input], outputs=modelOutput)
		

		with gzip.open('mnist.pkl.gz', 'rb') as f:
			train_set, valid_set, test_set = cPickle.load(f)

		toShufflePermutation = np.random.permutation(len(train_set[0]))
		x_trainToShuffle = np.array(train_set[0], dtype=theano.config.floatX)
		y_trainToShuffle = np.array(train_set[1], dtype=theano.config.floatX)

		x_train = x_trainToShuffle[toShufflePermutation].T
		y_trainPre = y_trainToShuffle[toShufflePermutation]
		#x_trainPre = train_set[0].T
		#y_train_pre = train_set[1]
		x_valid = np.array(valid_set[0].T, dtype=theano.config.floatX)
		y_valid = np.array(valid_set[1], dtype=theano.config.floatX)
		x_testPre = np.array(test_set[0].T, dtype=theano.config.floatX)
		y_test = np.array(test_set[1], dtype=theano.config.floatX)

		y_train = np.zeros((self.numberOfOutputs,x_train.shape[1]),dtype=theano.config.floatX)
		for i in range(y_train.shape[1]):
			y_train[y_trainPre[i].astype('int'),i]=np.float32(1.0)

		numberOfBatches = np.ceil(1.0*x_train.shape[1]/(1.0*self.batchSize)).astype('int64')

		for i in range(self.numberOfEpochs):

			for j in range(numberOfBatches):

				x_trainBatch = x_train[:,(j*self.batchSize):min(((j+1)*self.batchSize),x_train.shape[1])]
				y_trainBatch = y_train[:,(j*self.batchSize):min(((j+1)*self.batchSize),x_train.shape[1])]

				cur_cost = cost(x_trainBatch, y_trainBatch)
		
		self.validAccuracy = accuracyMeasure(mnistOutput(runForward(x_valid)), y_valid)[0]

class GAFit():

	def __init__(self, n_inputs, n_hidden, n_outputs, n_hidLayers, activation, L1_reg=0.00, L2_reg=0.0001, n_epochs=1000)

		with gzip.open('mnist.pkl.gz', 'rb') as f:
			train_set, valid_set, test_set = cPickle.load(f)

		self.L1 = L1_reg
		self.L2 = L2_reg
		self.numberOfEpochs = n_epochs
		self.numberOfInputs = n_inputs
		self.numberOfHiddenUnits = n_hidden
		self.numberOfOutputs = n_outputs
		self.numberOfHiddenLayers = n_hidLayers
		self.activation = activation
		self.totalNumberOfParameters = n_inputs*n_hidden + n_hidden + n_hidden*n_outputs + n_outputs + max(0,(n_hidLayers-1))*(n_hidden**2+n_hidden)
		self.x_train = np.array(train_set[0], dtype=theano.config.floatX).T
		self.y_train = np.array(train_set[1], dtype=theano.config.floatX)
		self.x_valid = np.array(valid_set[0], dtype=theano.config.floatX).T
		self.y_valid = np.array(valid_set[1], dtype=theano.config.floatX)

		y_trainOneHot = np.zeros((self.numberOfOutputs,self.x_train.shape[1]),dtype=theano.config.floatX)
		y_validOneHot = np.zeros((self.numberOfOutputs,self.x_valid.shape[1]),dtype=theano.config.floatX)
		
		for i in range(y_trainOneHot.shape[1]):
			y_trainOneHot[self.y_train[i].astype('int'),i]=np.float32(1.0)

		for i in range(y_validOneHot.shape[1]):
			y_validOneHot[self.y_valid[i].astype('int'),i]=np.float32(1.0)

		self.y_trainOneHot = y_trainOneHot
		self.y_validOneHot = y_validOneHot

	def GAEvaluate(self, individual):

		MLPIndividual = MLPGA(np.random.RandomState(1234), self.x_train, self.y_trainOneHot, self.numberOfInputs, self.numberOfHiddenUnits, self.numberOfOutputs, self.numberOfHiddenLayers, self.activation, np.asarray(individual))

		return accuracyMeasure(mnistOutput(MLPIndividual.output), self.y_train);

	def GAEvaluateValidationAccuracy(self, individual):

		MLPIndividual = MLPGA(np.random.RandomState(1234), self.x_valid, self.y_validOneHot, self.numberOfInputs, self.numberOfHiddenUnits, self.numberOfOutputs, self.numberOfHiddenLayers, self.activation, np.asarray(individual))
		return accuracyMeasure(mnistOutput(MLPIndividual.output), self.y_valid);

	def modelFit(self):

		creator.create("FitnessMax", base.Fitness, weights=(1.0,))
		creator.create("Individual", list, fitness=creator.FitnessMax)

		toolbox = base.Toolbox()
		toolbox.register("par_value", random.uniform, -10.0, 10.0)
		toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.par_value, self.totalNumberOfParameters)
		toolbox.register("population", tools.initRepeat, list, toolbox.individual)
		
		toolbox.register("evaluation", self.GAEvaluate)
		toolbox.register("mate", tools.cxSimulatedBinaryBounded, eta=5, low=-10, up=10)
		toolbox.register("mutate", tools.mutGaussian, mu=0.0, sigma=0.5, indpb=0.05)
		toolbox.register("select", tools.selTournament, tournsize=3)

		random.seed(64)

		pop = toolbox.population(n=300)

		CXPB, MUTPB, NGEN = 0.5, 0.2, self.numberOfEpochs

		fitnesses = list(map(toolbox.evaluation, pop))
	
		for ind, fit in zip(pop, fitnesses):
			ind.fitness.values = fit

		for g in range(NGEN):

			# Select the next generation individuals
			offspring = toolbox.select(pop, len(pop))
			# Clone the selected individuals
			offspring = list(map(toolbox.clone, offspring))

			# Apply crossover and mutation on the offspring
			for child1, child2 in zip(offspring[::2], offspring[1::2]):

				# cross two individuals with probability CXPB
				if random.random() < CXPB:
					toolbox.mate(child1, child2)

					# fitness values of the children
					# must be recalculated later
					del child1.fitness.values
					del child2.fitness.values

			for mutant in offspring:

				# mutate an individual with probability MUTPB
				if random.random() < MUTPB:
					toolbox.mutate(mutant)
					del mutant.fitness.values

			# Evaluate the individuals with an invalid fitness
			invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
			fitnesses = map(toolbox.evaluation, invalid_ind)
			for ind, fit in zip(invalid_ind, fitnesses):
				ind.fitness.values = fit

			# The population is entirely replaced by the offspring
			pop[:] = offspring

			# Gather all the fitnesses in one list and print the stats
			fits = [ind.fitness.values[0] for ind in pop]

			best_ind = tools.selBest(pop, 1)[0]
			self.validAccuracy = self.GAEvaluateValidationAccuracy(best_ind)[0]

def test_model():

	#model=SGDFit(n_inputs=784, n_hidden=32, n_outputs=10, n_hidLayers=11, activation=True)

	model=GAFit(n_inputs=784, n_hidden=32, n_outputs=10, n_hidLayers=7, activation=True)

	model.modelFit()

	print(model.validAccuracy)

if __name__ == '__main__':

	test_model()
