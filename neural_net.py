import numpy as np
import math
import random

beta = 1.5
np.set_printoptions(precision=5)

class Neural_Net:

	# create a neural network which is generated via a specific number of inputs, 
	# a specific number of layers, and a number of outputs. 
	def __init__(self, num_inputs, num_layers, layer_size, num_outputs):
		print("creating network with:\n\n\t{0} inputs,\n\t{1} layers of size {2},\n\t{3} outputs.".format(
			num_inputs, num_layers, layer_size, num_outputs)
		)
		self.num_layers = num_layers
		self.num_inputs = num_inputs
		self.layer_size = layer_size
		self.num_outputs = num_outputs
		self.init_weights()
		self.init_biases()
		self.init_neurons()
		
	# inits neural networks weights ranging from -2.0 to 2 to start.
	def init_weights(self):

		# create emptry array to store weights for all layers
		self.weights = [] 

		# create first set of weights from input layer to first hidden layer
		self.weights.append(np.random.uniform(low= -1.5, high = 1.5, size=(self.layer_size, self.num_inputs)))
		# for each hidden layer
		for layer in range(self.num_layers):
			# if it is the last layer, append weight matrix size (num_outputs, layer_size)
			if layer == self.num_layers - 1:
				self.weights.append(np.random.uniform(low= -1.5, high = 1.5, size=(self.num_outputs, self.layer_size)))
			# else, append weight matrix size (layer_size, layer_size)
			else:
				self.weights.append(np.random.uniform(low= -1.5, high= 1.5, size=(self.layer_size, self.layer_size)))

	# init biases to add to each result.
	def init_biases(self):
		self.biases = []
		# for each hidden layer + the output layer
		for i in range(self.num_layers + 1):
			# if output layer, append array of size num_outputs with biases between -5, 5
			if i == self.num_layers:
				self.biases.append(np.random.uniform(low = -5.0, high = 5.0, size = self.num_outputs))
			# else, append array of size layer_size with biases between -5, 5
			else:
				self.biases.append(np.random.uniform(low = -5.0, high = 5.0, size = self.layer_size))

	# init neurons.
	def init_neurons(self):
		self.neurons = []
		# for each hidden layer + the output layer
		for i in range(self.num_layers + 1):
			# if output layer, append array of size num_outputs with biases between -5, 5
			if i == self.num_layers:
				self.neurons.append(np.zeros(self.num_outputs))
			# else, append array of size layer_size with biases between -5, 5
			else:
				self.neurons.append(np.zeros(self.layer_size))

	# take in input array and convert to np array.
	def obtain_inputs(self, inputs):
		self.current_inputs = np.asarray(inputs)

	# basic feed forward algorithm to generate some output.
	# first take inputs, multiply by weight matrix, 
	# then add biases to result, 
	# then run through activation function.
	def feed_forward(self, inputs, layer):	
		self.neurons[layer] = np.matmul(self.weights[layer], inputs) + self.biases[layer]
		self.neurons[layer] = np.array(list(map(activ_function, self.neurons[layer])))

		if layer < self.num_layers:
			self.feed_forward(self.neurons[layer], layer + 1)

	# function to train this algorithm
	def train(self):
		self.feed_forward(self.current_inputs, 0)
		print(self.current_inputs, self.neurons[-1])

			
	def print_weights(self, layer):
		print(self.weights[layer])
		print()

	def print_biases(self):
		for layer in range(self.num_layers + 1):
			print(self.biases[layer])
			print()

	def print_all_weights(self):

		for layer in range(self.num_layers + 1):
			print("layer {0}:\n".format(layer - 1))
			self.print_weights(layer)

def sigmoid(x):
	if x >= 0 :
		z =  np.exp(-x)
		return 1 / (1 + z)
	else:
		z = np.exp(x)
		return z / (1 + z)

def activ_function(x):
	return beta * x * sigmoid(x)

def activ_deriv(x):
	return activ_function(x) + (sigmoid(x) * (beta - activ_function(x)))