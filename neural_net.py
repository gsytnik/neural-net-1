import numpy as np
import random

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
		self.init_neural_network()
		

	def init_neural_network(self):
		self.input_weights = np.asarray([[random.uniform(-2.0, 2) for i in range(self.num_inputs)] for i in range(self.layer_size)])
		self.hidden_weights = [] 
		for layer in range(self.num_layers):
			if layer == self.num_layers - 1:
				self.hidden_weights.append(np.random.uniform(low= -2.0, high = 2.0, size=(self.num_outputs, self.layer_size)))
			else:
				self.hidden_weights.append(np.random.uniform(low= -2.0, high= 2.0, size=(self.layer_size, self.layer_size)))
			
	def print_hidden_weights(self, layer):
		print(self.hidden_weights[layer])
		print()

	def print_input_weights(self):
		print(self.input_weights)

	def print_all_weights(self):
		print("input layer:\n")
		self.print_input_weights()

		for layer in range(self.num_layers):
			print("layer {0}:\n".format(layer))
			self.print_hidden_weights(layer)