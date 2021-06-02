from neural_net import Neural_Net
import numpy as np

def main():
	
	network = Neural_Net(9, 10, 10, 9)
	network.obtain_inputs([1, -1, 1, 0, 0, 0, -1, 1, 1])
	network.train()



if __name__ == '__main__':
    main()