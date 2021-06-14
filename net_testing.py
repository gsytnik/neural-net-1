from neural_net import Neural_Net
import numpy as np
from PIL import Image
from skimage.transform import resize
import random
import glob

def main():
	
	network = Neural_Net(720000, 4, 80, 1, 0.1)
	
	files = []
	for file in glob.glob('C:\\Users\\gsytn\\Documents\\SchoolAndCS\\ML\\neural-net-1\\test_data\\dataset\\training_set\\dogs\\*.jpg'):
		files.append(file)
	
	for file in glob.glob('C:\\Users\\gsytn\\Documents\\SchoolAndCS\\ML\\neural-net-1\\test_data\\dataset\\training_set\\cats\\*.jpg'):
		files.append(file)

	random.shuffle(files)
	
	network.load_training()

	for file in files:
		if 'cat' in file:
			expected = [0]
		else:
			expected = [1]

		img = Image.open(file)
		img = np.array(img)
		img = resize(img, (400, 600)).ravel()
		network.train(img, expected)
		if network.count > 200:
			break
	
	network.save_training()
		
	for file in glob.glob('C:\\Users\\gsytn\\Documents\\SchoolAndCS\\ML\\neural-net-1\\test_data\\dataset\\training_set\\dog\\*.jpg'):
		print("e")
		expected = [1]
		img = Image.open(file)
		img = np.array(img)
		img = resize(img, (400, 600)).ravel()
		network.run(img)

	


if __name__ == '__main__':
    main()