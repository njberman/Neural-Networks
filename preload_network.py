import random
import mnist_loader
import mynetwork as network
from os import listdir
from os.path import isfile, join
import pickle
import time
from datetime import datetime

training_data, validation_data, test_data = mnist_loader.load_data_wrapper(
    r'./data/mnist.pkl')

# num_of_networks = 20
# for i in range(num_of_networks):
#     start = time.time()
#     net = network.Network([784, 30, 10])
#     net.SGD(training_data, 10, 10, 3.0, test_data=test_data)
#     end = time.time()

#     print(
#         f'[{datetime.now()}] Network {i + 1} completed. Time taken: {round((end - start) / 60, 2)} min. ETA: {round(((end - start) * (num_of_networks - i)) / 60, 2)} min.')

mypath = './networks'
bestnetwork_path = './networks/' + str(max([float(f.replace('.pkl', ''))
                                            for f in listdir(mypath) if isfile(join(mypath, f))])) + '.pkl'

with open(bestnetwork_path, 'rb') as f:
    biases, weights = pickle.load(f)
    net = network.Network([784, 30, 10], biases=biases, weights=weights)


def increment():
    return random.randint(0, len(test_data))
