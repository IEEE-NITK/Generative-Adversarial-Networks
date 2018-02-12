import numpy as np


class DataLoader():

    def __init__(self, image_path, metadeta_path, transform, mode):
        # load data here
        self.input = np.ones((500, 784))
        self.y = np.ones((500, 10))
        self.image = image_path
        self.transform = transform
        self.mode = mode
        self.lines = open(metadeta_path, "r").readlines()
        self.num_data = int

    def next_batch(self, batch_size):
        idx = np.random.choice(500, batch_size)
        yield self.input[idx], self.y[idx]
