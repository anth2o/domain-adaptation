import math
import numpy as np
from keras.utils import Sequence
import matplotlib.pyplot as plt

class Generator(Sequence):
    def __init__(self, x, y, x_unlabelled, y_unlabelled, batch_size=32, print_lambda=True):
        self.x = x
        self.y = y['label']
        self.x_unlabelled = np.concatenate([x_unlabelled, x], axis=0)
        self.y_unlabelled = np.concatenate([y_unlabelled['domain'], y['domain']], axis=0)
        self.batch_size = batch_size
        self.indices = np.arange(self.x.shape[0])
        self.epochs = 0
        self.gamma = 1e-4
        self.print_lambda = print_lambda
        self.compute_lambda()
        np.random.shuffle(self.indices)
        self.indices_unlabelled = np.arange(self.x_unlabelled.shape[0])
        np.random.shuffle(self.indices_unlabelled)

    def __len__(self):
        return math.ceil(self.x.shape[0] / self.batch_size)

    def __getitem__(self, idx):
        inds = self.indices[idx * self.batch_size : (idx + 1) * self.batch_size]
        batch_x = self.x[inds]
        batch_y = self.y[inds]
        inds_unlabelled = self.indices_unlabelled[idx * self.batch_size : idx * self.batch_size + len(inds)]
        batch_x_unlabelled = self.x_unlabelled[inds_unlabelled]
        batch_y_unlabelled = self.y_unlabelled[inds_unlabelled]
        return [batch_x, batch_x_unlabelled, self._lambda * np.ones((batch_x.shape[0], 1))], \
            [batch_y, batch_y_unlabelled]
    
    def on_epoch_end(self):
        np.random.shuffle(self.indices)
        np.random.shuffle(self.indices_unlabelled)
        self.epochs += 1
        self.compute_lambda()

    def compute_lambda(self):
        self._lambda = 2.0 / (1 + np.exp(-self.gamma * self.epochs)) - 1
        if self.print_lambda:
            print('')
            print('Lambda value: ' + str(self._lambda))
