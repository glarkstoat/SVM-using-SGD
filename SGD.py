from typing import Tuple

import numpy as np


class SGD:
    def __init__(self, learning_rate):
        self.weight = None
        self.learning_rate = learning_rate

    def predict(self, features):
        if self.weight is None:
            print("You need to train first!")
            return None
        result = self.weight[0]
        result += sum(self.weight[i + 1] * features[i] for i in range(len(features) - 1))
        return result

    def train(self, xtrain, ytrain, epoch_count):
        if len(xtrain) != len(ytrain):
            print("Check your training data dimensions.")
        data_count = len(xtrain)
        for epoch in range(epoch_count):
            error = 0
            # todo should we iterate over all items??
            for i in range(data_count):
                itemx = xtrain[i]
                itemy = ytrain[i]
                if self.weight is None:
                    self.weight = np.zeros(len(itemx) + 1)
                prediction = self.predict(itemx)
                # we sum up squared errors
                current_error = prediction - itemy
                error += current_error ** 2
                # optimize weights
                self.weight[0] -= self.learning_rate * current_error
                self.weight[1:] -= self.learning_rate * current_error * itemx[0:]
            print(f"Epoch: {epoch}. Total error: {error}")
