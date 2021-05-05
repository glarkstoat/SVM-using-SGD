from typing import Tuple


class SGD:
    def __init__(self, initial_weight, learning_rate):
        self.weight = initial_weight
        self.learning_rate = learning_rate

    def predict(self, features):
        result = self.weight[0]
        result += sum(self.weight[i + 1] * features[i] for i in range(len(features) - 1))
        return result

    def train(self, training_data: Tuple[list, list], epoch_count):
        for epoch in range(epoch_count):
            error = 0
            for itemx, itemy in training_data:
                prediction = self.predict(itemx)
                # we sum up squared errors
                current_error = prediction - itemy
                error += current_error ** 2
                # optimize weights
                self.weight[0] -= self.learning_rate * current_error
                self.weight[1:] -= self.learning_rate * current_error * itemx[1:]
            print(f"Epoch: {epoch}. Total error: {error}")
