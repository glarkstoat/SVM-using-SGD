import numpy as np
import threading

from DataUtils import DataUtils


class ParallelSGD:
    def __init__(self, learning_rate, thread_count, regularization):
        self.learning_rate = learning_rate
        self.thread_count = thread_count
        self.weights = None  # map (thread_number, weights)
        self.examples_per_thread = None
        self.threads = []
        self.regularization = regularization

    def predict_for_thread(self, features, thread_number):
        if self.weights[thread_number] is None:
            print("You need to train first!")
            return None
        weight = self.weights[thread_number]
        return np.dot(weight, features)

    def train(self, xtrain, ytrain):
        self.examples_per_thread = xtrain.shape[1] / self.thread_count
        self.weights = {}
        for i in range(self.thread_count):
            self.weights[i] = np.zeros(xtrain.shape[1])
            thread = threading.Thread(target=self.train_threaded, args=(i, xtrain, ytrain))
            self.threads.append(thread)
            thread.start()

        for thread in self.threads:
            thread.join()

    @staticmethod
    def hinge_gradient(weight, sample, label, regularization):
        return 2 * regularization * weight - label * sample

    def train_threaded(self, thread_number, xtrain, ytrain):
        xtrain_shuffled, ytrain_shuffled = DataUtils.shuffle_data(xtrain, ytrain)
        for i in range(self.examples_per_thread):
            itemx, itemy = xtrain_shuffled[i], ytrain_shuffled[i]
            weight = self.weights[thread_number]
            prediction = self.predict_for_thread(itemx, thread_number)
            if itemy * prediction < 1:  # either within margin or incorrectly classified
                weight -= self.learning_rate * self.hinge_gradient(weight, itemx, itemy, self.regularization)

    def total_weight(self):
        return np.mean(self.weights.values())
