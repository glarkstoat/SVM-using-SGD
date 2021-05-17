import math
from datetime import time, datetime

import numpy as np
import threading

from tqdm import tqdm

from DataUtils import DataUtils


class ParallelSGD:
    def __init__(self, learning_rate, thread_count, regularization, loss_function, accuracy_function):
        self.learning_rate = learning_rate
        self.thread_count = thread_count
        self.weights = None  # map (thread_number, weights)
        self.examples_per_thread = None
        self.threads = []
        self.regularization = regularization
        self.losses = {}
        self.accuracies = {}
        self.loss_function = loss_function
        self.accuracy_function = accuracy_function

    def predict_for_thread(self, features, thread_number):
        if self.weights[thread_number] is None:
            print("You need to train first!")
            return None
        return np.dot(self.weights[thread_number], features)

    def train(self, xtrain, ytrain):
        self.examples_per_thread = math.floor(xtrain.shape[0] / self.thread_count)
        print(f"Examples per thread: {self.examples_per_thread}")
        self.weights = {}
        start = datetime.now()
        for i in range(self.thread_count):
            self.weights[i] = np.zeros(xtrain.shape[1])
            self.losses[i] = []
            self.accuracies[i] = []
            thread = threading.Thread(target=self.train_threaded, args=(i, xtrain, ytrain))
            self.threads.append(thread)
            thread.start()

        for thread in self.threads:
            thread.join()

        print(f"Finished in {datetime.now() - start}")
        return self.losses, self.accuracies

    @staticmethod
    def hinge_gradient(weight, sample, label, regularization):
        return 2 * regularization * weight - label * sample

    def train_threaded(self, thread_number, xtrain, ytrain):
        try:
            print(f"Thread {thread_number} spawned")
            xtrain_shuffled, ytrain_shuffled = DataUtils.shuffle_data(xtrain, ytrain)
            weight = self.weights[thread_number]
            for i in range(self.examples_per_thread):
                itemx, itemy = xtrain_shuffled[i], ytrain_shuffled[i]
                prediction = self.predict_for_thread(itemx, thread_number)
                # if itemy * prediction < 1:  # either within margin or incorrectly classified
                #     weight -= self.learning_rate * self.hinge_gradient(weight, itemx, itemy, self.regularization)
                if itemy * prediction < 1:
                    weight -= self.learning_rate * self.hinge_gradient(weight, itemx, itemy, self.regularization)
                self.losses[thread_number].append(self.loss_function(xtrain, ytrain, weight))
                self.accuracies[thread_number].append(self.accuracy_function(xtrain, ytrain, weight))
            self.weights[thread_number] = weight
            print(f"Thread {thread_number} finished")
        except Exception as e:
            print(e)

    def total_weight(self):
        return np.mean([np.array(item) for item in self.weights.values()], axis=0)
