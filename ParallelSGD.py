import math
from datetime import datetime

import numpy as np
import threading

from DataUtils import DataUtils


class ParallelSGD:
    def __init__(self, learning_rate, thread_count, regularization, loss_function, accuracy_function, collect_data=False):
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
        self.collect_data = collect_data
        self.runtime = None

    @staticmethod
    def predict(features, weight):
        return np.dot(weight, features)

    def train(self, xtrain, ytrain):
        self.examples_per_thread = math.floor(xtrain.shape[0] / self.thread_count)
        print(f"Examples per thread: {self.examples_per_thread}")
        self.weights = {}
        start = datetime.now()
        for i in range(self.thread_count):
            self.weights[i] = np.zeros(xtrain.shape[1])
            self.losses[i] = []
            self.accuracies[i] = []

        for i in range(self.thread_count):
            thread = threading.Thread(target=self.train_threaded, args=(i, xtrain, ytrain))
            self.threads.append(thread)
            thread.start()

        for thread in self.threads:
            thread.join()

        self.runtime = datetime.now() - start
        print(f"Finished in {self.runtime}")
        return self.losses, self.accuracies

    @staticmethod
    def hinge_gradient(weight, sample, label, regularization):
        return 2 * regularization * weight - label * sample

    def train_threaded(self, thread_number, xtrain, ytrain):
        try:
            xtrain_shuffled, ytrain_shuffled = DataUtils.shuffle_data(xtrain, ytrain)
            weight = self.weights[thread_number]
            loss = []
            accuracy = []
            learning_rate = self.learning_rate

            last_checkpoint = 0
            for i in range(self.examples_per_thread):
                itemx, itemy = xtrain_shuffled[i], ytrain_shuffled[i]
                prediction = self.predict(itemx, weight)
                if itemy * prediction < 1:
                    weight -= learning_rate * self.hinge_gradient(weight, itemx, itemy, self.regularization)

                if self.collect_data:
                    current_percentage = (thread_number * self.examples_per_thread + i) / (xtrain_shuffled.shape[0] / 100)
                    if current_percentage != last_checkpoint and current_percentage % 5 == 0:
                        last_checkpoint = current_percentage
                        loss.append(self.loss_function(xtrain, ytrain, weight))
                        accuracy.append(self.accuracy_function(xtrain, ytrain, weight))
                else:
                    if i == self.examples_per_thread - 1:
                        loss.append(self.loss_function(xtrain, ytrain, weight))
                        accuracy.append(self.accuracy_function(xtrain, ytrain, weight))

            self.weights[thread_number] = weight
            self.losses[thread_number] = loss
            self.accuracies[thread_number] = accuracy
        except Exception as e:
            print(e)

    def total_weight(self):
        return np.mean([np.array(item) for item in self.weights.values()], axis=0)
