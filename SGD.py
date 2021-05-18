import datetime
import numpy as np

from DataUtils import DataUtils


class SGD:
    """ Serial computation the losses and accuarcies per epoch, for a given training set 
        via mini-batch gradient descent. """

    def __init__(self, learning_rate, regularization, batch_size, epoch_count, loss_function, accuracy_function):
        self.learning_rate = learning_rate
        self.weight = None
        self.batch_size = batch_size
        self.epoch_count = epoch_count
        self.regularization = regularization
        self.loss_function = loss_function
        self.accuracy_function = accuracy_function
        self.runtime = None

    def train(self, xtrain, ytrain):
        """ Calculates the average gradient for a given batch
            and updates the weights with these averages after the
            batch has been processed. """

        self.weight = np.zeros(xtrain.shape[1])
        learning_rate = self.learning_rate
        start = datetime.datetime.now() # runtime

        losses, accuracies = [], []
        for epoch in range(self.epoch_count):
            learning_rate /= np.sqrt(epoch + 1)  # adaptive learning rate
            xtrain, ytrain = DataUtils.shuffle_data(xtrain, ytrain)

            # Loops through the batches
            for i in range(int(len(ytrain) / self.batch_size)):
                batch_start = i * self.batch_size
                batch_end = (i + 1) * self.batch_size

                # Sums up the gradients of the incorrectly classified samples
                # or the samples that lie within the margin
                grad = 0
                for sample, label in zip(xtrain[batch_start:batch_end],
                                         ytrain[batch_start:batch_end]):
                    prediction = self.predict(sample)
                    if label * prediction < 1:  # either within margin or incorrectly classified
                        grad += self.hinge_gradient(self.weight, sample, label, self.regularization)
                    """else: ## Technically required but makes results worse and computation longer
                        grad += 2 * self.regularization * self.weight"""

                # Weights are updated with average gradients after batch is completed
                self.weight -= learning_rate * grad / self.batch_size

            # Losses & accuracies after one epoch
            losses.append(self.loss_function(xtrain, ytrain, self.weight))
            accuracies.append(self.accuracy_function(xtrain, ytrain, self.weight))

        print(f"Finished in {datetime.datetime.now() - start}")

        return np.array(losses), np.array(accuracies)

    @staticmethod
    def hinge_gradient(weight, sample, label, regularization):
        return 2 * regularization * weight - label * sample

    def predict(self, sample):
        """ Calculates the confidence value for a given sample. """

        # If function is called externally the 1 for the offset term is manually added
        if sample.shape != self.weight.shape:
            sample = np.append(sample, 1)

        return np.dot(self.weight, sample)

    def total_weight(self):
        return self.weight