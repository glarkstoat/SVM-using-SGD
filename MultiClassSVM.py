# %%
from DataLoader import *
import matplotlib.pyplot as plt
from tqdm import tqdm
import datetime

from DataUtils import *


class MultiClassSVM:
    """ SVM that can handle multiple classes by using the multiclass hinge loss. 
        Returns the average accuracies and losses over all classes. """

    def __init__(self, lr=0.1, C=0.001, max_iters=5, 
                 batch_size=200, tqdm_toggle=False):
        self.lr = lr
        self.C = C
        self.max_iters = max_iters
        self.batch_size = batch_size
        self.tqdm_toggle = tqdm_toggle
        self.losses = []
        self.accuracies = []
        self.weights = None
        self.runtime = None  # in seconds

    def fit(self, xtrain, ytrain):

        n_samples = len(ytrain)
        classes = np.unique(ytrain).astype(int)

        # Add extra column of 1s to xtrain to account for bias term b
        xtrain = np.c_[xtrain, np.ones(xtrain.shape[0])]

        # Build dict containing the weights for each class
        self.weights = {c: np.zeros(xtrain.shape[1]) for c in classes}

        n_batches = int(n_samples / self.batch_size)
        if n_batches < 1:
            raise Exception("Batch size is greater than number of samples!")

        # Optimization
        self.losses, self.accuracies = self.minibatchGD(xtrain, ytrain, n_batches, self.lr)

        return self

    def minibatchGD(self, xtrain, ytrain, n_batches, lr):
        """ Calculates the average gradient for a given batch
            and updates the weights with these averages after the
            batch has been completed. """

        # Used to calculate runtime
        start = datetime.datetime.now()

        if self.tqdm_toggle:
            iterations = tqdm(range(1, self.max_iters + 1))
        else:
            iterations = range(1, self.max_iters + 1)

        losses, accuracies = [], []
        for epoch in iterations:
            lr /= np.sqrt(epoch)  # adaptive learning rate
            xtrain, ytrain = DataUtils.shuffle_data(xtrain, ytrain)

            # Loops through the batches
            for count in range(n_batches):
                batch_start = count * self.batch_size
                batch_end = (count + 1) * self.batch_size

                # resets the gradients for each class to 0 at the start of a batch
                gradients = {c: 0 for c in self.weights.keys()}

                # Loop through batches
                for x, y in zip(xtrain[batch_start:batch_end],
                                ytrain[batch_start:batch_end]):

                    # Loop through classes weights
                    for i in self.weights.keys():

                        inner_max = [[j, self.confidence(self.weights[j], x) - self.confidence(self.weights[y], x)]
                                     for j, weight in self.weights.items() if j != y]

                        # sorts the list by descending confidence score and returns the j that has the highest
                        # confidence
                        inner_max = sorted(inner_max, key=lambda row: row[1], reverse=True)
                        j = inner_max[0][0]  # class with maximum confidence

                        # central conditions whether to update gradient or not 
                        if inner_max[0][1] > -1:
                            if i != j and i != y:
                                continue  # do nothing
                            elif i == y:
                                gradients[i] += -x
                            elif i != y:
                                gradients[i] += x

                # Weights are updated after batch is completed
                for i in self.weights.keys():
                    self.weights[i] -= lr * gradients[i] / self.batch_size

            # Losses & accuracies after one epoch    
            losses.append(self.mc_hinge_loss(xtrain, ytrain))
            accuracies.append(self.accuracy(xtrain, ytrain))

        self.runtime = (datetime.datetime.now() - start).total_seconds()

        return np.array(losses), np.array(accuracies)

    def predict_class(self, sample):
        """ Returns class with highest confidence """

        max_confidence = 0
        best_class = 0
        for c, weight in self.weights.items():
            conf = self.confidence(weight.T, sample)
            if conf > max_confidence:
                max_confidence = conf
                best_class = c

        return best_class  # predicted label

    @staticmethod
    def confidence(weight, sample):
        return np.dot(weight.T, sample)

    def accuracy(self, features, labels):
        """ compute average classification error
            over all classes """
            
        # If function is called externally the 1s for the offset terms are manually added
        if features.shape[1] != self.weights[0].shape[0]:
            features = np.c_[features, np.ones(features.shape[0])]

        n_correct = 0
        for (x, y) in zip(features, labels):
            pred_class = self.predict_class(x)
            if pred_class == y:
                n_correct += 1

        return n_correct / len(labels)

    def mc_hinge_loss(self, features, labels):
        """ Calculates the average loss over all classes """

        loss = 0
        for x, y in zip(features, labels):
            max_inner = np.max([self.confidence(self.weights[j], x) - self.confidence(self.weights[y], x)
                                for j, weight in self.weights.items() if j != y])
            loss += max(0, 1 + max_inner)
        loss /= len(labels)

        return loss
