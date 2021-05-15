# %%
from DataLoader import *
import matplotlib.pyplot as plt
from tqdm import tqdm
import datetime
import seaborn as sns

from DataUtils import *

sns.set_style('darkgrid')
sns.set(color_codes=True)
sns.set_context('paper')


class LinearSVM:
    """        
        Parameters
        ----------
        lr : float
            Learning rate used by SGD optimization.
        
        C : float
            Regularization parameter.
            
        loss : str
            Loss used by optimization. Currently implemented : "hinge"
            
        max_iters : int
            Number of iterations i.e. epochs during optimization.
            
        batch_size : int
            Number of samples per batch when using minibatch GD.
            
        show_plot : bool
            If set to True the function plot_margin is called at the end of 
            training. This works for 2-dimensional features only. 
        
        tqdm_toggle : bool
            If set to True the training progress will be displayed via tqdm prints. 
            Not recommended when carrying out hyperparameter search.
        
        """

    def __init__(self, lr=0.5, C=0.01, loss="hinge",
                 max_iters=30, batch_size=20,
                 show_plot=False, tqdm_toggle=False):
        self.lr = lr
        self.C = C
        self.weights = None
        self.loss = loss
        self.max_iters = max_iters
        self.batch_size = batch_size
        self.losses = []
        self.accuracies = []
        self.show_plot = show_plot
        self.runtime = None
        self.tqdm_toggle = tqdm_toggle

    def fit(self, xtrain, ytrain, optimizer="minibatchGD"):

        n_samples = len(ytrain)

        # add extra column of 1s to xtrain and weights to account for bias term b
        xtrain = np.c_[xtrain, np.ones(xtrain.shape[0])]
        self.weights = np.zeros(xtrain.shape[1])
        # self.weights = np.random.normal(size=xtrain.shape[1])

        n_batches = int(n_samples / self.batch_size)
        if n_batches < 1:
            raise Exception("Batch size is greater than number of samples!")

        # Perform optimization
        if optimizer == "minibatchGD":
            self.losses, self.accuracies = self.minibatchGD(xtrain, ytrain, n_batches, self.lr)
        else:
            raise Exception("Invalid optimizer!")

        if self.show_plot:
            self.plot_margin(xtrain, ytrain)

        return self

    def minibatchGD(self, xtrain, ytrain, n_batches, lr):
        """ Calculates the average gradient for a given batch
            and updates the weights with these averages after the
            batch has been processed. """

        # Used to calculate runtime
        start = datetime.datetime.now()

        losses, accuracies = [], []
        if self.tqdm_toggle:
            iterations = tqdm(range(self.max_iters))
        else:
            iterations = range(self.max_iters)

        for epoch in iterations:

            lr /= np.sqrt(epoch + 1)  # adaptive learning rate
            xtrain, ytrain = DataUtils.shuffle_data(xtrain, ytrain)

            # Loops through the batches
            for i in range(n_batches):
                batch_start = i * self.batch_size
                batch_end = (i + 1) * self.batch_size

                # Sums up the gradients of the incorrectly classified samples
                # or the samples that lie within the margin
                grad = 0
                for sample, label in zip(xtrain[batch_start:batch_end],
                                         ytrain[batch_start:batch_end]):

                    prediction = self.predict(sample)
                    if label * prediction < 1:  # either within margin or incorrectly classified
                        grad += self.hinge_gradient(sample, label)
                    """else: ## Technically required but makes results worse and computation longer
                        grad += 2 * self.C * self.weights"""

                # Weights are updated with average gradients after batch is completed
                self.weights -= lr * grad / self.batch_size

                # Losses & accuracies after one epoch
            losses.append(self.hinge_loss(xtrain, ytrain, self.weights))
            accuracies.append(self.accuracy(xtrain, ytrain))

        self.runtime = (datetime.datetime.now() - start).total_seconds()

        return np.array(losses), np.array(accuracies)

    def predict(self, sample):
        """ Calculates the confidence value for a given sample. """

        # If function is called externally the 1 for the offset term is manually added
        if sample.shape != self.weights.shape:
            sample = np.append(sample, 1)

        return np.dot(self.weights, sample)

    def hinge_gradient(self, sample, label, loss="hinge"):
        if loss == "hinge":
            return 2 * self.C * self.weights - label * sample
        else:
            raise Exception("Gradient loss not defined.")

    def accuracy(self, features, labels):
        """ Computes the classification score for given feature vectors and their 
            respective labels. Requires a previously trained SVM. """

        # If function is called externally the 1s for the offset terms are manually added
        if features.shape[1] != self.weights.shape[0]:
            features = np.c_[features, np.ones(features.shape[0])]

        predictions = labels * np.dot(self.weights, features.T)
        n_correct = np.sum(predictions > 0)

        return n_correct / len(labels)

    @staticmethod
    def hinge_loss(features, labels, weights):
        # If function is called externally the 1s for the offset terms are manually added
        if features.shape[1] != weights.shape[0]:
            features = np.c_[features, np.ones(features.shape[0])]

        predictions = labels * np.dot(weights, features.T)
        return sum(filter(lambda x: x > 0, 1 - predictions)) / len(labels)  # equivalent to hingle loss

    def plot_margin(self, xtrain, ytrain):
        """ Adapted from IML's svm.ipynb. Credit to Prof. Tschiatschek.
            Only works with 2-dimensional features and two classes. """

        plt.scatter(xtrain[ytrain == -1, 0], xtrain[ytrain == -1, 1], label='Class -1', s=20, alpha=0.5, marker="x",
                    c="b")
        plt.scatter(xtrain[ytrain == 1, 0], xtrain[ytrain == 1, 1], label='Class 1', s=30, alpha=0.5, marker="x",
                    c="red")

        # decision boundary
        X, Y = np.meshgrid(np.linspace(-3, 3, 10), np.linspace(-4, 4, 10))
        Z = np.c_[X.ravel(), Y.ravel(), np.ones(100)].dot(self.weights)
        Z = Z.reshape(X.shape)
        plt.contour(X, Y, Z, levels=[-1, 0, 1], colors=["blue", "gray", "red"])
        plt.show()

# %%
