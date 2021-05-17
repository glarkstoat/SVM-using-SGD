# %%
from DataLoader import *
import matplotlib.pyplot as plt
from tqdm import tqdm
import datetime
import seaborn as sns

from DataUtils import *
from ParallelSGD import *

sns.set_style('darkgrid')
sns.set(color_codes=True)
sns.set_context('paper')


class LinearSVMParallel:
    """        
        Parameters
        ----------
        lr : float
            Learning rate used by SGD optimization.
        
        C : float
            Regularization parameter.
            
        show_plot : bool
            If set to True the function plot_margin is called at the end of 
            training. This works for 2-dimensional features only. 
        
        tqdm_toggle : bool
            If set to True the training progress will be displayed via tqdm prints. 
            Not recommended when carrying out hyperparameter search.
        
        """

    def __init__(self, lr=0.5, C=0.01, show_plot=False, tqdm_toggle=False):
        self.lr = lr
        self.C = C
        self.losses = {}
        self.accuracies = {}
        self.show_plot = show_plot
        self.runtime = None
        self.tqdm_toggle = tqdm_toggle
        self.optimizer = None

    def fit(self, xtrain, ytrain):
        n_samples = len(ytrain)
        # add extra column of 1s to xtrain and weights to account for bias term b
        xtrain = np.c_[xtrain, np.ones(xtrain.shape[0])]

        self.optimizer = ParallelSGD(learning_rate=self.lr,
                                     thread_count=4,
                                     regularization=self.C,
                                     loss_function=LinearSVMParallel.hinge_loss,
                                     accuracy_function=LinearSVMParallel.accuracy)
        self.losses, self.accuracies = self.optimizer.train(xtrain, ytrain)

        if self.show_plot:
            self.plot_margin(xtrain, ytrain)

        return self

    def predict(self, sample):
        """ Calculates the confidence value for a given sample. """

        # If function is called externally the 1 for the offset term is manually added
        if sample.shape != self.optimizer.total_weight().shape:
            sample = np.append(sample, 1)

        return np.dot(self.optimizer.total_weight(), sample)

    @staticmethod
    def accuracy(features, labels, weight):
        """ Computes the classification score for given feature vectors and their 
            respective labels. Requires a previously trained SVM. """

        # If function is called externally the 1s for the offset terms are manually added
        if features.shape[1] != weight.shape[0]:
            features = np.c_[features, np.ones(features.shape[0])]

        predictions = labels * np.dot(weight, features.T)
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
        Z = np.c_[X.ravel(), Y.ravel(), np.ones(100)].dot(self.optimizer.total_weight())
        Z = Z.reshape(X.shape)
        plt.contour(X, Y, Z, levels=[-1, 0, 1], colors=["blue", "gray", "red"])
        plt.show()

# %%
