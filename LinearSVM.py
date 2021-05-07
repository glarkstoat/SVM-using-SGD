#%%
from DataLoader import *
import matplotlib.pyplot as plt
from tqdm import tqdm
import datetime

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
            
        """
        
    def __init__(self, lr=5, C=0.1, loss="hinge", 
                 max_iters=100, batch_size=20, tol=0.99,
                 show_plot=False):
        self.lr = lr
        self.C = C
        self.weights = []
        self.loss = loss
        self.max_iters = max_iters
        self.batch_size = batch_size
        self.losses = []
        self.accuracies = []
        self.tol = tol
        self.show_plot = show_plot
        self.runtime = None

    def train(self, xtrain, ytrain, optimizer="minibatchGD"):
        
        n_samples = len(ytrain)
                
        # add extra column of 1s to xtrain and weights to account for bias term b
        xtrain = np.c_[xtrain, np.ones(xtrain.shape[0])]
        self.weights = np.zeros(xtrain.shape[1])
        #self.weights = np.random.normal(size=xtrain.shape[1])
                
        n_batches = int(n_samples / self.batch_size)
        if n_batches < 1:
            raise Exception("Batch size is greater than number of samples!")
        
        # Perform optimization
        if optimizer == "minibatchGD":
            self.losses, self.accuracies = self.minibatchGD(xtrain, ytrain, n_batches)
        else:
            raise Exception("Invalid optimizer!")
            
        if self.show_plot:
            self.plot_margin(xtrain, ytrain)
        
    def minibatchGD(self, xtrain, ytrain, n_batches):
        """ Calculates the average gradient for a given batch
            and updates the weights with these averages after the
            batch has been processed. """
        
        # Used to calculate runtime
        start = datetime.datetime.now()
        
        losses, accuracies = [], []
        for epoch in tqdm(range(self.max_iters)):

            self.lr /= np.sqrt(epoch+1) # adaptive learning rate
            xtrain, ytrain = self.shuffle_data(xtrain, ytrain)
            
            # Loops through the batches
            for i in range(n_batches):
                batch_start = i*self.batch_size
                batch_end = (i+1)*self.batch_size
                
                grad = 0
                # Sums up the gradients of the incorrectly classified samples
                # or the samples that lie within the margin
                for x,y in zip(xtrain[batch_start:batch_end], 
                               ytrain[batch_start:batch_end]):
                    
                    prediction = self.predict(x,y)
                    if prediction < 1: # either within margin or incorrectly classified
                        grad += self.gradient(x, y)
                    """else: ## Technically required but makes results worse
                        grad += 2 * self.C * self.weights"""
                
                # Weights are updated after batch is completed
                self.weights -= self.lr * grad / self.batch_size 
            
            # Losses & accuracies after one epoch    
            losses.append(self.hinge_loss(xtrain, ytrain, self.weights))            
            accuracies.append(self.accuracy(xtrain, ytrain))

        self.runtime = (datetime.datetime.now() - start).total_seconds() 
        
        return losses, accuracies
    
    def predict(self, sample, label):
        return label * np.dot(self.weights, sample)

    def gradient(self, sample, label, loss="hinge"):
        if loss == "hinge":
            return 2 * self.C * self.weights - label * sample
        else:
            raise Exception("Gradient loss not defined.")
        
    def accuracy(self, features, labels):
        """ compute classification error """
        
        n_correct = 0
        for (x, y) in zip(features, labels):
            pred = self.predict(x,y)
            if pred > 0:
                n_correct += 1
                
        return n_correct / len(labels)
        
    @staticmethod
    def hinge_loss(features, labels, weights):
        return np.sum([max(0, 1 - label * np.dot(weights, feature))
                       for feature, label in zip(features, labels)]) / len(labels)

    def plot_margin(self, xtrain, ytrain):
        
        plt.plot(xtrain[ytrain==-1,0], xtrain[ytrain==-1,1], '*', c='g', label='Class 1')
        plt.plot(xtrain[ytrain==1,0], xtrain[ytrain==1,1], '+', c='r', label='Class 2')
                
        # decision boundary
        X, Y = np.meshgrid(np.linspace(-6,6,10), np.linspace(-6,6,10))
        Z = np.c_[X.ravel(), Y.ravel()].dot(self.weights[:-1])
        Z = Z.reshape(X.shape)
        plt.contour(X, Y, Z, levels=[-1,0,1], colors=['green', 'blue', 'red'])
        plt.show()
        
    def shuffle_data(self, xtrain, ytrain):
        indices = np.random.permutation(len(ytrain))
        xtrain = xtrain[indices]
        ytrain = ytrain[indices]
        
        return xtrain, ytrain

# %%
