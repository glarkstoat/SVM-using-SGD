#%%
from DataLoader import *
import matplotlib.pyplot as plt

class LinearSVM:
    """        
        Parameters
        ----------
        lr : float
            Learning rate used by SGD optimization.
        
        C : float
            Regularization parameter.
        """
        
    def __init__(self, lr=0.3, C=0.1, loss="hinge", 
                 max_iters=100, batch_size=20, tol=0.99,
                 show_plot=True):
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

    def train(self, xtrain, ytrain):
        
        n_samples = len(ytrain)
        
        # add extra column of 1s to xtrain and weights
        xtrain = np.c_[xtrain, np.ones(xtrain.shape[0])]
        self.weights = np.ones(xtrain.shape[1])
        
        # reset number of iterations if dataset has low number of samples
        if self.max_iters > n_samples:
            self.max_iters = n_samples
        
        n_batches = int(n_samples / self.batch_size)
        if n_batches < 1:
            raise Exception("Batch size is greater than number of samples!")
        
        for epoch in range(self.max_iters): # epochs

            #self.lr /= np.sqrt(t+1) # adaptive learning rate
            
            # Shuffles the training data
            indices = np.random.permutation(n_samples)
            xtrain = xtrain[indices]
            ytrain = ytrain[indices]
            
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
                    """else:
                        grad += 2 * self.C * self.weights"""
                
                # Weights are updated after batch is completed
                self.weights -= self.lr * grad / self.batch_size   
            
            # Losses & accuracies after one epoch    
            self.losses.append(self.hinge_loss(xtrain, ytrain, self.weights))            
            self.accuracies.append(self.accuracy(xtrain, ytrain))
    
        if show_plot:
            self.plot_margin(xtrain, ytrain)
        
    def predict(self, sample, label):
        return label * np.dot(self.weights, sample)
    
    def update_weights(self, sample, label):
        self.weights -= self.lr * self.gradient(sample, label)

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

# %%
