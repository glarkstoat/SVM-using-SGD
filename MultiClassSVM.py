#%%
from DataLoader import *
import matplotlib.pyplot as plt
from tqdm import tqdm
import datetime

class MultiClassSVM:
    
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
        classes = np.unique(ytrain).astype(int)
        
        # add extra column of 1s to xtrain to account for bias term b
        xtrain = np.c_[xtrain, np.ones(xtrain.shape[0])]
        
        # build dict containing the weights for each class
        self.weights = {c : np.zeros(xtrain.shape[1]) for c in classes}
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
        for epoch in tqdm(range(1, self.max_iters+1)):
            
            print(f"\nEpoch: {epoch} / {self.max_iters} ... ")

            self.lr /= np.sqrt(epoch) # adaptive learning rate
            xtrain, ytrain = self.shuffle_data(xtrain, ytrain)
            
            # Loops through the batches
            for count in tqdm(range(n_batches), position=0, leave=True):
                batch_start = count*self.batch_size
                batch_end = (count+1)*self.batch_size
                
                # resets the gradients for each class to 0 at the start of a batch
                gradients = {c : 0 for c in self.weights.keys()}
                
                # Loop through batches
                for x,y in zip(xtrain[batch_start:batch_end], 
                               ytrain[batch_start:batch_end]):
                    
                    # Loop through classes weights
                    for i in self.weights.keys():
                        
                        inner_max = [[j, self.confidence(self.weights[j], x) - self.confidence(self.weights[y], x)]
                                                    for j, weight in self.weights.items() if j != y]
                        # sorts the list by descending confidence score and returns the j that has the highest cs
                        inner_max = sorted(inner_max, key=lambda row:row[1], reverse=True)
                        j = inner_max[0][0] # class with maximum confidence
                        
                        # central conditions whether to update gradient or not 
                        if inner_max[0][1] > -1:
                            if i != j and i != y:
                                continue # do nothing
                            elif i == y:
                                gradients[i] += -x
                            elif i != y:
                                gradients[i] += x

                # Weights are updated after batch is completed
                for i in self.weights.keys():
                    self.weights[i] -= self.lr * gradients[i] / self.batch_size
            
            # Losses & accuracies after one epoch    
            losses.append(self.mc_hinge_loss(xtrain, ytrain))            
            accuracies.append(self.accuracy(xtrain, ytrain))

        self.runtime = (datetime.datetime.now() - start).total_seconds() 
        
        return losses, accuracies
    
    def predict_class(self, sample):
        """ Returs class with highest confidence """
        
        max_confidence = 0; best_class = 0
        for c, weight in self.weights.items():
            conf = self.confidence(weight.T, sample)
            if conf > max_confidence:
                max_confidence = conf
                best_class = c
        
        return best_class # predicted label
    
    def confidence(self, weight, sample):
        return np.dot(weight.T, sample)
        
    def accuracy(self, features, labels):
        """ compute averge classification error
            over all classes """
        
        n_correct = 0
        for (x, y) in zip(features, labels):
            pred_class = self.predict_class(x)
            if pred_class == y:
                n_correct += 1
                
        return n_correct / len(labels)
        
    def mc_hinge_loss(self, features, labels):
        """ Calculates the average loss over all classes """
        
        loss = 0
        for x,y in zip(features, labels):
            max_inner = np.max([self.confidence(self.weights[j], x) - self.confidence(self.weights[y], x)
                          for j, weight in self.weights.items() if j != y])
            loss += max(0, 1 + max_inner)
        loss /= len(labels)
        
        return loss    

    def shuffle_data(self, xtrain, ytrain):
        indices = np.random.permutation(len(ytrain))
        xtrain = xtrain[indices]
        ytrain = ytrain[indices]
        
        return xtrain, ytrain