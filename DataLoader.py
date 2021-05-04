#%%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

class DataLoader:
    def __init__(self, path='datasets/'):
        self.path = path
    
    def get_toydata_tiny(self, train_split_size=0.7, normalize=True):
        """ Returns the train_test_split of the features and labels of the 
            tiny toydata set. """

        tiny = pd.read_csv(f"{self.path}toydata_tiny.csv", header=0)
        features = tiny[['x1', 'x2']]
        labels = tiny['y']
        
        xtrain, xtest, ytrain, ytest = train_test_split(features, labels, 
                                            train_size=train_split_size,
                                            random_state=42) # to keep results consistent
        if normalize == True:
            xtrain = StandardScaler().fit_transform(xtrain)

        return xtrain, xtest, ytrain, ytest

    def get_toydata_large(self, train_split_size=0.7, normalize=True):
        """ Returns the train_test_split of the features and labels of the 
            large toydata set. """
        
        large = pd.read_csv(f"{self.path}toydata_large.csv", header=0)
        features = large.iloc[:, :8]
        labels = large['y']
        
        xtrain, xtest, ytrain, ytest = train_test_split(features, labels, 
                                            train_size=train_split_size,
                                            random_state=42) 
        if normalize == True:
            xtrain = StandardScaler().fit_transform(xtrain)
            
        return xtrain, xtest, ytrain, ytest

    def get_MNIST(self, normalize=True):
        """ Returns the training, training-labels, the test and 
            the test-labels of the MNIST dataset.  """
        
        mnist = np.load(f"{self.path}mnist.npz")
        xtrain = mnist['train']
        
        if normalize == True:
            xtrain = StandardScaler().fit_transform(xtrain)
        
        return xtrain, mnist['train_labels'], mnist['test'], mnist['test_labels']

#%%