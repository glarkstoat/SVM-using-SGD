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
        features = np.array(tiny[['x1', 'x2']])
        labels = np.array(tiny['y'])
        
        xtrain, xtest, ytrain, ytest = train_test_split(features, labels, 
                                            train_size=train_split_size,
                                            random_state=42) # to keep results consistent
        if normalize == True:
            xtrain, xtest = self.normalization(xtrain, xtest)

        return xtrain, xtest, ytrain, ytest

    def get_toydata_large(self, train_split_size=0.7, normalize=True):
        """ Returns the train_test_split of the features and labels of the 
            large toydata set. """
        
        large = pd.read_csv(f"{self.path}toydata_large.csv", header=0)
        features = np.array(large.iloc[:, :8])
        labels = np.array(large['y'])
        
        xtrain, xtest, ytrain, ytest = train_test_split(features, labels, 
                                            train_size=train_split_size,
                                            random_state=42) 
        if normalize == True:
            xtrain, xtest = self.normalization(xtrain, xtest)

        return xtrain, xtest, ytrain, ytest

    def get_MNIST(self, normalize=True):
        """ Returns the training-features, the test-features, training-labels and 
            the test-labels of the MNIST dataset.  """
        
        mnist = np.load(f"{self.path}mnist.npz")
        
        # Data is transposed because it is stored (features, samples)
        # and not like toydata sets (samples, features)
        xtrain = mnist['train'].T
        xtest = mnist['test'].T
        
        if normalize == True:
            xtrain, xtest = self.normalization(xtrain, xtest)

        return xtrain, xtest, mnist['train_labels'], mnist['test_labels']

    def normalization(self, train, test):
        """ Fits a normalization model with xtrain and 
            applies transformation to both sets. """
            
        scaler = StandardScaler().fit(train)
        train = scaler.transform(train)
        test = scaler.transform(test)

        return train, test
#%%