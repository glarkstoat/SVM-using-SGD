#%%
import pandas as pd
import numpy as np

class DataLoader:
    def __init__(self, path='datasets/'):
        self.path = path
    
    def get_toydata_tiny(self):
        """ Returns the features and labels of the tiny toydata set """

        tiny = pd.read_csv(f"{self.path}toydata_tiny.csv", header=0)
        features = tiny[['x1', 'x2']]
        labels = tiny['y']
        
        return features, labels

    def get_toydata_large(self):
        """ Returns the features and labels of the large toydata set """
        large = pd.read_csv(f"{self.path}toydata_large.csv", header=0)
        features = large.iloc[:, :8]
        labels = large['y']
        
        return features, labels

    def get_MNIST(self):
        """ Returns the training, training-labels, the test and 
            the test-labels.  """
        mnist = np.load(f"{self.path}mnist.npz")
        
        return mnist['train'], mnist['train_labels'], mnist['test'], mnist['test_labels']

#%%