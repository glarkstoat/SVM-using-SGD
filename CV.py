#%%
import numpy as np

def cross_validation_score(estimator, xtrain, ytrain, k):
    """ Manual k-fold cross-validation on training set using a given estimator. 
        Returns the average classification score for all validation sets. 
        Due to implementation it's possible that the remainder of 
        len(xtrain) % k samples will not be used for CV. """
    
    fold_size = int(len(xtrain) / k)
    accuracy_scores = []

    for i in range(k):
        """ Iterates through all possible configurations for training- and
            validation sets. Estimator is fitted with the corresponding 
            training set for each iteration. """
            
        if i == 0: 
            """ First iteration. Validation set is the first fold, i.e. 
                        the first fold_size rows """
            clf = estimator.fit(xtrain[fold_size:],
                                        ytrain[fold_size:])
        elif i == k-1: 
            """ Last iteration. Validation set is the last fold, i.e. 
                        the last fold_size rows """
            clf = estimator.fit(xtrain[:i*fold_size],
                                        ytrain[:i*fold_size])
        else:
            """ Concatenates the array slices that make up the training set
            i.e. the whole training set except the validation set.
            Necessary if the validation set is in the middle, i.e. neither the 
            first nor the last fold. """
            clf = estimator.fit(np.concatenate((xtrain[: i * fold_size], 
                                        xtrain[(i+1) * fold_size :])),
                                np.concatenate((ytrain[: i * fold_size], 
                                        ytrain[(i+1) * fold_size :])))
        
        # Current location of validation set
        val_range = range(i * fold_size, (i+1) * fold_size)
        
        # Accuracy score using weights of current clf
        accuracy_scores.append(clf.accuracy(np.c_[xtrain[val_range], 
                                    np.ones(xtrain[val_range].shape[0])], 
                                       
                                       ytrain[val_range]))
    return np.mean(accuracy_scores) # cross-validation score
# %%
