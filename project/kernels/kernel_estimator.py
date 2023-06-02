import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats


class KernelEstimatedPDF(object):
    def __init__(self, kernel):
        self.kernel = kernel

    def predict(self, X,X_train=None):
        if X_train is None:
            #try to use self.X if it exists
            try:
                X_train = self.X
            except:
                raise ValueError("No training data provided")
        f = []
        for x in X:
            f.append(np.mean(self.kernel(x - X_train)))
        return np.array(f)


    def fit(self, X, n_folds = 5, lr = 0.001, epochs = 1000, learning_rate_scheduler=None, epsilon = 1e-3,batch_size = 1):
        #save the training data
        self.X = X
        #split the data into n_folds
        indexs = np.arange(len(X))
        np.random.shuffle(indexs)
        indexs_fold = np.array_split(indexs, n_folds)
        #note down the initial parameters
        initial_params = self.kernel.get_params()
        #if the learning rate scheduler is not provided, use a constant learning rate
        if learning_rate_scheduler is None:
            learning_rate_scheduler = lambda x: lr
        #for each fold, train the kernel
        for fold in range(n_folds):
            X_train = np.delete(X, indexs_fold[fold], axis=0)
            X_test = X[indexs_fold[fold]]
            #reset the parameters
            self.kernel.set_params(initial_params)
            #train the kernel
            for epoch in range(epochs):


