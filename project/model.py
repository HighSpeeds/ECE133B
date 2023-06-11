import numpy as np
import pandas as pd


from kernels.kernel_estimator import KernelEstimatedPDF
from kernels.gaussian_kernels import MultivariateGaussianKernel
from optimization import *


class model:

    def __init__(self,kernel):
        self.kernel = kernel
        self.kernel_estimator = KernelEstimatedPDF(kernel)

    def fit(self,X,kernel_estimator_params):
        #cast X to cupy array
        # X = X.array()
        self.X = np.array(X)
        self.kernel_estimator.fit(X,**kernel_estimator_params)
        self.second_mmt = self.kernel_estimator.get_pseudo_covariance()
        self.average_return = self.kernel_estimator.get_average_return()
        self.optimal_weights = optimize_portfolio(self.average_return,self.second_mmt)

    def transform(self,X):
        return self.X @ self.optimal_weights
    
    def fit_transform(self,X,kernel_estimator_params):
        self.fit(X,kernel_estimator_params)
        return self.transform(X)
    
    def save(self,path):
        self.kernel_estimator.save_params(path+"params.npz")
        np.savez(path+"weights.npz", self.optimal_weights)
    
    def load(self,path):
        self.kernel_estimator.load_params(path)
        self.optimal_weights = np.load(path)

    def get_weights(self):
        return self.optimal_weights

if __name__=="__main__":
    import pandas as pd
    import numpy as np
    import os
    import sklearn.decomposition as skd
    from optimization import optimize_portfolio

    sp500_df = pd.read_csv('sp500.csv')
    sp500_df_price = pd.read_csv('sp500_price.csv')
    sp500_df_price.set_index('date', inplace=True)
    sp500_df.set_index('date', inplace=True)
    # print(sp500_df.head())
    sp500_df = sp500_df.pct_change()
    sp500_df.dropna(inplace=True)
    changes = sp500_df.values
    #split into train and test
    train = changes[:int(changes.shape[0]*0.8)]
    test = changes[int(changes.shape[0]*0.8):]
    # print(train.shape)
    print(np.mean(train,axis=0).reshape(-1,1)@np.mean(train,axis=0).reshape(1,-1))
    initial_guess = np.cov(train.T)*train.shape[0]
    print(initial_guess)
    # # print(initial_guess)
    kernel = MultivariateGaussianKernel(initial_guess)
    Model = model(kernel)
    Model.fit(train,dict(n_folds=5, lr=0.0001, epochs=5, epsilon=1e-3, batch_size=20,multiprocessing = False))

    w = Model.get_weights()
    os.makedirs('runs/naiveKernel/',exist_ok=True)
    Model.save('runs/naiveKernel/')
    print(w)
    train_changes = train @ w
    test_changes = test @ w

    print("train sharpe ratio: ", np.mean(train_changes)/np.std(train_changes)*np.sqrt(252))
    print("test sharpe ratio: ", np.mean(test_changes)/np.std(test_changes)*np.sqrt(252))
    #save w
    np.save('weights/naiveKernel.npy',w)
    import matplotlib.pyplot as plt
    plt.plot(np.cumprod(1+changes@w))
    plt.axvline(x=int(changes.shape[0]*0.8),color='r')
    plt.plot(sp500_df_price['^GSPC'].values/sp500_df_price['^GSPC'].values[0])
    plt.show()