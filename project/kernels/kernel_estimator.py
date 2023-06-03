import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import tqdm
import multiprocessing as mp


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
        # print(X_train.shape)
        for x in X:
            f.append(np.mean(self.kernel(x - X_train)))
        return np.array(f)


    def fit(self, X, n_folds = 5, lr = 0.001, epochs = 1000, learning_rate_scheduler=None, epsilon = 1e-3,batch_size = 1,weight_func = lambda x: 1,
            multiprocessing = False, verbose = True,n_processes = 2):
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
        self.optimal_params = []
        for fold in range(n_folds):
            X_train = np.delete(X, indexs_fold[fold], axis=0)
            X_test = X[indexs_fold[fold]]
            #reset the parameters
            self.kernel.set_params(initial_params)
            #train the kernel
            for epoch in range(epochs):
                #shuffle the test data
                np.random.shuffle(X_test)
                #for each observation in the test data
                batch_gradient = 0
                prev_params = self.kernel.get_params()

                X_test_batched = np.array_split(X_test, len(X_test)//batch_size)
                for batch in tqdm.tqdm(X_test_batched):
                    F = []
                    X_centered = []
                    for x in batch:
                        F.append(self.predict([x],X_train)[0])
                        X_centered.append(x - X_train)
                    if multiprocessing:
                        with mp.Pool(n_processes) as pool:
                            G = pool.starmap(self.get_gradient, zip(X_centered, F))
                    else:
                        G = []
                        for i in range(len(batch)):
                            G.append(self.get_gradient(X_centered[i], F[i]))
                    #drop all the nan values
                    G_no_nan = []
                    for g in G:
                        if not np.isnan(g).any():
                            G_no_nan.append(g)
                    G = np.array(G_no_nan)
                    print(G)
                    # print(G.shape)
                    self.kernel.update_params({'R': -learning_rate_scheduler(epoch)*np.sum(G,axis=0)/len(X_train)})


                # for i,x in enumerate(tqdm.tqdm(X_test)):
                #     g = 0
                #     #get the prediction
                #     f = self.predict([x],X_train)[0]
                #     #for each observation in the training data
                #     g = self.kernel.get_gradient(x - X_train)
                #     # for x_train in tqdm.tqdm(X_train):
                #     #     #compute the gradient
                #     #     g += self.kernel.get_gradient(x - x_train)
                #     g/=len(X_train)
                #     #the gradient is equal to the 
                #     batch_gradient += -g * weight_func(i) / f
                #     #update the parameters
                #     if (i+1)%batch_size == 0:
                #         self.kernel.update_params({'R': -learning_rate_scheduler(epoch)*batch_gradient})
                #         batch_gradient = 0
                # self.kernel.update_params({'R': -learning_rate_scheduler(epoch)*batch_gradient})
                #if the parameters have converged, stop training
                if np.linalg.norm(prev_params['R'] - self.kernel.get_params()['R']) < epsilon:
                    print('Converged')
                    break
                print('Epoch {} finished'.format(epoch))
                print("Sigma: {}".format(self.kernel.get_params()['Sigma']))
                print("NLL :", -self.score(X_test,X_train,kernel=self.kernel))
            print('Fold {} finished'.format(fold))
            print("optimal Sigma: {}".format(self.kernel.get_params()['Sigma']))
            print("optimal R: {}".format(self.kernel.get_params()['R']))
            self.optimal_params.append(self.kernel.get_params())
        #set the parameters to the average of the optimal parameters
        self.Sigma = np.mean([params['Sigma'] for params in self.optimal_params], axis=0)
        self.kernel.set_params({'Sigma': self.Sigma})

    def get_gradient(self, X_centered, f):
        if f==0 or np.isnan(f):
            return np.nan
        g = self.kernel.get_gradient(X_centered)
        return -g / f

    def save_params(self, path):
        np.savez(path, self.kernel.get_params())

    def load_params(self, path):
        params = np.load(path)
        self.kernel.set_params({'Sigma': params['Sigma']})

    def score(self, X, X_train=None,kernel=None):
        #log likelihood
        return np.mean(np.log(self.predict(X, X_train)))



    def get_average_return(self):
        return np.mean(self.X, axis=0)

    def expected_value(self,w):
        return np.mean(self.X@w)

    def get_pseudo_covariance(self):
        kernel_second_mmt = self.kernel.get_params()['Sigma']
        X_average = self.get_average_return()
        return kernel_second_mmt+1/(self.X.shape[0])*self.X.T@self.X - (X_average@X_average.T)
                    
if __name__ == "__main__":
    import gaussian_kernels

    #draw samples from a normal distribution
    # X = np.random.normal(0,1,1000)
    kernel = gaussian_kernels.MultivariateGaussianKernel(np.array([[1, 0.5], [0.5, 1]]))
    
    data = stats.multivariate_normal.rvs(mean=[0, 0], cov=np.array([[5, -1.5], [-1.5, 5]]), size=2000)

    model = KernelEstimatedPDF(kernel)
    model.fit(data, n_folds=5, lr=0.01, epochs=10, epsilon=1e-3, batch_size=20  )
    print(model.score(data))
    print(model.get_pseudo_covariance())