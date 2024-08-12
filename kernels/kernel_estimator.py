import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import tqdm
import multiprocessing as mp
import pickle

class log:
    def __init__(self):
        pass

    def add(self,label,values):
        if not hasattr(self,label):
            setattr(self,label,[])
        getattr(self,label).append(values)

    def as_dict(self):
        return {label:getattr(self,label) for label in dir(self) if not label.startswith('__')}
    
    def save(self,filename):
        with open(filename,'wb') as f:
            pickle.dump(self.as_dict(),f)
    
    def get(self,label):
        return getattr(self,label)

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


    def fit_cross_validate(self, X, n_folds = 5, lr = 0.001, epochs = 1000, learning_rate_scheduler=None, epsilon = 1e-3,batch_size = 1,weight_func = lambda x: 1,
            verbose = True,n_processes = 2,multiprocess_folds=False,multiprocess_batch=False,log=True,log_filename=None):
        #assert that both multiprocessing options are not both true
        if multiprocess_folds and multiprocess_batch:
            raise ValueError("Cannot use both multiprocessing options at the same time")
        #if we log
        if log:
            #create a log object
            self.log = log()
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
            best_params,losses = self.train_once(X_train,X_test,lr = lr, epochs = epochs, learning_rate_scheduler=learning_rate_scheduler, epsilon = epsilon,batch_size = batch_size,weight_func = weight_func,
                multiprocessing = multiprocess_batch, verbose = verbose,n_processes = n_processes)
            
            self.optimal_params.append(best_params)
            if log:
                log.add(f'losses_fold{fold}',losses)
        #set the parameters to the average of the optimal parameters
        self.R = np.mean([params['R'] for params in self.optimal_params], axis=0)
        self.kernel.set_params({'R': self.R})
        #we use R instead of sigma because it is numerically more stable
        #if we log
        if log:
            #if the log filename is not provided, use the default
            if log_filename is None:
                log_filename = 'log.pkl'
            #save the log
            self.log.save(log_filename)

    def fit_no_cross_validate(self, X, lr = 0.001, epochs = 1000, learning_rate_scheduler=None, epsilon = 1e-3,batch_size = 1,weight_func = lambda x: 1,
                              train_test_split = 0.8, verbose = True,n_processes = 2,multiprocess_batch=False,log=True,log_filename=None):
        #if we log
        if log:
            #create a log object
            self.log = log()
        #save the training data
        self.X = X
        #split the data into n_folds
        indexs = np.arange(len(X))
        np.random.shuffle(indexs)
        indexs_train = indexs[:int(len(X)*train_test_split)]
        indexs_test = indexs[int(len(X)*train_test_split):]
        X_train = X[indexs_train]
        X_test = X[indexs_test]
        #note down the initial parameters
        initial_params = self.kernel.get_params()
        #if the learning rate scheduler is not provided, use a constant learning rate
        if learning_rate_scheduler is None:
            learning_rate_scheduler = lambda x: lr
        #reset the parameters
        self.kernel.set_params(initial_params)
        #train the kernel
        best_params,losses = self.train_once(X_train,X_test,lr = lr, epochs = epochs, learning_rate_scheduler=learning_rate_scheduler, epsilon = epsilon,batch_size = batch_size,weight_func = weight_func,
            multiprocessing = multiprocess_batch, verbose = verbose,n_processes = n_processes)
        
        self.optimal_params.append(best_params)
        if log:
            log.add(f'losses',losses)
            #save the log
            if log_filename is None:
                log_filename = 'log.pkl'
            self.log.save(log_filename)
        
    def fit(self,X,fit_method,train_test_split = 0.8, n_folds = 5,
            lr = 0.001, epochs = 1000, learning_rate_scheduler=None, epsilon = 1e-3,batch_size = 1,weight_func = lambda x: 1,
                            verbose = True,n_processes = 2,multiprocess_batch=False,log=True,log_filename=None):
        if fit_method == 'cross_validate':
            self.fit_cross_validate(X, n_folds = n_folds, lr = lr, epochs = epochs, learning_rate_scheduler=learning_rate_scheduler, epsilon = epsilon,batch_size = batch_size,weight_func = weight_func,
                              train_test_split = train_test_split, verbose = verbose,n_processes = n_processes,multiprocess_batch=multiprocess_batch,log=log,log_filename=log_filename)
        elif fit_method == 'no_cross_validate':
            self.fit_no_cross_validate(X, lr = lr, epochs = epochs, learning_rate_scheduler=learning_rate_scheduler, epsilon = epsilon,batch_size = batch_size,weight_func = weight_func,
                                train_test_split = train_test_split, verbose = verbose,n_processes = n_processes,multiprocess_batch=multiprocess_batch,log=log,log_filename=log_filename)
        



    def train_once(self,X_train,X_test,lr = 0.001, epochs = 1000, learning_rate_scheduler=None, epsilon = 1e-3,batch_size = 1,weight_func = lambda x: 1,
            multiprocessing = False, verbose = True,n_processes = 2):
        #note down the initial parameters
        # initial_params = self.kernel.get_params()
        #if the learning rate scheduler is not provided, use a constant learning rate
        if learning_rate_scheduler is None:
            learning_rate_scheduler = lambda x: lr
        #train the kernel
        losses = []
        best_params = self.kernel.get_params()
        for epoch in range(epochs):
            #shuffle the test data
            np.random.shuffle(X_test)
            #for each observation in the test data
            batch_gradient = 0
            prev_params = self.kernel.get_params()

            X_test_batched = np.array_split(X_test, len(X_test)//batch_size)
            NLL = 0
            n=0
            for batch in X_test_batched:
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
                # print(G)
                # print(G.shape)
                self.kernel.update_params({'R': -learning_rate_scheduler(epoch)*np.sum(G,axis=0)/len(X_train)})
                NLL += -np.sum(np.log(F))
                n+=len(batch)
            losses.append(NLL/n)
            #set best parameters to the current parameters if the loss is lower
            if NLL/n < np.min(losses):
                best_params = self.kernel.get_params()
            losses.append(NLL/n)  
            if verbose:
                print(f"Epoch {epoch}/{epochs} finished. NLL: {NLL/n}")
            #if the parameters have converged, stop training
            if np.linalg.norm(prev_params['R'] - self.kernel.get_params()['R']) < epsilon:
                print('Converged')
                break
        #return the best parameters and the losses
        return best_params, losses

            
        
        

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