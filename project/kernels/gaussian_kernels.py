import numpy as np 
import scipy.stats as stats
from base_kernel import kernel


class MultivariateGaussianKernel(kernel):

    def __init__(self, Sigma_init):
        super().__init__()
        self.Sigma = Sigma_init
        #Cholesky decomposition of the covariance matrix
        L = np.linalg.cholesky(self.Sigma)
        #we want it in terms of R
        self.R = L.T

    def __call__(self, x):
        return stats.multivariate_normal.pdf(x, cov=self.Sigma)
    
    def get_gradient(self, x):
        p = self(x)
        R_inv = np.linalg.inv(self.R)
        x = x.reshape(-1, 1)
        gradient = -p*(R_inv.T - np.linalg.inv(self.Sigma) @ x @ x.T @ R_inv)
        return gradient
    
    def update_params(self, params_delta: dict):
        """Update the parameters of the kernel.
        """
        if np.isnan(params_delta['R']).any():
            print('nan')
            return
        self.R = self.R + params_delta['R']
        self.Sigma = self.R.T @ self.R

    def set_params(self, params: dict):
        """Set the parameters of the kernel.
        """
        #if just R in params, we can update Sigma
        if 'R' in params.keys():
            self.R = params['R']
            self.Sigma = self.R.T @ self.R
        #if just Sigma in params, we can update R
        elif 'Sigma' in params.keys():
            self.Sigma = params['Sigma']
            #Cholesky decomposition of the covariance matrix
            L = np.linalg.cholesky(self.Sigma)
            #we want it in terms of R
            self.R = L.T

if __name__ == "__main__":
    kernel = MultivariateGaussianKernel(np.array([[1, 0.5], [0.5, 1]]))
    
    data = stats.multivariate_normal.rvs(mean=[0, 0], cov=np.array([[5, -1], [-1, 5]]), size=10000)
    for epoch in range(10):
        g = 0
        for x in data:
            p = kernel(x)
            gradient = kernel.get_gradient(x)
            kernel.update_params({'R': 0.001/p*gradient})
        print(kernel.get_params())
        nll = 0
        for x in data:
            p = kernel(x)
            nll += -np.log(p)
            # # gradient = kernel.get_gradient(x)
            # kernel.update_params({'R': -0.01/p*gradient})
        print(nll)
