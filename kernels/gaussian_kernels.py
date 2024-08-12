import numpy as np 
import scipy.stats as stats
# from kernels.base_kernel import kernel

class kernel:
    """Base class for all kernels.
    """
    def __init__(self):
        pass

    def update_params(self, params_delta: dict):
        """Update the parameters of the kernel.
        """
        for key, value in params_delta.items():
            param = getattr(self, key)
            setattr(self, key, param + value)

    def set_params(self, params: dict):
        """Set the parameters of the kernel.
        """
        for key, value in params.items():
            setattr(self, key, value)

    def get_params(self):
        """Get the parameters of the kernel.
        """
        return {key: getattr(self, key) for key in self.__dict__.keys() if not key.startswith('__')}

class MultivariateGaussianKernel(kernel):

    def __init__(self, Sigma_init):
        super().__init__()
        self.Sigma = Sigma_init
        print(self.Sigma)
        #Cholesky decomposition of the covariance matrix
        L = np.linalg.cholesky(self.Sigma)
        #we want it in terms of R
        self.R = L.T
        self.calculate_invs()

    def __call__(self, x):
        # print(self.Sigma)
        # if np.any(np.isnan(self.Gaussian.pdf(x))):
        #     print('nan')
        #     print(self.Sigma)
        #     print(self.R)
        #     raise ValueError
        
        return self.Gaussian.pdf(x)
    
    def calculate_invs(self):
        self.R_inv = np.linalg.inv(self.R)
        self.Sigma_inv = self.R_inv @ self.R_inv.T
        self.Gaussian = stats.multivariate_normal(cov=self.Sigma,allow_singular=True)
    
    def get_gradient(self, x):
        #if x is one dimensional:
        p = self(x)
        if len(x.shape) == 1:
            # p = self(x)
            # R_inv = np.linalg.inv(self.R)
            x = x.reshape(-1, 1)
            gradient = -p*(self.R_inv.T - self.Sigma_inv @ x @ x.T @ self.R_inv)
        #if x is two dimensional:
        else:
            gradient = -self.R_inv.T*np.sum(p) + self.Sigma_inv @ \
                np.sum(p.reshape(-1,1,1)*( x.reshape(x.shape[0],x.shape[1],1) @ x.reshape(x.shape[0], 1, x.shape[1])), axis=0) @ self.R_inv
        return gradient
    

    def update_params(self, params_delta: dict):
        """Update the parameters of the kernel.
        """
        if type(params_delta['R'])==float or type(params_delta['R'])==np.float64:
            # print('nan')
            return
        if np.isnan(params_delta['R']).any():
            print('nan')
            return
        #zero out the values below the diagonal
        try:
            params_delta['R'] = np.triu(params_delta['R'])
        except:
            print(params_delta['R'])
            print(type(params_delta['R']))
            raise ValueError
        y = np.diagonal(self.R)
        x = np.log(y)
        self.R = self.R + params_delta['R']
        # #if the value on the diagonal is negative, we set it to 0.001
        # self.R[np.diag_indices_from(self.R)] = np.maximum(np.diagonal(self.R))
        
        self.Sigma = self.R.T @ self.R
        # print(self.Sigma)
        #check if sigma is positive definite
        
        self.calculate_invs()

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
            try:
                #Cholesky decomposition of the covariance matrix
                L = np.linalg.cholesky(self.Sigma)
                #we want it in terms of R
                self.R = L.T
            except:
                print(self.Sigma)
                # raise ValueError
        
        self.calculate_invs()

    def randomized_init(self):
        self.R = np.random.randn(self.R.shape[0], self.R.shape[1])
        self.R[np.diag_indices_from(self.R)] = np.maximum(np.diagonal(self.R), 0.001)
        self.R = np.triu(self.R)
        self.Sigma = self.R.T @ self.R
        self.calculate_invs()

if __name__ == "__main__":
    initial_guess = 0.5*np.ones((10, 10))+0.5*np.eye(10)
    kernel = MultivariateGaussianKernel(initial_guess)
    
    data = stats.multivariate_normal.rvs(mean=[0]*10, cov=np.eye(10)+0.5*np.ones((10, 10)),
                                          size=10000)
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