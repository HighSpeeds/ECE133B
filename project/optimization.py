import numpy as np
import scipy.optimize as opt
# import matplotlib.pyplot as plt


def optimize_portfolio(mu,sigma):
    """optimize based on sharpe ratio

    Args:
        mu (_type_): _description_
        sigma (_type_): _description_
    """
    func = lambda y: y.T@sigma@y
    cons = ({'type': 'eq', 'fun': lambda y: mu.T@y-1},
            {'type': 'ineq', 'fun': lambda y: y})
    res = opt.minimize(func, np.ones(mu.shape[0])/mu.shape[0], constraints=cons)
    y = res.x
    w = y/np.sum(y)
    print("target sharpe ratio: ", mu.T@w/np.sqrt(w.T@sigma@w)*np.sqrt(252))
    print("optimal weights: ", w)
    return w

if __name__ == "__main__":
    mu = np.array([0.1,0.2,0.3])
    sigma = np.array([[0.1,0.05,0.01],[0.05,0.2,0.03],[0.01,0.03,0.3]])
    w = optimize_portfolio(mu,sigma)
    print(w)