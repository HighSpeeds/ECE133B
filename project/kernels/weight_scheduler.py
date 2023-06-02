import numpy as np

class cosineAnnealing:
    """Cosine annealing scheduler for learning rate.
    """
    def __init__(self, T_max, eta_min=0, last_epoch=-1):
        self.T_max = T_max
        self.eta_min = eta_min
        self.last_epoch = last_epoch
        self.last_lr = 1
    
    def __call__(self, epoch):
        self.last_epoch = epoch
        self.last_lr = self.eta_min + (1 + np.cos(np.pi * epoch / self.T_max)) / 2 * (1 - self.eta_min)
        return self.last_lr
    


#test
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    scheduler = cosineAnnealing(10)
    lrs = []
    for i in range(100):
        lrs.append(scheduler(i))
    plt.plot(lrs)
    plt.show()