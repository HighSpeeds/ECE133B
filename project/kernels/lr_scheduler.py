import numpy as np

class exponential_decay:
    def __init__(self,lr0,decay_rate):
        self.lr0 = lr0
        self.decay_rate = decay_rate
    def __call__(self,epoch):
        return self.lr0*np.exp(-self.decay_rate*epoch)
    
class cosineAnnealing:
    def __init__(self,lr0):
        self.lr0 = lr0
    def __call__(self,epoch):
        return self.lr0/2*(np.cos(np.pi*epoch/10)+1)
    


#test
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    scheduler = cosineAnnealing(10)
    lrs = []
    for i in range(100):
        lrs.append(scheduler(i))
    plt.plot(lrs)
    plt.show()