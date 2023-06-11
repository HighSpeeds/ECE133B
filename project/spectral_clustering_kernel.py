import pandas as pd
import numpy as np
import sklearn.decomposition as skd
import sklearn.cluster as skc
from optimization import optimize_portfolio
import matplotlib.pyplot as plt
from model import *

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

cov_train = np.cov(train.T)
mu_train = np.mean(train,axis=0)
# plt.imshow(cov_train)
# plt.show()
correlation_train = np.corrcoef(train.T)

# fig, axs = plt.subplots(1,2)
# plt.sca(axs[0])
# plt.imshow(correlation_train)
# plt.title("Unclustered")

clustering = skc.SpectralClustering(n_clusters=30,affinity='nearest_neighbors')
affinity  = -correlation_train
#set main diagonal to 0
np.fill_diagonal(affinity,0)
print(np.any(np.isnan(affinity)))
print(np.any(np.isinf(affinity)))
clustering.fit(affinity)
print(clustering.labels_)
cluster_weights = {}
clusters = {}
cluster_returns_train = []
cluster_returns_test = []
for i in range(len(np.unique(clustering.labels_))):
    print("Cluster ",i)
    train_cluster = train[:,clustering.labels_==i]
    test_cluster = test[:,clustering.labels_==i]
    cov_train_cluster = np.cov(train_cluster.T)
    # mu_train_cluster = np.mean(train_cluster,axis=0)
    initial_guess = cov_train_cluster*train.shape[0]
    print(initial_guess)
    # # print(initial_guess)
    kernel = MultivariateGaussianKernel(initial_guess)
    Model = model(kernel)
    Model.fit(train_cluster,dict(n_folds=5, lr=0.0001, epochs=5, epsilon=1e-3, batch_size=20,multiprocessing = False))
    weights = Model.get_weights()
    cluster_weights[i] = weights
    clusters[i] = list(sp500_df.columns[clustering.labels_==i])
    print("Cluster weights: ",cluster_weights[i])
    print("Cluster stocks: ",clusters[i])
    print("cluster sharpe train:", round(np.mean(train_cluster @ cluster_weights[i])/np.std(train_cluster @ cluster_weights[i])*np.sqrt(252),3))
    print("cluster sharpe test:", round(np.mean(test_cluster @ cluster_weights[i])/np.std(test_cluster @ cluster_weights[i])*np.sqrt(252),3))
    cluster_returns_train.append(train_cluster @ cluster_weights[i])
    cluster_returns_test.append(test_cluster @ cluster_weights[i])

import pickle 

with open('weights/Kernel_clusters.pkl', 'wb') as f:
    pickle.dump(clusters, f)

with open('weights/Kernel_cluster_weights.pkl', 'wb') as f:
    pickle.dump(cluster_weights, f)


#create a new matrix of cluster returns
cluster_returns_train = np.array(cluster_returns_train).T
cluster_returns_test = np.array(cluster_returns_test).T
cluster_returns = np.concatenate((cluster_returns_train,cluster_returns_test),axis=0)
print(cluster_returns.shape)
#get the covariance matrix of the cluster returns
cov_cluster_returns_train = np.cov(cluster_returns_train.T)
cov_cluster_returns_test = np.cov(cluster_returns_test.T)

#optimize the portfolio of cluster returns
# cluster_returns_weights = optimize_portfolio(np.mean(cluster_returns_train,axis=0),cov_cluster_returns_train)

initial_guess = cov_cluster_returns_train*train.shape[0]
print(initial_guess)
# # print(initial_guess)
kernel = MultivariateGaussianKernel(initial_guess)
Model = model(kernel)
Model.fit(cluster_returns_train,dict(n_folds=5, lr=0.0001, epochs=5, epsilon=1e-3, batch_size=20,multiprocessing = True))
cluster_returns_weights = Model.get_weights()


print("Cluster returns weights: ",cluster_returns_weights)
print("Cluster returns sharpe train:", round(np.mean(cluster_returns_train @ cluster_returns_weights)/np.std(cluster_returns_train @ cluster_returns_weights)*np.sqrt(252),3))
print("Cluster returns sharpe test:", round(np.mean(cluster_returns_test @ cluster_returns_weights)/np.std(cluster_returns_test @ cluster_returns_weights)*np.sqrt(252),3))
plt.plot(np.cumprod(1+cluster_returns @ cluster_returns_weights),label="Clustered")
plt.axvline(x=int(changes.shape[0]*0.8),color='r')
plt.plot(sp500_df_price['^GSPC'].values/sp500_df_price['^GSPC'].values[0])
print("Yearly return train clustered:", 
      round(100*(np.cumprod(1+cluster_returns_train @ cluster_returns_weights)[-1]**(252/cluster_returns_train.shape[0])-1),3))
print("Yearly return test clustered:", 
      round(100*(np.cumprod(1+cluster_returns_test @ cluster_returns_weights)[-1]**(252/cluster_returns_test.shape[0])-1),3))
print("S&P500 yearly return train:", 100*((sp500_df_price['^GSPC'].values[cluster_returns_train.shape[0]]/sp500_df_price['^GSPC'].values[0])**(252/cluster_returns_train.shape[0])-1))
print("S&P500 yearly return test:", 100*((sp500_df_price['^GSPC'].values[-1]/sp500_df_price['^GSPC'].values[cluster_returns_train.shape[0]])**(252/cluster_returns_test.shape[0])-1))

#save the clusters and cluster weights

np.save('weights/cluster_returns_weights.npy',cluster_returns_weights)
#creat the overall weights
overall_weights = np.zeros((changes.shape[1],))
for i in range(len(np.unique(clustering.labels_))):
    overall_weights[clustering.labels_==i] = cluster_weights[i]*cluster_returns_weights[i]
print("Overall weights: ",overall_weights)
np.save('weights/cluster_overall_weights.npy',overall_weights)
plt.show()
