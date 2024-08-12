import pandas as pd
import numpy as np
import sklearn.decomposition as skd
import sklearn.cluster as skc
from optimization import optimize_portfolio
import matplotlib.pyplot as plt

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
affinity  = -correlation_train
#set main diagonal to 0
np.fill_diagonal(affinity,0)
# from scipy import csgraph
import scipy.linalg

L = (affinity.shape[0]-1)*np.eye(affinity.shape[0])-affinity

# csgraph.laplacian(affinity, normed=True)
# print(L)
# print(affinity)
eigenvalues,_= scipy.linalg.eig(L)
# eigenvalues = eigenvalues[-50:]
eigenvalues = np.sort(eigenvalues)[5:50]
indexs = np.arange(5,50)
index_largest_gap = np.argmax(np.abs(np.diff(eigenvalues)))
fig,axs = plt.subplots(2,1,sharex=True,figsize=(10,5))
plt.sca(axs[0])
plt.plot(indexs,
    eigenvalues,"-x")
plt.title("Eigenvalues")
plt.ylabel("Eigenvalue")
plt.sca(axs[1])
plt.title("Eigenvalue differences")
plt.stem(indexs[1:],np.diff(eigenvalues))
plt.xlabel("Eigenvalue index")
plt.ylabel("Eigenvalue difference")
plt.savefig("eigenvalues.png",dpi = 300)
plt.show()
n_clusters = index_largest_gap + 6
print(n_clusters)

# # plt.show()
# clustering = skc.SpectralClustering(n_clusters=n_clusters,affinity='nearest_neighbors')
# #set main diagonal to 0
# np.fill_diagonal(affinity,0)
# print(np.any(np.isnan(affinity)))
# print(np.any(np.isinf(affinity)))
# clustering.fit(affinity)
# print(clustering.labels_)
# for i in range(len(np.unique(clustering.labels_))):
#     print("Cluster ",i)
#     for ticker in list(sp500_df.columns[clustering.labels_==i]):
#         print(ticker)
#     print("-----------------")

# #reorder the correlation matrix
# correlation_train = correlation_train[clustering.labels_.argsort(),:]
# correlation_train = correlation_train[:,clustering.labels_.argsort()]


# plt.sca(axs[1])
# plt.title("Clustered")
# plt.imshow(correlation_train)
# #draw the lines
# x=0
# for i in range(1,len(np.unique(clustering.labels_))+1):
#     xnew=x+(clustering.labels_==i-1).sum()
#     plt.plot([x,xnew],[x,x],color='r')
#     plt.plot([x,x],[x,xnew],color='r')
#     plt.plot([x,xnew],[xnew,xnew],color='r')
#     plt.plot([xnew,xnew],[x,xnew],color='r')
#     x=xnew
# plt.xlim([0,correlation_train.shape[0]])
# plt.ylim([correlation_train.shape[0],0])
# plt.show()