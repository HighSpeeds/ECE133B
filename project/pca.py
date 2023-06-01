import pandas as pd
import numpy as np
import sklearn.decomposition as skd

sp500_df = pd.read_csv('sp500.csv')
sp500_df.set_index('date', inplace=True)
# print(sp500_df.head())
sp500_df = sp500_df.pct_change()
sp500_df.dropna(inplace=True)
changes = sp500_df.values
pca = skd.PCA(n_components=100)
pca.fit(changes)
print(pca.explained_variance_ratio_)
#get the number of components that explain 90% of the variance
n_components = 0
variance = 0
for i in range(len(pca.explained_variance_ratio_)):
    variance += pca.explained_variance_ratio_[i]
    n_components += 1
    if variance >= 0.9:
        break
#get the sharpe ratio of each component
sharpe_ratios = []
for i in range(n_components):
    sharpe_ratios.append(np.mean(pca.components_[i,:])/np.std(pca.components_[i,:]))
print(sharpe_ratios)
