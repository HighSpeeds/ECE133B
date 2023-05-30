import pandas as pd
import numpy as np
import sklearn.decomposition as skd

sp500_df = pd.read_csv('sp500.csv')
sp500_df.set_index('date', inplace=True)
# print(sp500_df.head())
sp500_df = sp500_df.pct_change()
sp500_df.dropna(inplace=True)
changes = sp500_df.values
pca = skd.PCA(n_components=10)
pca.fit(changes)
print(pca.explained_variance_ratio_)
print(pca.singular_values_)
#for each component, print the top 5 stocks
for i in range(10):
    print('Component', i)
    #argsort returns the indices that would sort the array
    #[-5:] gets the last 5 indices
    #[::-1] reverses the array
    #the result is the indices of the top 5 stocks
    top5 = np.argsort(np.abs(pca.components_[i]))[-5:][::-1]
    print(sp500_df.columns[top5])
    print(pca.components_[i][top5])
    print()