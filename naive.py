import pandas as pd
import numpy as np
import sklearn.decomposition as skd
from optimization import optimize_portfolio

sp500_df = pd.read_csv('sp500.csv')
sp500_df_price = pd.read_csv('sp500_price.csv')
sp500_df_price.set_index('date', inplace=True)

sp500_df.set_index('date', inplace=True)
#make sure the index is the same
sp500_df_price = sp500_df_price[sp500_df_price.index.isin(sp500_df.index)]
# print(sp500_df.head())
sp500_df = sp500_df.pct_change()
sp500_df.dropna(inplace=True)
changes = sp500_df.values
sp500_changes = sp500_df_price.pct_change()
sp500_changes.dropna(inplace=True)
sp500_changes = sp500_changes.values
#split into train and test
train = changes[:int(changes.shape[0]*0.8)]
test = changes[int(changes.shape[0]*0.8):]
sp500_train = sp500_changes[:int(changes.shape[0]*0.8)]
sp500_test = sp500_changes[int(changes.shape[0]*0.8):]
# print(train.shape)
cov_train = np.cov(train.T)
mu_train = np.mean(train,axis=0)
print(cov_train)
print(mu_train)
cov_test = np.cov(test.T)
mu_test = np.mean(test,axis=0)

w = optimize_portfolio(mu_train,cov_train)
print(w)
train_changes = train @ w
test_changes = test @ w

print("train sharpe ratio: ", np.mean(train_changes)/np.std(train_changes)*np.sqrt(252))
print("test sharpe ratio: ", np.mean(test_changes)/np.std(test_changes)*np.sqrt(252))
print("naive train sharpe ratio: ", np.mean(sp500_train)/np.std(sp500_train)*np.sqrt(252))
print("naive test sharpe ratio: ", np.mean(sp500_test)/np.std(sp500_test)*np.sqrt(252))
#save w
np.save('weights/naive.npy',w)
import matplotlib.pyplot as plt
plt.plot(np.cumprod(1+changes@w))
plt.axvline(x=int(changes.shape[0]*0.8),color='r')
plt.plot(sp500_df_price['^GSPC'].values/sp500_df_price['^GSPC'].values[0])
plt.show()

