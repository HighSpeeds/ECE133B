

import yfinance as yf
import pandas as pd
import multiprocessing as mp
import tqdm

def get_data(ticker):
	print(ticker)
	ticker_stock = yf.Ticker(ticker)
	try:
		ticker_df = ticker_stock.history(end = '2020-01-01', start = '2010-01-01')
		#make the date column a column
		ticker_df.reset_index(inplace=True)
		#keep only date and close columns
		ticker_df = ticker_df[['Date', 'Close']]
		#for each nan value, replace with the previous value
		ticker_df['Close'].fillna(method='ffill', inplace=True)
		#rename columns
		ticker_df.columns = ['date', ticker]
		# print(ticker_df.head())
		return ticker_df
	except:
		return None

def get_sp500():
	stock_tickers = pd.read_csv('constituents_csv.csv')
	s_and_p = stock_tickers['Symbol'].tolist()
	print(s_and_p)
	sp500_dfs = []
	for ticker in tqdm.tqdm(s_and_p):
		df = get_data(ticker)
		if df is not None:
			sp500_dfs.append(df)

	sp500_df = sp500_dfs[0]
	for df in tqdm.tqdm(sp500_dfs[1:]):
		if df.shape[0] == sp500_df.shape[0]:
			sp500_df = sp500_df.merge(df, on='date', how='left')
	
	return sp500_df

def get_sp500_price():
	ticker = '^GSPC'
	df = get_data(ticker)
	return df

if __name__ == '__main__':
	sp500_df = get_sp500()
	sp500_df.to_csv('sp500.csv', index=False)
	sp500_price = get_sp500_price()
	sp500_price.to_csv('sp500_price.csv', index=False)


