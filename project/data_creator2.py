

import yfinance as yf
import pandas as pd
import multiprocessing as mp
import tqdm

def get_data(ticker):
	print(ticker)
	ticker_stock = yf.Ticker(ticker)
	ticker_df = ticker_stock.history(period='10y')
	#make the date column a column
	ticker_df.reset_index(inplace=True)
	#keep only date and close columns
	ticker_df = ticker_df[['Date', 'Close']]
	#for each nan value, replace with the previous value
	ticker_df['Close'].fillna(method='ffill', inplace=True)
	#rename columns
	ticker_df.columns = ['date', ticker]
	return ticker_df

def get_sp500():
	stock_tickers = pd.read_csv('constituents_csv.csv')
	s_and_p = stock_tickers['Symbol'].tolist()
	print(s_and_p)
	sp500_dfs = []
	for ticker in tqdm.tqdm(s_and_p):
		sp500_dfs.append(get_data(ticker))

	sp500_df = sp500_dfs[0]
	for df in tqdm.tqdm(sp500_dfs[1:]):
		sp500_df = sp500_df.merge(df, on='date', how='inner')
	
	return sp500_df

if __name__ == '__main__':
	sp500_df = get_sp500()
	sp500_df.to_csv('sp500.csv', index=False)


