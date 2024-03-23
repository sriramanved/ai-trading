"""
A module that produces a catalog containing historical price and technical analysis indicator data of all stocks on the S&P 500
"""
import numpy as np
import pandas as pd
import pandas_ta as ta
import yfinance as yf
import math
import color_codes

stock_catalog = {}

# Retrieve stock tickers for all 503 stocks on the S&P 500
snp500_stock_tickers = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0].loc[:, 'Symbol'].tolist()

# Download all stocks at once and worry about reliability later
print(f'{color_codes.TEAL_COLOR_CODE}[  Downloading stock data and calculating technical analysis indicators for all stocks on the S&P 500...  ]{color_codes.RESET_COLOR_CODE}')
downloaded_stocks = yf.download(snp500_stock_tickers, period='200d')

# Calculate technical analysis indicators for stock tickers; also initiate reliability protocols to ensure all stock tickers have been successfully downloaded
for stock_ticker in downloaded_stocks['Adj Close'].columns.get_level_values(0).tolist():
  # Check for case where current stock ticker failed to download at all; if so, redownload the stock and populate the data frame with the redownloaded data
  if math.isnan(downloaded_stocks['Adj Close'][stock_ticker].iloc[-1]):
    redownloaded_stock_data = yf.download(stock_ticker, period='200d')
    # Skip this stock ticker if unable to redownload stock
    if redownloaded_stock_data.empty:
      continue
    downloaded_stocks.loc[:, ('Adj Close', stock_ticker)] = redownloaded_stock_data['Adj Close']
    downloaded_stocks.loc[:, ('Close', stock_ticker)] = redownloaded_stock_data['Close']
    downloaded_stocks.loc[:, ('High', stock_ticker)] = redownloaded_stock_data['High']
    downloaded_stocks.loc[:, ('Low', stock_ticker)] = redownloaded_stock_data['Low']
    downloaded_stocks.loc[:, ('Open', stock_ticker)] = redownloaded_stock_data['Open']
    downloaded_stocks.loc[:, ('Volume', stock_ticker)] = redownloaded_stock_data['Volume']

  # Skip stock ticker if the corresponding stock has not been listed for 200 days, or if it's been delisted
  if math.isnan(downloaded_stocks['Open'][stock_ticker].iloc[0]) or downloaded_stocks['Volume'][stock_ticker].iloc[-1] == 0:
    continue

  stock_ticker_data_frame = pd.DataFrame()
  stock_ticker_data_frame['Open'] = downloaded_stocks['Open'][stock_ticker]
  stock_ticker_data_frame['High'] = downloaded_stocks['High'][stock_ticker]
  stock_ticker_data_frame['Low'] = downloaded_stocks['Low'][stock_ticker]
  stock_ticker_data_frame['Close'] = downloaded_stocks['Close'][stock_ticker]
  stock_ticker_data_frame['Adj Close'] = downloaded_stocks['Adj Close'][stock_ticker]
  stock_ticker_data_frame['Volume'] = downloaded_stocks['Volume'][stock_ticker]
  if not stock_ticker_data_frame.empty:
    stock_ticker_data_frame['sma_5'] = ta.sma(close=stock_ticker_data_frame['Close'], length=5, append=True)
    stock_ticker_data_frame['sma_200'] = ta.sma(close=stock_ticker_data_frame['Close'], length=200, append=True)
    stock_ticker_data_frame['rsi2'] = ta.rsi(close=stock_ticker_data_frame['Close'], length=2)
    stock_ticker_data_frame['5d_high'] = stock_ticker_data_frame['High'].rolling(window=5).max()
    stock_ticker_data_frame['5d_low'] = stock_ticker_data_frame['Low'].rolling(window=5).min()
    # Check that stock_ticker is a stock that we can perform our trading strategies on
    if (stock_ticker_data_frame['sma_5'].iloc[-1] == None or
        stock_ticker_data_frame['sma_200'].iloc[-1] == None or
        stock_ticker_data_frame['rsi2'].iloc[-1] == None or
        stock_ticker_data_frame['5d_high'].iloc[-1] == None or
        stock_ticker_data_frame['5d_low'].iloc[-1] == None):
      continue
    stock_catalog[stock_ticker] = stock_ticker_data_frame
print(stock_catalog)