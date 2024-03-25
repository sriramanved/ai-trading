"""
A module containing functions that implement several well-known trading strategies
"""

import numpy as np
import pandas as pd
import pandas_ta as ta
import yfinance as yf
import wsb.stock_parser as sp

def calculate_stock_transactions_R3():
  """
  Implementation of the R3 trading strategy
  
  Buy a stock if...
  - its closing price is above the 200-day moving average
  - its 2-day RSI drops three days in a row and the first day's drop is from an RSI below 60
  - its 2-day RSI is currently below 10

  Sell the stock if its 2-day RSI is above 70
  """
  buy_list = []
  sell_list = []
  for stock_ticker in sp.stock_catalog:
    stock_ticker_data_frame = sp.stock_catalog[stock_ticker]

    # Evaluate whether or not to buy under the R3 trading strategy
    first_condition = stock_ticker_data_frame['Close'].iloc[-1] > stock_ticker_data_frame['sma_200'].iloc[-1]
    second_condition = (stock_ticker_data_frame['rsi2'].iloc[-3] > stock_ticker_data_frame['rsi2'].iloc[-2] and
                        stock_ticker_data_frame['rsi2'].iloc[-2] > stock_ticker_data_frame['rsi2'].iloc[-1] and
                        stock_ticker_data_frame['rsi2'].iloc[-3] < 60)
    third_condition = stock_ticker_data_frame['rsi2'].iloc[-1] < 10
    if first_condition and second_condition and third_condition:
      buy_list.append(stock_ticker)

    # Evaluate whether or not to sell under the R3 trading strategy
    elif stock_ticker_data_frame['rsi2'].iloc[-1] > 70:
      sell_list.append(stock_ticker)
  return buy_list, sell_list

def calculate_stock_transactions_D5():
  """
  Implementation of the Double Five trading strategy
  
  Buy a stock if...
  - its closing price is above the 200-day moving average
  - its closing price is at a five-day low

  Sell the stock if its closing price is at a five-day high
  """
  buy_list = []
  sell_list = []
  for stock_ticker in sp.stock_catalog:
    stock_ticker_data_frame = sp.stock_catalog[stock_ticker]
    first_condition = stock_ticker_data_frame['Close'].iloc[-1] > stock_ticker_data_frame['sma_200'].iloc[-1]
    second_condition = stock_ticker_data_frame['Close'].iloc[-1] <= stock_ticker_data_frame['5d_low'].iloc[-1]
    if first_condition and second_condition:
      buy_list.append(stock_ticker)
    elif stock_ticker_data_frame['Close'].iloc[-1] >= stock_ticker_data_frame['5d_high'].iloc[-1]:
      sell_list.append(stock_ticker)
  return buy_list, sell_list

def calculate_stock_transactions_MDU2():
  """
  Implementation of the Multiple Days Up and Multiple Days Down trading strategy
  
  Buy a stock if...
  - its closing price is above the 200-day moving average
  - its closing price is below the 5-day moving average
  - its closing price must have dropped at least four out of the last five trading days

  Sell the stock if its closing price is above the 5-day moving average
  """
  buy_list = []
  sell_list = []
  for stock_ticker in sp.stock_catalog:
    stock_ticker_data_frame = sp.stock_catalog[stock_ticker]
    first_condition = stock_ticker_data_frame['Close'].iloc[-1] > stock_ticker_data_frame['sma_200'].iloc[-1]
    second_condition = stock_ticker_data_frame['Close'].iloc[-1] < stock_ticker_data_frame['sma_5'].iloc[-1]
    num_drops = 0
    for i in range(-5, 0):
      if stock_ticker_data_frame['Close'].iloc[i] < stock_ticker_data_frame['Close'].iloc[i - 1]:
        num_drops += 1
    third_condition = num_drops >= 4
    if first_condition and second_condition and third_condition:
      buy_list.append(stock_ticker)
    elif stock_ticker_data_frame['Close'].iloc[-1] > stock_ticker_data_frame['sma_5'].iloc[-1]:
      sell_list.append(stock_ticker)
  return buy_list, sell_list

def calculate_stock_transactions_3DHL():
  """
  Implementation of the 3 Day High/Low trading strategy
  
  Buy a stock if...
  - its closing price is above the 200-day moving average
  - its high must be lower for the third day in a row
  - its low must be lower for the third day in a row

  Sell the stock if its 2-day RSI is above 70
  """
  buy_list = []
  sell_list = []
  for stock_ticker in sp.stock_catalog:
    stock_ticker_data_frame = sp.stock_catalog[stock_ticker]
    first_condition = stock_ticker_data_frame['Close'].iloc[-1] > stock_ticker_data_frame['sma_200'].iloc[-1]
    second_condition = (stock_ticker_data_frame['High'].iloc[-4] > stock_ticker_data_frame['High'].iloc[-3] and
                        stock_ticker_data_frame['High'].iloc[-3] > stock_ticker_data_frame['High'].iloc[-2] and
                        stock_ticker_data_frame['High'].iloc[-2] > stock_ticker_data_frame['High'].iloc[-1])
    third_condition = (stock_ticker_data_frame['Low'].iloc[-4] > stock_ticker_data_frame['Low'].iloc[-3] and
                       stock_ticker_data_frame['Low'].iloc[-3] > stock_ticker_data_frame['Low'].iloc[-2] and
                       stock_ticker_data_frame['Low'].iloc[-2] > stock_ticker_data_frame['Low'].iloc[-1])
    if first_condition and second_condition and third_condition:
      buy_list.append(stock_ticker)
    elif stock_ticker_data_frame['rsi2'].iloc[-1] > 70:
      sell_list.append(stock_ticker)
  return buy_list, sell_list