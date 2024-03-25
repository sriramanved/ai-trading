"""
A module that provides an interface to interact with Alpaca account
"""
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
import config
import wsb.color_codes as color_codes
import wsb.alpaca_account_manager as aam
import wsb.db as db
import wsb.stock_parser as sp
import yfinance as yf

trading_client = TradingClient(config.ALPACA_KEY, config.ALPACA_SECRET_KEY, paper=False)
DB = db.DatabaseDriver()

def get_available_cash():
  """
  Retrieve available cash in Alpaca account
  """
  return float(trading_client.get_account().cash)

def get_buying_power():
  """
  Retrieve buying power of Alpaca account
  """
  return float(trading_client.get_account().buying_power)

def get_equity():
  """
  Retrieve total equity of Alpaca account
  """
  return float(trading_client.get_account().equity)

def get_open_positions():
  """
  Retrieve all open positions owned by Alpaca account
  """
  return trading_client.get_all_positions()

def get_open_position_given_stock_ticker(stock_ticker):
  """
  Retrieve the position object associated with the given stock ticker; return None object if [stock_ticker] isn't owned by Alpaca account
  """
  try:
    return trading_client.get_open_position(stock_ticker)
  except Exception as e:
    return None

def execute_buy_order(stock_ticker, quantity):
  """
  Execute a buy order that buys [quantity] shares of [stock_ticker] through Alpaca
  """
  try:
    market_order_data = MarketOrderRequest(
                        symbol=stock_ticker,
                        qty=quantity,
                        side=OrderSide.BUY,
                        time_in_force=TimeInForce.GTC
                        )
    market_order = trading_client.submit_order(order_data=market_order_data)
    print(market_order)
    print(f'{color_codes.GREEN_COLOR_CODE}Successfully bought {quantity} shares of {stock_ticker}{color_codes.RESET_COLOR_CODE}')
    return True
  except Exception as e:
    print(f'{color_codes.RED_COLOR_CODE}An error happened while attempting to buy {quantity} shares of {stock_ticker}{color_codes.RESET_COLOR_CODE}')
    return False

def execute_sell_order(stock_ticker, quantity):
  """
  Execute a sell order that sells [quantity] shares of [stock_ticker] through Alpaca
  """
  try:
    market_order_data = MarketOrderRequest(
                        symbol=stock_ticker,
                        qty=quantity,
                        side=OrderSide.SELL,
                        time_in_force=TimeInForce.GTC
                        )
    market_order = trading_client.submit_order(order_data=market_order_data)
    print(market_order)
    print(f'{color_codes.GREEN_COLOR_CODE}Successfully sold {quantity} shares of {stock_ticker}{color_codes.RESET_COLOR_CODE}')
    return True
  except Exception as e:
    print(f'{color_codes.RED_COLOR_CODE}An error happened while attempting to sell {quantity} shares of {stock_ticker}{color_codes.RESET_COLOR_CODE}')
    return False
  
def calculate_percent_change(current_equity, initial_equity):
  return (current_equity - initial_equity) / initial_equity * 100

def calculate_performance_logs():
  """
  Calculate performance logs (ex. 1D, 5D performance) of Alpaca account
  """
  # Insert today's starting equity into database
  DB.insert_log(float(trading_client.get_account().last_equity))
  performance_logs = {}
  current_id = DB.get_current_id()
  snp500_stock_data = yf.download('^GSPC', period='6d')

  # Calculate 1D profit/loss
  bot_percent_change_1D = calculate_percent_change(get_equity(), float(trading_client.get_account().last_equity))
  snp500_percent_change_1D = calculate_percent_change(snp500_stock_data['Close'].iloc[-1], snp500_stock_data['Close'].iloc[-2])
  performance_logs['1D'] = (float('{:.2f}'.format(bot_percent_change_1D)), float('{:.2f}'.format(snp500_percent_change_1D)))

  # Calculate 5D profit/loss
  if DB.get_log_count() >= 5:
    last_equity_5D = DB.get_log_by_id(current_id - 4)['equity'] # Equity of account at 4pm EST five days ago
    bot_percent_change_5D = calculate_percent_change(get_equity(), last_equity_5D)
    snp500_percent_change_5D = calculate_percent_change(snp500_stock_data['Close'].iloc[-1], snp500_stock_data['Close'].iloc[-6])
    performance_logs['5D'] = (float('{:.2f}'.format(bot_percent_change_5D)), float('{:.2f}'.format(snp500_percent_change_5D)))
  return performance_logs