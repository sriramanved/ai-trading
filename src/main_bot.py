"""
A module that compiles lists of stocks to buy and sell, executes buy and sell orders, and sends update emails
"""
import numpy as np
import pandas as pd
import pandas_ta as ta
import yfinance as yf
import stock_parser as sp
import trading_strategies as ts
import alpaca_account_manager as aam
import config
import email_manager
import color_codes
from datetime import datetime
import db

DB = db.DatabaseDriver()
current_date = datetime.now()
formatted_date = current_date.strftime('%m/%d/%y')

def dollar_amount_to_color_html(dollar_amount):
  """
  Create an HTML table data cell that displays [dollar_amount] in green if [dollar_amount] > 0, red if [dollar_amount] < 0, and black otherwise
  """
  # Convert dollar_amount to a float in case it is passed in as a string
  dollar_amount = float(dollar_amount)
  return f'<td style="color:#33cc33">+${"{:.2f}".format(dollar_amount)}</td>' if dollar_amount > 0 else (f'<td style="color:#ff0000">-${"{:.2f}".format(abs(dollar_amount))}</td>' if dollar_amount < 0 else f'<td>${"{:.2f}".format(dollar_amount)}</td>')

def percentage_amount_to_color_html(percentage_amount):
  """
  Create an HTML table data cell that displays [percentage_amount] in green if [percentage_amount] > 0, red if [percentage_amount] < 0, and black otherwise
  """
  # Convert percentage_amount to a float in case it is passed in as a string
  percentage_amount = float(percentage_amount)
  return f'<td style="color:#33cc33">+{"{:.2f}".format(percentage_amount)}%</td>' if percentage_amount > 0 else (f'<td style="color:#ff0000">-{"{:.2f}".format(abs(percentage_amount))}%</td>' if percentage_amount < 0 else f'<td>{"{:.2f}".format(percentage_amount)}%</td>')

# Generate a dict for buy recommendations where key = stock ticker and value = number of strategies that rated the stock ticker as a BUY
buy_dict = {}
print(f'{color_codes.TEAL_COLOR_CODE}[  Generating buy dict...  ]{color_codes.RESET_COLOR_CODE}')
buy_list_r3, _ = ts.calculate_stock_transactions_R3()
buy_list_d5, _ = ts.calculate_stock_transactions_D5()
buy_list_mdu2, _ = ts.calculate_stock_transactions_MDU2()
buy_list_3dhl, _ = ts.calculate_stock_transactions_3DHL()
for buy_list in [buy_list_r3, buy_list_d5, buy_list_mdu2, buy_list_3dhl]:
  for stock_ticker in buy_list:
    buy_dict[stock_ticker] = buy_dict.get(stock_ticker, 0) + 1

# Sort buy_dict by value in descending order; this ensures that the stocks with more BUY recommendations appear earlier in the dict and that the stocks with fewer BUY recommendations appear later in the dict
# NOTE: sorting buy_dict by value in descending order guarantees that we first buy stocks with more BUY recommendations
buy_dict = dict(sorted(buy_dict.items(), key=lambda item: item[1], reverse=True))

# Execute buy and sell orders
if config.PRINT_STOCKS_ONLY:
  # Print out buy list
  print(f'\n{color_codes.BOLD_HIGHLIGHTED_GREEN_COLOR_CODE}Buy List:{color_codes.RESET_COLOR_CODE}')
  for stock_ticker in sorted(list(buy_dict.keys())):
    print(stock_ticker)
  print('\n')
else:
  # Initialize HTML string used to form body of update email
  body_html = '<head><style>th, td {padding: 5px; text-align: center;}</style></head><body>'
  
  # Strategy for selling: sell all open positions in order to maximize buying power for stocks with BUY recommendations
  print(f'{color_codes.TEAL_COLOR_CODE}[  Executing sell orders...  ]{color_codes.RESET_COLOR_CODE}')
  body_html += '<p><span style="color:#ff0000"><span style="font-size:18px">Sell:</span></span></p>'
  num_stocks_sold = 0
  for stock_ticker in [position.symbol for position in aam.get_open_positions()]:
    position_on_alpaca = aam.get_open_position_given_stock_ticker(stock_ticker)
    if position_on_alpaca is None:
      continue
    stock_ticker_price = float(position_on_alpaca.current_price)
    quantity = float(position_on_alpaca.qty)

    if aam.execute_sell_order(stock_ticker, quantity):
      num_stocks_sold += 1
      # Print out table header in email if we sold at least one stock
      if num_stocks_sold == 1:
        body_html += '<table border="1" cellpadding="1" cellspacing="1" style="width:500px"><tbody><tr><td style="font-weight: bold;">Asset</td><td style="font-weight: bold;">Quantity</td><td style="font-weight: bold;">Price Per Share</td><td style="font-weight: bold;">Total Market Value</td><td style="font-weight: bold;">All-Time Profit/Loss</td><td style="font-weight: bold;">% All-Time Profit/Loss</td></tr>'
      body_html += f'<tr><td>{stock_ticker}</td><td>{quantity}</td><td>${"{:.2f}".format(stock_ticker_price)}</td><td>${"{:.2f}".format(stock_ticker_price * quantity)}</td>{dollar_amount_to_color_html(position_on_alpaca.unrealized_pl)}{percentage_amount_to_color_html(position_on_alpaca.unrealized_plpc)}</tr>'
  if num_stocks_sold >= 1:
    body_html += '</tbody></table>'
  else:
    body_html += '<p>None</p>'

  # Execute all buy orders
  print(f'{color_codes.TEAL_COLOR_CODE}[  Executing buy orders...  ]{color_codes.RESET_COLOR_CODE}')
  body_html += '<p><span style="font-size:18px"><span style="color:#33cc33">Buy:</span></span></p>'
  num_stocks_bought = 0
  for stock_ticker in list(buy_dict.keys()):
    # Allocate [config.MAX_PERCENT_ALLOCATED_PER_TRADE]% of available cash towards buying [stock_ticker]
    allocated_amount = config.MAX_PERCENT_ALLOCATED_PER_TRADE / 100 * aam.get_buying_power()
    stock_ticker_price = sp.stock_catalog[stock_ticker]['Close'].iloc[-1]
    quantity = allocated_amount // stock_ticker_price
    if quantity <= 0:
      print(f'{color_codes.RED_COLOR_CODE}Skipped buying {stock_ticker} because one share of it exceeds allocated spending{color_codes.RESET_COLOR_CODE}')
      continue # Not worth buying less than one share of a stock
    if aam.execute_buy_order(stock_ticker, quantity):
      num_stocks_bought += 1
      # Print out table header in email if we bought at least one stock
      if num_stocks_bought == 1:
        body_html += '<table border="1" cellpadding="1" cellspacing="1" style="width:500px"><tbody><tr><td style="font-weight: bold;">Asset</td><td style="font-weight: bold;">Quantity</td><td style="font-weight: bold;">Price Per Share</td><td style="font-weight: bold;">Total Market Value</td></tr>'
      body_html += f'<tr><td>{stock_ticker}</td><td>{quantity} shares</td><td>${"{:.2f}".format(float(stock_ticker_price))}</td><td>${"{:.2f}".format(stock_ticker_price * quantity)}</td></tr>'
  if num_stocks_bought >= 1:
    body_html += '</tbody></table>'
  else:
    body_html += '<p>None</p>'

  # Display all open positions in update email
  body_html += '<p><span style="color:#0000ff"><span style="font-size:18px">Positions:</span></span></p><table border="1" cellpadding="1" cellspacing="1" style="width:500px"><tbody><tr><td style="font-weight: bold;">Asset</td><td style="font-weight: bold;">Quantity</td><td style="font-weight: bold;">Price Per Share</td><td style="font-weight: bold;">Total Market Value</td><td style="font-weight: bold;">All-Time Profit/Loss</td><td style="font-weight: bold;">% All-Time Profit/Loss</td><td style="font-weight: bold;">Today\'s Profit/Loss</td><td style="font-weight: bold;">% Today\'s Profit/Loss</td></tr>'
  for position in aam.get_open_positions():
    body_html += f'<tr><td>{position.symbol}</td><td>{float(position.qty)} shares</td><td>${"{:.2f}".format(float(position.current_price))}</td><td>${"{:.2f}".format(float(position.market_value))}</td>{dollar_amount_to_color_html(position.unrealized_pl)}{percentage_amount_to_color_html(position.unrealized_plpc)}{dollar_amount_to_color_html(position.unrealized_intraday_pl)}{percentage_amount_to_color_html(position.unrealized_intraday_plpc)}</tr>'
  body_html += '</tbody></table><p><span style="color:#005e55"><span style="font-size:18px">Portfolio Balance Information:</span></span></p><table border="1" cellpadding="1" cellspacing="1" style="width:500px"><tbody>'

  # Display all portfolio balance information
  body_html += f'<tr><td style="font-weight: bold;">Available Cash</td><td>${"{:.2f}".format(aam.get_available_cash())}</td></tr>'
  body_html += f'<tr><td style="font-weight: bold;">Total Equity</td><td>${"{:.2f}".format(float(aam.get_equity()))}</td></tr>'
  body_html += '</tbody></table><p><span style="color:#f2981b"><span style="font-size:18px">Performance Logs:</span></span></p><table border="1" cellpadding="1" cellspacing="1" style="width:500px"><tbody><tr><td style="font-weight: bold;">Period</td><td style="font-weight: bold;">Bot Performance</td><td style="font-weight: bold;">S&P 500 Performance</td></tr>'
  
  # Display all performance logs in update email
  performance_logs = aam.calculate_performance_logs()
  for period in performance_logs:
    body_html += f'<tr><td>{period}</td>'

    period_bot_performance = performance_logs[period][0]
    body_html += percentage_amount_to_color_html(period_bot_performance)

    period_snp500_performance = performance_logs[period][1]
    body_html += f'{percentage_amount_to_color_html(period_snp500_performance)}</tr>'
  body_html += '</tbody></table><p>&nbsp;</p></body>'

  # Send update email
  email_manager.send_email('charleswang2021@gmail.com', f'{formatted_date} Report', body_html)