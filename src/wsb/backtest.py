"""
A module that compiles lists of stocks to buy and sell, executes buy and sell orders, and sends update emails
"""
import numpy as np
import pandas as pd
import stock_parser as sp
import trading_strategies as ts
import color_codes
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

def main():
    start_date = datetime.strptime('2022-07-01', '%Y-%m-%d')
    end_date = datetime.strptime('2023-11-27', '%Y-%m-%d')
    current_date = start_date
    initial_cash = 10000
    cash = initial_cash
    portfolio = {}

    # Lists to store dates and total values for plotting
    dates = []
    total_values = []

    print(f'Initial cash balance: ${initial_cash:.2f}')

    while current_date <= end_date:
        formatted_date = current_date.strftime('%Y-%m-%d')
        print(f'Processing date: {formatted_date}')

        # Generate buy recommendations
        buy_dict = {}
        buy_list_r3, sell_list_r3 = ts.calculate_stock_transactions_R3()
        buy_list_d5, sell_list_d5 = ts.calculate_stock_transactions_D5()
        buy_list_mdu2, sell_list_mdu2 = ts.calculate_stock_transactions_MDU2()
        buy_list_3dhl, sell_list_3dhl = ts.calculate_stock_transactions_3DHL()
        for buy_list in [buy_list_r3, buy_list_d5, buy_list_mdu2, buy_list_3dhl]:
            for stock_ticker in buy_list:
                buy_dict[stock_ticker] = buy_dict.get(stock_ticker, 0) + 1

        # Sell stocks
        for stock_ticker in list(portfolio.keys()):
            if stock_ticker in sell_list_r3 or stock_ticker in sell_list_d5 or stock_ticker in sell_list_mdu2 or stock_ticker in sell_list_3dhl:
                stock_price = sp.stock_catalog[stock_ticker]['Close'].loc[:current_date].iloc[-1]
                quantity = portfolio[stock_ticker]
                cash += stock_price * quantity
                del portfolio[stock_ticker]
                print(f'Sold {quantity} shares of {stock_ticker} at ${stock_price:.2f} each. Cash balance: ${cash:.2f}')

        # Buy stocks
        for stock_ticker in list(buy_dict.keys()):
            if stock_ticker not in portfolio:
                stock_price = sp.stock_catalog[stock_ticker]['Close'].loc[:current_date].iloc[-1]
                if stock_price <= 0:
                    continue
                max_affordable_shares = int(cash // stock_price)
                if max_affordable_shares > 0:
                    portfolio[stock_ticker] = max_affordable_shares
                    cash -= stock_price * max_affordable_shares
                    print(f'Bought {max_affordable_shares} shares of {stock_ticker} at ${stock_price:.2f} each. Cash balance: ${cash:.2f}')

        # Calculate total portfolio value
        portfolio_value = sum(sp.stock_catalog[ticker]['Close'].loc[:current_date].iloc[-1] * quantity for ticker, quantity in portfolio.items())
        total_value = cash + portfolio_value

        print(f'Date: {formatted_date}, Cash: ${cash:.2f}, Portfolio Value: ${portfolio_value:.2f}, Total Value: ${total_value:.2f}')

        # Append data for plotting
        dates.append(current_date)
        total_values.append(total_value)

        current_date += timedelta(days=1)

    # Plotting the results
    plt.figure(figsize=(10, 5))
    plt.plot(dates, total_values, label='Total Portfolio Value')
    plt.xlabel('Date')
    plt.ylabel('Total Value ($)')
    plt.title('Portfolio Value Over Time')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()