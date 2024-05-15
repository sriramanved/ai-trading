from __future__ import annotations

import datetime
import threading
import time

import alpaca_trade_api as tradeapi
import gym
import numpy as np
import pandas as pd
import torch

# from finrl.meta.data_processors.processor_alpaca import AlpacaProcessor
from finrl.meta.paper_trading.common import AgentPPO
from alpacaprocess import AlpacaProcessor


class CustomPaperTradingAlpaca:
    def __init__(
        self,
        ticker_list,
        time_interval,
        drl_lib,
        agent,
        cwd,
        net_dim,
        state_dim,
        action_dim,
        API_KEY,
        API_SECRET,
        API_BASE_URL,
        tech_indicator_list,
        turbulence_thresh=30,
        max_stock=1e2,
        latency=None,
    ):
        # load agent
        self.drl_lib = drl_lib
        if agent == "ppo":
            if drl_lib == "elegantrl":
                agent_class = AgentPPO
                agent = agent_class(net_dim, state_dim, action_dim)
                actor = agent.act
                # load agent
                try:
                    cwd = cwd + "/actor.pth"
                    print(f"| load actor from: {cwd}")
                    actor.load_state_dict(
                        torch.load(cwd, map_location=lambda storage, loc: storage)
                    )
                    self.act = actor
                    self.device = agent.device
                except BaseException:
                    raise ValueError("Fail to load agent!")

            elif drl_lib == "stable_baselines3":
                from stable_baselines3 import PPO

                try:
                    # load agent
                    self.model = PPO.load(cwd)
                    print("Successfully load model", cwd)
                except:
                    raise ValueError("Fail to load agent!")

            else:
                raise ValueError(
                    "The DRL library input is NOT supported yet. Please check your input."
                )

        else:
            raise ValueError("Agent input is NOT supported yet.")

        # connect to Alpaca trading API
        try:
            self.alpaca = tradeapi.REST(API_KEY, API_SECRET, API_BASE_URL, "v2")
        except:
            raise ValueError(
                "Fail to connect Alpaca. Please check account info and internet connection."
            )

        # read trading time interval
        if time_interval == "1D" or time_interval == "1Day" or time_interval == "day":
            self.time_interval = 60 * 60 * 24
        if time_interval == "1Min" or time_interval == "1MIN" or time_interval == "min":
            self.time_interval = 60
        if time_interval == "15Min" or time_interval == "15MIN":
            self.time_interval = 60 * 15

        # read trading settings
        self.tech_indicator_list = tech_indicator_list
        self.turbulence_thresh = turbulence_thresh
        self.max_stock = max_stock

        # initialize account
        self.stocks = np.asarray([0] * len(ticker_list))  # stocks holding
        self.stocks_cd = np.zeros_like(self.stocks)
        self.cash = None  # cash record
        self.stocks_df = pd.DataFrame(
            self.stocks, columns=["stocks"], index=ticker_list
        )
        self.asset_list = []
        self.price = np.asarray([0] * len(ticker_list))
        self.stockUniverse = ticker_list
        self.turbulence_bool = 0
        self.equities = []

    def test_latency(self, test_times=10):
        total_time = 0
        for i in range(0, test_times):
            time0 = time.time()
            self.get_state()
            time1 = time.time()
            temp_time = time1 - time0
            total_time += temp_time
        latency = total_time / test_times
        print("latency for data processing: ", latency)
        return latency

    def run(self):

        # Wait for market to open.
        print("Waiting for market to open...")
        self.awaitMarketOpen()
        print("Market opened.")

        self.trade()
        print("Trading complete.")
        last_equity = float(self.alpaca.get_account().last_equity)
        cur_time = time.time()
        self.equities.append([cur_time, last_equity])
        time.sleep(self.time_interval)

    def awaitMarketOpen(self):
        isOpen = self.alpaca.get_clock().is_open

    def trade(self):
        print("getting state")
        state = self.get_state()
        # print("Current state: ", state)
        # time.sleep(10)

        if self.drl_lib == "elegantrl":
            with torch.no_grad():
                s_tensor = torch.as_tensor((state,), device=self.device)
                a_tensor = self.act(s_tensor)
                action = a_tensor.detach().cpu().numpy()[0]
            action = (action * self.max_stock).astype(int)

        elif self.drl_lib == "stable_baselines3":

            action = self.model.predict(state, deterministic=True)
            print(action)
            action = action[0]
            print("Predicted action: ", action)
            time.sleep(10)

        else:
            raise ValueError(
                "The DRL library input is NOT supported yet. Please check your input."
            )

        self.stocks_cd += 1
        if self.turbulence_bool == 0:
            print("here")
            min_action = 0  # stock_cd
            threads = []
            for index in np.where(action < -min_action)[0]:  # sell_index:
                sell_num_shares = min(self.stocks[index], -action[index])
                qty = abs(int(sell_num_shares))
                respSO = []
                tSubmitOrder = threading.Thread(
                    target=self.submitOrder(
                        qty, self.stockUniverse[index], "sell", respSO
                    )
                )
                tSubmitOrder.start()
                threads.append(tSubmitOrder)  # record thread for joining later
                self.cash = float(self.alpaca.get_account().cash)
                print("cash: ", self.cash)
                self.stocks_cd[index] = 0

            for x in threads:  #  wait for all threads to complete
                x.join()

            threads = []
            for index in np.where(action > min_action)[0]:  # buy_index:
                print("cash: ", self.cash)
                if self.cash < 0:
                    tmp_cash = 0
                else:
                    tmp_cash = self.cash
                buy_num_shares = min(
                    tmp_cash // self.price[index], abs(int(action[index]))
                )
                if buy_num_shares != buy_num_shares:  # if buy_num_change = nan
                    qty = 0  # set to 0 quantity
                else:
                    qty = abs(int(buy_num_shares))
                respSO = []
                tSubmitOrder = threading.Thread(
                    target=self.submitOrder(
                        qty, self.stockUniverse[index], "buy", respSO
                    )
                )
                tSubmitOrder.start()
                threads.append(tSubmitOrder)  # record thread for joining later
                self.cash = float(self.alpaca.get_account().cash)
                self.stocks_cd[index] = 0

            for x in threads:  #  wait for all threads to complete
                x.join()

        else:  # sell all when turbulence
            threads = []
            positions = self.alpaca.list_positions()
            for position in positions:
                if position.side == "long":
                    orderSide = "sell"
                else:
                    orderSide = "buy"
                qty = abs(int(float(position.qty)))
                respSO = []
                tSubmitOrder = threading.Thread(
                    target=self.submitOrder(qty, position.symbol, orderSide, respSO)
                )
                tSubmitOrder.start()
                threads.append(tSubmitOrder)  # record thread for joining later

            for x in threads:  #  wait for all threads to complete
                x.join()

            self.stocks_cd[:] = 0

    def get_state(self):
        alpaca = AlpacaProcessor(api=self.alpaca)
        price, tech, turbulence = alpaca.fetch_latest_data(
            ticker_list=self.stockUniverse,
            time_interval="1D",
            tech_indicator_list=self.tech_indicator_list,
        )
        turbulence_bool = 1 if turbulence >= self.turbulence_thresh else 0

        turbulence = (
            self.sigmoid_sign(turbulence, self.turbulence_thresh) * 2**-5
        ).astype(np.float32)

        positions = self.alpaca.list_positions()
        stocks = [0] * len(self.stockUniverse)
        for position in positions:
            ind = self.stockUniverse.index(position.symbol)
            stocks[ind] = abs(int(float(position.qty)))

        stocks = np.asarray(stocks, dtype=float)
        cash = float(self.alpaca.get_account().cash)
        self.cash = cash
        self.stocks = stocks
        self.turbulence_bool = turbulence_bool
        self.price = price
        print("cash: ", cash)
        amount = np.array(self.cash * (2**-12), dtype=np.float32)
        scale = np.array(2**-6, dtype=np.float32)
        # state is wrong dimension 14 here, 11 when trained
        print("amount: ", self.cash)
        print("price: ", price)
        print("stocks: ", self.stocks)
        print("tech: ", tech)
        state = np.hstack(
            (
                self.cash,
                # turbulence,
                # self.turbulence_bool,
                price,
                self.stocks,
                # self.stocks_cd,
                tech,
            )
        ).astype(np.float32)
        state[np.isnan(state)] = 0.0
        state[np.isinf(state)] = 0.0
        # print(len(self.stockUniverse))
        return state

    def submitOrder(self, qty, stock, side, resp):
        if qty > 0:
            try:
                self.alpaca.submit_order(stock, qty, side, "market", "day")
                print(
                    "Market order of | "
                    + str(qty)
                    + " "
                    + stock
                    + " "
                    + side
                    + " | completed."
                )
                resp.append(True)
            except:
                print(
                    "Order of | "
                    + str(qty)
                    + " "
                    + stock
                    + " "
                    + side
                    + " | did not go through."
                )
                resp.append(False)
        else:
            """
            print(
                "Quantity is 0, order of | "
                + str(qty)
                + " "
                + stock
                + " "
                + side
                + " | not completed."
            )
            """
            print(
                "Quantity is 0, order of | "
                + str(qty)
                + " "
                + stock
                + " "
                + side
                + " | not completed."
            )
            resp.append(True)

    @staticmethod
    def sigmoid_sign(ary, thresh):
        def sigmoid(x):
            return 1 / (1 + np.exp(-x * np.e)) - 0.5

        return sigmoid(ary / thresh) * thresh
