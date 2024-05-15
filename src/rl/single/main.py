from custom_alpaca import CustomPaperTradingAlpaca
import pandas as pd
from stable_baselines3 import PPO
from finrl.config import INDICATORS

ticker_list = (
    pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]
    .loc[:, "Symbol"]
    .tolist()
)

ticker_list = ["AAPL"]

model_type = "ppo"

drl_lib = "stable_baselines3"
cwd = "./" + "trained_models/" + "agent_" + model_type + ".zip"

agent = "ppo"

time_interval = "1Day"  # placeholder not sure if needed
net_dim = 0  # only needed for elegantrl
state_dim = 0  # only needed for elegantrl and rllib
action_dim = 0  # only needed for elegantrl and rllib

API_KEY = "PKCRH84U78I1M7MNUUJA"
API_SECRET = "Nf0k1JRhhHVV9llkcX7OmASG9kslcY1SoUpOoV2n"
API_BASE_URL = "https://paper-api.alpaca.markets"
tech_indicator_list = INDICATORS

paperTrader = CustomPaperTradingAlpaca(
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
)

if __name__ == "__main__":
    print("Start running paper trading")
    paperTrader.run()
    print("Paper trading complete")
