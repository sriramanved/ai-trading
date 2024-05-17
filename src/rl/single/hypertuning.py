import optuna
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise
import numpy as np
from stable_baselines3.common.evaluation import evaluate_policy
import os
import pandas as pd

from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv

# from customenv import StockTradingEnv
# from finrl.env.env_stocktrading import StockTradingEnv
from finrl.agents.stablebaselines3.models import DRLAgent
from stable_baselines3.common.logger import configure
from finrl import config_tickers
from finrl.main import check_and_make_directories
from finrl.config import INDICATORS, TRAINED_MODEL_DIR, RESULTS_DIR
import numpy as np
from stable_baselines3.common.noise import NormalActionNoise


check_and_make_directories([TRAINED_MODEL_DIR])

train = pd.read_csv("train_data_single.csv")

# If you are not using the data generated from part 1 of this tutorial, make sure
# it has the columns and index in the form that could be make into the environment.
# Then you can comment and skip the following two lines.
train = train.set_index(train.columns[0])
train.index.names = [""]

stock_dimension = len(train.tic.unique())
state_space = 1 + 2 * stock_dimension + len(INDICATORS) * stock_dimension
print(len(INDICATORS))
print(state_space)
print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")

buy_cost_list = sell_cost_list = [0.001] * stock_dimension
num_stock_shares = [0] * stock_dimension

env_kwargs = {
    "hmax": 10,
    "initial_amount": 100000,
    "num_stock_shares": num_stock_shares,
    "buy_cost_pct": buy_cost_list,
    "sell_cost_pct": sell_cost_list,
    "state_space": state_space,
    "stock_dim": stock_dimension,
    "tech_indicator_list": INDICATORS,
    "action_space": stock_dimension,
    "reward_scaling": 1e-4,
}


e_train_gym = StockTradingEnv(df=train, **env_kwargs)

def objective(trial):
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-3)
    buffer_size = trial.suggest_int("buffer_size", 50000, 1000000)
    batch_size = trial.suggest_int("batch_size", 64, 256)
    tau = trial.suggest_loguniform("tau", 1e-3, 1e-1)
    action_space = e_train_gym.action_space.shape[-1]
    action_noise = NormalActionNoise(
        mean=np.zeros(action_space), sigma=0.1 * np.ones(action_space)
    )

    model_ddpg = DDPG(
        "MlpPolicy",
        e_train_gym,
        action_noise=action_noise,
        learning_rate=learning_rate,
        buffer_size=buffer_size,
        batch_size=batch_size,
        tau=tau,
        gamma=0.99,
        train_freq=(1, "step"),
        gradient_steps=1,
        verbose=0,
    )
    env_kwargs = {
        "hmax": 100,
        "initial_amount": 100000,
        "num_stock_shares": num_stock_shares,
        "buy_cost_pct": buy_cost_list,
        "sell_cost_pct": sell_cost_list,
        "state_space": state_space,
        "stock_dim": stock_dimension,
        "tech_indicator_list": INDICATORS,
        "action_space": stock_dimension,
        "reward_scaling": 1e-3,
    }
    trade = pd.read_csv("trade_data_single.csv")
    trade = trade.set_index(trade.columns[0])
    trade.index.names = ['']        
    
    e_trade_gym = StockTradingEnv(
        df=trade, turbulence_threshold=70, risk_indicator_col="vix", **env_kwargs
    )

    

    model_ddpg.learn(total_timesteps=100000)
    mean_reward, _ = evaluate_policy(model_ddpg, e_trade_gym, n_eval_episodes=10)
    return mean_reward


study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100)
best_params = study.best_params

print(best_params)
