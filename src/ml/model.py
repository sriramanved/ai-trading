"""
DNN for predicting what stock to buy using tensorflow keras

Output 1 if stock is a buy, 0 if stock is a hold, -1 if stock is a sell

Input: stock data, technical indicators, and turbulence index
"""

import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.python.keras.layers import Input, Dense, Dropout
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.callbacks import ModelCheckpoint

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split




