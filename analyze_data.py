import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split

df = pd.read_csv('./data/train.csv')

train_df, test_df = train_test_split(df, test_size=0.2)
