import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split

from clean_data import *

df = pd.read_csv('./data/train.csv')

# CLEAN DATA
df = remove_constant_columns(df)
df = remove_duplicate_columns(df)

train_df, test_df = train_test_split(df, test_size=0.2)
combine = [train_df, test_df]
