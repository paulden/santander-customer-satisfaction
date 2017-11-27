import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score as auc

from clean_data import *

df = pd.read_csv('./data/train.csv')

# CLEAN DATA
print("Cleaning data...")
df = remove_constant_columns(df)
df = remove_duplicate_columns(df)

train_df, test_df = train_test_split(df, test_size=0.2)

X_train = train_df.drop(['ID', 'TARGET'], axis=1)
X_test = test_df.drop(['ID', 'TARGET'], axis=1)

y_train = train_df['TARGET']
y_test = test_df['TARGET']

print("Classifying using GradientBoostingClassifier...")
gb_classifier = GradientBoostingClassifier(random_state=1)

gb_classifier.fit(X_train, y_train)
score = gb_classifier.score(X_test, y_test)

print("Score from classifier: %s" % score)

print("Computing AUC...")
auc_score = auc(y_test, gb_classifier.predict(X_test))

print("AUC score: %s" % auc_score)
