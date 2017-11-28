import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score

from xgboost import XGBClassifier

from clean_data import *
from select_features import *


df = pd.read_csv('./data/train.csv')

# CLEAN DATA
print("Cleaning data...\n")
df = remove_constant_columns(df)
df = remove_duplicate_columns(df)

values_df = pd.DataFrame(df['TARGET'].value_counts())
values_df['Percentage'] = 100 * values_df['TARGET'] / df.shape[0]
print(values_df)

train_df, test_df = train_test_split(df, test_size=0.2)

X_train = train_df.drop(['ID', 'TARGET'], axis=1)
X_test = test_df.drop(['ID', 'TARGET'], axis=1)

y_train = train_df['TARGET']
y_test = test_df['TARGET']

# Feature selection
print("\nSelecting most important features to use using ExtraTreesClassifier")
X_train, X_test = extra_trees(X_train, y_train, X_test)

print(X_train.shape, X_test.shape)

# GradientBoostingClassifier
print("\nClassifying using GradientBoostingClassifier...")
gb_classifier = GradientBoostingClassifier(random_state=1)

gb_classifier.fit(X_train, y_train)
gb_predictions = gb_classifier.predict(X_test)

print("Computing scores...")
gb_accuracy_score = accuracy_score(y_test, gb_predictions)
gb_auc_score = roc_auc_score(y_test, gb_predictions)
print("AUC score: %s\n" % gb_auc_score)

# XGBClassifier
print("Classifying using XGBClassifier...")
xgb_classifier = XGBClassifier()

xgb_classifier.fit(X_train, y_train, eval_metric='auc')
xgb_predictions = xgb_classifier.predict_proba(X_test)[:, 1]

print("Computing scores...")
xgb_auc_score = roc_auc_score(y_test, xgb_predictions, average='macro')
print("AUC score: %s\n" % xgb_auc_score)
