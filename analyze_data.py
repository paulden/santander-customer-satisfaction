import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel

from xgboost import XGBClassifier

from clean_data import *

from time import time


df = pd.read_csv('./data/train.csv')

print(df.shape)

# Clean data
print("Cleaning data...\n")
df = remove_constant_columns(df)
df = remove_duplicate_columns(df)

values_df = pd.DataFrame(df['TARGET'].value_counts())
values_df['Percentage'] = 100 * values_df['TARGET'] / df.shape[0]
print(values_df)


# Split data between train and test
train_df, test_df = train_test_split(df, test_size=0.2)

X_train = train_df.drop(['ID', 'TARGET'], axis=1)
X_test = test_df.drop(['ID', 'TARGET'], axis=1)

y_train = train_df['TARGET']
y_test = test_df['TARGET']


# Feature selection
start_time_fs = time()
extra_trees_classifier = ExtraTreesClassifier(random_state=1)
selector = extra_trees_classifier.fit(X_train, y_train)

features_selector = SelectFromModel(selector, prefit=True)

X_train = features_selector.transform(X_train)
X_test = features_selector.transform(X_test)

print("Feature selection complete: %s features selected in %.2f seconds.\n" % (X_train.shape[1], time() - start_time_fs))


scores = {}

# GradientBoostingClassifier
start_time_gb = time()
gb_classifier = GradientBoostingClassifier(random_state=1)

gb_classifier.fit(X_train, y_train)
gb_predictions = gb_classifier.predict_proba(X_test)[:, 1]

gb_auc_score = roc_auc_score(y_test, gb_predictions)

scores['GradientBoostingClassifier'] = [gb_auc_score, time() - start_time_gb]

# XGBClassifier
start_time_xgb = time()
xgb_classifier = XGBClassifier()

xgb_classifier.fit(X_train, y_train, eval_metric='auc')
xgb_predictions = xgb_classifier.predict_proba(X_test)[:, 1]

xgb_auc_score = roc_auc_score(y_test, xgb_predictions, average='macro')

scores['XGBoost'] = [xgb_auc_score, time() - start_time_xgb]

for model in scores:
    print("%s - AUC score: %.5f - Time to fit and predict: %.2f seconds." % (model, scores[model][0], scores[model][1]))
