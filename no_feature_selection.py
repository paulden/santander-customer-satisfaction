import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from time import time


# Load data
df = pd.read_csv('./data/train.csv')
df_test = pd.read_csv('./data/test.csv')

# Clean data
print("\nCleaning data...\n")

constant_columns = []
duplicate_columns = []
columns_length = len(df.columns)

for column in df.columns:
    if df[column].std() == 0:
        constant_columns += [column]

for i in range(columns_length - 1):
    original_column_values = df[df.columns[i]].values
    for j in range(i + 1, columns_length):
        column_values_to_compare = df[df.columns[j]].values
        if np.array_equal(original_column_values, column_values_to_compare):
            duplicate_columns += [df.columns[j]]

columns_to_remove = list(set(duplicate_columns) | set(constant_columns))

df.drop(columns_to_remove, axis=1, inplace=True)
df_test.drop(columns_to_remove, axis=1, inplace=True)

values_df = pd.DataFrame(df['TARGET'].value_counts())
values_df['Percentage'] = 100 * values_df['TARGET'] / df.shape[0]
print(values_df)


# Split data between train and test
train_df, test_df = train_test_split(df, test_size=0.2, random_state=1)

X_train = train_df.drop(['ID', 'TARGET'], axis=1)
X_test = test_df.drop(['ID', 'TARGET'], axis=1)
test = df_test.drop(['ID'], axis=1)

y_train = train_df['TARGET']
y_test = test_df['TARGET']


scores = {}

# GradientBoostingClassifier
start_time_gb = time()
gb_classifier = GradientBoostingClassifier(random_state=1)

gb_classifier.fit(X_train, y_train)
gb_predictions = gb_classifier.predict_proba(X_test)[:, 1]

gb_auc_score = roc_auc_score(y_test, gb_predictions)

scores['GradientBoostingClassifier'] = (gb_auc_score, time() - start_time_gb)

# RandomForestClassifier
start_time_rf = time()
rf_classifier = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=1)

rf_classifier.fit(X_train, y_train)
rf_predictions = rf_classifier.predict_proba(X_test)[:, 1]

rf_auc_score = roc_auc_score(y_test, rf_predictions)

scores['RandomForestClassifier'] = (rf_auc_score, time() - start_time_rf)

# XGBClassifier
start_time_xgb = time()
xgb_classifier = XGBClassifier()

xgb_classifier.fit(X_train, y_train, eval_metric='auc')
xgb_predictions = xgb_classifier.predict_proba(X_test)[:, 1]

xgb_auc_score = roc_auc_score(y_test, xgb_predictions)

scores['XGBoost'] = (xgb_auc_score, time() - start_time_xgb)


# Find best estimator
score_max = 0
best_model = None

for model in scores:
    score = scores[model][0]
    elapsed_time = scores[model][1]

    print("Results for %s:" % model)
    print("AUC score: %.5f - Time to fit and predict: %.2f seconds.\n" % (score, elapsed_time))
    if score > score_max:
        best_model = model
        score_max = score

print("Best model using AUC score: %s (AUC %s)" % (best_model, score_max))

# Create submission with test predictions
if best_model == 'GradientBoostingClassifier':
    target = gb_classifier.predict_proba(test)[:, 1]
elif best_model == 'XGBoost':
    target = rf_classifier.predict_proba(test)[:, 1]
elif best_model == 'RandomForestClassifier':
    target = xgb_classifier.predict_proba(test)[:, 1]

# submission = pd.DataFrame({"ID": df_test['ID'], "TARGET": target})
# submission.to_csv("data/submission.csv", index=False)
