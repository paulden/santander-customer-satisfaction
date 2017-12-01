import numpy as np
import pandas as pd

from sklearn.ensemble import ExtraTreesClassifier
from xgboost import XGBClassifier

from sklearn.feature_selection import SelectFromModel

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

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


# Feature selection
start_time_fs = time()
clf = ExtraTreesClassifier(random_state=1)
selector = clf.fit(X_train, y_train)

features_selector = SelectFromModel(selector, prefit=True)

X_train = features_selector.transform(X_train)
X_test = features_selector.transform(X_test)
test = features_selector.transform(test)

print("\nFeature selection complete: %s features selected in %.2f seconds.\n" % (X_train.shape[1], time() - start_time_fs))


# XGBClassifier
start_time = time()
classifier = XGBClassifier()

classifier.fit(X_train, y_train, eval_metric='auc')
predictions = classifier.predict_proba(X_test)[:, 1]

# Measure mean cross-validation with 10-folds
scores = cross_val_score(classifier, X_train, y_train, cv=10, scoring='roc_auc')
print("Mean AUC score with 10-folds cross-validation: %.5f" % scores.mean())

# Use Grid Search to optimize parameters
parameters = [{'n_estimators': [20, 50, 100, 200],
               'max_depth': [1, 2, 5],
               'learning_rate': [0.1, 0.5, 0.01]}]
grid_search = GridSearchCV(classifier, param_grid=parameters, scoring='roc_auc', cv=5)
grid_search.fit(X_train, y_train)
best_score = grid_search.best_score_
best_params = grid_search.best_params_

print("Best score found: %.5f" % best_score)
print(best_params)
