import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.model_selection import train_test_split

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
clf.fit(X_train, y_train)

main_features = clf.feature_importances_
importance = pd.DataFrame(main_features, index=X_train.columns, columns=['importance'])
importance['std'] = np.std([tree.feature_importances_ for tree in clf.estimators_], axis=0)

importance = importance.nlargest(20, 'importance')

print(importance.head())

x = range(importance.shape[0])
y = importance.iloc[:, 0]
yerr = importance.iloc[:, 1]

plt.bar(x, y, yerr=yerr, align='center')
plt.xticks(x, importance.index, rotation=45)
plt.xlabel('Feature label')
plt.ylabel('Importance')
plt.show()
