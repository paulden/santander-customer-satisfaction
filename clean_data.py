import pandas as pd
import numpy as np


def remove_constant_columns(df):
    constant_columns = []

    for column in df.columns:
        if df[column].std() == 0:
            constant_columns += [column]

    df.drop(constant_columns, axis=1, inplace=True)

    return df


def remove_duplicate_columns(df):
    duplicate_columns = []
    columns_length = len(df.columns)

    for i in range(columns_length - 1):
        original_column_values = df[df.columns[i]].values
        for j in range(i + 1, columns_length):
            column_values_to_compare = df[df.columns[j]].values
            if np.array_equal(original_column_values, column_values_to_compare):
                duplicate_columns += [df.columns[j]]

    df.drop(duplicate_columns, axis=1, inplace=True)

    return df
