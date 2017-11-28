import pandas as pd

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel


def extra_trees(X_train, y_train, X_test):
    extra_trees_classifier = ExtraTreesClassifier()
    selector = extra_trees_classifier.fit(X_train, y_train)

    features_selector = SelectFromModel(selector, prefit=True)

    X_train = features_selector.transform(X_train)
    X_test = features_selector.transform(X_test)

    return X_train, X_test