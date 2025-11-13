import pandas as pd
import sys

from sklearn import tree
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import f1_score, mean_squared_error
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

def classifier_accuracy(u, v):
    u = list(u)
    v = list(v)
    assert len(u) == len(v)
    return sum(x == y for x, y in zip(u, v)) / len(u)


def decision_tree_classifier_overfit_self():
    print(sys._getframe().f_code.co_name, end='. ')
    df = pd.read_csv('../datasets/winequality-red.csv', sep=';')
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X, y)
    y_pred = clf.predict(X)
    print("Accuracy:", classifier_accuracy(y, y_pred))


def random_forest_classifier_depth10_self():
    print(sys._getframe().f_code.co_name, end='. ')
    df = pd.read_csv('../datasets/winequality-red.csv', sep=';')
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    clf = RandomForestClassifier(random_state=42, max_depth=10, n_estimators=100)
    clf.fit(X, y)
    y_pred = clf.predict(X)
    print("Accuracy:", classifier_accuracy(y, y_pred))


def random_forest_classifier_overfit_self():
    print(sys._getframe().f_code.co_name, end='. ')
    df = pd.read_csv('../datasets/winequality-red.csv', sep=';')
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    clf = RandomForestClassifier(random_state=42, max_depth=100, n_estimators=100)
    clf.fit(X, y)
    y_pred = clf.predict(X)
    print("Accuracy:", classifier_accuracy(y, y_pred))


def split_dataset(X, y):
    SPLIT_STEP = 5
    X_train = X.loc[X.index % SPLIT_STEP != 0]
    X_pred = X.loc[X.index % SPLIT_STEP == 0]
    y_train = y.loc[y.index % SPLIT_STEP != 0]
    y_ref = y.loc[y.index % SPLIT_STEP == 0]
    return X_train, y_train, X_pred, y_ref


def read_wine_dataset():
    df = pd.read_csv('../datasets/winequality-red.csv', sep=';')
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    return X, y


def decision_tree_classifier_depth10():
    print(sys._getframe().f_code.co_name, end='. ')
    X_train, y_train, X_pred, y_ref = split_dataset(*read_wine_dataset())
    clf = DecisionTreeClassifier(random_state=42, max_depth=10)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_pred)
    print("Accuracy:", classifier_accuracy(y_ref, y_pred))


def random_forest_classifier_depth10():
    print(sys._getframe().f_code.co_name, end='. ')
    X_train, y_train, X_pred, y_ref = split_dataset(*read_wine_dataset())
    clf = RandomForestClassifier(random_state=42, max_depth=10, n_estimators=100)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_pred)
    print("Accuracy:", classifier_accuracy(y_ref, y_pred))


def random_forest_binary_classification():
    print(sys._getframe().f_code.co_name, end='. ')
    df = pd.read_csv('../datasets/magic04.data')
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    X_train, y_train, X_pred, y_ref = split_dataset(X, y)
    clf = RandomForestClassifier(random_state=42, max_depth=10, n_estimators=100)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_pred)
    print("F1-score:", f1_score(y_ref, y_pred, pos_label='h'))


def decision_tree_regressor_overfit_self():
    print(sys._getframe().f_code.co_name, end='. ')
    df = pd.read_csv('../datasets/Folds5x2_pp.csv')
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    clf = DecisionTreeRegressor(max_depth=1000)
    clf.fit(X, y)
    y_pred = clf.predict(X)
    print("Error:", mean_squared_error(y, y_pred))



def decision_tree_regressor():
    print(sys._getframe().f_code.co_name, end='. ')
    df = pd.read_csv('../datasets/Folds5x2_pp.csv')
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    X_train, y_train, X_pred, y_ref = split_dataset(X, y)
    clf = DecisionTreeRegressor(max_depth=10)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_pred)
    print("Error:", mean_squared_error(y_ref, y_pred))



def random_forest_regressor_overfit_self():
    print(sys._getframe().f_code.co_name, end='. ')
    df = pd.read_csv('../datasets/Folds5x2_pp.csv')
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    clf = RandomForestRegressor(n_estimators=100, max_depth=1000)
    clf.fit(X, y)
    y_pred = clf.predict(X)
    print("Error:", mean_squared_error(y, y_pred))



def random_forest_regressor(max_depth):
    df = pd.read_csv('../datasets/Folds5x2_pp.csv')
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    X_train, y_train, X_pred, y_ref = split_dataset(X, y)
    clf = RandomForestRegressor(max_depth=max_depth, n_estimators=100)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_pred)
    print("Error:", mean_squared_error(y_ref, y_pred))


def random_forest_regressor_depth5():
    print(sys._getframe().f_code.co_name, end='. ')
    random_forest_regressor(max_depth=5)


def random_forest_regressor_depth10():
    print(sys._getframe().f_code.co_name, end='. ')
    random_forest_regressor(max_depth=10)


def run_tests():
    decision_tree_classifier_overfit_self()
    decision_tree_classifier_depth10()

    random_forest_binary_classification()

    random_forest_classifier_overfit_self()
    random_forest_classifier_depth10()
    random_forest_classifier_depth10_self()

    decision_tree_regressor_overfit_self()
    decision_tree_regressor()

    random_forest_regressor_overfit_self()
    random_forest_regressor_depth5()
    random_forest_regressor_depth10()


if __name__ == "__main__":
    run_tests()

