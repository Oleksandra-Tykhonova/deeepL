import numpy as np
from sklearn.datasets import load_iris


def load_split_standardized(n_train_per_class=35):
    iris = load_iris()
    X, y = iris.data, iris.target

    rng = np.random.default_rng(0)
    train_idx = []
    test_idx = []
    for c in (0, 1, 2):
        class_idx = np.where(y == c)[0]
        shuffled = rng.permutation(class_idx)
        train_idx.append(shuffled[:n_train_per_class])
        test_idx.append(shuffled[n_train_per_class:])

    train_idx = np.concatenate(train_idx)
    test_idx = np.concatenate(test_idx)

    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0, ddof=0)
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    return X_train, y_train, X_test, y_test
