# pylint: disable=all

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from metrics import *
from tree.base import DecisionTree
from ensemble.gradientBoosted import GradientBoostedRegressor
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression

np.random.seed(42)

X, y= make_regression(
    n_features=3,
    n_informative=3,
    noise=10,
    tail_strength=10,
    random_state=42,
)

X = pd.DataFrame(X)
y = pd.Series(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

GBR = GradientBoostedRegressor(n_estimators=25, learning_rate=0.1)

GBR.fit(X_train, y_train)

y_pred = GBR.predict(X_test)

rmse_ = rmse(y_test, pd.Series(y_pred))
print(f"RMSE: {rmse_:.2f}")
print(f"MAE: {mae(y_test, pd.Series(y_pred)):.2f}")
