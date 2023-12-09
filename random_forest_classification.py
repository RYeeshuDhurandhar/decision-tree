import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from metrics import *

from tree.randomForest import RandomForestClassifier
from tree.randomForest import RandomForestRegressor


###Write code here

########### RandomForestClassifier ################### 
from sklearn.datasets import make_classification

np.random.seed(42)

print("------------------------")
print("RandomForestClassifier")
print("------------------------")

# Generate a synthetic classification dataset
X, y = make_classification(n_samples=30, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1, random_state=42)
X = pd.DataFrame(X)
y = pd.Series(y, dtype="category")

for criteria in ['information_gain', 'gini_index']:
    Classifier_RF = RandomForestClassifier(3, criterion = criteria) 
    Classifier_RF.fit(X, y) 
    y_hat = Classifier_RF.predict(X)
    Classifier_RF.plot()
    print('Criteria :', criteria)
    print('Accuracy: ', accuracy(y_hat, y))
    for cls in y.unique():
        print('Precision: ', precision(y_hat, y, cls))
        print('Recall: ', recall(y_hat, y, cls))

print("------------------------")


########### RandomForestRegressor ###################

from sklearn.datasets import make_regression

print("------------------------")
print("RandomForestRegressor")
print("------------------------")

X, y= make_regression(n_features=3, n_informative=3, noise=10, tail_strength=10, random_state=42)
X = pd.DataFrame(X)
y = pd.Series(y)

Regressor_RF = RandomForestRegressor(3, criterion = criteria)
Regressor_RF.fit(X, y)
y_hat = Regressor_RF.predict(X)
Regressor_RF.plot()
print('Criteria :', criteria)
print('RMSE: ', rmse(y_hat, y))
print('MAE: ', mae(y_hat, y))
