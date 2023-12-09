# pylint: disable=all

"""
The current code given is for the Assignment 2.
> Classification
> Regression
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from metrics import *

from ensemble.bagging import BaggingClassifier
from tree.base import DecisionTree

from sklearn.tree import DecisionTreeClassifier

N = 50
P = 2
NUM_OP_CLASSES = 2
n_estimators = 7
X = pd.DataFrame(np.abs(np.random.randn(N, P)))
y = pd.Series(np.random.randint(NUM_OP_CLASSES, size=N), dtype="category")

criteria = 'information_gain'
tree = DecisionTreeClassifier
Classifier_B = BaggingClassifier(base_estimator=tree, n_estimators=n_estimators, criterion='gini', n_jobs = -1)
Classifier_B.fit(X, y)
y_hat = Classifier_B.predict(X)
[fig1, fig2] = Classifier_B.plot(X, y, 'plot')

print('Criteria :', criteria)
print('Accuracy: ', accuracy(y_hat, y))
for cls in y.unique():
    print('Precision: ', precision(y_hat, y, cls))
    print('Recall: ', recall(y_hat, y, cls))

print("Plots saved as q4_{}_fig1.png and q4_{}_fig1.png".format('plot', 'plot'))
