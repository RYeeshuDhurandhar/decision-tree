import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from metrics import *

from ml3 import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

np.random.seed(42)

########### AdaBoostClassifier on Real Input and Discrete Output ###################

N = 30
P = 2
NUM_OP_CLASSES = 2
n_estimators = 3
X = pd.DataFrame(np.abs(np.random.randn(N, P)))
y = pd.Series(np.random.randint(NUM_OP_CLASSES, size=N), dtype="category")
y = y.cat.rename_categories([-1, 1])  # Changing 0 to -1 for adaboost

criteria = 'information_gain'
tree = DecisionTreeClassifier
Classifier_AB = AdaBoostClassifier(base_estimator=tree, n_estimators=n_estimators)
Classifier_AB.fit(X, y)
y_hat = Classifier_AB.predict(X)
[fig1, fig2] = Classifier_AB.plot(X, y, "RIDO")
print('Criteria :', criteria)
print('Accuracy: ', accuracy(y_hat, y))
for cls in y.unique():
    print('Precision: ', precision(y_hat, y, cls))
    print('Recall: ', recall(y_hat, y, cls))



########### AdaBoostClassifier on Classification dataset ###################

from sklearn.datasets import make_classification

# np.random.seed(42)

# # Generate a synthetic classification dataset
# X, y = make_classification(n_samples=30, n_features=2, n_informative=2,
#                            n_redundant=0, n_clusters_per_class=1, random_state=42)
# X = pd.DataFrame(X)
# y = pd.Series(y, dtype="category")
# y = y.cat.rename_categories([-1, 1])  # Changing 0 to -1 for AdaBoost

# criteria = 'information_gain' 

# # Create an AdaBoostClassifier with a decision tree as the base estimator
# n_estimators = 3
# ada_boost = AdaBoostClassifier(base_estimator=DecisionTreeClassifier, n_estimators=n_estimators)

# # Fit the model to the data
# ada_boost.fit(X, y)

# # Plot the individual and common decision surfaces of estimators
# y_hat = ada_boost.predict(X)
# [fig1, fig2] = ada_boost.plot(X, y, "classification")


# print('Criteria :', criteria)
# print('Accuracy: ', accuracy(y_hat, y))
# for cls in y.unique():
#     print('Precision: ', precision(y_hat, y, cls))
#     print('Recall: ', recall(y_hat, y, cls))
