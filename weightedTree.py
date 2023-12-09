# pylint: disable=all

import pandas as pd
import numpy as np
import random
import random
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification

np.random.seed(42)

# Importing dataset and dividing it into features X and output y
X, y = make_classification(n_features=2, n_redundant=0, n_informative=2, random_state=1, n_clusters_per_class=2, class_sep=0.5)

df = pd.DataFrame(X)
df[2] = y

# shuffle the DataFrame
df_shuffled = df.sample(frac=1, random_state=2).reset_index(drop=True)

X = df_shuffled[[0,1]]
y = df_shuffled[[2]]
y = y.squeeze()

# Defining split ratio and storing length of y
split = 0.7
len_y = len(y)

# Spliting dataset into train and test
X_train, y_train = X.loc[:split*len_y], y.loc[:split*len_y]
X_test, y_test = X.loc[split*len_y+1:].reset_index(drop=True), y.loc[split*len_y+1:].reset_index(drop=True)

Weights = np.random.uniform(low=0.0, high=1.0, size=len(y_train))

trees = []

# Training and testing for both the criteria 'gini_index' and 'information_gain'
for criteria in ['gini_index', 'information_gain']:
    tree = DecisionTree(criterion=criteria)
    trees.append(tree)
    tree.fit(X_train, y_train, Weights)
    print('\n')
    tree.plot()               # Uncomment this to see the learnt decision tree              
    y_hat = tree.predict(X_test)
    print('\nCriteria :', criteria)
    print('Accuracy: ', accuracy(y_hat, y_test))
    for cls in y.unique():
        print(f'Precision for class {cls}:', precision(y_hat, y_test, cls))
        print(f'Recall for class {cls}:', recall(y_hat, y_test, cls))

# create the decision tree classifier with information gain as criterion
tree = DecisionTreeClassifier(criterion='entropy')

# fit the classifier to the data
tree.fit(X_train, y_train)
y_pred = tree.predict(X_test)
print('\nAccuracy (sk-learn): ', accuracy(y_pred, y_test))

x_min, x_max = X.iloc[:, 0].min() - 0.5, X.iloc[:, 0].max() + 0.5
y_min, y_max = X.iloc[:, 1].min() - 0.5, X.iloc[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
mesh = pd.DataFrame(np.c_[xx.ravel(), yy.ravel()])

# Use your trained decision tree to predict the class for each point in the meshgrid
predictions = trees[1].predict(mesh)

# Reshape the predicted classes to the same shape as the meshgrid
predictions = predictions.values.reshape(xx.shape)

# Plot the decision boundary and the training data
plt.contourf(xx, yy, predictions, alpha=0.6)
plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y, cmap='cool', alpha=0.6)
# plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=['red', 'green', 'blue'], alpha=0.8)
plt.title("Decision Tree Boundary")
plt.xlabel("feature1")
plt.ylabel("feature2")
plt.show()
