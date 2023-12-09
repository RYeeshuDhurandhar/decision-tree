import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *

np.random.seed(42)

# Imprting dataset and dividing it into features X and output y
from sklearn.datasets import make_classification
X, y = make_classification(n_features=2, n_redundant=0, n_informative=2, random_state=1, n_clusters_per_class=2, class_sep=0.5)

# Visualizing the input data using scatter plot
import matplotlib.pyplot as plt
plt.scatter(X[:, 0], X[:, 1], c=y)
# plt.show()   # Uncomment this to see the plot. Close that plot window to run remaining code.

# Converting X and y to dataset and series to use them easily
X = pd.DataFrame(X)
y = pd.Series(y)

# Defining split ratio and storing length of y
split = 0.7
len_y = len(y)

# Spliting dataset into train and test
X_train, y_train = X.loc[:split*len_y], y.loc[:split*len_y]
X_test, y_test = X.loc[split*len_y+1:].reset_index(drop=True), y.loc[split*len_y+1:].reset_index(drop=True)

# Training and testing for both the criteria 'gini_index' and 'information_gain'
for criteria in ['gini_index', 'information_gain']:
    tree = DecisionTree(criterion=criteria)
    tree.fit(X_train, y_train)
    # tree.plot()               # Uncomment this to see the learnt decision tree              
    y_hat = tree.predict(X_test)
    print('\nCriteria :', criteria)
    print('Accuracy: ', accuracy(y_hat, y_test))
    for cls in y.unique():
        print(f'Precision for class {cls}:', precision(y_hat, y_test, cls))
        print(f'Recall for class {cls}:', recall(y_hat, y_test, cls))

# Implimenting 5 fold cross validation
def cross_validation(X, y, folds=5, depths=[3, 4, 5, 6]):
    assert(len(X) > 0)
    assert(len(X) == len(y))

    max_depth = max(depths)

    # Defining dataframe to store the folds and repective accurcies for different depths
    cross_val_table = pd.DataFrame(index=list(range(1,folds+1)), columns=depths)
    cross_val_table.index.name = "Fold Number" 
    cross_val_table.columns.name = "Depths" 

    # partition_fraction is the fraction of data used for validation
    partition_fraction = int(len(X)//folds)

    for fold in range(folds):
        # Making a series of boolean values representing the partition for train and validation
        folds_bool = pd.Series([False for i in range(len(X))])
        folds_bool.loc[range(fold*partition_fraction, (fold+1)*partition_fraction)] = True

        # Deviding the data for train and test for any specific folds_bool
        X_train, y_train = X[~folds_bool].reset_index(drop=True), y[~folds_bool].reset_index(drop=True)
        X_test, y_test = X[folds_bool].reset_index(drop=True), y[folds_bool].reset_index(drop=True)

        # Creating decision tree 
        tree = DecisionTree(max_depth=max_depth)
        tree.fit(X_train, y_train)

        # testing on validation data for different depths
        for depth in depths:
            y_hat = tree.predict(X_test, max_depth=depth)
            cross_val_table.loc[fold+1][depth] = accuracy(y_hat, y_test)
    
    # Showing the performance measures
    cross_val_table.loc["mean"] = cross_val_table.mean()
    print(cross_val_table)
    print(f'Best Mean Accuracy = {cross_val_table.loc["mean"].max()}')
    print(f'Optimum Depth of the tree = {np.argmax(pd.Series(cross_val_table.loc["mean"])) + min(depths)}')

# Calling the function for cross validation
cross_validation(X, y, folds=3, depths=list(range(3, 5)))
