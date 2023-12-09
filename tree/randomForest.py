import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
from matplotlib.colors import ListedColormap
import os

class RandomForestClassifier():
    def __init__(self, n_estimators=100, criterion='gini_index', max_depth=None):
        '''
        :param estimators: DecisionTree
        :param n_estimators: The number of trees in the forest.
        :param criterion: The function to measure the quality of a split.
        :param max_depth: The maximum depth of the tree.
        '''
        dic = { # dictionary to map the criterion to the sklearn criterion 
            'gini_index' : 'gini',
            'information_gain' : 'entropy' 
        }
        self.estimators = [DecisionTreeClassifier(criterion=dic[criterion], max_depth=max_depth) for _ in range(n_estimators)] 
        self.n_estimators = n_estimators    # number of trees
        self.criterion = criterion          # criterion for the tree
        self.max_depth = max_depth          # max depth of the tree 
        self.data = []                      # storing the data for prediction

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Function to train and construct the RandomForestClassifier
        Inputs:
        X: pd.DataFrame with rows as samples and columns as features (shape of X is N X P) where N is the number of samples and P is the number of columns.
        y: pd.Series with rows corresponding to output variable (shape of Y is N)
        """
        self.trainX = X 
        self.trainY = y 

        m = 2   # number of features to be selected for each tree 

        for i, est in enumerate(self.estimators):  # looping for all estimators 
            sample_X = X.sample(n = m, replace = True, axis = 1, random_state = i) # selecting m features
            est.fit(sample_X, y)    # fitting the tree 
            self.data.append((sample_X.to_numpy(), y.to_numpy(), sample_X.columns)) # storing the data for prediction 

    def predict(self, X): 
        """
        Funtion to run the RandomForestClassifier on a data point
        Input:
        X: pd.DataFrame with rows as samples and columns as features
        Output:
        y: pd.Series with rows corresponding to output variable. THe output variable in a row is the prediction for sample in corresponding row in X.
        """
        y_preds = []
        
        for i, est in enumerate(self.estimators):
            y_preds.append(est.predict(X[self.data[i][2]])) 

        y_preds = pd.DataFrame(y_preds).T 

        # majority vote (mode)
        return y_preds.mode(axis = 1)[0] 

       
    def plot(self):
        """
        Function to plot for the RandomForestClassifier.
        It creates three figures

        1. Creates a figure with 1 row and `n_estimators` columns. Each column plots the learnt tree. If using your sklearn, this could a matplotlib figure.
        If using your own implementation, it could simply invole print functionality of the DecisionTree you implemented in assignment 1 and need not be a figure.

        2. Creates a figure showing the decision surface for each estimator

        3. Creates a figure showing the combined decision surface

        """
        X_train = self.trainX.to_numpy() 
        y_train = self.trainY.to_numpy()

        x_min, x_max = X_train[:, 0].min() - 0.5, X_train[:, 0].max() + 0.5
        y_min, y_max = X_train[:, 1].min() - 0.5, X_train[:, 1].max() + 0.5

        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))

        fig1, ax = plt.subplots(figsize = (100, 100))

        # figure 1 
        for i in range(self.n_estimators):
            ax = plt.subplot(1, self.n_estimators, i+1)
            plot_tree(self.estimators[i], ax = ax)

        # figure 2: decision surface for each estimator 
        fig2, axs = plt.subplots(1, self.n_estimators, figsize=(20, 5))

        for i in range(self.n_estimators):
            xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
            y_preds = self.estimators[i].predict(np.c_[xx.ravel(), yy.ravel()]) # predicting the values 
            Z = y_preds.reshape(xx.shape) 
            axs[i].contourf(xx, yy, Z, cmap=plt.cm.RdYlBu)  
            axs[i].scatter(X_train[:, 0], X_train[:, 1], c=y_train, s=20, edgecolor='k')
            axs[i].set_title(f'Estimator {i}')
            

        # figure 3 
        fig3, ax = plt.subplots(figsize=(5, 5))

        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
        Z = np.zeros_like(xx)

        for i in range(self.n_estimators):
            Z += self.estimators[i].predict(np.column_stack((xx.ravel(), yy.ravel()))).reshape(xx.shape)
        Z = Z.reshape(xx.shape)
        ax.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu)
        ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, s=20, edgecolor='k')
        ax.set_title(f'Combined Estimator')
        

        fig1.savefig(os.path.join("figures", "q5_classifier_fig1.png"))
        fig2.savefig(os.path.join("figures", "q5_classifier_fig2.png"))
        fig3.savefig(os.path.join("figures", "q5_classifier_fig3.png"))

        return [fig1, fig2, fig3]



class RandomForestRegressor():
    def __init__(self, n_estimators=100, criterion='variance', max_depth=None):
        '''
        :param n_estimators: The number of trees in the forest.
        :param criterion: The function to measure the quality of a split.
        :param max_depth: The maximum depth of the tree.
        '''

        self.estimators = [DecisionTreeRegressor(max_depth=max_depth) for _ in range(n_estimators)]
        self.n_estimators = n_estimators
        self.criteria = criterion
        self.max_depth = max_depth 
        self.data = []

    def fit(self, X, y):
        """
        Function to train and construct the RandomForestRegressor
        Inputs:
        X: pd.DataFrame with rows as samples and columns as features (shape of X is N X P) where N is the number of samples and P is the number of columns.
        y: pd.Series with rows corresponding to output variable (shape of Y is N)
        """
        self.trainX = X
        self.trainY = y

        m = int(X.shape[1] /2) # number of features to sample => m = P/2

        for est in self.estimators:
            sample_X = X.sample(n = m, replace = True, axis = 1) # sample m features 
            sample_Y = y[sample_X.index]   # sample the corresponding output variable
            est.fit(sample_X,sample_Y)
            self.data.append((sample_X.to_numpy(), sample_Y.to_numpy(), sample_X.columns)) 


    def predict(self, X):
        """
        Funtion to run the RandomForestRegressor on a data point
        Input:
        X: pd.DataFrame with rows as samples and columns as features
        Output:
        y: pd.Series with rows corresponding to output variable. THe output variable in a row is the prediction for sample in corresponding row in X.
        """
        y_preds = [] # list to store the predictions of each estimator 
        
        for i, est in enumerate(self.estimators):
            y_preds.append(est.predict(X[self.data[i][2]])) # predict for each estimator 
        
        y_preds = pd.DataFrame(y_preds).T # convert to a dataframe 
        return y_preds.mean(axis = 1) # return the mean of all the predictions 

        
    def plot(self):
        """
        Function to plot for the RandomForestRegressor.
        It creates three figures

        1. Creates a figure with 1 row and `n_estimators` columns. Each column plots the learnt tree. If using your sklearn, this could a matplotlib figure.
        If using your own implementation, it could simply invole print functionality of the DecisionTree you implemented in assignment 1 and need not be a figure.

        2. Creates a figure showing the decision surface/estimation for each estimator. Similar to slide 9, lecture 4

        3. Creates a figure showing the combined decision surface/prediction

        """
        X_train = self.trainX.to_numpy() 
        y_train = self.trainY.to_numpy()
        x_min, x_max = X_train[:, 0].min() - 0.5, X_train[:, 0].max() + 0.5
        y_min, y_max = X_train[:, 1].min() - 0.5, X_train[:, 1].max() + 0.5

        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))

        # figure 1
        fig1, ax = plt.subplots(figsize=(20, 5))
        
        # plot each tree in the forest 
        for i in range(self.n_estimators):
            ax = plt.subplot(1, self.n_estimators, i+1)
            plot_tree(self.estimators[i], ax = ax)

        # figure 2
        fig2, axs = plt.subplots(1, self.n_estimators, figsize=(20, 5))

        # figure 3 
        fig3, ax = plt.subplots(figsize=(5, 5))

        fig1.savefig(os.path.join("figures", "q5_regressor_fig1.png"))
        fig2.savefig(os.path.join("figures", "q5_regressor_fig2.png"))
        fig3.savefig(os.path.join("figures", "q5_regressor_fig3.png"))

        return [fig1, fig2, fig3]
 
