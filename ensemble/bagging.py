# pylint: disable=all

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tree.base import DecisionTree
from sklearn import tree as sktree
from sklearn.utils.extmath import weighted_mode
import os
import time
from time import perf_counter
from joblib import Parallel, delayed

class BaggingClassifier():
    def __init__(self, base_estimator=DecisionTree, n_estimators=5, max_depth=50, criterion="information_gain", n_jobs = 1):
        '''
        :param base_estimator: The base estimator model instance from which the bagged ensemble is built (e.g., DecisionTree(), LinearRegression()).
                               You can pass the object of the estimator class
        :param n_estimators: The number of estimators/models in ensemble.
        '''
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.criterion = criterion
        self.n_jobs = n_jobs
        self.estimator_trees = []
        self.datas = []

    # Revised for 3b
    def fit_helper(self, X, y):
        estimator_trees_temp = []
        X_sampled = X.sample(frac=1, axis='rows', replace=True)
        y_sampled = y[X_sampled.index]
        X_sampled = X_sampled.reset_index(drop=True)
        y_sampled = y_sampled.reset_index(drop=True)

        # Learning decision tree estimators on each sampled data
        tree = self.base_estimator(criterion=self.criterion)
        tree.fit(X_sampled, y_sampled)

        # self.trees.append(tree)
        estimator_trees_temp.append(tree)
        estimator_trees_temp.append([X_sampled, y_sampled])
        return estimator_trees_temp

    def fit(self, X, y):
        """
        Function to train and construct the BaggingClassifier
        Inputs:
        X: pd.DataFrame with rows as samples and columns as features (shape of X is N X P) where N is the number of samples and P is the number of columns.
        y: pd.Series with rows corresponding to output variable (shape of Y is N)
        """
        if self.n_jobs == 1:
            t1_start = perf_counter()
            for n in range(self.n_estimators):
                estimator_trees_temp = self.fit_helper(X, y)
                self.estimator_trees.append(estimator_trees_temp[0])
                self.datas.append(estimator_trees_temp[1])
            t1_end = perf_counter()
            print("n_jobs = 1, elapsed time:", t1_end-t1_start, "seconds")
        else:
            t1_start = perf_counter()

            result = Parallel(n_jobs=self.n_jobs, prefer = "threads")(delayed(self.fit_helper)(X, y) for i in range(self.n_estimators))
            
            t1_end = perf_counter()
            print(f"n_jobs = {self.n_jobs}, elapsed time:", t1_end-t1_start, "seconds")

            for res in result:
                self.estimator_trees.append(res[0]) 
                self.datas.append(res[1])
            

    def predict(self, X):
        """
        Funtion to run the BaggingClassifier on a data point
        Input:
        X: pd.DataFrame with rows as samples and columns as features
        Output:
        y: pd.Series with rows corresponding to output variable. THe output variable in a row is the prediction for sample in corresponding row in X.
        """
        y_hat_all_models = None

        for i, tree in enumerate(self.estimator_trees):
            if y_hat_all_models is None:
                y_hat_all_models = pd.Series(tree.predict(X)).to_frame()
            else:
                y_hat_all_models[i] = tree.predict(X)
        return y_hat_all_models.mode(axis=1)[0]
    
    def plot(self, X, y, name=""): 
        """
        Function to plot the decision surface for BaggingClassifier for each estimator(iteration).
        Creates two figures
        Figure 1 consists of 1 row and `n_estimators` columns and should look similar to slide #16 of lecture
        The title of each of the estimator should be iteration number
        Figure 2 should also create a decision surface by combining the individual estimators and should look similar to slide #16 of lecture
        Reference for decision surface: https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html
        This function should return [fig1, fig2]
        """

        X = X.to_numpy()
        y = y.to_numpy() 

        fig1, axes1 = plt.subplots(1, len(self.estimator_trees), figsize=(5*len(self.estimator_trees), 4)) # figure for individual estimators

        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

        # define the grid 
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02)) 

        for i, tree in enumerate(self.estimator_trees): 
            print("-----------------------------")
            print("Decision Tree {}".format(i+1)) # printing the decision tree number 
            print("-----------------------------")

            print(sktree.export_text(tree)) # printing the tree 

            # flatten each grid to a vector 
            r1, r2 = xx.flatten(), yy.flatten()
            r1, r2 = r1.reshape((len(r1), 1)), r2.reshape((len(r2), 1)) 

            # horizontal stack vectors to create x1,x2 input for the model
            grid = np.hstack((r1,r2))
            grid = pd.DataFrame(grid) 

            y_preds = np.array(tree.predict(grid)) # predicting the output
            zz = y_preds.reshape(xx.shape) # reshaping back into grid 

            cs = axes1[i].contourf(xx, yy, zz, alpha=0.4, cmap=plt.cm.RdYlBu)
            fig1.colorbar(cs, ax=axes1[i], shrink=0.9) 

            axes1[i].scatter(X[:, 0], X[:, 1], s=30, c=y, alpha=0.8, cmap=plt.cm.RdYlBu)

            axes1[i].set_title("Decision Surface Tree: " + str(i+1))  
            axes1[i].legend() 
        
        fig1.tight_layout() # to adjust the subplots 

        # figure for the common decision surface
        fig2, axes2 = plt.subplots(1, 1, figsize=(5, 4))  
        
        r1, r2 = xx.flatten(), yy.flatten()
        r1, r2 = r1.reshape((len(r1), 1)), r2.reshape((len(r2), 1))
        grid = np.hstack((r1,r2))
        grid = pd.DataFrame(grid)
        y_hat = np.array(self.predict(grid))
        zz = y_hat.reshape(xx.shape)
        axes2.contourf(xx, yy, zz, alpha=0.4, cmap=plt.cm.RdYlBu)
        axes2.scatter(X[:, 0], X[:, 1], s=30, c=y, alpha=0.8, cmap=plt.cm.RdYlBu)

        axes2.set_xlabel("X1") 
        axes2.set_ylabel("X2")

        axes2.legend(loc="lower right") 
        axes2.set_title("Common Decision Surface") 

        fig2.colorbar(cs, ax=axes2, shrink=0.9) 

        # saving the figures 
        fig1.savefig(os.path.join("figures", "q4_{}_fig1.png".format(name)))
        fig2.savefig(os.path.join("figures", "q4_{}_fig2.png".format(name)))

        return fig1, fig2
