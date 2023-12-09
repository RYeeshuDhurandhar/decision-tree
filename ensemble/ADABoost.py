from sklearn.tree import DecisionTreeClassifier
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils.extmath import weighted_mode
from sklearn import tree as sktree
import os


class AdaBoostClassifier():
    # Optional Arguments: Type of estimator
    def __init__(self, base_estimator=DecisionTreeClassifier, n_estimators=5, max_depth=1, criterion="entropy"):
        '''
        :param base_estimator: The base estimator model instance from which the boosted ensemble is built (e.g., DecisionTree, LinearRegression).
                               If None, then the base estimator is DecisionTreeClassifier(max_depth=1).
                               You can pass the object of the estimator class
        :param n_estimators: The maximum number of estimators at which boosting is terminated. In case of perfect fit, the learning procedure may be stopped early.
        '''
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators

        self.estimator_trees = [] # list of trees
        self.estimator_alphas = [] # list of alphas
        self.max_depth = 1 # max depth of the tree
        self.criterion = "entropy" # criterion for the tree
        

    def fit(self, X, y):
        """
        Function to train and construct the AdaBoostClassifier
        Inputs:
        X: pd.DataFrame with rows as samples and columns as features (shape of X is N X P) where N is the number of samples and P is the number of columns.
        y: pd.Series with rows corresponding to output variable (shape of Y is N)
        """
        # initializing all weights, w_i = 1/n 
        weights = np.ones(len(y))
        weights /= len(y) 

        # looping for all estimators
        for n in (range(self.n_estimators)):
            
            # creating a tree
            tree = self.base_estimator(criterion=self.criterion, max_depth=self.max_depth)

            tree.fit(X, y, sample_weight=weights) # fitting the tree using current weights 

            y_preds = pd.Series(tree.predict(X)) # predicting the output
            wrong_preds = (y_preds != y) # finding the wrong predictions 

            err_m = np.sum(weights[wrong_preds])/np.sum(weights) # calculating weighted error
            alpha_m = 0.5*np.log((1-err_m)/err_m) # calculating alpha

            # updating the weights of the samples
            weights[~wrong_preds] *= np.exp(-alpha_m) # predicted correctly
            weights[wrong_preds] *= np.exp(alpha_m) # predicted incorrectly

            weights = weights/np.sum(weights) # normalizing weights to 1

            # storing the trees and alphas
            self.estimator_trees.append(tree)
            self.estimator_alphas.append(alpha_m)
    

    def predict(self, X):
        """
        Input:  
        X: pd.DataFrame with rows as samples and columns as features
        Output:
        y: pd.Series with rows corresponding to output variable. THe output variable in a row is the prediction for sample in corresponding row in X.
        """
        final_pred = None  # to store final predictions 

        # looping over all trees and alphas 
        for alpha_m, tree in zip(self.estimator_alphas, self.estimator_trees):
            h_m = pd.Series(tree.predict(X))    # prediction of each classifier 
            h_m_alpha = h_m * alpha_m           # weighted predictions of each classifier 

            if final_pred is None: 
                final_pred = h_m_alpha     # for the first prediction
            else:
                final_pred += h_m_alpha       
        
        return final_pred.apply(np.sign) 
    

    def plot(self, X, y, name=""): 
        """
        Function to plot the decision surface for AdaBoostClassifier for each estimator(iteration).
        Creates two figures 
        Figure 1 consists of 1 row and `n_estimators` columns
        The title of each of the estimator should be associated alpha (similar to slide#38 of course lecture on ensemble learning)
        Further, the scatter plot should have the marker size corresponnding to the weight of each point.

        Figure 2 should also create a decision surface by combining the individual estimators

        Reference for decision surface: https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html

        This function should return [fig1, fig2]
        """

        X = X.to_numpy()
        y = y.to_numpy() 

        # figure for individual estimators 
        fig1, axes1 = plt.subplots(1, len(self.estimator_trees), figsize=(5*len(self.estimator_trees), 4)) # figure for individual estimators

        # define bounds of domain 
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1 

        # define the grid 
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02)) 

        # looping over all trees 
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
        fig1.savefig(os.path.join("figures", "q3_{}_fig1.png".format(name)))
        fig2.savefig(os.path.join("figures", "q3_{}_fig2.png".format(name)))

        return fig1, fig2
