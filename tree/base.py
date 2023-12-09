# pylint: disable=all

"""
The current code given is for the Assignment 1.
You will be expected to use this to make trees for:
> discrete input, discrete output
> real input, real output
> real input, discrete output
> discrete input, real output
"""
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tree.utils import information_gain, gini_gain

np.random.seed(42)

# class for the tree node 
class Node: 
    def __init__(self, decision_attr = None, value = None, depth = None):

        self.value = value              # value of leaf node  
        self.depth = depth              # depth of current node in tree
        self.decision_attr = decision_attr  # attribute to split on 
        self.child_nodes = {}           # stores child nodes 

        # to identify clssification or regression, one would be None 
        self.prob = None                # for classification 
        self.mean = None                # for regression 


    def traverse_tree(self, X, max_depth = np.inf):
        '''
        Recursive function to traverse and return value at max_depth
        '''
        
        a = self.decision_attr

        # base cases 
        if (a == None): return self.value 
        if (self.depth >= max_depth): return self.value

        # for classification 
        if (self.mean == None):
            # check if already trained 
            if (X[a] in self.child_nodes):
                next_level = self.child_nodes[X[a]]
            else:
                max_prob_child, max_prob = max(self.child_nodes.items(), key=lambda x:x[1].prob)
                next_level = self.child_nodes[max_prob_child]

            return next_level.traverse_tree(X.drop(a), max_depth = max_depth) 


        # for regression
        else:
            cur_node_mean = self.mean   # mean of current node 

            if (X[a] <= cur_node_mean): cn = "low"
            else: cn = "high"
            
            # function call on appropriate child node 
            next_level = self.child_nodes[cn]
            return next_level.traverse_tree(X, max_depth=max_depth)


############################################################################################################

@dataclass 
class DecisionTree: 
    # criterion: Literal["information_gain", "gini_index"]  # criterion won't be used for regression
    # max_depth: int  # The maximum depth the tree can grow to

    def __init__(self, criterion="information_gain", max_depth=10):
        """
        Put all infromation to initialize your tree here.
        Inputs:
        > criterion : {"information_gain", "gini_index"} # criterion won't be used for regression
        > max_depth : The maximum depth the tree can grow to 
        """
        self.root = None                    # root node 
        self.max_depth = max_depth          # max depth tree can grow to, defaul value = 10
        self.task_type = None               # to determine classification or regression 
        
        self.criterion = criterion          # determines the best split 
        self.n_samples = None               # len(X) => number of rows in X, track the no of sampels
        self.cols = None                    # store column names in X 


    def fit(self, X: pd.DataFrame, y: pd.Series, Weights = None) -> None:
        if Weights is None:
            Weights = np.ones(y.size)
        """
        Function to train and construct the decision tree
        Inputs:
        X: pd.DataFrame with rows as samples and columns as features (shape of X is N X P) where N is the number of samples and P is the number of columns.
        y: pd.Series with rows corresponding to output variable (shape of Y is N)
        """

        self.task_type = y.dtype
        self.n_samples = len(X)
        self.cols = y.name

        self.root = self.build_tree(X, y, Weights, None, depth = 0)      ###
        self.root.prob = 1 

    def build_tree(self, X, Y, Weights = None, parent_node = None, depth = 0):

        if Weights is None:
            Weights = np.ones(Y.size)        
        '''
        Recursive function to build tree.
        parent_node: caller of the function 
        depth: current depth
        '''
        if (Y.nunique() == 1):
            lc = Y.values[0]
            return Node(value = lc, depth = depth)

        if (len(X.columns) <= 0 or 
            depth >= self.max_depth):
            
            return Node(value = Y.mode(dropna=True)[0] 
                if str(Y.dtype) == 'category' else Y.mean(), 
                    depth = depth)


        max_ig = - np.inf # highest information gain
        max_ig_mean = None # mean of the feature with highest IG
        
        # determining the best column 
        for c in list(X.columns): 
            col_mean_val = None 
            cur_col_ig = None 
            column_ig = None 

            if (str(Y.dtype) == "category" and self.criterion == "information_gain"):
                column_ig = information_gain(Y, X[c], Weights)
            elif (str(Y.dtype) == "category" and self.criterion == "gini_index"): 
                column_ig = gini_gain(Y, X[c], Weights)
            else: 
                column_ig = information_gain(Y, X[c], Weights) 


            cur_col_ig = column_ig

            # real/continuous values [information gain, mean]
            if (type(column_ig) == tuple):
                cur_col_ig, col_mean_val = column_ig[0], column_ig[1] 

            if (cur_col_ig > max_ig):
                max_ig = cur_col_ig 
                max_ig_mean = col_mean_val
                best_split = c              # feature with highest IG (best split)


        # best column
        node = Node(decision_attr = best_split)
        best_col_data = X[best_split]       
        
        # for discrete
        if (str(best_col_data.dtype) == "category"):
            X = X.drop(best_split, axis=1) # to avoid overfitting

            # group unique values 
            best_split_classes = best_col_data.groupby(best_col_data).count() 
            df_temp = pd.DataFrame({'best_col_data' : best_col_data, 'Weights':Weights})
            for val, count in best_split_classes.items():
                frows = (best_col_data == val) # bool mask to filter rows
                weights_temp = df_temp.loc[df_temp['best_col_data'] == val]['Weights']
                if (count > 0):
                    node.child_nodes[val] = self.build_tree(X[frows], Y[frows], weights_temp, node, depth+1)
                    node.child_nodes[val].prob = len(X[frows])/self.n_samples 

        # for continuous/real
        else:
            # mean of the best_split
            node.mean = max_ig_mean 

            # filtering rows based on max_ig_mean
            l = (best_col_data <= max_ig_mean) 
            h = (best_col_data >= max_ig_mean)
            df_temp = pd.DataFrame({'best_col_data' : best_col_data, 'Weights':Weights})
            weights_l_temp = df_temp.loc[df_temp['best_col_data'] <= max_ig_mean]['Weights']
            weights_h_temp = df_temp.loc[df_temp['best_col_data'] >= max_ig_mean]['Weights']

            # creating child nodes on the current node
            node.child_nodes["low"] = self.build_tree(X[l], Y[l], weights_l_temp, node, depth+1)
            node.child_nodes["high"] = self.build_tree(X[h], Y[h], weights_h_temp, node, depth+1)

        node.value = Y.mode(dropna=True)[0] if (str(Y.dtype) == "category") else Y.mean()
        node.depth = depth

        return node


    def predict(self, X: pd.DataFrame, max_depth = np.inf) -> pd.Series:
        """
        Funtion to run the decision tree on a data point
        Input:
        X: pd.DataFrame with rows as samples and columns as features
        Output:
        y: pd.Series with rows corresponding to output variable. THe output variable in a row is the prediction for sample in corresponding row in X.
        """
        y_pred = [] # predicted values for each row in X 

        for i in X.index: 
            # prediction for each row in X
            y_pred.append(self.root.traverse_tree(X.loc[i], max_depth = max_depth)) 
        
        # return predicted values 
        return pd.Series(y_pred, name = self.cols)
    
    def plot(self, node=None, depth=0):
        """
        Function to plot the tree
        Output Example:
        ?(X1 > 4)
            Y: ?(X2 > 7)
                Y: Class A
                N: Class B
            N: Class C
        Where Y => Yes and N => No
        """

        # base case: reached leaf node 
        if (node == None):
            node = self.root
        if (node.decision_attr == None):
            print("    " * depth + "     val = " + str(node.value) + ", depth = " + str(node.depth))
            return
        
        for cn in node.child_nodes:
            # for classification
            if (node.child_nodes[cn].prob != None): 
                print("    " * depth + "  ?(X" + str(node.decision_attr) + " = " + str(cn) + "):")

            # for regression 
            else:
                if (cn == "low"): 
                    print("    " * depth + "  ?(X" + str(node.decision_attr) + " <= " + str(node.mean) + "):")
                elif (cn == "high"):
                    print("    " * depth + "  ?(X" + str(node.decision_attr) + " > " + str(node.mean) + "):")
            
            self.plot(node.child_nodes[cn], depth + 1)
            
            
