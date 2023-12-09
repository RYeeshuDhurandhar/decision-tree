# pylint: disable=all

# Note: All the comments are from assignment 1, the weights are implemented in the code here

import numpy as np
import pandas as pd

def entropy(Y: pd.Series, Weights = None) -> float:
    """
    Function to calculate the entropy for discrete output
    Inputs:
    > Y: pd.Series of Labels
    Outpus:
    > Returns the entropy as a float
    """
    if Weights is None:
        Weights = np.ones(Y.size)
    assert(Y.size > 0)
    assert(Weights.size > 0)
    assert len(Y) == len(Weights)
    df = pd.DataFrame({'Y': Y, 'Weights': Weights})
    en = 0

    # Making groups of labels in Y and calculating entropy
    group_sum = df.groupby('Y').sum()
    weights_sum = df['Weights'].sum()
    for v in list(group_sum.index):
        p = group_sum.loc[v]['Weights']/weights_sum
        if p > 0: en -= p*np.log2(p)            # Entropy = - summation(p*np.log2(p) )
    return en


def calWeightedVar(Y, Weights):
    average = np.average(Y, weights=Weights)
    var = np.average((Y-average)**2, weights=Weights)
    return var
    
def information_gain(Y, attr, Weights = None):
    """
    Function to calculate the information gain

    Inputs:
    > Y: pd.Series of Labels
    > attr: pd.Series of attribute at which the gain should be calculated
    Outputs:
    > Return the information gain as a float
    """
    if Weights is None:
        Weights = np.ones(Y.size)

    assert(Y.size == attr.size)
    assert(Y.size == Weights.size)
    assert(Y.size > 0)

    df = pd.DataFrame({'Y': Y, 'Weights': Weights, 'attr':attr})
    weights_sum = df['Weights'].sum()

    # For DFDO
    if str(attr.dtype) == 'category' and str(Y.dtype) == 'category':
        unique_values = attr.unique()

        # Gain = E(Y) - summation(|Sv|*E(Sv) / |S|)
        gain = entropy(Y, Weights)                   
        for v in unique_values:
            gain -= df.loc[attr == v]['Weights'].sum() * entropy(df.loc[attr == v]['Y'], df.loc[attr == v]['Weights']) / weights_sum
    
    # For RFDO
    elif str(attr.dtype) == 'float64' and str(Y.dtype) == 'category':
        best_split_val = None
        best_gain = -np.inf
        attr_sorted = attr.sort_values()
        df_sorted = df.sort_values(by = 'attr')
        low = attr_sorted.index[0]
        # finding best split and best gain by spliting with mid values of each consequtive values in attr_sorted
        for high in attr_sorted.index[1:]:
            mid = (attr_sorted[low] + attr_sorted[high])/2

            # Gain = E(Y) - summation(|Sv|*E(Sv) / |S|)
            # Here, v has only two values for '<= mid' and '> mid'
      
            gain_temp = entropy(Y, Weights) - df_sorted.loc[attr <= mid]['Weights'].sum() *entropy(df_sorted.loc[attr <= mid]['Y'], df_sorted.loc[attr <= mid]['Weights'])/weights_sum - df_sorted.loc[attr > mid]['Weights'].sum() *entropy(df_sorted.loc[attr > mid]['Y'], df_sorted.loc[attr > mid]['Weights'])/weights_sum

            # Storing the best_gain and best_split_val
            if gain_temp > best_gain:
                best_gain = gain_temp
                best_split_val = mid
            
            low = high
        gain = (best_gain, best_split_val)


    # for DFRO
    elif str(attr.dtype) == "category" and str(Y.dtype) == 'float64':
        unique_values = list(attr.unique())

        # Gain = Var(Y) - summation(|Sv|*Var(Sv) / |S|) = reduction in variance
        gain = calWeightedVar(Y, Weights)   # initialized with weighted varience

        for v in unique_values:
            gain -= df.loc[attr == v]['Weights'].sum() * calWeightedVar(df.loc[attr == v]['Y'], df.loc[attr == v]['Weights']) / weights_sum

    # for RFRO
    else:
        best_split_val = None
        best_gain = -np.inf
        attr_sorted = attr.sort_values()

        df_sorted = df.sort_values(by = 'attr')

        low = attr_sorted.index[0]

        for high in attr_sorted.index[1:]:
            mid = (attr_sorted[low] + attr_sorted[high])/2

            # Gain = Var(Y) - summation(|Sv|*Var(Sv) / |S|) = reduction in variance
            # Here, v has only two values for '<= mid' and '> mid'
            gain_temp = calWeightedVar(df_sorted['Y'], df_sorted['Weights']) - df_sorted.loc[attr <= mid]['Weights'].sum() * calWeightedVar(df_sorted.loc[attr <= mid]['Y'], df_sorted.loc[attr <= mid]['Weights'])/weights_sum - df_sorted.loc[attr > mid]['Weights'].sum() * calWeightedVar(df_sorted.loc[attr > mid]['Y'], df_sorted.loc[attr > mid]['Weights'])/weights_sum

            # Storing the best_gain and best_split_val
            if gain_temp > best_gain:
                best_gain = gain_temp
                best_split_val = mid
            
            low = high
        gain = (best_gain, best_split_val)

    return gain


def gini_index(Y: pd.Series, Weights = None) -> float:
    """
    Function to calculate the gini index

    Inputs:
    > Y: pd.Series of Labels
    Outpus:
    > Returns the gini index as a float
    """
    if Weights is None:
        Weights = np.ones(Y.size)
    assert(Y.size > 0)
    assert(Weights.size > 0)
    assert len(Y) == len(Weights)
    df = pd.DataFrame({'Y': Y, 'Weights': Weights})
    gini = 1

    # Gini index = 1 - summation(p**2), where p is the weighted probability of a class
    group_sum = df.groupby('Y').sum()
    weights_sum = df['Weights'].sum()

    for v in list(group_sum.index):
        gini -= (group_sum.loc[v]['Weights']/weights_sum)**2

    return gini


def gini_gain(Y, attr, Weights = None):
    # For DFDO
    gini_gain = 0
    if Weights is None:
        Weights = np.ones(Y.size)
    df = pd.DataFrame({'Y': Y, 'Weights': Weights, 'attr':attr})
    weights_sum = df['Weights'].sum()

    if str(attr.dtype) == 'category'and str(Y.dtype) == 'category':       # 
        unique_values = list(attr.unique())

        # gini_gain = G(Y) - summation(|Sv|*G(Sv) / |S|)
        # here, G = weighted gini index
        # |Sv|, |S| = weighted sum of Sv and S
        gini_gain = gini_index(Y, Weights)
        for v in unique_values:
            gini_gain -= df.loc[attr == v]['Weights'].sum() * gini_index(df.loc[attr == v]['Y'], df.loc[attr == v]['Weights']) / weights_sum

    # For RFDO
    elif str(attr.dtype) == 'float64' and str(Y.dtype) == 'category':
        best_split_val = None
        best_gain = -np.inf
        attr_sorted = attr.sort_values()    #

        df_sorted = df.sort_values(by = 'attr')

        low = attr_sorted.index[0]  #

        for high in attr_sorted.index[1:]:
            mid = (attr_sorted[low] + attr_sorted[high])/2  #

            # Gain = G(Y) - summation(|Sv|*G(Sv) / |S|)
            # Here, v has only two values for '<= mid' and '> mid'
            gain_temp = gini_index(Y, Weights) - df_sorted.loc[attr <= mid]['Weights'].sum() *gini_index(df_sorted.loc[attr <= mid]['Y'], df_sorted.loc[attr <= mid]['Weights'])/weights_sum - df_sorted.loc[attr > mid]['Weights'].sum() *gini_index(df_sorted.loc[attr > mid]['Y'], df_sorted.loc[attr > mid]['Weights'])/weights_sum

            # Storing the best_gain and best_split_val
            if gain_temp > best_gain:
                best_gain = gain_temp
                best_split_val = mid
            
            low = high
        gini_gain = (best_gain, best_split_val)
    return gini_gain
