import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *
import os
from datetime import datetime
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier

# Create a folder ('stored_plots') to store the plots 
if not os.path.exists("stored_plots"):
    os.makedirs("stored_plots")
stored_plots = os.path.abspath("stored_plots")

np.random.seed(42)

def run_dt_experiment(N = [30, ], M = [5, ], exp="DT", task=["dido", ]): 
    """
    Runs an experiment to measure the time taken for fitting and predicting using decision trees.
    
    Parameters:
    - N (list): a list of integers representing the number of samples used in the experiment.
    - M (list): a list of integers representing the number of features used in the experiment.
    - exp (str): the type of decision tree to use (DT or sklearn).
    - task (list): a list of strings representing the type of task being performed.
    
    Returns:
    - output (np.ndarray): an array containing the results of the experiment.
    """
    # { N, M, time to learn, time to predict }
    output = np.zeros((len(task), len(N), len(M), 4)) 

    # to track progress of experiment
    total_iterations = len(task)*len(N)*len(M)
    current_iteration = 0 

    for t in task:
        for n in N:
            for m in M:
                # generate data 
                X, y = create_data(N = n, M = m, task_type = t) 
                
                # my DT implementation
                if (exp == "DT"): 
                    tree = DecisionTree()
                
                # sklearn DT implementation
                else: 
                    if (t == "dido" or t == "rido"):
                        tree = DecisionTreeClassifier(criterion = "entropy")  
                    elif (t == "diro" or t == "riro"):
                        tree = DecisionTreeRegressor() # default criterion is gini

                # finding time for learning and predicting
                start1 = datetime.now() 
                tree.fit(X, y)
                end1 = datetime.now()
                start2 = datetime.now()
                tree.predict(X)
                end2 = datetime.now()

                avg_time_to_learn = (end1 - start1).total_seconds()
                avg_time_to_predict = (end2 - start2).total_seconds()

                output[task.index(t), N.index(n), M.index(m)] = np.array([n, m, avg_time_to_learn, avg_time_to_predict])

                current_iteration += 1
                print(f'{(current_iteration/total_iterations)*100}% complete.')

    # plotting learning 
    plt.figure()

    if (len(N) > 1 or len(M) > 1):
        # if N is varied
        if (len(N) > 1): 
            for t in task:
                plt.plot(output[task.index(t), :, 0, 0], output[task.index(t), :, 0, 2], label = t)

            plt.title(exp + " : learning plot")
            plt.xlabel("N")
            plt.ylabel("Time (secs)")
        
        # if M is varied
        else:
            for t in task:
                plt.plot(output[task.index(t), 0, :, 1], output[task.index(t), 0, :, 2], label=t)

            plt.title(exp + " : learning plot")
            plt.xlabel("M")
            plt.ylabel("Time (secs)")
    
    plt.legend()
    plt.savefig(os.path.join(stored_plots, "learning.png"))
    plt.show()


    # ploting predicting
    plt.figure()

    if (len(N) > 1 or len(M) > 1):
        if (len(N) > 1): # if N is varied
            for t in task:
                plt.plot(output[task.index(t), :, 0, 0], output[task.index(t), :, 0, 3], label = t)

            plt.title(exp + " : predicting plot")
            plt.xlabel("N")
            plt.ylabel("Time (secs)")
        
        else: # if M is varied
            for t in task:
                plt.plot(output[task.index(t), 0, :, 1], output[task.index(t), 0, :, 3], label=t)

            plt.title(exp + " : predicting plot")
            plt.xlabel("M")
            plt.ylabel("Time (secs)")
    
    plt.legend()
    plt.savefig(os.path.join(stored_plots, "prediction.png"))
    plt.show()
    return output

'''Function to create fake data (take inspiration from usage.py)'''
def create_data(N=30, M=5, task_type = "dido"):
    P=5 # number of features

    if (task_type == "dido"):
        X = pd.DataFrame({i: pd.Series(np.random.randint(P, size=N), dtype="category") for i in range(M)})
        y = pd.Series(np.random.randint(P, size=N), dtype="category")
    elif (task_type == "diro"):
        X = pd.DataFrame({i: pd.Series(np.random.randint(P, size=N), dtype="category") for i in range(M)})
        y = pd.Series(np.random.randn(N))
    elif (task_type == "rido"):
        X = pd.DataFrame(np.random.randn(N, M))
        y = pd.Series(np.random.randint(P, size=N), dtype="category")
    elif (task_type == "riro"):
        X = pd.DataFrame(np.random.randn(N, M))
        y = pd.Series(np.random.randn(N))
    
    return X, y


'''
Run the following experiments one at a time by uncommenting 
'''

'''for varying N'''
# run_dt_experiment(N = list(range(30, 100, 5)), task = ["dido", "diro", "rido", "riro"], exp="DT")
# run_dt_experiment(N = list(range(30, 100, 5)), task = ["dido", "diro", "rido", "riro"], exp="Sklearn")

'''for varying M'''
# run_dt_experiment(M = list(range(5, 40, 2)), task = ["dido", "diro", "rido", "riro"], exp="DT")
# run_dt_experiment(M = list(range(5, 40, 2)), task = ["dido", "diro", "rido", "riro"], exp="Sklearn")
