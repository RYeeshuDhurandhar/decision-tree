import pandas as pd
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *
from sklearn.tree import DecisionTreeRegressor

np.random.seed(42)

# Defining max depth and split ratio
max_depth = 10
split = 0.7

# Loading dataset
auto_eff = pd.read_excel('auto_efficiency_data.xlsx')

# Dividing dataset into featues and output 
X = auto_eff[['displacement', 'horsepower', 'weight', 'acceleration']]
y = auto_eff['mpg']

len_y = len(y)

# Train - test split
X_train, y_train = X.loc[:split*len_y], y.loc[:split*len_y]
X_test, y_test = X.loc[split*len_y+1:].reset_index(drop=True), y.loc[split*len_y+1:].reset_index(drop=True)

# Training the decision tree
tree = DecisionTree(criterion="information_gain", max_depth=max_depth)
tree.fit(X_train, y_train)
# tree.plot()               # Uncomment this to see the learnt decision tree

# Creating dataframe to store the rmse and mae value at different depths
results_table = pd.DataFrame(index=['rmse', 'mae'])
results_table.index.name = "Measures" 
results_table.columns.name = "Depths"

# calculating rmse and mae at different depths and storing it to results_table
for depth in range(1, max_depth+1):
    y_hat = tree.predict(X_test, max_depth=depth)
    results_table.loc['rmse', depth] = rmse(y_hat, y_test)
    results_table.loc['mae', depth] = mae(y_hat, y_test)

# Printing the results and least rmse and mae and corresponding depths
print(results_table)
print(f'\nLeast rmse = {results_table.loc["rmse"].min()}, at Depth = {results_table.loc["rmse"].idxmin()}')
print(f'Least mae = {results_table.loc["mae"].min()}, at Depth = {results_table.loc["mae"].idxmin()}')


# Sci-kit learn Decision Tree Regressor for using criterion='squared_error'
dt = DecisionTreeRegressor(criterion='squared_error', max_depth=max_depth, random_state=0)
dt.fit(X_train, y_train)
y_hat = pd.Series(dt.predict(X_test))

print("\nUsing criterion='squared_error'")
print('Sklearn rmse: ', rmse(y_hat, y_test))
print('Sklearn mae: ', mae(y_hat, y_test))

# Sci-kit learn Decision Tree Regressor for using criterion='absolute_error'
dt = DecisionTreeRegressor(criterion='absolute_error', max_depth=max_depth, random_state=0)
dt.fit(X_train, y_train)
y_hat = pd.Series(dt.predict(X_test))

print("\nUsing criterion='absolute_error'")
print('Sklearn rmse: ', rmse(y_hat, y_test))
print('Sklearn mae: ', mae(y_hat, y_test))
