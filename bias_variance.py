import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils import resample

np.random.seed(1234)

def get_bias(predicted_values, true_values):
    return np.round(np.mean((predicted_values - true_values) ** 2), 0)

def get_variance(values):
    return np.round(np.var(values), 0)

# Define the data
x = np.linspace(0, 10, 50)
eps = np.random.normal(0, 5, 50)
y = x**2 + 1 + eps

# Define the number of bootstrap samples and depths to consider
n_samples = 500
max_depths = range(1, 10)

# Define arrays to store the bias and variance values
biases = np.zeros(len(max_depths))
variances = np.zeros(len(max_depths))

# Loop over the depths
for i, depth in enumerate(max_depths):
    # Define arrays to store the predicted y values for each bootstrap sample
    y_preds = np.zeros((n_samples, len(x)))
    
    # Loop over the bootstrap samples
    for j in range(n_samples):
        # Create a bootstrap sample of the data
        x_sample, y_sample = resample(x, y)
        
        # Fit a decision tree to the bootstrap sample
        tree = DecisionTreeRegressor(max_depth=depth)
        tree.fit(x_sample.reshape(-1, 1), y_sample)
        
        # Make predictions on the full dataset using the fitted tree
        y_preds[j, :] = tree.predict(x.reshape(-1, 1))
    
    # compute bias as the squared difference between the mean prediction and the true y values
    bias = get_bias(y_preds, y)

    # compute variance as the mean of the squared differences between the predictions and the mean prediction
    variance = get_variance(y_preds)
    
    # Store the bias and variance values
    biases[i] = bias
    variances[i] = variance

# Plot the bias and variance as a function of tree depth
plt.plot(max_depths, biases, label='bias')
plt.plot(max_depths, variances, label='variance')
plt.xlabel('Tree depth')
plt.ylabel('Error')
plt.legend()
plt.show() 
