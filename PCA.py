
# Load libraries
import importlib_resources
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.linalg import svd 
from Data_preprocessing import *
from scipy.linalg import svd 

# Subtract mean value from data
# We do not use the mean centering:
# Y = X_standardized - np.ones((N, 1)) * X_standardized.mean(axis=0)
# Because we use the standardized X, which already include mean centering

# PCA by computing SVD of Y
U, S, Vh = svd(X_standardized, full_matrices=False)

#Transpose V
V = Vh.T

# Compute variance explained by principal components 
rho = (S * S) / (S * S).sum()

# Plot variance explained
threshold = 0.90

# Plot variance explained
plt.figure()
plt.plot(range(1, len(rho) + 1), rho, "x-")
plt.plot(range(1, len(rho) + 1), np.cumsum(rho), "o-")
plt.plot([1, len(rho)], [threshold, threshold], "k--")
plt.title("Variance explained by principal components")
plt.xlabel("Principal component")
plt.ylabel("Variance explained")
plt.legend(["Individual", "Cumulative", "Threshold"])
plt.grid()
plt.show()


