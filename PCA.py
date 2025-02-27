
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

# Plot principal directions
plt.figure(figsize=(12, 6))
plt.imshow(V, cmap='viridis', aspect='auto')
plt.colorbar()
plt.yticks(range(len(V)), [f'PC{i+1}' for i in range(len(V))])
plt.xticks(range(X_standardized.shape[1]), df.columns, rotation=90)  # Assumes data.columns has attribute names 
plt.xlabel('Attributes')
plt.ylabel('Principal Components')
plt.title('Principal Directions (PCA Components)')
plt.grid(False)
plt.show()


# Plot variance explained
plt.figure()
plt.plot(range(1, len(rho) + 1), rho, color='cornflowerblue', linestyle='-')
plt.plot(range(1, len(rho) + 1), np.cumsum(rho), color='palevioletred', linestyle='-')
plt.plot([1, len(rho)], [threshold, threshold], color='firebrick', linestyle='--')
plt.xticks(range(0, len(rho) + 1, 1))
plt.title("Variance explained by principal components")
plt.xlabel("Principal component")
plt.ylabel("Variance explained")
plt.legend(["Individual", "Cumulative", "Threshold"])
plt.grid()
plt.show()





### Data projection of Scatterplot
# Project data onto the principal components
Z = X_standardized @ V

# Target attribute
y = df['chd'] 

# Scatter plot of the first two principal components
plt.figure(figsize=(10, 6))
scatter = plt.scatter(Z[:, 0], Z[:, 1], c=y, cmap='viridis')  # Color by class label chd
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('Data projected onto first two principal components')
plt.colorbar(scatter, label='CHD (Target)')
plt.grid()
plt.show()






### Correlation Circle (PCA1 and PCA2)
# Get feature names (Assumes data is loaded as a DataFrame)
feature_names = df.columns.drop('chd')  # Drop target variable if it's in the DataFrame

# Calculate the loadings for the first two principal components
# Loadings = V * sqrt(Eigenvalue)
loadings = V[:, :2] * S[:2]  # Use first 2 components

# Normalize the loadings to lie within the unit circle (correlation)
loadings = loadings / np.sqrt(np.sum(loadings**2, axis=0))

# Plot the correlation circle
plt.figure(figsize=(8, 8))

# Draw the unit circle
circle = plt.Circle((0, 0), 1, color='gray', fill=False)
plt.gca().add_artist(circle)

# Plot the vectors for each attribute
for i, feature in enumerate(feature_names):
    plt.arrow(0, 0, loadings[i, 0], loadings[i, 1], 
              color='blue', alpha=0.8, head_width=0.05, head_length=0.05)
    plt.text(loadings[i, 0] * 1.1, loadings[i, 1] * 1.1, 
             feature, color='red', ha='center', va='center')

# Label the axes
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('Correlation Circle (PC1 vs PC2)')
plt.axhline(0, color='grey', linestyle='--')
plt.axvline(0, color='grey', linestyle='--')
plt.grid()
plt.xlim(-1, 1)
plt.ylim(-1, 1)
plt.show()