
# Load libraries
import importlib_resources
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.linalg import svd 
from Data_preprocessing import *
from scipy.linalg import svd 

# Assuming X_standardized is a NumPy array
X_standardized_df = pd.DataFrame(X_standardized, columns=attributeNames)  # attribute_names should be your column names

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
plt.plot(range(1, len(rho) + 1), rho, "x-", color='cornflowerblue')
plt.plot(range(1, len(rho) + 1), np.cumsum(rho), "x-", color='mediumvioletred')
plt.plot([1, len(rho)], [threshold, threshold], color='firebrick', linestyle='--')
plt.xticks(range(1, len(rho)+1, 1))
plt.xlim(1, len(rho))
plt.yticks(np.arange(0, max(np.cumsum(rho)) + 0.1, 0.1))  
plt.title("Variance explained by principal components")
plt.xlabel("Principal component")
plt.ylabel("Variance explained")
plt.legend(["Individual", "Cumulative", "Threshold"])
plt.grid()
plt.show()


### Data projection of Scatterplot
# Project data onto the principal components
Z = U * S

classLabels = X[:, -1]  # -1 takes the last column
# Then determine which classes are in the data by finding the set of
# unique class labels
classNames = np.unique(classLabels)

classDict = dict(zip(classNames, range(len(classNames))))
C = len(classNames)
y =  np.array([classDict[cl] for cl in classLabels])

# Indices of the principal components to be plotted
i = 0
j = 1

# Change the colors for the PCA plot
colors = ["blue", "orange"]

# Plot PCA of the data
plt.figure()
plt.title("PCA")
for c in range(C):
    # select indices belonging to class c:
    class_mask = y == c
    plt.plot(Z[class_mask, i], Z[class_mask, j], "o", alpha=0.5, color=colors[c])
plt.legend(np.array(["CHD = 0", "CHD = 1"]) )
plt.xlabel("PC{0}".format(i + 1))
plt.ylabel("PC{0}".format(j + 1))
plt.grid()
# Output result to screen
plt.show()


### Interpret PC1, PC2, and PC3 Coefficients
# Define the principal components to plot
pcs = [0, 1, 2]  # PC1, PC2, and PC3 (0-based indexing)
legendStrs = ["PC" + str(e + 1) for e in pcs]
colors = ['royalblue', 'mediumseagreen', 'crimson']  # Colors for each component
bw = 0.2  # Bar width
r = np.arange(1, len(attributeNames) + 1)  # Number of attributes (make sure it's the length of attribute_names)

# Plot bar graphs for each component
plt.figure(figsize=(12, 6))
for i, pc in enumerate(pcs):
    plt.bar(r + i * bw, V[:, pc], color=colors[i], width=bw, label=legendStrs[i], alpha=0.7)

# Fix x-ticks to use all attribute names
plt.xticks(r + bw, attributeNames)  # Adjust if needed
plt.xlabel("Attributes")
plt.ylabel("Component Coefficients")
plt.title("PCA Component Coefficients for PC1, PC2, and PC3")
plt.legend()
plt.grid(True)
plt.show()


### Correlation Circle (PCA1 and PCA2)
# Calculate the loadings for the first two principal components
loadings = V[:, :2] * S[:2]  # Use first 2 components

# Normalize the loadings to lie within the unit circle (correlation)
loadings = loadings / np.sqrt(np.sum(loadings**2, axis=0))

# Plot the correlation circle
plt.figure(figsize=(8, 8))

# Draw the unit circle
circle = plt.Circle((0, 0), 1, color='gray', fill=False)
plt.gca().add_artist(circle)

# Plot the vectors for each attribute
for i, feature in enumerate(df.columns):
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