# Load libraries
import importlib_resources
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import zscore

# Load data
url = "https://hastie.su.domains/ElemStatLearn/datasets/SAheart.data"
df = pd.read_csv(url)  


# Remove the first column "row.names"
df = df.drop(columns = "row.names", axis = 1)

# Extract attribute names
attributeNames = df.columns.tolist()

# Convert categorical 'famhist' to numerical values
df["famhist"] = df["famhist"].astype("category").cat.codes 

# Extract feature matrix X
X = df.iloc[:,:].values  # Exclude row names

# Compute dataset properties
N, M = X.shape  # Number of observations and attributes

# Standardize data
X_standardized = zscore(X, ddof=1)