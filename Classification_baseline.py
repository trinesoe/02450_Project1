# Baseline classification
# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn.dummy import DummyClassifier
from Data_preprocessing import *

# Define attribute names
attribute_names = ['sbp', 'tobacco', 'ldl', 'adiposity', 'typea', 'obesity', 'alcohol', 'age', 'famhist']
class_name = ['chd']

# Extract X and y
X = df[attribute_names].to_numpy(dtype=np.float32)
y = df[class_name].to_numpy(dtype=np.float32).ravel()

# Standardize data
X_standardized = zscore(X, ddof=1)
X = X_standardized

# Define cross-validation parameters
K1 = 10  # Outer fold
CV_outer = model_selection.KFold(n_splits=K1, shuffle=True, random_state=42)

# Initialize arrays
Error_test_outer = np.zeros(K1)

# Outer loop
k_outer = 0
for train_outer_index, test_outer_index in CV_outer.split(X):
    X_train_outer = X[train_outer_index]
    y_train_outer = y[train_outer_index]
    X_test_outer = X[test_outer_index]
    y_test_outer = y[test_outer_index]

    # Find the most frequent class in the training set
    values, counts = np.unique(y_train_outer, return_counts=True)
    most_frequent_class = values[np.argmax(counts)]

    # Predict the most frequent class for all test samples
    # y_test_pred = np.full_like(y_test_outer, fill_value=most_frequent_class)
    y_test_pred = np.array([most_frequent_class] * len(y_test_outer))

    # Compute test error
    Error_test_outer[k_outer] = np.sum(y_test_pred != y_test_outer) / len(y_test_outer)

    print(f"Fold {k_outer + 1}: Most Frequent Class = {int(most_frequent_class)}, E_test = {Error_test_outer[k_outer]:.4f}")

    k_outer += 1

# Store results in DataFrame
df_results = pd.DataFrame({
    "Outer Fold": np.arange(1, K1 + 1),
    "E_test": Error_test_outer
})

# Compute estimated generalization error
generalization_error_baseline = np.mean(Error_test_outer)

print(df_results)
print(f"Estimated Generalization Error (Baseline): {generalization_error_baseline:.4f}")
