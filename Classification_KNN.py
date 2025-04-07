# KNN Classification with Two-Layer Cross-Validation
# Correct KNN
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from Data_preprocessing import *


# Define attribute names
attribute_names = ['sbp', 'tobacco', 'ldl', 'adiposity', 'typea', 'obesity', 'alcohol', 'age', 'famhist']
class_name = ['chd']

# Extract X and y
X = df[attribute_names].to_numpy(dtype=np.float32)
y = df[class_name].to_numpy(dtype=np.float32).ravel()

# Standardize data
X_standardized = zscore(X, ddof=1)

## Overwrite X_standardized as X
X = X_standardized

# Define cross-validation parameters
min_k = 1  # Minimum number of neighbors
max_k = 30  # Maximum number of neighbors

k_values = list(range(min_k,max_k+1))

K1 = 10  # Outer folds
K2 = 10  # Inner folds


# Initialize KFold for outer CV
CV_outer = KFold(n_splits=K1, shuffle=True, random_state=42)
CV_inner = KFold(n_splits=K2, shuffle=True, random_state=42)
Error_test_outer = np.zeros(K1)
best_k_values = np.zeros(K1, dtype=int)
N = len(y)  # Total number of data points

i_outer = 0

for train_outer_index, test_outer_index in CV_outer.split(X):
    X_train_outer = X[train_outer_index]
    y_train_outer = y[train_outer_index]
    X_test_outer = X[test_outer_index]
    y_test_outer = y[test_outer_index]

    data_outer_test_length = float(len(y_test_outer))

    # Reset validation errors for each outer fold
    validation_errors = np.zeros(len(k_values))

    # Inner loop
    i_inner = 0    
    for train_inner_index, test_inner_index in CV_inner.split(X_train_outer):
        X_train_inner = X_train_outer[train_inner_index]
        y_train_inner = y_train_outer[train_inner_index]
        X_test_inner = X_train_outer[test_inner_index]
        y_test_inner = y_train_outer[test_inner_index]
        
        for k in range(min_k, max_k + 1): #1, len(k_values)
            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(X_train_inner, y_train_inner)
            y_val_pred = knn.predict(X_test_inner)
            validation_errors[k - min_k] += np.sum(y_val_pred != y_test_inner) / len(y_test_inner)
    i_inner +=1

    # Select best k (minimum validation error)
    best_k = np.argmin(validation_errors)
    best_k_values[i_outer] = k_values[best_k]  # Set the actual k value
    
    # Train final model using the best k on the entire outer training set
    best_model = KNeighborsClassifier(n_neighbors=best_k_values[i_outer])
    best_model.fit(X_train_outer, y_train_outer)
    
    # Compute test error
    y_test_pred = best_model.predict(X_test_outer)
    Error_test_outer[i_outer] = np.sum(y_test_pred != y_test_outer) / len(y_test_outer)
    
    print(f"Fold {i_outer + 1}: Best k = {best_k_values[i_outer]}, Error = {Error_test_outer[i_outer]:.4f}")
    
    i_outer += 1

# Create summary table
summary_df = pd.DataFrame({
    'Fold': np.arange(1, K1 + 1),
    'Best k': best_k_values,
    'Error': Error_test_outer 
})

# Compute estimated generalization error
generalization_error_logistic_model = np.sum(np.multiply(Error_test_outer, data_outer_test_length)) * (1/N)
print(summary_df)
print(f"Estimated Generalization Error: {generalization_error_logistic_model:.4f}")
