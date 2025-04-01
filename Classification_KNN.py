# KNN Classification with Two-Layer Cross-Validation

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from Data_preprocessing import *  # Ensure this imports your dataset

# Define attribute names
attribute_names = ['sbp', 'tobacco', 'ldl', 'adiposity', 'typea', 'obesity', 'alcohol', 'age', 'famhist']
class_name = ['chd']

# Extract X and y
X = df[attribute_names].to_numpy(dtype=np.float32)
y = df[class_name].to_numpy(dtype=np.float32).ravel()

# Define cross-validation parameters
min_k = 1  # Minimum number of neighbors
max_k = 40  # Maximum number of neighbors
K1 = 10  # Outer folds
K2 = 10  # Inner folds

# Initialize KFold for outer CV
CV_outer = KFold(n_splits=K1, shuffle=True, random_state=42)
Error_test_outer = np.zeros(K1)
Best_k_values = np.zeros(K1, dtype=int)
N = len(y)  # Total number of data points

i_outer = 0

for train_outer_index, test_outer_index in CV_outer.split(X):
    X_train_outer, X_test_outer = X[train_outer_index], X[test_outer_index]
    y_train_outer, y_test_outer = y[train_outer_index], y[test_outer_index]
    
    # Inner cross-validation to find the best k
    CV_inner = KFold(n_splits=K2, shuffle=True, random_state=42)
    validation_errors = np.zeros(max_k - min_k + 1)
    
    for train_inner_index, val_inner_index in CV_inner.split(X_train_outer):
        X_train_inner, X_val_inner = X_train_outer[train_inner_index], X_train_outer[val_inner_index]
        y_train_inner, y_val_inner = y_train_outer[train_inner_index], y_train_outer[val_inner_index]
        
        for k in range(min_k, max_k + 1):
            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(X_train_inner, y_train_inner)
            y_val_pred = knn.predict(X_val_inner)
            validation_errors[k - min_k] += np.sum(y_val_pred != y_val_inner) / len(y_val_inner)
    
    # Compute mean validation error for each k
    validation_errors /= K2
    
    # Select best k (minimum validation error)
    best_k = np.argmin(validation_errors) + min_k
    Best_k_values[i_outer] = best_k
    
    # Train final model using the best k on the entire outer training set
    best_model = KNeighborsClassifier(n_neighbors=best_k)
    best_model.fit(X_train_outer, y_train_outer)
    
    # Compute test error
    y_test_pred = best_model.predict(X_test_outer)
    Error_test_outer[i_outer] = np.sum(y_test_pred != y_test_outer) / len(y_test_outer)
    
    print(f"Fold {i_outer + 1}: Best k = {best_k}, Test Error = {Error_test_outer[i_outer]:.4f}")
    
    i_outer += 1

# Compute estimated generalization error
E_gen = np.sum(Error_test_outer * len(y_test_outer)) / N
print(f"Estimated Generalization Error: {E_gen:.4f}")

# Create summary table
summary_df = pd.DataFrame({
    'Fold': np.arange(1, K1 + 1),
    'Best k': Best_k_values,
    'Test Error': Error_test_outer * 100  # Convert to percentage
})

summary_df.loc['Mean'] = ['-', '-', np.mean(Error_test_outer) * 100]
print(summary_df.to_string(index=False))
