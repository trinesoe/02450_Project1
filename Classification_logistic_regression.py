# Classification logistic regression
# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from Data_preprocessing import *  # Ensure this imports your dataset

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
K = 10  # Both outer and inner folds
lambda_interval = np.logspace(-1, 2, 20)  # Smaller range for better precision

# Initialize KFold for outer CV
CV = model_selection.KFold(n_splits=K, shuffle=True, random_state=42)

# Store lambda* and test errors
lambda_star = np.zeros(K)
Error_test_outer = np.zeros(K)

# Outer loop
k_outer = 0  

for train_outer_index, test_outer_index in CV.split(X):
    X_train_outer = X[train_outer_index]
    y_train_outer = y[train_outer_index]
    X_test_outer = X[test_outer_index]
    y_test_outer = y[test_outer_index]


    data_outer_test_length = float(len(y_test_outer))

    # Inner cross-validation
    CV_inner = model_selection.KFold(n_splits=K, shuffle=True, random_state=42)
    validation_errors = np.zeros(len(lambda_interval))

    for train_inner_index, val_inner_index in CV_inner.split(X_train_outer):
        X_train_inner = X_train_outer[train_inner_index]
        y_train_inner = y_train_outer[train_inner_index]
        X_val_inner = X_train_outer[val_inner_index]
        y_val_inner = y_train_outer[val_inner_index]

        for s in range(0,len(lambda_interval)):
            mdl = LogisticRegression(penalty="l2", C=1 / lambda_interval[s])
            mdl.fit(X_train_inner, y_train_inner)
            y_val_pred = mdl.predict(X_val_inner)
            validation_errors[s] = np.sum(y_val_pred != y_val_inner) / len(y_val_inner)

        #for s, lambda_val in enumerate(lambda_interval):
            #mdl = LogisticRegression(penalty="l2", C=1/lambda_val, solver='liblinear')
            #mdl.fit(X_train_inner, y_train_inner)
            #y_val_pred = mdl.predict(X_val_inner)

            #validation_errors[s] += np.sum(y_val_pred != y_val_inner) / len(y_val_inner)

    # Compute mean validation error for each lambda
    #validation_errors /= K  # Ensure averaging over folds

    # Select best lambda (minimum validation error)
    best_lambda_idx = np.argmin(validation_errors)
    lambda_star[k_outer] = lambda_interval[best_lambda_idx]

    # Train final model using the best lambda on the entire outer training set
    best_model = LogisticRegression(penalty="l2", C=1/lambda_star[k_outer], solver='liblinear')
    best_model.fit(X_train_outer, y_train_outer)

    # Compute test error
    y_test_pred = best_model.predict(X_test_outer)
    Error_test_outer[k_outer] = np.sum(y_test_pred != y_test_outer) / len(y_test_outer)

    # Debugging output
    print(f"Fold {k_outer + 1}: Lambda* = {lambda_star[k_outer]:.6f}, E_test = {Error_test_outer[k_outer]:.4f}")

    # Increment outer loop counter
    k_outer += 1  

# Store results in DataFrame
df_results = pd.DataFrame({
    "Outer Fold": np.arange(1, K + 1),
    "Lambda*": lambda_star,
    "E_test": Error_test_outer
})

# Compute estimated generalization error
generalization_error_logistic_model = np.sum(np.multiply(Error_test_outer,data_outer_test_length)) * (1/N)
print(df_results)
print(f"Estimated Generalization Error: {generalization_error_logistic_model:.4f}")
