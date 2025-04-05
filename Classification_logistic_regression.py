# Classification logistic regression
# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
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
K1 = 10 # K for outer fold
K2 = 10 # K for inner fold

# Set lambda interval
lambda_interval = np.logspace(-1, 3, 20)  # Smaller range for better precision

# Initialize KFold for outer CV and inner CV
CV_outer = model_selection.KFold(n_splits=K1, shuffle=True, random_state=42)
CV_inner = model_selection.KFold(n_splits=K2, shuffle=True, random_state=42)


# Store lambda and test errors
lambda_optimal = np.zeros(K1)
Error_test_outer = np.zeros(K1)
validation_errors = np.zeros(len(lambda_interval))
summed_eval_inner = np.zeros(len(lambda_interval)) #bruges til at gemme summerne af Eval_M_s_j

# Outer loop
k_outer = 0  

for train_outer_index, test_outer_index in CV_outer.split(X):
    X_train_outer = X[train_outer_index]
    y_train_outer = y[train_outer_index]
    X_test_outer = X[test_outer_index]
    y_test_outer = y[test_outer_index]


    data_outer_test_length = float(len(y_test_outer)) # |D^{par}_i|

    # Inner loop   
    k_inner=0
    for train_inner_index, test_inner_index in CV_inner.split(X_train_outer):
        X_train_inner = X_train_outer[train_inner_index]
        y_train_inner = y_train_outer[train_inner_index]
        X_test_inner = X_train_outer[test_inner_index]
        y_test_inner = y_train_outer[test_inner_index]


        for s in range(0,len(lambda_interval)):
            mdl = LogisticRegression(penalty="l2", C=1 / lambda_interval[s])
            mdl.fit(X_train_inner, y_train_inner)
            y_val_est = mdl.predict(X_test_inner)
            validation_errors = np.sum(y_val_est != y_test_inner) / len(y_test_inner)

            summed_eval_inner[s]+=validation_errors
        k_inner +=1
        


    opt_val_err = np.min(summed_eval_inner)
    best_lambda = np.argmin(summed_eval_inner)
    lambda_optimal[k_outer]=lambda_interval[best_lambda]
    

    # Train final model using the best lambda on the entire outer training set
    best_model = LogisticRegression(penalty="l2", C=1/lambda_optimal[k_outer], solver='liblinear')
    best_model.fit(X_train_outer, y_train_outer)

    # Compute test error
    y_test_pred = best_model.predict(X_test_outer)
    Error_test_outer[k_outer] = np.sum(y_test_pred != y_test_outer) / len(y_test_outer)

    # Debugging output
    print(f"Fold {k_outer + 1}: Lambda* = {lambda_optimal[k_outer]:.6f}, E_test = {Error_test_outer[k_outer]:.4f}")

    # Increment outer loop counter
    k_outer += 1  

# Store results in DataFrame
df_results = pd.DataFrame({
    "Outer Fold": np.arange(1, K1 + 1),
    "Lambda*": lambda_optimal,
    "E_test": Error_test_outer
})

# Compute estimated generalization error
generalization_error_logistic_model = np.sum(np.multiply(Error_test_outer,data_outer_test_length)) * (1/N)
print(df_results)
print(f"Estimated Generalization Error: {generalization_error_logistic_model:.4f}")

