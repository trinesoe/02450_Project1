##################################################
#### Classification + statistical evaluation #####
##################################################

# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from dtuimldmtools import mcnemar
from sklearn.metrics import accuracy_score 
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

# Number of neighbors
min_k = 1  # Minimum number of neighbors
max_k = 35  # Maximum number of neighbors
k_values = list(range(min_k,max_k+1))

# Set lambda interval
lambda_interval = np.logspace(-1, 3, 20)  

# Initialize KFold for outer CV and inner CV
CV_outer = model_selection.KFold(n_splits=K1, shuffle=True, random_state=42)
CV_inner = model_selection.KFold(n_splits=K2, shuffle=True, random_state=42)


# For logistic regression
lambda_optimal = np.zeros(K1)
Error_test_outer_log = np.zeros(K1)
val_errors_log = np.zeros(len(lambda_interval))
summed_eval_inner = np.zeros(len(lambda_interval)) 

# For KNN
Error_test_outer_knn = np.zeros(K1)
best_k_values = np.zeros(K1, dtype=int)
N = len(y)  # Total number of data points

# For the baseline model
Error_test_outer_baseline = np.zeros(K1)


# Initialize values for statistical evaluation
yhat_baseline = []
yhat_logreg = []
yhat_knn = []
y_true = []

# Outer loop
k_outer = 0  

for train_outer_index, test_outer_index in CV_outer.split(X):
    X_train_outer = X[train_outer_index]
    y_train_outer = y[train_outer_index]
    X_test_outer = X[test_outer_index]
    y_test_outer = y[test_outer_index]
    

    data_outer_test_length = float(len(y_test_outer)) 
    val_errors_knn = np.zeros(len(k_values))

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
            y_val_pred_log = mdl.predict(X_test_inner)
            val_errors_log = np.sum(y_val_pred_log != y_test_inner) / len(y_test_inner)

            summed_eval_inner[s]+=val_errors_log

        for k in range(min_k, max_k + 1): #1, len(k_values)
            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(X_train_inner, y_train_inner)
            y_val_pred_knn = knn.predict(X_test_inner)
            val_errors_knn[k - min_k] += np.sum(y_val_pred_knn != y_test_inner) / len(y_test_inner)


        k_inner +=1
    
    ############################
    #####  Baseline model ######
    ############################

    # Find the most frequent class in the training set
    values, counts = np.unique(y_train_outer, return_counts=True)
    most_frequent_class = values[np.argmax(counts)]

    # Predict the most frequent class for all test samples
    y_test_pred_base = np.array([most_frequent_class] * len(y_test_outer))

    # Compute test error
    Error_test_outer_baseline[k_outer] = np.sum(y_test_pred_base != y_test_outer) / len(y_test_outer)

    # Save yhat 
    yhat_baseline.append(y_test_pred_base)


    ##################################################
    ##### The best model for logistic regression #####
    ##################################################
    
    # Select the best lambda
    opt_val_err = np.min(summed_eval_inner)
    best_lambda = np.argmin(summed_eval_inner)
    lambda_optimal[k_outer]=lambda_interval[best_lambda]
    
    # Train final model using the best lambda on the entire outer training set
    best_model = LogisticRegression(penalty="l2", C=1/lambda_optimal[k_outer], solver='liblinear')
    best_model.fit(X_train_outer, y_train_outer)

    # Compute test error
    y_test_pred_log = best_model.predict(X_test_outer)
    Error_test_outer_log[k_outer] = np.sum(y_test_pred_log != y_test_outer) / len(y_test_outer)

    # Save yhat 
    yhat_logreg.append(y_test_pred_log)


    ##################################
    ##### The best model for KNN #####
    ##################################

    # Select best k (minimum validation error)
    best_k = np.argmin(val_errors_knn)
    best_k_values[k_outer] = k_values[best_k]  # Set the actual k value
    
    # Train final model using the best k on the entire outer training set
    best_model_knn = KNeighborsClassifier(n_neighbors=best_k_values[k_outer])
    best_model_knn.fit(X_train_outer, y_train_outer)
    
    # Compute test error
    y_test_pred_knn = best_model_knn.predict(X_test_outer)
    Error_test_outer_knn[k_outer] = np.sum(y_test_pred_knn != y_test_outer) / len(y_test_outer)

    # Save yhat 
    yhat_knn.append(y_test_pred_knn)

    # Save ytrue
    y_true.append(y_test_outer)

    # Increment outer loop counter
    k_outer += 1  



###################################
##### Summery of the results ######
###################################

# Store results in DataFrame for logistic
df_results = pd.DataFrame({
    "Outer Fold": np.arange(1, K1 + 1),
    "Lambda*": lambda_optimal,
    "E_test": Error_test_outer_log
})

# Compute estimated generalization error for the logistic regression model
generalization_error_logistic_model = np.sum(np.multiply(Error_test_outer_log,data_outer_test_length)) * (1/N)
print(df_results)
print(f"Estimated Generalization Error: {generalization_error_logistic_model:.4f}")

# Create summary table
summary_df = pd.DataFrame({
    'Fold': np.arange(1, K1 + 1),
    'Best k': best_k_values,
    'Error': Error_test_outer_knn 
})

# Compute estimated generalization error for the KNN model
generalization_error_knn = np.sum(np.multiply(Error_test_outer_knn, data_outer_test_length)) * (1/N)
print(summary_df)
print(f"Estimated Generalization Error: {generalization_error_knn:.4f}")

# Store results in DataFrame
df_results = pd.DataFrame({
    "Outer Fold": np.arange(1, K1 + 1),
    "E_test": Error_test_outer_baseline
})

# Compute estimated generalization error for the baseline model
generalization_error_baseline = np.mean(Error_test_outer_baseline)

print(df_results)
print(f"Estimated Generalization Error (Baseline): {generalization_error_baseline:.4f}")



###########################################
##### Statistical analysis - SETUP I ######
###########################################
# Concatenate all predictions and true values for McNemar's test
yhat_baseline = np.concatenate(yhat_baseline)
yhat_logreg = np.concatenate(yhat_logreg)
yhat_knn = np.concatenate(yhat_knn)
y_true = np.concatenate(y_true)

# Compute accuracy for each model
accuracy_baseline = accuracy_score(y_true, yhat_baseline)
accuracy_logreg = accuracy_score(y_true, yhat_logreg)
accuracy_knn = accuracy_score(y_true, yhat_knn)

print(f"Accuracy of Baseline model: {accuracy_baseline:.4f}")
print(f"Accuracy of Logistic Regression model: {accuracy_logreg:.4f}")
print(f"Accuracy of KNN model: {accuracy_knn:.4f}")

# McNemar's test function for pairwise comparisons
alpha = 0.05


# Create a summary table for McNemar's test results
mcnemar_results = pd.DataFrame({
    'Comparison': [
        'Baseline vs Logistic Regression',
        'Baseline vs KNN',
        'Logistic Regression vs KNN'
    ],
    'Theta Estimate': [
        round(thetahat_BL_LR, 4),
        round(thetahat_BL_KNN, 4),
        round(thetahat_LR_KNN, 4)
    ],
    'Confidence Interval': [
        f"({CI_BL_LR[0]:.4f}, {CI_BL_LR[1]:.4f})",
        f"({CI_BL_KNN[0]:.4f}, {CI_BL_KNN[1]:.4f})",
        f"({CI_LR_KNN[0]:.4f}, {CI_LR_KNN[1]:.4f})"
    ],
    'p-value': [
        f"{p_BL_LR:.4f}",
        f"{p_BL_KNN:.4f}",
        f"{p_LR_KNN:.4f}"
    ]
})

print(mcnemar_results.to_string(index=False))