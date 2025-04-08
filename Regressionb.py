import numpy as np
import pandas as pd
import matplotlib.pyplot as plt  # Importing pyplot from matplotlib
from sklearn.preprocessing import StandardScaler
from scipy.linalg import svd
from scipy.io import loadmat
import scipy.stats as st
from sklearn import model_selection
from dtuimldmtools import (
    draw_neural_net,
    train_neural_net,
    visualize_decision_boundary,
    rlr_validate
)
import torch
plt.rcParams.update({'font.size': 12})
from Data_preprocessing import *


#Load dataset (no famhist)
# attribute_names = ['sbp','tobacco','chd','adiposity','typea','obesity','alcohol','age', 'famhist']
attribute_names = ['sbp','tobacco','chd','adiposity','typea','obesity','alcohol','age']
class_name      = ['ldl']

# Adjust X and y to keep selected attributes
X = df[attribute_names].values.astype(np.float32)
y = df[class_name].values.astype(np.float32).ravel()

# Compute dataset properties
N, M = X.shape  # Number of observations and attributes

# Standardize data
X = StandardScaler().fit_transform(df[attribute_names])
y = StandardScaler().fit_transform(df[class_name])


# Define model parameters
K = 10
max_iter = 10000
lambdas = np.power(10.,range(-1,3)) #10^-1 til 10^3 før: -5 til 9
best_lambdas =  np.empty((K,1))
loss_fn = torch.nn.MSELoss()
hidden_units_range = range(1,10) # 1 to 9 hidden units

# Initialize lists to store errors
error_ann_all_folds  = []    
error_baseline = []         
error_rlr = np.empty((K,1))
best_lambdas = np.empty((K,1))
min_ann_errors = np.empty(K)   

# Cross-validation loop
CV = model_selection.KFold(K,shuffle=True, random_state=42) # K-fold cross-validation

for fold_idx, (train_index, test_index) in enumerate(CV.split(X,y)): 
    print('\nCrossvalidation fold: {0}/{1}'.format(fold_idx + 1, K))    
    
    X_train = X[train_index,:]
    y_train = y[train_index]
    X_test = X[test_index,:]
    y_test = y[test_index] 


    # Optimize the regularization parameter (lambda) 
    opt_val_err, opt_lambda, mean_w_vs_lambda, train_err_vs_lambda, test_err_vs_lambda = rlr_validate(X_train, y_train, lambdas, 10)
    error_rlr[fold_idx] = min(test_err_vs_lambda)
    best_lambdas[fold_idx] = lambdas[np.argmin(test_err_vs_lambda)]


    # ANN training
    
    # Convert data to PyTorch tensors
    X_train_tensor = torch.Tensor(X_train)
    y_train_tensor = torch.Tensor(y_train).reshape(-1, 1)
    X_test_tensor = torch.Tensor(X_test)
    y_test_tensor = torch.Tensor(y_test).reshape(-1, 1)

    ann_fold_errors  = []
    for n_hidden in hidden_units_range:
        model = lambda: torch.nn.Sequential(
                            torch.nn.Linear(M, n_hidden), #M features to H hiden units
                            # 1st transfer function, either Tanh or ReLU:
                            torch.nn.Tanh(),                            #torch.nn.ReLU(),
                            torch.nn.Linear(n_hidden, 1), # H hidden units to 1 output neuron
                            )

        net, final_loss, learning_curve = train_neural_net(model,
                                                        loss_fn,
                                                        X=X_train_tensor,
                                                        y=y_train_tensor,
                                                        n_replicates=1,
                                                        max_iter=max_iter)
    
        y_pred = net(X_test_tensor) 
        error_rate = loss_fn(y_pred, y_test_tensor).item()
        ann_fold_errors.append( error_rate) # store error rate for current CV fold 

    
    # Store errors
    error_ann_all_folds.append(ann_fold_errors)
    min_ann_errors[fold_idx] = min(ann_fold_errors)
    baseline_prediction = torch.mean(y_train_tensor).repeat(len(y_test_tensor))
    error_baseline.append(loss_fn(baseline_prediction, y_test_tensor).item())

# Summarize results
error_baseline = np.array(error_baseline)
error_rlr = error_rlr.squeeze()
best_h_per_fold = [np.argmin(err) + 1 for err in error_ann_all_folds]



print("\nFold Summary:")
for i in range(K):
    print(f"Fold {i+1}: Best ANN Error = {min_ann_errors[i]:.4f} (h={best_h_per_fold[i]}), "
          f"RLR Error = {error_rlr[i]:.4f}, Lambda = {best_lambdas[i, 0]:.1e}, "
          f"Baseline = {error_baseline[i]:.4f}")




# Statistical evaluation

def conf_int(data):
    return st.t.interval(0.95, len(data)-1, loc=np.mean(data), scale=st.sem(data))

print("\n95% Confidence Intervals:")
print(f"ANN - RLR: {np.round(conf_int(min_ann_errors - error_rlr), 6)}")
print(f"ANN - Baseline: {np.round(conf_int(min_ann_errors - error_baseline), 6)}")
print(f"RLR - Baseline: {np.round(conf_int(error_rlr - error_baseline), 6)}")

print("\nP-values (one-sample t-tests):")
print(f"ANN vs RLR:      {st.ttest_1samp(min_ann_errors - error_rlr, 0).pvalue:.6f}")
print(f"ANN vs Baseline:   {st.ttest_1samp(min_ann_errors - error_baseline, 0).pvalue:.6f}")
print(f"RLR vs Baseline: {st.ttest_1samp(error_rlr - error_baseline, 0).pvalue:.6f}")


