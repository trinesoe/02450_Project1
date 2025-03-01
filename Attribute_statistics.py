
# Load libraries
import importlib_resources
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Load the data
from Data_preprocessing import *


##### Missing values #####
missing_values  = np.isnan(X)
missing_val_sum= np.sum(missing_values)
print("Number of missing values:", missing_val_sum)

##### Summary statistics #####
cases_with_chd = np.count_nonzero(X[:,9] == 1)
percent_with_chd = (cases_with_chd/N)*100
print("Number of cases with chd:", cases_with_chd)
print("Percent with chd:",percent_with_chd)


# Basic statistics calculated for each attribute
mean_x = np.round(np.mean(X, axis=0), 2)
median_x = np.round(np.median(X, axis=0), 2)
std_x = np.round(np.std(X, axis=0, ddof=1), 2)
min_x = np.round(np.min(X, axis=0), 2)
max_x = np.round(np.max(X, axis=0), 2)
range_x = np.round(max_x - min_x, 2)

# Summerizes all of the statistic in a dataframe
summary_table = pd.DataFrame({
    "Mean": mean_x,
    "Median": median_x,
    "Std Dev": std_x,
    "Min": min_x,
    "Max": max_x,
    "Range": range_x
}, index=attributeNames[:10])  

# Print the table
print(summary_table)

# Correlation matrix #
correlation_matrix_df = np.corrcoef(X, rowvar=False)
print(correlation_matrix_df)

# Plot the correlation matrix
sns.heatmap(correlation_matrix_df, cmap="YlGnBu", annot=True, fmt=".2f",xticklabels=attributeNames[:10], yticklabels=attributeNames[:10])
plt.title("Correlation matrix heatmap")
plt.show()