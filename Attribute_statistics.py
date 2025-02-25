
# Load libraries
import importlib_resources
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Load data
url = "https://hastie.su.domains/ElemStatLearn/datasets/SAheart.data"
df = pd.read_csv(url)  
print(df) 


# Remove the first column "row.names"
df = df.drop(columns = "row.names", axis = 1)
print(df)

# Change the famhist column from absent and present to 0 and 1
df["famhist"] = df["famhist"].replace({"Absent": 0, "Present": 1})

# Extract the attribute names:
attributeNames = np.asarray(df.columns)
print(attributeNames)

# Convert the pandas dataframe (df) into numpy arrays
raw_data = df.values
print(raw_data)

cols = range(0, 10)
X = raw_data[:, cols]
N, M = X.shape
print(X.shape)

##### Missing values #####
#df.isnull().sum()
# Note: Here is no missing values
missing_values  = np.isnan(X)
missing_val_sum= np.sum(missing_values)
print("Number of missing values:", missing_val_sum)

##### Summary statistics #####
cases_with_chd = np.count_nonzero(X[:,9] == 1)
percent_with_chd = (cases_with_chd/N)*100
print("Number of cases with chd:", cases_with_chd)
print("Percent with chd:",percent_with_chd)


#axis = 0 calculates along the column
mean_x = np.round(np.mean(X, axis=0), 2)
median_x = np.round(np.median(X, axis=0), 2)
std_x = np.round(np.std(X, axis=0, ddof=1), 2)
min_x = np.round(np.min(X, axis=0), 2)
max_x = np.round(np.max(X, axis=0), 2)
range_x = np.round(max_x - min_x, 2)

# Print results
print("Mean:", mean_x)
print("Median:", median_x)
print("Standard Deviation:", std_x)
print("Min:", min_x)
print("Max:", max_x)
print("Range:", range_x)


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

# Plot the correlation matrix #
# YlGnBu
# coolwarm
sns.heatmap(correlation_matrix_df, cmap="YlGnBu", annot=True, fmt=".2f",xticklabels=attributeNames[:10], yticklabels=attributeNames[:10])
plt.title("Correlation Matrix Heatmap")
plt.show()