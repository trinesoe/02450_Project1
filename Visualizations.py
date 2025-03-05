import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import zscore
import seaborn as sns
from Data_preprocessing import *


# Prepare data for plotting by removing binary variables
plot_data = df[["sbp", "tobacco", "ldl", "adiposity", "typea", "obesity", "alcohol", "age"]]
X_plot = plot_data.values
X_plot_standardized = zscore(X_plot, ddof=1)
attributeNames_plot = ["SBP", "Tobacco", "LDL", "Adiposity", "Type A", "Obesity", "Alcohol", "Age"]
M_plot = len(attributeNames_plot)
N_plot = X_plot.shape[0]





### Basic Boxplot ##############################################################
plt.figure(figsize=(12, 6))
plt.boxplot(X_plot_standardized)
plt.xticks(range(1, M_plot+1), attributeNames_plot, rotation=45)
plt.title("Boxplot of Standardized Data")
plt.show(block=False)





### Standardized Boxplot with colors ######################################################
sns.set_style("whitegrid")

# Create a colorful boxplot
plt.figure(figsize=(12, 6))
box = plt.boxplot(X_plot_standardized, patch_artist=True,
                  medianprops={"color": "darkred", "linewidth": 1.3})  # Set the thickness of the median line

# Define a color palette
colors = sns.color_palette("husl", len(box["boxes"]))

# Color each box in the plot
for patch, color in zip(box["boxes"], colors):
    patch.set_facecolor(color)

# Customize ticks and labels
plt.xticks(range(1, M_plot+1), attributeNames_plot, rotation=45, fontsize=12)
plt.yticks(fontsize=12)
plt.title("Standardized Boxplot of Attributes", fontsize=16, fontweight="bold")
plt.xlabel("Attributes", fontsize=14)
plt.ylabel("Standardized Values", fontsize=14)
plt.savefig("plots/standardized_boxplot.png")
plt.show(block=False) 





### Boxplot with colors (not standardized) ######################################################
sns.set_style("whitegrid")

# Create a colorful boxplot
plt.figure(figsize=(12, 6))
box = plt.boxplot(X_plot, patch_artist=True,
                  medianprops={"color": "darkred", "linewidth": 1.3})  # Set the thickness of the median line

# Define a color palette
colors = sns.color_palette("husl", len(box["boxes"]))

# Color each box in the plot
for patch, color in zip(box["boxes"], colors):
    patch.set_facecolor(color)

# Customize ticks and labels
plt.xticks(range(1, M_plot+1), attributeNames_plot, rotation=45, fontsize=12)
plt.yticks(fontsize=12)
plt.title("Boxplot of Attributes", fontsize=16, fontweight="bold")
plt.xlabel("Attributes", fontsize=14)
plt.ylabel("Values", fontsize=14)

# Show the plot
plt.savefig("plots/boxplot.png")
plt.show(block=False)





### Histrograms of attributes ######################################################
plt.figure(figsize=(12, 6))

# Calculate grid dimensions
u = np.floor(np.sqrt(M_plot))
v = np.ceil(float(M_plot) / u)

# Define a color palette for the histograms
colors = sns.color_palette("husl", M_plot)

# Loop through each attribute to create a subplot
for i in range(M_plot):
    plt.subplot(int(u), int(v), i + 1)  # Create subplot in grid
    plt.hist(X_plot[:, i], color=colors[i], bins=20, edgecolor='black', alpha=0.7)  # Create histogram with color
    plt.xlabel(attributeNames_plot[i], fontsize=14)  # Label for the x-axis
    plt.ylim(0, N_plot / 2)  # Set y-axis limits

plt.suptitle("Histograms of Attributes", fontsize=16, fontweight="bold")
plt.tight_layout()
plt.savefig("plots/histograms.png")
plt.show(block=False)






### Histrograms of attributes (using standardized data) ######################################################
plt.figure(figsize=(12, 6))

# Calculate grid dimensions
u = np.floor(np.sqrt(M_plot))
v = np.ceil(float(M_plot) / u)

# Define a color palette for the histograms
colors = sns.color_palette("husl", M_plot)

# Loop through each attribute to create a subplot
for i in range(M_plot):
    plt.subplot(int(u), int(v), i + 1)  # Create subplot in grid
    plt.hist(X_plot_standardized[:, i], color=colors[i], bins=20, edgecolor='black', alpha=0.7)  # Create histogram with color
    plt.xlabel(attributeNames_plot[i], fontsize=14)  # Label for the x-axis
    plt.ylim(0, N_plot / 2)  # Set y-axis limits

plt.suptitle("Histograms of Attributes (standardized)", fontsize=16, fontweight="bold")

# Adjust layout to avoid overlap
plt.tight_layout()
#plt.savefig("plots/histograms.png")
plt.show(block=False)






### Boxplot of attributes grouped by CHD presence (not standardized) ######################################################

numeric_cols = ["sbp", "tobacco", "ldl", "adiposity", "typea", "obesity", "alcohol", "age"]

# Set Seaborn style
sns.set_style("whitegrid")

# Create figure with subplots (2 rows, 4 columns)
fig, axes = plt.subplots(2, 4, figsize=(16, 8))

# Define color palette (same as previous boxplots)
colors = sns.color_palette("husl", len(numeric_cols))

# Loop through attributes and plot boxplots for CHD=0 and CHD=1 
for i, (ax, col, attr_name, color) in enumerate(zip(axes.flatten(), numeric_cols, attributeNames_plot, colors)):
    sns.boxplot(
        x="chd", 
        y=col, 
        data=df,  
        ax=ax, 
        palette=[colors[5], colors[0]]  
    )
    
    ax.set_title(attr_name, fontsize=14, fontweight="bold")
    ax.set_xlabel("")  # Remove x-axis label
    ax.set_ylabel("Values", fontsize=12)  # Use raw values
    ax.set_xticklabels(["No CHD", "CHD"], fontsize=10)  # Keep labels clear

# Adjust layout and show the plot
plt.suptitle("Boxplots of Attributes Grouped by CHD Presence", fontsize=16, fontweight="bold")
plt.tight_layout()
plt.savefig("plots/boxplots_chd.png")
plt.show(block=False)




#### Barplot of family history ######################################################

# Set Seaborn style
sns.set_style("whitegrid")

# Create temporary copies for plotting
df_plot_hist = df.copy()
df_plot_hist["famhist"] = df_plot_hist["famhist"].replace({0: "Not Present", 1: "Present"})
df_plot_hist["chd"] = df_plot_hist["chd"].replace({0: "No CHD", 1: "CHD"})

# Create figure with two subplots
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# --- First plot: Family history count ---
sns.countplot(x="famhist", data=df_plot_hist, palette=["gray"], ax=axes[0], width=0.4, edgecolor="black", 
              order=["Not Present", "Present"])
axes[0].set_xlabel("Family History of CHD", fontsize=14)
axes[0].set_ylabel("Count", fontsize=14)
axes[0].set_title("Family History Distribution", fontsize=14, fontweight="bold")

# --- Second plot: Family history grouped by CHD ---
sns.countplot(x="famhist", hue="chd", data=df_plot_hist, 
              palette=[colors[0], colors[5]], ax=axes[1], 
              width=0.4, edgecolor="black", 
              order=["Not Present", "Present"])  # Change order here

axes[1].set_xlabel("Family History of CHD", fontsize=14)
axes[1].set_ylabel("Count", fontsize=14)
axes[1].set_title("Family History by CHD Status", fontsize=14, fontweight="bold")
axes[1].legend(title="CHD Status")


# Adjust layout
plt.tight_layout()
plt.savefig("plots/family_history.png")
plt.show()



