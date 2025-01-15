import pandas as pd 
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt 
from ydata_profiling import ProfileReport
import math

x_train = pd.read_csv("x_train.csv")
profile = ProfileReport(x_train)
profile

# colonnes numériques seulement
numeric_cols = x_train.select_dtypes(include=[np.number])

# matrice de corrélation
correlation_matrix = numeric_cols.corr()

plt.figure(figsize=(20, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='YlGnBu')
plt.title("Correlation of Features")
plt.show()
