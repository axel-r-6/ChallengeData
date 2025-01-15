import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

x_train = pd.read_csv("x_train.csv")

# Colonnes catégoriques
categorical_cols = ['venue', 'action', 'trade']

plt.figure(figsize=(15, 10))

for i, col in enumerate(categorical_cols):
    plt.subplot(2, 2, i + 1)
    sns.countplot(data=x_train, x=col, order=x_train[col].value_counts().index)
    plt.title(f'Diagramme à barres pour {col}')

plt.tight_layout()
plt.show()
