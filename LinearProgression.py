import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

x_train = pd.read_csv("X_train.csv")
y_train = pd.read_csv("y_train.csv")
x_test = pd.read_csv("X_test.csv")

print(x_train.dtypes)

# Identifier les colonnes catégorielles et booléennes
categorical_cols = ['venue', 'action', 'side']
boolean_cols = ['trade']

# Préparer les transformations pour les colonnes catégorielles et booléennes
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(), categorical_cols),
        ('bool', 'passthrough', boolean_cols)
    ],
    remainder='passthrough'
)

# Appliquer les transformations
x_train_transformed = preprocessor.fit_transform(x_train.drop(columns=['obs_id']))

# Vérifier les dimensions après transformation
print(f"x_train_transformed shape: {x_train_transformed.shape}")

# Aplatir les séquences de 100 événements en un vecteur de caractéristiques
def flatten_sequences(df):
    return preprocessor.transform(df.drop(columns=['obs_id'])).flatten()

X_train_flat = np.array([
    flatten_sequences(group) for _, group in x_train.groupby('obs_id')
])

# Vérifier les dimensions après aplatissement
print(f"X_train_flat shape: {X_train_flat.shape}")

# Encoder les étiquettes de classe en one-hot
one_hot_encoder = OneHotEncoder(sparse_output=False)
y_train_one_hot = one_hot_encoder.fit_transform(y_train['eqt_code_cat'].values.reshape(-1, 1))

# Diviser les données en ensemble d'entraînement et ensemble de validation
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
    X_train_flat, y_train_one_hot, test_size=0.2, random_state=42
)

# Vérifier les dimensions après division
print(f"X_train_split shape: {X_train_split.shape}")
print(f"X_val_split shape: {X_val_split.shape}")

# Créer un pipeline avec standardisation des données et régression linéaire
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('linear_regression', LinearRegression())
])

pipeline.fit(X_train_split, y_train_split)

y_pred_proba = pipeline.predict(X_val_split)

# Convertir les probabilités prédites en classes
y_pred = np.argmax(y_pred_proba, axis=1)
y_val = np.argmax(y_val_split, axis=1)

accuracy = accuracy_score(y_val, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%") # a peu pres 21% pour la regression lineaire

x_test_transformed = preprocessor.fit_transform(x_test.drop(columns=['obs_id']))

X_test_flat = np.array([
    flatten_sequences(group) for _, group in x_test.groupby('obs_id')
])

y_pred_test_proba = pipeline.predict(X_test_flat)

# Convertir les probabilités prédites en classes
y_pred_test = np.argmax(y_pred_test_proba, axis=1)

# Créer un DataFrame pour les résultats
y_test = pd.DataFrame({
    'obs_id': x_test['obs_id'].unique(),
    'eqt_code_cat': y_pred_test
})


y_test.to_csv('y_test_LR.csv', index=False)

print("Les prédictions pour les données de test ont été sauvegardées dans 'y_test_LR.csv'.")