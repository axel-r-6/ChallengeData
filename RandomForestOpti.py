import pandas as pd
import time
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from scipy.stats import randint, uniform

x_train = pd.read_csv("X_train.csv")
y_train = pd.read_csv("y_train.csv")
x_test = pd.read_csv("X_test.csv")

columns_to_use = ['obs_id', 'venue', 'action', 'trade', 'bid', 'ask', 'price', 'bid_size', 'ask_size', 'flux']
categorical_cols = ['venue', 'action']
log_abs_cols = ['flux']
log_plus_one_cols = ['bid_size', 'ask_size']
pass_trough_cols = ['trade', 'bid', 'ask', 'price']

preprocessor = ColumnTransformer(
    transformers=[
        ('obs_id', 'passthrough', ['obs_id']),
        ('cat', OneHotEncoder(), categorical_cols),
        ('passthrough', 'passthrough', pass_trough_cols),
        ('log1p', FunctionTransformer(np.log1p), log_plus_one_cols),
        ('log', FunctionTransformer(lambda x: np.log(np.abs(x))), log_abs_cols)
    ]
)

# Enlever données avec bid/ask size négatif
data_with_negative_bid_ask_size = x_train.loc[(x_train['bid_size'] < 0) | (x_train['ask_size'] < 0), 'obs_id'].unique()
x_train = x_train.loc[~x_train['obs_id'].isin(data_with_negative_bid_ask_size)]
y_train = y_train.loc[~y_train['obs_id'].isin(data_with_negative_bid_ask_size)]

x_train_transformed = preprocessor.fit_transform(x_train[columns_to_use])

one_hot_enc_cols = []
for col in categorical_cols:
    for i in range(x_train[col].nunique()):
        one_hot_enc_cols.append(f"{col}_{i}")

del x_train

x_train_transformed = pd.DataFrame(
    x_train_transformed,
    columns=['obs_id'] + one_hot_enc_cols + pass_trough_cols + log_plus_one_cols + log_abs_cols
)

# Remplacer log(flux) = -inf (donc flux = 0) par 0
x_train_transformed['flux'] = x_train_transformed['flux'].replace(-np.inf, 0)

features = x_train_transformed.columns[1:]

# Plus rapide pour flatten
x_train_flat = x_train_transformed.groupby('obs_id').apply(lambda x: x.values[:, 1:].flatten())
y_train = y_train.loc[x_train_flat.index, 'eqt_code_cat']
del x_train_transformed

x_train_flat = np.stack(x_train_flat.values, axis=0).astype(np.float32)

X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
    x_train_flat, y_train, test_size=0.2, random_state=42
)

print(f"X_train_split shape: {X_train_split.shape}")
print(f"X_val_split shape: {X_val_split.shape}")
print(f"y_train_split shape: {y_train_split.shape}")
print(f"y_val_split shape: {y_val_split.shape}")

t0 = time.time()

rf_model = RandomForestClassifier(random_state=42, n_jobs=-1, criterion='entropy')

# Define the hyperparameter grid for random search
param_dist = {
    'n_estimators': randint(100, 1000),
    'max_depth': randint(5, 50),
    'min_samples_split': randint(2, 10),
    'min_samples_leaf': randint(1, 10),
    'max_features': uniform(0.1, 0.9),
}

# Perform random search with 5-fold cross-validation
classifier = RandomizedSearchCV(
    rf_model,
    param_distributions=param_dist,
    n_iter=20,
    scoring='accuracy',
    cv=5,
    verbose=2,
    random_state=42,
    n_jobs=-1
)

y_pred = classifier.predict(X_val_split)

accuracy = accuracy_score(y_val_split, y_pred)
print(f"Test Accuracy: {accuracy}")

# Classification report détaillée
report = classification_report(y_val_split, y_pred, target_names=[f"Class {i}" for i in range(y_train.nunique())])
print("Classification Report:")
print(report)

x_test_transformed = preprocessor.transform(x_test[columns_to_use])

one_hot_enc_cols = []
for col in categorical_cols:
    for i in range(x_test[col].nunique()):
        one_hot_enc_cols.append(f"{col}_{i}")

x_test_transformed = pd.DataFrame(
    x_test_transformed,
    columns=['obs_id'] + one_hot_enc_cols + pass_trough_cols + log_plus_one_cols + log_abs_cols
)

# Remplacer log(flux) = -inf (donc flux = 0) par 0
x_test_transformed['flux'] = x_test_transformed['flux'].replace(-np.inf, 0)

# Plus rapide pour flatten
x_test_flat = x_test_transformed.groupby('obs_id').apply(lambda x: x.values[:, 1:].flatten())
del x_test_transformed

x_test_flat = np.stack(x_test_flat.values, axis=0).astype(np.float32)

# Prédire sur les données de test
y_pred_test = classifier.predict(x_test_flat)

# Sauvegarde sous le bon format
y_test = pd.DataFrame({'obs_id': x_test['obs_id'].unique(), 'eqt_code_cat': y_pred_test})
y_test.to_csv('y_test_RF.csv', index=False)