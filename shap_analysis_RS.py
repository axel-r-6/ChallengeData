import shap
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer

model = tf.keras.models.load_model('model_rs')

# Charger les données de test
x_test = pd.read_csv("x_test.csv")

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

# Enlever données avec bid ask size négatif
data_with_negative_bid_ask_size = x_test.loc[
    (x_test['bid_size'] < 0) | (x_test['ask_size'] < 0), 'obs_id'
].unique()

x_test = x_test.loc[~x_test['obs_id'].isin(data_with_negative_bid_ask_size)]

# Prétraiter les données de test
x_train_transformed = preprocessor.fit_transform(x_test[columns_to_use])

one_hot_enc_cols = []
for col in categorical_cols:
    for i in range(x_test[col].nunique()):
        one_hot_enc_cols.append(col + '_' + str(i))

x_train_transformed = pd.DataFrame(
    x_train_transformed,
    columns=['obs_id'] + one_hot_enc_cols + pass_trough_cols + log_plus_one_cols + log_abs_cols
)

# Remplacer log(flux) = -inf par 0
x_train_transformed['flux'] = x_train_transformed['flux'].replace(-np.inf, 0)

# Transformer en tenseur
X_train_tensor = x_train_transformed.groupby('obs_id').apply(lambda x: x.values[:, 1:])
X_train_tensor = np.stack(X_train_tensor.values, axis=0).astype(np.float32)

# Dimensions
print(X_train_tensor.shape)

background_data = shap.sample(X_train_tensor, 100)

explainer = shap.DeepExplainer(model, background_data)

shap_values = explainer.shap_values(X_train_tensor)

shap.plots.beeswarm(shap_values)
