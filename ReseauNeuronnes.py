import time
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import pickle
from tensorflow.keras import layers, models
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer

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

# Enlever données avec bid ask size négatif
data_with_negative_bid_ask_size = x_train.loc[
    (x_train['bid_size'] < 0) | (x_train['ask_size'] < 0), 'obs_id'
].unique()

x_train = x_train.loc[~x_train['obs_id'].isin(data_with_negative_bid_ask_size)]
y_train = y_train.loc[~y_train['obs_id'].isin(data_with_negative_bid_ask_size)]

x_train_transformed = preprocessor.fit_transform(x_train[columns_to_use])

one_hot_enc_cols = []
for col in categorical_cols:
    for i in range(x_train[col].nunique()):
        one_hot_enc_cols.append(col + '_' + str(i))

del x_train

x_train_transformed = pd.DataFrame(
    x_train_transformed,
    columns=['obs_id'] + one_hot_enc_cols + pass_trough_cols + log_plus_one_cols + log_abs_cols
)

# Remplacer log(flux) = -inf par 0
x_train_transformed['flux'] = x_train_transformed['flux'].replace(-np.inf, 0)

# Transformer en tenseur
X_train_tensor = x_train_transformed.groupby('obs_id').apply(lambda x: x.values[:, 1:])
y_train = y_train.loc[X_train_tensor.index, 'eqt_code_cat']
X_train_tensor = np.stack(X_train_tensor.values, axis=0).astype(np.float32)
del x_train_transformed

# Transformer les labels en dummies
y_train_one_hot = pd.get_dummies(y_train).values

X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
    X_train_tensor, y_train_one_hot, test_size=0.2, random_state=42
)

print(f"X_train_split shape: {X_train_split.shape}")
print(f"X_val_split shape: {X_val_split.shape}")
print(f"y_train_split shape: {y_train_split.shape}")
print(f"y_val_split shape: {y_val_split.shape}")

# Définir un modèle GRU
class MultiClassGRUClassifier(models.Model):
    def __init__(self, num_classes):
        super(MultiClassGRUClassifier, self).__init__()
        self.gru = layers.Bidirectional(layers.GRU(64, return_sequences=False), merge_mode='concat')
        self.dropout = layers.Dropout(0.3)
        self.dense1 = layers.Dense(64, activation='selu')
        self.dense2 = layers.Dense(num_classes, activation='softmax')

    def call(self, inputs):
        x = self.gru(inputs)
        x = self.dropout(x)
        x = self.dense1(x)
        x = self.dense2(x)
        return x

batch_size = 1000
time_steps = 100
num_classes = y_train_one_hot.shape[1]
total_batches = 10000

# Initialiser le modèle
model = MultiClassGRUClassifier(num_classes)

# Définir l'optimisateur, la loss fonction et les autres métriques
optimizer = tf.keras.optimizers.Adam(learning_rate=3e-3)
loss_fn = tf.keras.losses.CategoricalCrossentropy()
train_loss_metric = tf.keras.metrics.Mean(name='train_loss')
train_accuracy_metric = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')
val_loss_metric = tf.keras.metrics.Mean(name='val_loss')
val_accuracy_metric = tf.keras.metrics.CategoricalAccuracy(name='val_accuracy')

# Stocker les métriques
train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []

print("Début de l'entraînement avec early stopping...")
validation_interval = 100
patience = 5
best_val_loss = float('inf')
wait = 0

t0 = time.time()

for batch_num in range(total_batches):
    indices = np.random.choice(X_train_split.shape[0], batch_size, replace=False)
    X_batch = X_train_split[indices]
    y_batch = y_train_split[indices]

    with tf.GradientTape() as tape:
        predictions = model(X_batch, training=True)
        loss = loss_fn(y_batch, predictions)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss_metric(loss)
    train_accuracy_metric(y_batch, predictions)

    train_losses.append(train_loss_metric.result().numpy())
    train_accuracies.append(train_accuracy_metric.result().numpy())

    if (batch_num + 1) % validation_interval == 0:
        val_predictions = model(X_val_split, training=False)
        val_loss = loss_fn(y_val_split, val_predictions)
        val_loss_metric(val_loss)
        val_accuracy_metric(y_val_split, val_predictions)

        val_losses.append(val_loss_metric.result().numpy())
        val_accuracies.append(val_accuracy_metric.result().numpy())

        t1 = time.time()
        total = t1 - t0
        print("Temps d'exécution :", round(total))

        print(f"Batch {batch_num + 1}/{total_batches}: "
              f"Train Loss = {train_loss_metric.result():.4f}, "
              f"Train Accuracy = {train_accuracy_metric.result():.4f}, "
              f"Validation Loss = {val_loss_metric.result():.4f}, "
              f"Validation Accuracy = {val_accuracy_metric.result():.4f}")

        if val_loss_metric.result() < best_val_loss:
            best_val_loss = val_loss_metric.result()
            wait = 0
        else:
            wait += 1
            print(f'Early stopping patience count : {wait}/{patience}')

        val_loss_metric.reset_states()
        val_accuracy_metric.reset_states()

        if wait >= patience:
            print(f"Early stopping triggered at batch {batch_num + 1}. Best Validation Loss : {best_val_loss:.4f}")
        t0 = time.time()

# Évaluation finale sur les données de validation
print("\nÉvaluation finale sur les données de validation...")
val_predictions = model(X_val_split, training=False)
y_val_pred = np.argmax(val_predictions, axis=1)
y_val_labels = np.argmax(y_val_split, axis=1)

val_accuracy_score = accuracy_score(y_val_labels, y_val_pred)
print(f"Final Validation Accuracy: {val_accuracy_score:.4f}")

# Sauvegarder le modèle au format TensorFlow SavedModel pour une analyse Shap
model.save('model_rs', save_format='tf')

# Graphiques des métriques de validation et d'entraînement
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label="Training Loss")
plt.plot(np.arange(0, total_batches, validation_interval), val_losses, label="Validation Loss")
plt.xlabel("Batch Number")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label="Training Accuracy")
plt.plot(np.arange(0, total_batches, validation_interval), val_accuracies, label="Validation Accuracy")
plt.xlabel("Batch Number")
plt.ylabel("Accuracy")
plt.title("Training and Validation Accuracy")
plt.legend()

plt.tight_layout()
plt.show()
plt.savefig('Training & Validation Metrics.png')

# Classification report détaillé
report = classification_report(y_val_labels, y_val_pred, target_names=[f"Class {i}" for i in range(num_classes)])
print("Classification Report:")
print(report)

# Processer les données de test
x_test_transformed = preprocessor.transform(x_test[columns_to_use])
one_hot_enc_cols = []

for col in categorical_cols:
    for i in range(x_test[col].nunique()):
        one_hot_enc_cols.append(col + '_' + str(i))

x_test_transformed = pd.DataFrame(
    x_test_transformed,
    columns=['obs_id'] + one_hot_enc_cols + pass_trough_cols + log_plus_one_cols + log_abs_cols
)

x_test_transformed['flux'] = x_test_transformed['flux'].replace(-np.inf, 0)

X_test_tensor = x_test_transformed.groupby('obs_id').apply(lambda x: x.values[:, 1:])
X_test_tensor = np.stack(X_test_tensor.values, axis=0).astype(np.float32)

# Prédire sur les données de test
test_predictions = model(X_test_tensor, training=False)
y_test_pred = np.argmax(test_predictions, axis=1)

# Sauvegarde sous le bon format
y_test = pd.DataFrame({'obs_id': x_test['obs_id'].unique(), 'eqt_code_cat': y_test_pred})
y_test.to_csv('y_test_GRU_NN.csv', index=False)