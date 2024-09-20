import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, log_loss
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping

# 1. Chargement des données
df = pd.read_csv('dataSet/newDataSet.csv')

# 2. Prétraitement des données
X = df.drop('Attack Type', axis=1)  # Variables explicatives
y = df['Attack Type']  # Variable cible (DDoS, Intrusion, Malware)

# Encoder la variable cible (si elle est catégorielle)
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Variables pour stocker les résultats
epochs = 50
train_accuracies = []
test_accuracies = []
train_log_losses = []
test_log_losses = []

# 3. Division des données en ensemble d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# 4. Création du modèle pyramidal avec régularisation et dropout
model = Sequential()
model.add(Dense(128, input_dim=X_train.shape[1], activation='relu', kernel_regularizer=l2(0.001)))  # Régularisation L2
model.add(Dropout(0.5))  # Dropout à 50% sur la première couche
model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.001)))
model.add(Dropout(0.5))  # Dropout à 50% sur la deuxième couche
model.add(Dense(32, activation='relu', kernel_regularizer=l2(0.001)))
model.add(Dropout(0.5))  # Dropout à 50% sur la troisième couche
model.add(Dense(len(np.unique(y)), activation='softmax'))  # Couche de sortie avec softmax pour la classification multi-classes

# 5. Compilation du modèle
model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])

# 6. Ajout du Early Stopping (arrêt précoce) pour stopper l'entraînement si le modèle commence à overfitter
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# 7. Entraînement du modèle avec validation
history = model.fit(X_train, y_train, epochs=epochs, validation_data=(X_test, y_test), callbacks=[early_stopping], verbose=1)

# 8. Création des graphiques
fig, ax = plt.subplots(1, 2, figsize=(14, 6))

# Graphique pour Accuracy
ax[0].plot(history.history['accuracy'], label='Training Accuracy', marker='o')
ax[0].plot(history.history['val_accuracy'], label='Test Accuracy', marker='x')
ax[0].set_title('Training vs Test Accuracy')
ax[0].set_xlabel('Epoch')
ax[0].set_ylabel('Accuracy')
ax[0].legend()
ax[0].grid(True)

# Graphique pour Log Loss
ax[1].plot(history.history['loss'], label='Training Log Loss', marker='o')
ax[1].plot(history.history['val_loss'], label='Test Log Loss', marker='x')
ax[1].set_title('Training vs Test Log Loss')
ax[1].set_xlabel('Epoch')
ax[1].set_ylabel('Log Loss')
ax[1].legend()
ax[1].grid(True)

plt.tight_layout()
plt.show()