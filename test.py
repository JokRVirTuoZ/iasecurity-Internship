import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# 1. Chargement des données
df = pd.read_csv('dataSet/newDataSet.csv')

# 2. Prétraitement des données
X = df.drop('Attack Type', axis=1)  # Variables explicatives (25 features)
# y = df['Attack Type']  # Variable cible (DDoS, Intrusion, Malware)
y = pd.get_dummies(df['Attack Type'], prefix='Attack_Type')
#TODO ONE-HOT

# Encoder la variable cible (si elle est catégorielle)
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Normalisation des features (important pour les réseaux de neurones)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Division des données en ensemble d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=1)
#TODO balance y_train y_test
count1 = [0, 0, 0]
for ycount in y_train:

    match ycount:
        case 0:
            count1[0] += 1
        case 1:
            count1[1] += 1
        case 2:
            count1[2] += 1
percent1 = [0, 0, 0]
for foo in range(3):
    percent1[foo] = 100*count1[foo]/len(y_train)
count2 = [0, 0, 0]
for ycount in y_test:

    match ycount:
        case 0:
            count2[0] += 1
        case 1:
            count2[1] += 1
        case 2:
            count2[2] += 1
percent2 = [0,0,0]
for foo in range(3):
    percent2[foo] = 100*count2[foo]/len(y_test)

# 4. Création du modèle de réseau de neurones
model = Sequential()

# Couche d'entrée et première couche cachée (64 neurones, activation ReLU)
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))

# Ajout d'une deuxième couche cachée avec moins de neurones (32 neurones)
model.add(Dense(32, activation='relu'))

# Ajout d'une couche cachée supplémentaire pour capturer plus de complexité (16 neurones)
model.add(Dense(16, activation='relu'))

# Couche de sortie (3 neurones pour la classification multi-classes)
model.add(Dense(4, activation='softmax'))  # 3 classes à prédire: Malware, Intrusion, DDoS

# 5. Compilation du modèle avec une fonction de perte et un optimiseur adaptés
model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(learning_rate=0.0001), metrics=['accuracy'])
#TODO categorical_crossentropy
# 6. Ajout du Early Stopping pour stopper l'entraînement si le modèle commence à overfitter
#early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# 7. Entraînement du modèle sur plusieurs époques avec validation
# history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stopping], verbose=1)
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test), verbose=1)
# 8. Création des graphiques pour visualiser accuracy et loss au fil des époques
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
ax[1].plot(history.history['loss'], label='Training Loss', marker='o')
ax[1].plot(history.history['val_loss'], label='Test Loss', marker='x')
ax[1].set_title('Training vs Test Loss')
ax[1].set_xlabel('Epoch')
ax[1].set_ylabel('Loss')
ax[1].legend()
ax[1].grid(True)

plt.tight_layout()
plt.show()