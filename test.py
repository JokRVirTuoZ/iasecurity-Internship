import pandas as pd
import numpy as np
from fontTools.ttLib.tables.G_D_E_F_ import table_G_D_E_F_
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from parser2 import parser2

# 1. Chargement des données
train_data = pd.read_csv("dataSet/KDDTrain+.txt" , sep = "," , encoding = 'utf-8')
test_data = pd.read_csv("dataSet/KDDTest+.txt" , sep = "," , encoding = 'utf-8')

columns = (
    ['duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land', 'wrong_fragment', 'urgent',
     'hot',
     'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell', 'su_attempted', 'num_root',
     'num_file_creations',
     'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login', 'is_guest_login', 'count', 'srv_count',
     'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
     'srv_diff_host_rate',
     'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
     'dst_host_same_src_port_rate',
     'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
     'dst_host_srv_rerror_rate', 'attack', 'level'])
train_data.column = columns
test_data.column = columns


train_data, log_train = parser2(train_data)
test_data, log_test = parser2(test_data)

# 2. Séparer les features et les labels
X_train = train_data.iloc[:, :-2]  # Toutes les colonnes sauf les deux dernières (features)
y_train = train_data.iloc[:, -2]   # Avant-dernière colonne (cible)

X_test = test_data.iloc[:, :-2]    # Toutes les colonnes sauf les deux dernières (features)
y_test = test_data.iloc[:, -2]     # Avant-dernière colonne (cible)


# 3. One-Hot Encoding pour les labels (si Attack Type contient des valeurs comme 0, 1, 2)
encoder = OneHotEncoder(sparse_output=False)
y_train = encoder.fit_transform(y_train.values.reshape(-1, 1))
y_test = encoder.transform(y_test.values.reshape(-1, 1))

# 4. Normalisation des données (car les réseaux de neurones fonctionnent mieux avec des données normalisées)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 5. Construction du modèle pyramidal
model = Sequential()

# Couche d'entrée (nombre de neurones = nombre de features)
input_dim = X_train.shape[1]  # Nombre de features

# Première couche cachée
model.add(Dense(units=64, activation='relu', input_dim=input_dim))

# Deuxième couche cachée (réduire le nombre de neurones)
model.add(Dense(units=32, activation='relu'))

# Troisième couche cachée
model.add(Dense(units=16, activation='relu'))

# Couche de sortie avec softmax (3 classes : Malware, DDoS, Intrusion)
model.add(Dense(units=3, activation='softmax'))

# 6. Compilation du modèle
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 7. Entraînement du modèle
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# 8. Évaluation du modèle sur les données de test
loss, accuracy = model.evaluate(X_test, y_test)

print(f'Accuracy on test set: {accuracy}')
print(f'Loss on test set: {loss}')

# 9. Tracer l'accuracy et la loss sur les epochs
import matplotlib.pyplot as plt

# Tracer l'accuracy
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Tracer la loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()