import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt

# 1. Chargement des données
train = pd.read_csv('dataSet/KDDTrain_final.csv')

# 2. Prétraitement des données
# Encodage de la variable cible avec One-Hot Encoding
encoder = OneHotEncoder(sparse_output=False)
y = encoder.fit_transform(train['class'].values.reshape(-1, 1))

# Variables explicatives
X = train.drop(['class'], axis=1).values

# 3. Mélange des données et stratification pour assurer une répartition équilibrée des classes
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=train['class'], random_state=1)

# 4. Normalisation des données
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 5. Construction du modèle pyramidal
model = Sequential()
input_dim = X_train.shape[1]  # Nombre de features

model.add(Dense(units=64, activation='relu', input_shape=(input_dim,)))
model.add(Dense(units=32, activation='relu'))
model.add(Dense(units=16, activation='relu'))
model.add(Dense(units=y.shape[1], activation='softmax'))  # Nombre de classes

# 6. Compilation du modèle
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 7. Entraînement du modèle
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# 8. Évaluation du modèle sur les données de test
loss, accuracy = model.evaluate(X_test, y_test)

print(f'Accuracy on test set: {accuracy}')
print(f'Loss on test set: {loss}')

# 9. Tracer l'accuracy et la loss sur les epochs
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
