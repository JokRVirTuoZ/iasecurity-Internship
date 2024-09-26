import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten, MaxPooling1D
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# 1. Chargement des données
train = pd.read_csv('dataSet/KDDTrain_final.csv')

# 2. Prétraitement des données
# Encodage de la variable cible avec One-Hot Encoding
encoder = OneHotEncoder(sparse_output=False)
y = encoder.fit_transform(train.iloc[:, -1].values.reshape(-1, 1))  # Cible dans l'avant-dernière colonne

# Variables explicatives
X = train.drop(train.columns[-1], axis=1).values  # Suppression de la colonne cible

# 3. Mélange des données et stratification pour assurer une répartition équilibrée des classes
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=train.iloc[:, -1], random_state=1)

# 4. Normalisation des données
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# CNN requiert une 3ème dimension (samples, time steps, features), on ajoute donc une dimension aux features
X_train = np.expand_dims(X_train, axis=2)
X_test = np.expand_dims(X_test, axis=2)

# 5. Construction du modèle CNN
model = Sequential()

# Couches convolutionnelles pour traiter les données tabulaires
model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(X_train.shape[1], 1)))
model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(filters=32, kernel_size=2, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())  # Aplatir les résultats de convolution pour passer aux couches denses
model.add(Dense(units=64, activation='relu'))
model.add(Dense(units=y.shape[1], activation='softmax'))  # Nombre de classes pour la classification

# 6. Compilation du modèle
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 7. Entraînement du modèle
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test))

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

# 10. Prédictions sur les données de test
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

# 11. Matrice de confusion
conf_matrix = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=encoder.categories_[0], yticklabels=encoder.categories_[0])
plt.title('Confusion Matrix')
plt.ylabel('True Class')
plt.xlabel('Predicted Class')
plt.show()

# 12. Rapport de classification
print(classification_report(y_true, y_pred_classes))
