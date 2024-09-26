import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
import seaborn as sns

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

# 10. Prédictions sur l'ensemble de test
y_pred = model.predict(X_test)

# Transformer les prédictions et les vraies valeurs en classes (au lieu de one-hot encoding)
y_test_classes = np.argmax(y_test, axis=1)
y_pred_classes = np.argmax(y_pred, axis=1)

# 11. Affichage du rapport de classification (precision, recall, f1-score, support)
target_names = encoder.categories_[0]
report = classification_report(y_test_classes, y_pred_classes, target_names=target_names)
print(report)

# 12. Calcul des métriques (Precision, Recall, F1-Score) par classe
precision = precision_score(y_test_classes, y_pred_classes, average=None)
recall = recall_score(y_test_classes, y_pred_classes, average=None)
f1 = f1_score(y_test_classes, y_pred_classes, average=None)
support = np.bincount(y_test_classes)

# Tracer les graphiques pour Precision, Recall, F1-Score, et Support
x = np.arange(len(target_names))

plt.figure(figsize=(10, 6))

# Precision
plt.bar(x - 0.3, precision, width=0.2, label='Precision', color='b')

# Recall
plt.bar(x - 0.1, recall, width=0.2, label='Recall', color='g')

# F1-Score
plt.bar(x + 0.1, f1, width=0.2, label='F1-Score', color='r')

# Support (on l'ajoute en tant que texte au-dessus des barres)
for i in range(len(support)):
    plt.text(x[i] + 0.15, f1[i] + 0.02, str(support[i]), ha='center', color='black')

# Labels
plt.xticks(x, target_names)
plt.xlabel('Classes')
plt.ylabel('Scores')
plt.title('Precision, Recall, F1-Score, and Support for Each Class')
plt.legend()
plt.show()

# 13. Matrice de confusion
conf_matrix = confusion_matrix(y_test_classes, y_pred_classes)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=target_names, yticklabels=target_names)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Class')
plt.ylabel('True Class')
plt.show()
