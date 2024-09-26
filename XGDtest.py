import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
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

# 5. Conversion de y_train et y_test pour XGBoost (étiquettes simples, pas One-Hot)
y_train_labels = np.argmax(y_train, axis=1)
y_test_labels = np.argmax(y_test, axis=1)

# 6. Construction du modèle XGBoost avec suivi de l'accuracy et de la loss
model = XGBClassifier(n_estimators=100, learning_rate=0.01, max_depth=6, use_label_encoder=False)

# 7. Surveillance du processus d'entraînement avec 'mlogloss' et 'mlogerror'
eval_set = [(X_train, y_train_labels), (X_test, y_test_labels)]

model.fit(X_train, y_train_labels, eval_set=eval_set, eval_metric=["mlogloss", "mlogerror"], verbose=True)

# 8. Récupération des résultats d'entraînement
results = model.evals_result()

# 9. Affichage des courbes de Loss et Accuracy
# Loss
epochs = len(results['validation_0']['mlogloss'])
x_axis = range(0, epochs)

plt.figure(figsize=(10, 5))
plt.plot(x_axis, results['validation_0']['mlogloss'], label='Train Loss')
plt.plot(x_axis, results['validation_1']['mlogloss'], label='Test Loss')
plt.title('XGBoost Log Loss')
plt.xlabel('Epochs')
plt.ylabel('Log Loss')
plt.legend()
plt.show()

# Accuracy (1 - error)
plt.figure(figsize=(10, 5))
plt.plot(x_axis, 1 - np.array(results['validation_0']['mlogerror']), label='Train Accuracy')
plt.plot(x_axis, 1 - np.array(results['validation_1']['mlogerror']), label='Test Accuracy')
plt.title('XGBoost Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# 10. Matrice de confusion
y_pred = model.predict(X_test)
conf_matrix = confusion_matrix(y_test_labels, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=encoder.categories_[0], yticklabels=encoder.categories_[0])
plt.title('Confusion Matrix')
plt.ylabel('True Class')
plt.xlabel('Predicted Class')
plt.show()

# 11. Rapport de classification
print(classification_report(y_test_labels, y_pred))
