# Importation des bibliothèques nécessaires
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss

# 1. Chargement des données
# Remplacez 'votre_dataset.csv' par le chemin de votre fichier de données
df = pd.read_csv('dataSet/newDataSet.csv')

# 2. Prétraitement des données
# Supposons que la colonne cible soit 'label' et les autres colonnes soient des caractéristiques
X = df.drop('Attack Type', axis=1)  # Variables explicatives
y = df['Attack Type']               # Variable cible (DDoS, Intrusion, Malware)

# Encodage si nécessaire (pour transformer les labels en numérique)
#y = pd.get_dummies(y).values

# 3. Division des données en ensemble d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=None)

# 4. Entraînement du modèle Random Forest
rf_model = RandomForestClassifier(n_estimators=25, random_state=None)
rf_model.fit(X_train, y_train)

# 5. Évaluation sur l'ensemble de test
y_pred_train = rf_model.predict(X_train)
y_pred_test = rf_model.predict(X_test)

# 6. Calcul de la précision et du log loss
train_accuracy = accuracy_score(y_train, y_pred_train)
test_accuracy = accuracy_score(y_test, y_pred_test)

# Pour log_loss, il est nécessaire de passer les probabilités plutôt que les prédictions
y_proba_train = rf_model.predict_proba(X_train)
y_proba_test = rf_model.predict_proba(X_test)

train_log_loss = log_loss(y_train, y_proba_train)
test_log_loss = log_loss(y_test, y_proba_test)

# Affichage des résultats
print(f"Train Accuracy: {train_accuracy:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Train Log Loss: {train_log_loss:.4f}")
print(f"Test Log Loss: {test_log_loss:.4f}")

# 7. Visualisation des performances (Accuracy et Log Loss)
# Création de deux figures pour Accuracy et Log Loss
fig, ax = plt.subplots(1, 2, figsize=(12, 5))

# Graphique pour Accuracy
ax[0].plot(['Train', 'Test'], [train_accuracy, test_accuracy], marker='o', color='b', label='Accuracy')
ax[0].set_title('Accuracy for Train and Test')
ax[0].set_ylabel('Accuracy')
ax[0].set_ylim(0, 1)

# Graphique pour Log Loss
ax[1].plot(['Train', 'Test'], [train_log_loss, test_log_loss], marker='o', color='r', label='Log Loss')
ax[1].set_title('Log Loss for Train and Test')
ax[1].set_ylabel('Log Loss')
ax[1].set_ylim(0, max(train_log_loss, test_log_loss) + 0.1)

# Affichage des graphiques
plt.tight_layout()
plt.show()