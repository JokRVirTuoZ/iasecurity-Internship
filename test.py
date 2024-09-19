import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss

# 1. Chargement des données
df = pd.read_csv('dataSet/newDataSet.csv')

# 2. Prétraitement des données
X = df.drop('Attack Type', axis=1)  # Variables explicatives
y = df['Attack Type']  # Variable cible (DDoS, Intrusion, Malware)

# Variables pour stocker les résultats
ind = 0
test_sizes = np.arange(0.01, 0.51, 0.01)
train_accuracies = []
test_accuracies = []
train_log_losses = []
test_log_losses = []

for test_size in test_sizes:
    # 3. Division des données en ensemble d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=1)

    # 4. Entraînement du modèle Random Forest
    rf_model = RandomForestClassifier(n_estimators=100, random_state=1)
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

    # Stocker les résultats
    train_accuracies.append(train_accuracy)
    test_accuracies.append(test_accuracy)
    train_log_losses.append(train_log_loss)
    test_log_losses.append(test_log_loss)

    ind += 1
    print(f"iteration :{ind}")
    print(f"train_accuracy :{train_accuracy}")
    print(f"test_accuracy :{test_accuracy}")
    print(f"train_log_loss :{train_log_loss}")
    print(f"test_log_loss :{test_log_loss}")


# Création des plots
fig, ax = plt.subplots(1, 2, figsize=(14, 6))

# Graphique pour Accuracy
ax[0].plot(test_sizes, train_accuracies, label='Training Accuracy', marker='o')
ax[0].plot(test_sizes, test_accuracies, label='Test Accuracy', marker='x')
ax[0].set_title('Training vs Test Accuracy')
ax[0].set_xlabel('Test Size')
ax[0].set_ylabel('Accuracy')
ax[0].legend()
ax[0].grid(True)

# Graphique pour Log Loss
ax[1].plot(test_sizes, train_log_losses, label='Training Log Loss', marker='o')
ax[1].plot(test_sizes, test_log_losses, label='Test Log Loss', marker='x')
ax[1].set_title('Training vs Test Log Loss')
ax[1].set_xlabel('Test Size')
ax[1].set_ylabel('Log Loss')
ax[1].legend()
ax[1].grid(True)

plt.tight_layout()
plt.show()