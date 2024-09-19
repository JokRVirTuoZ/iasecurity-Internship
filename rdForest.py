import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns

# Chargement des données
df = pd.read_csv('dataSet/dataSet.csv')

# Gestion des valeurs manquantes (exemple : imputation par la moyenne)
imputer = SimpleImputer(strategy='mean')
df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

# Séparation des features et de la cible
X = df_imputed.drop('Attack Type', axis=1)
y = df_imputed['Attack Type']

# Encodage des labels
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# Échelle les features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Création du pipeline
pipeline = Pipeline([
    ('scaler', scaler),
    ('rf', RandomForestClassifier(random_state=42))
])

# Grille de recherche avec une plus grande variété de paramètres
param_grid = {
    'rf__n_estimators': [50, 100, 200, 500],
    'rf__max_depth': [5, 10, 15, 20],
    'rf__min_samples_split': range(2, 101),  # Integer range for valid values
    'rf__min_samples_leaf': [1, 2, 4]
}

# Recherche par grille avec validation croisée stratifiée
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=cv, n_jobs=-1, verbose=2)
grid_search.fit(X_scaled, y_encoded)

# Meilleur modèle
best_model = grid_search.best_estimator_

# Prédictions sur l'ensemble de test
y_pred = best_model.predict(X_scaled)

# Évaluation du modèle
print(f"Best parameters: {grid_search.best_params_}")
print(f"Accuracy: {accuracy_score(y_encoded, y_pred)}")
print(f"Precision: {precision_score(y_encoded, y_pred, average='weighted')}")
print(f"Recall: {recall_score(y_encoded, y_pred, average='weighted')}")
print(f"F1-score: {f1_score(y_encoded, y_pred, average='weighted')}")
print(f"ROC AUC: {roc_auc_score(y_encoded, best_model.predict_proba(X_scaled), multi_class='ovr')}")

# Matrice de confusion
cm = confusion_matrix(y_encoded, y_pred)
sns.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# Importance des features
importances = best_model.named_steps['rf'].feature_importances_
indices = np.argsort(importances)[::-1]
plt.figure(figsize=(12, 6))
plt.bar(range(X.shape[1]), importances[indices])
plt.xticks(range(X.shape[1]), X.columns[indices], rotation=90)
plt.xlabel('Features')
plt.ylabel('Importance')
plt.title('Feature Importances')
plt.show()