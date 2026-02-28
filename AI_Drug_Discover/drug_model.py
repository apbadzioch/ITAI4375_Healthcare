"""
1. Load the Delaney solubility dataset
2. Compute molecular descriptors using RDKit
3. Train a Random Forest regression model
4. Evaluate performance
5. Save the trained model to disk
"""

# Import needed libraries and frameworks
import pandas as pd
import joblib
from rdkit import Chem
from rdkit.Chem import Descriptors
# import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

# ---------------------------------------------------------
# 1. Load and inspect the data
url = "https://raw.githubusercontent.com/dataprofessor/data/master/delaney.csv"
data = pd.read_csv(url)

data.dropna(inplace=True)
print(f"Initial dataset size: {len(data)} molecules")

#
# Compute chemical features using RDKit
data['Mol'] = data['SMILES'].apply(Chem.MolFromSmiles)

data = data[data["Mol"].notna()].copy()
print(f"Valid molecules after RDKit parsing: {len(data)}")

# 2. Compute molecular descriptors
# MolWT
data['MolWt'] = data['Mol'].apply(Descriptors.MolWt)
# LogP
data['LogP'] = data['Mol'].apply(Descriptors.MolLogP)

# target column
y_col = "measured log(solubility:mol/L)"

# from notebook for practice
"""
plt.figure()
plt.scatter(data["MolWt"], data[y_col], alpha=0.6)
plt.xlabel("Molecular Weight (MolWt)")
plt.ylabel("Measured log solubility (log mols/L)")
plt.title("MolWt vs Solubility")
plt.show()

plt.figure()
plt.scatter(data["LogP"], data[y_col], alpha=0.6)
plt.xlabel("LogP")
plt.ylabel("Measured log solubility (log mols/L)")
plt.title("LogP vs Solubility")
plt.show()

plt.figure()
plt.hist(data["MolWt"], bins=30)
plt.xlabel("MolWt")
plt.ylabel("Count")
plt.title("Distribution of MolWt")
plt.show()

plt.figure()
plt.hist(data["LogP"], bins=30)
plt.xlabel("LogP")
plt.ylabel("Count")
plt.title("Distribution of LogP")
plt.show()
"""

# --- TRAIN/TEST SPLIT ---
X = data[["MolWt", "LogP"]]
y = data[y_col]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --- TRAIN RANDOM FOREST MODEL ---
model = RandomForestRegressor(
    random_state=42,
    n_estimators=100,
    n_jobs=-1
)
model.fit(X_train, y_train)

# --- EVALUATE MODEL PERFORMANCE ---
y_pred = model.predict(X_test)

print("Model Performance:")
print("R2 Score:", r2_score(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))

# --- SAVE TRAINED MODEL ---
joblib.dump(model, "solubility_model.joblib")
print("Model saved as solubility_model.joblib")

