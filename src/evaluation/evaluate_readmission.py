"""
Evaluation script for the Hospital Readmission Prediction Model.

This script:
- Loads the trained XGBoost readmission model
- Loads the preprocessed patient dataset
- Generates predictions
- Computes evaluation metrics (precision, recall, accuracy, F1)
- Prints a confusion matrix

Ensure you have already run:
    - preprocessing/preprocess_hospital.py
    - models/train_readmission_model.py
"""

import pandas as pd
from sklearn.metrics import (
    precision_score,
    recall_score,
    accuracy_score,
    f1_score,
    confusion_matrix
)
import joblib

# -----------------------------
# Load Model and Data
# -----------------------------
MODEL_PATH = "src/models/readmission_model.pkl"
DATA_PATH = "src/data/patient_preprocessed.csv"

print("Loading model and data...")

model = joblib.load(MODEL_PATH)
data = pd.read_csv(DATA_PATH)

# Separate features and labels
X = data.drop("readmitted", axis=1)
y = data["readmitted"]

# convert object dtype columns to pandas 'category' so XGBoost accepts them
obj_cols = X.select_dtypes(include=["object"]).columns
for c in obj_cols:
    X[c] = X[c].astype("category")

# ensure target is numeric if necessary
if y.dtype == "object":
    y = y.astype("category").cat.codes

# -----------------------------
# Make Predictions
# -----------------------------
print("Running model predictions...")
preds = model.predict(X)

# -----------------------------
# Evaluation Metrics
# -----------------------------
precision = precision_score(y, preds)
recall = recall_score(y, preds)
accuracy = accuracy_score(y, preds)
f1 = f1_score(y, preds)
cm = confusion_matrix(y, preds)

# -----------------------------
# Display Results
# -----------------------------
print("\n================ Evaluation Results ================")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")

print("\nConfusion Matrix:")
print(cm)

print("\nEvaluation complete")
