import pandas as pd
from sklearn.metrics import precision_score, recall_score
import joblib

model = joblib.load("src/models/student_dropout_model.pkl")
data = pd.read_csv("src/data/student_preprocessed.csv")

X = data.drop("dropped_out", axis=1)
y = data["dropped_out"]

preds = model.predict(X)

print("Precision:", precision_score(y, preds))
print("Recall:", recall_score(y, preds))
