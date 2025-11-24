import pandas as pd
import xgboost as xgb
import joblib

data = pd.read_csv("src/data/patient_preprocessed.csv")

X = data.drop("readmitted", axis=1)
y = data["readmitted"]

# convert object dtype columns to pandas 'category'
obj_cols = X.select_dtypes(include=["object"]).columns
for c in obj_cols:
    X[c] = X[c].astype("category")

# ensure target is numeric (if it's categorical strings convert to codes)
if y.dtype == "object":
    y = y.astype("category").cat.codes

# enable_categorical=True so XGBoost accepts pandas categorical dtype
model = xgb.XGBClassifier(max_depth=4, learning_rate=0.05, n_estimators=150, enable_categorical=True)

model.fit(X, y)

joblib.dump(model, "src/models/readmission_model.pkl")
print("Hospital readmission model trained and saved!")
