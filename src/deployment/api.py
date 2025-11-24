from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()

model = joblib.load("src/models/readmission_model.pkl")

@app.post("/predict_readmission")
def predict(data: dict):
    df = pd.DataFrame([data])
    prediction = model.predict(df)[0]
    return {"readmission_risk": int(prediction)}
