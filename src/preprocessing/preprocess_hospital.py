import pandas as pd
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("src/data/sample_patient_data.csv")

df.fillna(df.median(numeric_only=True), inplace=True)

scaler = StandardScaler()
df[["lab_score", "previous_admissions"]] = scaler.fit_transform(
    df[["lab_score", "previous_admissions"]]
)

df.to_csv("src/data/patient_preprocessed.csv", index=False)
print("Hospital preprocessing complete!")
