import pandas as pd
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv("src/data/sample_student_data.csv")

# Handle missing values
df.fillna(df.mean(), inplace=True)

# Normalize numerical columns
scaler = MinMaxScaler()
df[["attendance_rate", "avg_grade", "lms_logins"]] = scaler.fit_transform(
    df[["attendance_rate", "avg_grade", "lms_logins"]]
)

df.to_csv("src/data/student_preprocessed.csv", index=False)
print("Student preprocessing complete!")
