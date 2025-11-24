import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

data = pd.read_csv("src/data/student_preprocessed.csv")

X = data.drop("dropped_out", axis=1)
y = data["dropped_out"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, max_depth=5)
model.fit(X_train, y_train)

joblib.dump(model, "src/models/student_dropout_model.pkl")
print("Student dropout model trained and saved!")
