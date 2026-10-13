from pathlib import Path

import joblib
import pandas as pd

base_dir = Path(__file__).resolve().parent.parent
csv_path = base_dir / "processed_student_data.csv"
model_path = base_dir / "model" / "deployment_model.pkl"

df = pd.read_csv(csv_path)
X = df.drop(columns=["G3"])
X = pd.get_dummies(X)

model = joblib.load(model_path)
prediction = model.predict(X.iloc[[0]])

print("Prediction :", round(prediction[0], 2))
print("Actual Value :", df["G3"].iloc[0])