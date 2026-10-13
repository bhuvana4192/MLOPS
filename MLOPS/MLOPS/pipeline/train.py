from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

base_dir = Path(__file__).resolve().parent.parent
csv_path = base_dir / "processed_student_data.csv"
model_dir = base_dir / "model"
model_dir.mkdir(exist_ok=True)


df = pd.read_csv(csv_path)
X = df.drop(columns=["G3"])
y = df["G3"]

X = pd.get_dummies(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

joblib.dump(model, model_dir / "deployment_model.pkl")

print("Model training completed")