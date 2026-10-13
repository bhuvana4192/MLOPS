import pandas as pd
import joblib

df=pd.read_csv("processed_student_data.csv")
print("Dataset loaded successfully")

X=df.drop(columns=["G3"])
X=pd.get_dummies(X)

model=joblib.load("tuned_random_forest.pkl")

print("Deployment model loaded successfully")

expected_columns=model.feature_names_in_

for column in expected_columns:
    if column not in X.columns:
        X[column] = 0
        
X=X[expected_columns]

predictions=model.predict(X)

df["Predicted_G3"]=predictions.round(2)
df.to_csv("batch_predictions.csv", index=False)

print(df[["G3", "Predicted_G3"]].head())
print("\nTotal students :", len(df))