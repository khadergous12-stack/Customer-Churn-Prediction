from fastapi import FastAPI
import joblib, pandas as pd
from src.preprocess import preprocess_infer
from src.features import add_features

app = FastAPI()

model = joblib.load("models/churn_model.pkl")
columns = joblib.load("models/columns.pkl")

@app.get("/")
def home():
    return {"message": "API Running 🚀"}

@app.post("/predict")
def predict(data: dict):
    df = pd.DataFrame([data])

    df = add_features(df)
    df = preprocess_infer(df)

    for col in columns:
        if col not in df.columns:
            df[col] = 0

    df = df[columns]

    pred = model.predict(df)[0]
    prob = model.predict_proba(df)[0][1]

    return {"prediction": int(pred), "probability": float(prob)}