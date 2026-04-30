import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib
import os

ENC_PATH = "models/encoders.pkl"

def load_data():
    df = pd.read_csv("data/telco.csv")
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df.fillna(0, inplace=True)
    return df

def preprocess_train(df):
    df = df.copy()
    df.drop("customerID", axis=1, inplace=True)

    df["Churn"] = df["Churn"].map({"Yes":1, "No":0})

    encoders = {}
    for col in df.select_dtypes(include="object").columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    os.makedirs("models", exist_ok=True)
    joblib.dump(encoders, ENC_PATH)

    X = df.drop("Churn", axis=1)
    y = df["Churn"]
    return X, y

def preprocess_infer(df):
    df = df.copy()
    encoders = joblib.load(ENC_PATH)

    for col, le in encoders.items():
        if col in df.columns:
            df[col] = df[col].map(
                lambda x: le.transform([x])[0] if x in le.classes_ else 0
            )

    return df