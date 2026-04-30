import joblib
import shap
import pandas as pd
import numpy as np

from src.preprocess import load_data, preprocess_train, preprocess_infer
from src.features import add_features


def build_explainer():
    df = load_data()
    df = add_features(df)

    X, y = preprocess_train(df)
    model = joblib.load("models/churn_model.pkl")

    # ✅ convert background to numpy (IMPORTANT FIX)
    background = X.sample(50).values

    # ✅ FORCE callable function
    def predict_fn(data):
        data = pd.DataFrame(data, columns=X.columns)
        return model.predict_proba(data)[:, 1]

    # ✅ FORCE KernelExplainer (NO AUTO DETECTION)
    explainer = shap.KernelExplainer(predict_fn, background)

    return explainer, X.columns


def explain_single(input_df):
    explainer, cols = build_explainer()

    input_df = add_features(input_df)
    input_df = preprocess_infer(input_df)

    columns = joblib.load("models/columns.pkl")

    for col in columns:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[columns]

    # ✅ convert to numpy
    shap_values = explainer.shap_values(input_df.values)

    return shap_values, input_df