import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from src.preprocess import load_data, preprocess_train
from src.features import add_features
import os

def train():
    df = load_data()
    df = add_features(df)

    X, y = preprocess_train(df)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = XGBClassifier(eval_metric="logloss")
    model.fit(X_train, y_train)

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/churn_model.pkl")
    joblib.dump(X.columns.tolist(), "models/columns.pkl")

    print("✅ Model + encoders + columns saved")

if __name__ == "__main__":
    train()