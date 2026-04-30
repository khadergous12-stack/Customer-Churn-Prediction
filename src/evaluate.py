import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from src.preprocess import load_data, preprocess_train
from src.features import add_features

os.makedirs("outputs", exist_ok=True)


def evaluate():
    df = load_data()
    df = add_features(df)

    X, y = preprocess_train(df)
    model = joblib.load("models/churn_model.pkl")

    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]

    # ===============================
    # CONFUSION MATRIX (FINAL)
    # ===============================
    cm = confusion_matrix(y, y_pred)
    tn, fp, fn, tp = cm.ravel()

    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)

    cm_percent = cm / cm.sum() * 100

    plt.figure(figsize=(7, 6))

    labels = np.array([
        [f"TN\n{tn}\n({cm_percent[0,0]:.1f}%)",
         f"FP\n{fp}\n({cm_percent[0,1]:.1f}%)"],
        [f"FN\n{fn}\n({cm_percent[1,0]:.1f}%)",
         f"TP\n{tp}\n({cm_percent[1,1]:.1f}%)"]
    ])

    sns.heatmap(
        cm,
        annot=labels,
        fmt="",
        cmap="Blues",
        xticklabels=["No Churn", "Churn"],
        yticklabels=["No Churn", "Churn"]
    )

    plt.title("Confusion Matrix (Model Performance)")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.figtext(
        0.5, -0.05,
        f"Accuracy: {accuracy:.2f} | Precision: {precision:.2f} | Recall: {recall:.2f} | F1: {f1:.2f}",
        ha="center",
        fontsize=10
    )

    plt.tight_layout()
    plt.savefig("outputs/confusion_matrix.png", bbox_inches="tight")
    plt.show()

    # ===============================
    # ROC CURVE
    # ===============================
    fpr, tpr, _ = roc_curve(y, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], '--', color='gray')

    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()

    plt.savefig("outputs/roc_curve.png")
    plt.show()

    # ===============================
    # PR CURVE
    # ===============================
    precision_curve, recall_curve, _ = precision_recall_curve(y, y_prob)

    plt.figure(figsize=(6, 5))
    plt.plot(recall_curve, precision_curve)

    plt.title("Precision-Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")

    plt.savefig("outputs/pr_curve.png")
    plt.show()

    # ===============================
    # FEATURE IMPORTANCE
    # ===============================
    importances = model.feature_importances_
    indices = np.argsort(importances)

    plt.figure(figsize=(7, 5))
    plt.barh(range(len(indices)), importances[indices])
    plt.yticks(range(len(indices)), X.columns[indices])

    plt.title("Feature Importance")
    plt.xlabel("Importance Score")

    plt.savefig("outputs/feature_importance.png")
    plt.show()

    print("✅ All professional graphs generated successfully!")


if __name__ == "__main__":
    evaluate()