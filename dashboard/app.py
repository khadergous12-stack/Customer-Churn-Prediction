# =========================
# FIX IMPORT PATH
# =========================
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# =========================
# IMPORTS
# =========================
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from src.preprocess import preprocess_infer
from src.features import add_features
from src.explain import explain_single

# =========================
# LOAD MODEL + DATA
# =========================
model = joblib.load("models/churn_model.pkl")
columns = joblib.load("models/columns.pkl")

df = pd.read_csv("data/telco.csv")
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df.fillna(0, inplace=True)
df["Churn"] = df["Churn"].map({"Yes":1, "No":0})

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Churn Dashboard", layout="wide")
st.title("🚀 Customer Churn Prediction Dashboard")

# =========================
# SIDEBAR FILTERS
# =========================
st.sidebar.header("🔎 Filters")

contract_filter = st.sidebar.multiselect(
    "Contract Type",
    df["Contract"].unique(),
    default=df["Contract"].unique()
)

payment_filter = st.sidebar.multiselect(
    "Payment Method",
    df["PaymentMethod"].unique(),
    default=df["PaymentMethod"].unique()
)

filtered_df = df[
    (df["Contract"].isin(contract_filter)) &
    (df["PaymentMethod"].isin(payment_filter))
]

# =========================
# TABS
# =========================
tab1, tab2 = st.tabs(["📊 Analytics", "🔮 Prediction"])

# =========================
# 📊 ANALYTICS TAB
# =========================
with tab1:

    st.subheader("📌 Key Metrics")

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Customers", len(filtered_df))
    c2.metric("Churn Rate", f"{filtered_df['Churn'].mean():.2%}")
    c3.metric("Avg Monthly Charges", f"{filtered_df['MonthlyCharges'].mean():.2f}")

    st.divider()

    st.subheader("📊 Customer Behavior Analysis")

    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize=(4,3))
        sns.countplot(x="Churn", data=filtered_df, ax=ax)
        ax.set_title("Churn Distribution")
        st.pyplot(fig)
        st.caption("High churn count indicates retention issues.")

    with col2:
        fig, ax = plt.subplots(figsize=(4,3))
        sns.boxplot(x="Churn", y="MonthlyCharges", data=filtered_df, ax=ax)
        ax.set_title("Charges vs Churn")
        st.pyplot(fig)
        st.caption("Higher charges may lead to churn.")

    st.divider()

    st.subheader("📈 Tenure Impact")

    col1, col2 = st.columns([2,1])

    with col1:
        fig, ax = plt.subplots(figsize=(6,3))
        sns.histplot(data=filtered_df, x="tenure", hue="Churn", bins=30, kde=True, ax=ax)
        st.pyplot(fig)

    with col2:
        st.markdown("""
        ### Insights
        - New customers churn more  
        - Long tenure customers are stable  
        - Early engagement is critical  
        """)

    st.divider()

    st.subheader("📌 Feature Correlation")

    col1, col2 = st.columns([2,1])

    with col1:
        fig, ax = plt.subplots(figsize=(6,4))
        sns.heatmap(
            filtered_df.select_dtypes(include=["int64","float64"]).corr(),
            cmap="coolwarm",
            annot=True,
            ax=ax
        )
        st.pyplot(fig)

    with col2:
        st.markdown("""
        ### Insights
        - Tenure reduces churn  
        - Charges slightly increase churn  
        """)

# =========================
# 🔮 PREDICTION TAB
# =========================
with tab2:

    st.subheader("🧠 Predict Customer Churn")

    with st.form("form"):

        c1, c2 = st.columns(2)

        with c1:
            tenure = st.number_input("Tenure", 1, 72, 12)
            monthly = st.number_input("Monthly Charges", 10.0, 150.0, 70.0)
            total = st.number_input("Total Charges", 0.0, 10000.0, 1000.0)

        with c2:
            contract = st.selectbox("Contract", ["Month-to-month","One year","Two year"])
            payment = st.selectbox("Payment",
                ["Electronic check","Mailed check","Bank transfer (automatic)","Credit card (automatic)"]
            )

        submit = st.form_submit_button("Predict")

    # =========================
    # RUN ONLY AFTER CLICK
    # =========================
    if submit:

        input_df = pd.DataFrame([{
            "tenure": tenure,
            "MonthlyCharges": monthly,
            "TotalCharges": total,
            "Contract": contract,
            "PaymentMethod": payment
        }])

        input_df = add_features(input_df)
        input_df = preprocess_infer(input_df)

        for col in columns:
            if col not in input_df.columns:
                input_df[col] = 0

        input_df = input_df[columns]

        pred = model.predict(input_df)[0]
        prob = model.predict_proba(input_df)[0][1]

        # =========================
        # RESULT
        # =========================
        st.subheader("📊 Result")

        c1, c2 = st.columns(2)
        c1.metric("Churn Probability", f"{prob:.2f}")

        if pred == 1:
            c2.error("High Risk Customer")
        else:
            c2.success("Low Risk Customer")

        st.progress(int(prob * 100))

        # =========================
        # RECOMMENDATION
        # =========================
        st.subheader("💡 Recommendation")

        if prob > 0.7:
            st.write("Offer discounts or improve support.")
        elif prob > 0.4:
            st.write("Monitor customer behavior.")
        else:
            st.write("Customer is stable.")

        # =========================
        # SHAP (ONLY HERE)
        # =========================
        st.divider()
        st.subheader("🔍 Why this prediction?")

        shap_values, processed_df = explain_single(pd.DataFrame([{
            "tenure": tenure,
            "MonthlyCharges": monthly,
            "TotalCharges": total,
            "Contract": contract,
            "PaymentMethod": payment
        }]))

        vals = shap_values[0] if isinstance(shap_values, list) else shap_values

        importance = vals[0]
        features = processed_df.columns

        sorted_idx = np.argsort(importance)
        sorted_features = features[sorted_idx]
        sorted_values = importance[sorted_idx]

        colors = ["#FF4B4B" if v > 0 else "#00C49F" for v in sorted_values]

        fig, ax = plt.subplots(figsize=(7,5))
        ax.barh(sorted_features, sorted_values, color=colors)

        ax.set_title("Feature Contribution (SHAP)", fontsize=13)
        ax.set_xlabel("Impact on Prediction")

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        st.pyplot(fig)

        # =========================
        # AUTO INSIGHTS
        # =========================
        st.markdown(f"""
        ### 💡 Key Drivers:
        - 🔴 **{sorted_features[-1]}** increases churn risk  
        - 🟢 **{sorted_features[0]}** reduces churn  

        👉 Model decision is mainly based on these features.
        """)