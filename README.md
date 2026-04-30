# 🚀 Customer Churn Prediction System with Explainable AI

## 📌 Project Overview
Customer churn is a major challenge in telecom and service industries. This project predicts whether a customer is likely to leave (churn) using machine learning and explains the reason behind each prediction using Explainable AI (SHAP).

The system combines predictive modeling with an interactive dashboard to help understand customer behavior and support better decision-making.

---

## 🎯 Objectives
- Predict customer churn using machine learning  
- Analyze customer behavior patterns  
- Provide interactive visual insights  
- Explain predictions using SHAP  

---

## 🧠 Key Features

### 🔮 Prediction System
- Predicts churn probability for each customer  
- Classifies customers as High Risk or Low Risk  

### 📊 Interactive Dashboard
- Built using Streamlit  
- Includes:
  - Churn distribution  
  - Charges vs churn  
  - Tenure analysis  
  - Correlation heatmap  

### 🔍 Explainable AI (SHAP)
- Shows feature contribution for each prediction  
- Identifies factors increasing or decreasing churn  
- Makes model decisions transparent  

### 📈 Model Evaluation
- Confusion Matrix  
- ROC Curve (AUC Score)  
- Precision-Recall Curve  
- Feature Importance  

---

## 🛠️ Tech Stack
- Python  
- XGBoost  
- Pandas, NumPy  
- Matplotlib, Seaborn  
- Streamlit  
- SHAP  

---

## 📊 Dataset
Telco Customer Churn Dataset with:
- Customer demographics  
- Subscription details  
- Billing information  
- Service usage  

---

## 🧩 Project Structure
Customer-Churn-Prediction/
│── data/  
│── models/  
│── src/  
│── dashboard/  
│── outputs/  
│── requirements.txt  
│── README.md  

---

## ▶️ How to Run

Install dependencies:
pip install -r requirements.txt  

Train model:
python -m src.train  

Evaluate model:
python -m src.evaluate  

Run dashboard:
streamlit run dashboard/app.py  

---

## 💡 Key Insights
- Low tenure customers churn more  
- High monthly charges increase churn  
- Long-term customers are stable  
- Contract type strongly impacts churn  

---

## 🔍 Explainability
SHAP explains predictions:
- Positive values → increase churn  
- Negative values → reduce churn  

---

## 🎯 Results
- ROC-AUC ≈ 0.96  
- Strong prediction performance  
- Clear explainability  

---

## 🚀 Future Work
- Deploy online  
- Add recommendation system  
- Improve UI  

---

## 🙌 Conclusion
This project shows how ML + Explainable AI can provide accurate and interpretable solutions.

---

## 📬 Contact
Open for feedback and collaboration.
