# 💳 Credit Risk Prediction Demo

This project is a Streamlit-based demo system for predicting credit card default risk.

It demonstrates how machine learning models can be applied in a financial risk control scenario, including probability comparison, adjustable decision thresholds, and model performance interpretation.

---

## 🔍 Project Overview

This demo is built using the UCI Credit Card dataset.

The system allows users to:

- View dataset overview and class distribution
- Randomly sample customer records
- Compare default probabilities across three models
- Adjust decision threshold dynamically
- Observe how business strategy affects predictions

---

## 🤖 Models Used

Three supervised learning models are implemented:

- K-Nearest Neighbors (KNN)
- Logistic Regression
- XGBoost

All models are trained as pipelines and exported using `joblib`.

---

## 🎯 Key Features

### 1️⃣ Multi-Model Probability Comparison

For the same customer record, the system shows:

- Default probability
- Risk level (Low / Medium / High)
- Final decision based on threshold

This highlights how different models may produce different risk assessments.

---

### 2️⃣ Adjustable Decision Threshold

Users can modify the decision threshold.

Lower threshold:
- Higher Recall
- Fewer missed defaulters
- More conservative risk control

Higher threshold:
- Higher Precision
- Fewer false alarms
- More aggressive business strategy

---

### 3️⃣ Risk Visualization

- Probability progress bar
- Risk classification display
- Model decision summary

---

## 📊 Dataset

UCI Credit Card Default Dataset

Target variable:
- default.payment.next.month
  - 0 = Normal
  - 1 = Default

The dataset contains financial, demographic, and payment history features.

---

## 🛠 Tech Stack

- Python
- Pandas
- Scikit-learn
- XGBoost
- Streamlit
- Joblib

---

## 🚀 Deployment

The application is deployed using Streamlit Cloud.

---

## 💡 Business Insight

This demo illustrates how model probability must be converted into business decisions using a decision threshold.

In financial risk control scenarios:

- Lowering threshold reduces False Negatives (avoiding bad debt)
- Raising threshold reduces False Positives (protecting good customers)

This reflects the trade-off between risk control and revenue growth.

---

## 📎 Author

Built as part of a machine learning portfolio project.# credit-risk-prediction-demo
