<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=180&section=header&text=Customer%20Churn%20Predictor&fontSize=38&fontColor=fff&animation=twinkling&fontAlignY=38&desc=Predict%20which%20customers%20will%20leave%20%E2%80%94%20before%20they%20do&descAlignY=56&descSize=14" width="100%"/>

[![Python](https://img.shields.io/badge/Python-3.10-f97316?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3-f97316?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Joblib](https://img.shields.io/badge/Joblib-Saved-f97316?style=for-the-badge&logoColor=white)]()
[![Status](https://img.shields.io/badge/Status-Production%20Ready-22c55e?style=for-the-badge&logoColor=white)]()

</div>

---

## What This Project Does

A telecom company loses money every time a customer leaves. This model predicts **which customers are likely to churn** — so the company can take action before they leave.

**Real business value:** Retaining one customer costs 5x less than acquiring a new one.

---

## The Result

| Customer | Tenure | Contract | Prediction | Churn Probability |
|----------|--------|----------|------------|-------------------|
| Customer 1 | 8 months | Month-to-month | **Will Churn** | High |
| Customer 2 | 36 months | One year | **Will Stay** | Low |
| Customer 3 | 60 months | Two year | **Will Stay** | Very Low |

**Pattern discovered:** Short tenure + Month-to-month contract + Electronic check = highest churn risk.

---

## Full Pipeline — 15 Steps

```
Load Data → EDA → One-Hot Encoding → Feature/Target Split
→ Target Encoding (Yes/No → 1/0) → Train-Test Split
→ StandardScaler → LogisticRegression → Predict
→ Evaluate → Multi-customer prediction → Save with Joblib
```

---

## What Makes This Production-Ready

### 1. Missing Column Fix
```python
for col in training_columns:
    if col not in df_final.columns:
        df_final[col] = 0
```
New customers may not have all contract types. This ensures the model never crashes on unseen data. This is real production thinking — not tutorial code.

### 2. Predict Multiple Customers at Once
```python
new_customers = pd.DataFrame([
    {'tenure': 8, 'monthly_charges': 95.5, 'contract_type': 'Month-to-month', ...},
    {'tenure': 36, 'monthly_charges': 65.2, 'contract_type': 'One year', ...},
    {'tenure': 60, 'monthly_charges': 42.0, 'contract_type': 'Two year', ...}
])
```
Batch prediction — exactly how real systems work.

### 3. Model Saved with Joblib
```python
joblib.dump(model, "churn_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(training_columns, "training_columns.pkl")
```
Three separate files — model, scaler, and column names. This is the correct way to save a pipeline. If you only save the model and forget the scaler, predictions will be wrong.

---

## Features Used

| Feature | Type | What it tells us |
|---------|------|-----------------|
| tenure | Numerical | How long customer has stayed |
| monthly_charges | Numerical | How much they pay per month |
| total_charges | Numerical | Lifetime value |
| avg_monthly_gb_download | Numerical | Usage pattern |
| avg_calls_per_month | Numerical | Engagement level |
| customer_service_calls | Numerical | Frustration indicator |
| contract_type | Categorical → One-hot | Month-to-month = higher risk |
| paperless_billing | Categorical → One-hot | Digital engagement |
| payment_method | Categorical → One-hot | Electronic check = higher risk |

---

## Skills Demonstrated

[![Feature Engineering](https://img.shields.io/badge/Feature%20Engineering-One--Hot%20Encoding-f97316?style=flat-square)]()
[![Pipeline](https://img.shields.io/badge/Pipeline-End%20to%20End-f97316?style=flat-square)]()
[![Deployment](https://img.shields.io/badge/Deployment-Joblib%20pkl-f97316?style=flat-square)]()
[![Production](https://img.shields.io/badge/Production-Missing%20Column%20Fix-22c55e?style=flat-square)]()
[![Batch Prediction](https://img.shields.io/badge/Batch-Multi%20Customer%20Predict-22c55e?style=flat-square)]()

---

## How to Run

```bash
# Clone the repo
git clone https://github.com/ather-ops/ML-with-Scikit-Learn

# Install dependencies
pip install pandas scikit-learn matplotlib joblib

# Add your dataset
# Place Customer_churn.csv in the same folder

# Run
python customer_churn.py
```

---

## Load Saved Model

```python
import joblib
import pandas as pd

# Load
model = joblib.load('churn_model.pkl')
scaler = joblib.load('scaler.pkl')
training_columns = joblib.load('training_columns.pkl')

# Predict new customer
new_customer = pd.DataFrame([{
    'tenure': 12,
    'monthly_charges': 80.0,
    'total_charges': 960.0,
    'avg_monthly_gb_download': 25.0,
    'avg_calls_per_month': 45,
    'customer_service_calls': 4,
    'contract_type': 'Month-to-month',
    'paperless_billing': 'Yes',
    'payment_method': 'Electronic check'
}])

# Encode and predict
# (use encode_new_customer function from the notebook)
```

---

## Project Structure

```
ML-with-Scikit-Learn/
├── customer_churn.py        # Full pipeline
├── Customer_churn.csv       # Dataset
├── churn_model.pkl          # Saved model
├── scaler.pkl               # Saved scaler
├── training_columns.pkl     # Saved column names
└── README.md
```

---

<div align="center">

**Part of the ML with Scikit-Learn curriculum**

[![GitHub](https://img.shields.io/badge/GitHub-ather--ops-f97316?style=for-the-badge&logo=github&logoColor=white)](https://github.com/ather-ops)
[![Live App](https://img.shields.io/badge/Live%20App-Rain%20Predictor-f97316?style=for-the-badge&logo=streamlit&logoColor=white)](https://rain-predictor-app.streamlit.app)

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=100&section=footer" width="100%"/>

</div>
