# Customer Churn Predictor

A logistic regression model that predicts which telecom customers are likely to cancel their subscription — before they do. Built as part of the [ML-with-Scikit-Learn](https://github.com/ather-ops/ML-with-Scikit-Learn) curriculum.

---

## The Problem

A telecom company loses money every time a customer leaves. Retaining an existing customer costs significantly less than acquiring a new one. This model gives the business a list of at-risk customers so action can be taken before the cancellation happens.

---

## Result

| Customer | Tenure | Contract | Prediction | Risk |
|----------|--------|----------|------------|------|
| Customer 1 | 8 months | Month-to-month | Will Churn | High |
| Customer 2 | 36 months | One year | Will Stay | Low |
| Customer 3 | 60 months | Two year | Will Stay | Very Low |

Pattern identified: short tenure combined with a month-to-month contract and electronic check payment is the strongest predictor of churn.

---

## Pipeline

```
Load Data → EDA → One-Hot Encoding → Feature/Target Split
→ Target Encoding (Yes/No → 1/0) → Train-Test Split
→ StandardScaler → LogisticRegression → Predict
→ Evaluate → Batch Prediction → Save with Joblib
```

---

## Features

| Feature | Type | Signal |
|---------|------|--------|
| tenure | Numerical | How long the customer has stayed |
| monthly_charges | Numerical | Amount paid per month |
| total_charges | Numerical | Lifetime value |
| avg_monthly_gb_download | Numerical | Usage pattern |
| avg_calls_per_month | Numerical | Engagement level |
| customer_service_calls | Numerical | Frustration indicator — high count correlates with churn |
| contract_type | Categorical | Month-to-month carries the highest risk |
| paperless_billing | Categorical | Digital engagement signal |
| payment_method | Categorical | Electronic check correlates with higher churn |

---

## Production Details

### Missing Column Handling

New customers may not have every contract type represented in their data. Without this fix, the model crashes on unseen category combinations.

```python
for col in training_columns:
    if col not in df_final.columns:
        df_final[col] = 0
```

Any column the model was trained on but is absent in new data gets filled with zero. This is the difference between tutorial code and code that works in production.

### Batch Prediction

The model accepts multiple customers in a single call — which is how real systems operate.

```python
new_customers = pd.DataFrame([
    {'tenure': 8,  'monthly_charges': 95.5, 'contract_type': 'Month-to-month', ...},
    {'tenure': 36, 'monthly_charges': 65.2, 'contract_type': 'One year',       ...},
    {'tenure': 60, 'monthly_charges': 42.0, 'contract_type': 'Two year',       ...}
])
```

### Saving the Pipeline

Three separate files are saved — not one. Saving only the model and discarding the scaler is a common mistake that produces silently wrong predictions.

```python
joblib.dump(model,            'churn_model.pkl')
joblib.dump(scaler,           'scaler.pkl')
joblib.dump(training_columns, 'training_columns.pkl')
```

---

## Loading the Saved Model

```python
import joblib
import pandas as pd

model            = joblib.load('churn_model.pkl')
scaler           = joblib.load('scaler.pkl')
training_columns = joblib.load('training_columns.pkl')

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

# Encode using encode_new_customer() then pass through scaler and model
```

---

## Project Structure

```
ML-with-Scikit-Learn/
├── customer_churn.py          # Full pipeline
├── Customer_churn.csv         # Dataset
├── churn_model.pkl            # Saved model
├── scaler.pkl                 # Saved scaler
├── training_columns.pkl       # Saved column names
└── README.md
```

---

## Getting Started

```bash
git clone https://github.com/ather-ops/ML-with-Scikit-Learn
pip install pandas scikit-learn matplotlib joblib
# Place Customer_churn.csv in the project folder
python customer_churn.py
```

---

## License

MIT. Use freely.

---

## Author

[ather-ops](https://github.com/ather-ops)
