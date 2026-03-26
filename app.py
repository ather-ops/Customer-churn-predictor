import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os
from datetime import datetime

st.set_page_config(page_title="Churn Prediction", layout="wide")

# Simple CSS
st.markdown("""
<style>
    .main-header {
        background-color: #1f77b4;
        padding: 1rem;
        text-align: center;
        color: white;
        margin-bottom: 2rem;
        border-radius: 5px;
    }
    .prediction-box {
        border: 2px solid #1f77b4;
        padding: 1.5rem;
        text-align: center;
        margin: 1rem 0;
        border-radius: 10px;
    }
    .risk-high {
        background-color: #ffcccc;
        border-color: #ff0000;
    }
    .risk-low {
        background-color: #ccffcc;
        border-color: #00aa00;
    }
    .risk-medium {
        background-color: #ffffcc;
        border-color: #ffaa00;
    }
    .stButton button {
        background-color: #1f77b4;
        color: white;
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header"><h1>Customer Churn Prediction System</h1></div>', unsafe_allow_html=True)

# Initialize session state
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'model' not in st.session_state:
    st.session_state.model = None
if 'scaler' not in st.session_state:
    st.session_state.scaler = None
if 'training_cols' not in st.session_state:
    st.session_state.training_cols = None

# Sidebar
with st.sidebar:
    st.subheader("Data Upload")
    uploaded_file = st.file_uploader("Choose CSV file", type=['csv'])
    
    st.markdown("---")
    st.subheader("Model Status")
    
    if os.path.exists("churn_model.pkl"):
        try:
            st.session_state.model = joblib.load("churn_model.pkl")
            st.session_state.scaler = joblib.load("scaler.pkl")
            st.session_state.training_cols = joblib.load("training_columns.pkl")
            st.session_state.model_trained = True
            st.success("Model ready")
        except Exception as e:
            st.warning(f"Model error: {str(e)}")
    else:
        st.info("No model found")

# Main content
if uploaded_file:
    try:
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file)
        
        if 'customer_id' in df.columns:
            df = df.drop('customer_id', axis=1)
            st.info("Removed customer_id column (not used for prediction)")
        
        has_churn = 'churn' in df.columns
        st.success(f"Loaded: {len(df)} rows, {len(df.columns)} columns")
        
        st.subheader("Data Preview")
        st.dataframe(df.head(), use_container_width=True)
        
        if has_churn:
            st.subheader("Model Training")
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Total Records:** {len(df)}")
                st.write(f"**Features:** {len(df.columns) - 1}")
            with col2:
                churn_count = (df['churn'] == 'Yes').sum() if df['churn'].dtype == 'object' else df['churn'].sum()
                st.write(f"**Churn Count:** {churn_count}")
                st.write(f"**Churn Rate:** {(churn_count/len(df)*100):.1f}%")
            
            if st.button("Train Model", use_container_width=True):
                with st.spinner("Training model..."):
                    try:
                        df_clean = df.copy()
                        for col in df_clean.columns:
                            if col != 'churn':
                                if df_clean[col].dtype == 'object':
                                    df_clean[col].fillna(df_clean[col].mode()[0] if not df_clean[col].mode().empty else 'Unknown', inplace=True)
                                else:
                                    df_clean[col].fillna(df_clean[col].median(), inplace=True)
                        
                        cat_cols = ['contract_type', 'paperless_billing', 'payment_method']
                        existing_cats = [c for c in cat_cols if c in df_clean.columns]
                        
                        if existing_cats:
                            df_encoded = pd.get_dummies(df_clean, columns=existing_cats, drop_first=False)
                        else:
                            df_encoded = df_clean.copy()
                        
                        X = df_encoded.drop('churn', axis=1)
                        y = df_encoded['churn']
                        
                        if y.dtype == 'object':
                            y = y.map({'Yes': 1, 'No': 0})
                        
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                        
                        scaler = StandardScaler()
                        X_train_scaled = scaler.fit_transform(X_train)
                        X_test_scaled = scaler.transform(X_test)
                        
                        model = LogisticRegression(max_iter=1000, random_state=42)
                        model.fit(X_train_scaled, y_train)
                        
                        y_pred = model.predict(X_test_scaled)
                        accuracy = accuracy_score(y_test, y_pred)
                        
                        joblib.dump(model, "churn_model.pkl")
                        joblib.dump(scaler, "scaler.pkl")
                        joblib.dump(X.columns.tolist(), "training_columns.pkl")
                        
                        st.session_state.model = model
                        st.session_state.scaler = scaler
                        st.session_state.training_cols = X.columns.tolist()
                        st.session_state.model_trained = True
                        
                        st.success(f"Model trained! Accuracy: {accuracy:.2%}")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write("**Confusion Matrix**")
                            cm = confusion_matrix(y_test, y_pred)
                            cm_df = pd.DataFrame(cm, index=['Actual No Churn', 'Actual Churn'], columns=['Predicted No Churn', 'Predicted Churn'])
                            st.dataframe(cm_df)
                        with col2:
                            st.write("**Classification Report**")
                            report = classification_report(y_test, y_pred, output_dict=True)
                            st.dataframe(pd.DataFrame(report).transpose().round(3))
                    except Exception as e:
                        st.error(f"Training error: {str(e)}")
        
        else:
            st.subheader("Make Predictions")
            if st.session_state.model_trained:
                st.write("### Single Customer Prediction")
                col1, col2 = st.columns(2)
                with col1:
                    tenure = st.number_input("Tenure (months)", 0, 100, 12)
                    monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 200.0, 65.0)
                    total_charges = st.number_input("Total Charges ($)", 0.0, 10000.0, 780.0)
                    avg_gb = st.number_input("Avg Monthly GB Download", 0.0, 200.0, 25.0)
                with col2:
                    avg_calls = st.number_input("Avg Calls per Month", 0, 200, 45)
                    service_calls = st.number_input("Customer Service Calls", 0, 20, 2)
                    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
                    paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
                    payment = st.selectbox("Payment Method", ["Electronic check", "Credit card", "Bank transfer", "Mailed check"])
                
                if st.button("Predict Customer", use_container_width=True):
                    customer = pd.DataFrame([{
                        'tenure': tenure, 'monthly_charges': monthly_charges, 'total_charges': total_charges,
                        'avg_monthly_gb_download': avg_gb, 'avg_calls_per_month': avg_calls,
                        'customer_service_calls': service_calls, 'contract_type': contract,
                        'paperless_billing': paperless, 'payment_method': payment
                    }])
                    
                    try:
                        customer_encoded = pd.get_dummies(customer, columns=['contract_type', 'paperless_billing', 'payment_method'])
                        for col in st.session_state.training_cols:
                            if col not in customer_encoded.columns:
                                customer_encoded[col] = 0
                        customer_encoded = customer_encoded[st.session_state.training_cols]
                        customer_scaled = st.session_state.scaler.transform(customer_encoded)
                        
                        pred = st.session_state.model.predict(customer_scaled)[0]
                        prob = st.session_state.model.predict_proba(customer_scaled)[0]
                        
                        risk_class = "risk-high" if prob[1] >= 0.7 else ("risk-medium" if prob[1] >= 0.3 else "risk-low")
                        risk_text = "High Risk" if prob[1] >= 0.7 else ("Medium Risk" if prob[1] >= 0.3 else "Low Risk")
                        
                        st.markdown(f'<div class="prediction-box {risk_class}"><h2>{risk_text} of Churn</h2><h3>Probability: {prob[1]:.1%}</h3></div>', unsafe_allow_html=True)
                        
                        fig = go.Figure(go.Indicator(
                            mode="gauge+number", value=prob[1] * 100, title={"text": "Churn Risk Score"},
                            gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#1f77b4"},
                                   "steps": [{"range": [0, 30], "color": "#ccffcc"}, {"range": [30, 70], "color": "#ffffcc"}, {"range": [70, 100], "color": "#ffcccc"}],
                                   "threshold": {"line": {"color": "red", "width": 4}, "thickness": 0.75, "value": 70}}))
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"Prediction error: {str(e)}")
            else:
                st.warning("No trained model found. Please upload training data first.")
    except Exception as e:
        st.error(f"Error: {str(e)}")
else:
    st.info("Please upload a CSV file to begin")
