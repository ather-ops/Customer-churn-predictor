"""
Customer Churn Prediction System
"""

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
        
        # Data Preview
        st.subheader("Data Preview")
        st.dataframe(df.head(), use_container_width=True)
        
        if has_churn:
            # Training Mode
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
                        # Data cleaning
                        df_clean = df.copy()
                        for col in df_clean.columns:
                            if col != 'churn':
                                if df_clean[col].dtype == 'object':
                                    mode_val = df_clean[col].mode()
                                    if not mode_val.empty:
                                        df_clean[col].fillna(mode_val[0], inplace=True)
                                    else:
                                        df_clean[col].fillna('Unknown', inplace=True)
                                else:
                                    df_clean[col].fillna(df_clean[col].median(), inplace=True)
                        
                        # Encode categorical features
                        cat_cols = ['contract_type', 'paperless_billing', 'payment_method']
                        existing_cats = [c for c in cat_cols if c in df_clean.columns]
                        
                        if existing_cats:
                            df_encoded = pd.get_dummies(df_clean, columns=existing_cats, drop_first=False)
                        else:
                            df_encoded = df_clean.copy()
                        
                        # Features and target
                        X = df_encoded.drop('churn', axis=1)
                        y = df_encoded['churn']
                        
                        if y.dtype == 'object':
                            y = y.map({'Yes': 1, 'No': 0})
                        
                        # Split data
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                        
                        # Scale features
                        scaler = StandardScaler()
                        X_train_scaled = scaler.fit_transform(X_train)
                        X_test_scaled = scaler.transform(X_test)
                        
                        # Train model
                        model = LogisticRegression(max_iter=1000, random_state=42)
                        model.fit(X_train_scaled, y_train)
                        
                        # Predict
                        y_pred = model.predict(X_test_scaled)
                        accuracy = accuracy_score(y_test, y_pred)
                        
                        # Save model
                        joblib.dump(model, "churn_model.pkl")
                        joblib.dump(scaler, "scaler.pkl")
                        joblib.dump(X.columns.tolist(), "training_columns.pkl")
                        
                        st.session_state.model = model
                        st.session_state.scaler = scaler
                        st.session_state.training_cols = X.columns.tolist()
                        st.session_state.model_trained = True
                        
                        st.success(f"Model trained! Accuracy: {accuracy:.2%}")
                        
                        # Show metrics
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write("**Confusion Matrix**")
                            cm = confusion_matrix(y_test, y_pred)
                            cm_df = pd.DataFrame(cm, 
                                               index=['Actual No Churn', 'Actual Churn'],
                                               columns=['Predicted No Churn', 'Predicted Churn'])
                            st.dataframe(cm_df)
                        with col2:
                            st.write("**Classification Report**")
                            report = classification_report(y_test, y_pred, output_dict=True)
                            report_df = pd.DataFrame(report).transpose()
                            st.dataframe(report_df.round(3))
                            
                    except Exception as e:
                        st.error(f"Training error: {str(e)}")
        
        else:
            # Prediction Mode
            st.subheader("Make Predictions")
            
            if st.session_state.model_trained:
                # Single Prediction
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
                    # Create customer data
                    customer = pd.DataFrame([{
                        'tenure': tenure,
                        'monthly_charges': monthly_charges,
                        'total_charges': total_charges,
                        'avg_monthly_gb_download': avg_gb,
                        'avg_calls_per_month': avg_calls,
                        'customer_service_calls': service_calls,
                        'contract_type': contract,
                        'paperless_billing': paperless,
                        'payment_method': payment
                    }])
                    
                    try:
                        # Encode
                        cat_cols = ['contract_type', 'paperless_billing', 'payment_method']
                        existing_cats = [c for c in cat_cols if c in customer.columns]
                        
                        if existing_cats:
                            customer_encoded = pd.get_dummies(customer, columns=existing_cats)
                        else:
                            customer_encoded = customer.copy()
                        
                        # Add missing columns
                        for col in st.session_state.training_cols:
                            if col not in customer_encoded.columns:
                                customer_encoded[col] = 0
                        
                        # Reorder columns
                        customer_encoded = customer_encoded[st.session_state.training_cols]
                        
                        # Scale
                        customer_scaled = st.session_state.scaler.transform(customer_encoded)
                        
                        # Predict
                        pred = st.session_state.model.predict(customer_scaled)[0]
                        prob = st.session_state.model.predict_proba(customer_scaled)[0]
                        
                        # Show result
                        if pred == 1:
                            if prob[1] >= 0.7:
                                risk_class = "risk-high"
                                risk_text = "High Risk"
                            else:
                                risk_class = "risk-medium"
                                risk_text = "Medium Risk"
                        else:
                            risk_class = "risk-low"
                            risk_text = "Low Risk"
                        
                        st.markdown(f"""
                        <div class="prediction-box {risk_class}">
                            <h2>{risk_text} of Churn</h2>
                            <h3>Churn Probability: {prob[1]:.1%}</h3>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Gauge chart
                        fig = go.Figure(go.Indicator(
                            mode="gauge+number",
                            value=prob[1] * 100,
                            title={"text": "Churn Risk Score"},
                            gauge={
                                "axis": {"range": [0, 100]},
                                "bar": {"color": "#1f77b4"},
                                "steps": [
                                    {"range": [0, 30], "color": "#ccffcc"},
                                    {"range": [30, 70], "color": "#ffffcc"},
                                    {"range": [70, 100], "color": "#ffcccc"}
                                ],
                                "threshold": {
                                    "line": {"color": "red", "width": 4},
                                    "thickness": 0.75,
                                    "value": 70
                                }
                            }
                        ))
                        fig.update_layout(height=300)
                        st.plotly_chart(fig, use_container_width=True)
                        
                    except Exception as e:
                        st.error(f"Prediction error: {str(e)}")
                
                # Batch Prediction
                st.write("---")
                st.write("### Batch Prediction")
                
                if st.button("Predict All Customers", use_container_width=True):
                    with st.spinner("Processing predictions..."):
                        try:
                            # Remove customer_id if exists
                            df_pred = df.copy()
                            if 'customer_id' in df_pred.columns:
                                df_pred = df_pred.drop('customer_id', axis=1)
                            
                            # Encode all customers
                            cat_cols = ['contract_type', 'paperless_billing', 'payment_method']
                            existing_cats = [c for c in cat_cols if c in df_pred.columns]
                            
                            if existing_cats:
                                df_encoded = pd.get_dummies(df_pred, columns=existing_cats)
                            else:
                                df_encoded = df_pred.copy()
                            
                            # Add missing columns
                            for col in st.session_state.training_cols:
                                if col not in df_encoded.columns:
                                    df_encoded[col] = 0
                            
                            df_encoded = df_encoded[st.session_state.training_cols]
                            df_scaled = st.session_state.scaler.transform(df_encoded)
                            
                            # Predict
                            predictions = st.session_state.model.predict(df_scaled)
                            probabilities = st.session_state.model.predict_proba(df_scaled)
                            
                            # Create results
                            results = df.copy()
                            results['Predicted_Churn'] = ['Yes' if p == 1 else 'No' for p in predictions]
                            results['Churn_Probability'] = [f"{p[1]:.1%}" for p in probabilities]
                            
                            # Add risk level
                            risk = []
                            for p in probabilities:
                                if p[1] < 0.3:
                                    risk.append('Low')
                                elif p[1] < 0.7:
                                    risk.append('Medium')
                                else:
                                    risk.append('High')
                            results['Risk_Level'] = risk
                            
                            # Summary
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("High Risk Customers", risk.count('High'))
                            with col2:
                                st.metric("Medium Risk Customers", risk.count('Medium'))
                            with col3:
                                st.metric("Low Risk Customers", risk.count('Low'))
                            
                            # Show results
                            st.write("### Prediction Results")
                            show_cols = ['customer_id', 'tenure', 'monthly_charges', 'contract_type', 
                                       'Predicted_Churn', 'Churn_Probability', 'Risk_Level']
                            available = [c for c in show_cols if c in results.columns]
                            st.dataframe(results[available], use_container_width=True)
                            
                            # Download
                            csv = results.to_csv(index=False)
                            st.download_button(
                                "Download Predictions (CSV)",
                                csv,
                                f"churn_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                "text/csv"
                            )
                            
                        except Exception as e:
                            st.error(f"Batch prediction error: {str(e)}")
            else:
                st.warning("No trained model found. Please upload training data with 'churn' column first.")
    
    except Exception as e:
        st.error(f"Error reading file: {str(e)}")
        st.info("Please make sure your CSV file has headers and data.")

else:
    st.info("Please upload a CSV file to begin")
    
    st.markdown("""
    ### How to Use:
    
    **Step 1: Train the Model**
    - Upload a CSV file with a 'churn' column
    - Click 'Train Model'
    
    **Step 2: Make Predictions**
    - Upload your customer data
    - Click 'Predict All Customers'
    
    **Step 3: Download Results**
    - Click 'Download Predictions'
    
    ### Required Columns:
    - tenure, monthly_charges, total_charges
    - avg_monthly_gb_download, avg_calls_per_month
    - customer_service_calls, contract_type
    - paperless_billing, payment_method
    - churn (for training only)
    """)

st.markdown("---")
st.markdown("Customer Churn Prediction System | Powered by Machine Learning")
