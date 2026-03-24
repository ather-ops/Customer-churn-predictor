"""
Streamlit App for Customer Churn Prediction
Uses the pipeline from customer_churn.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
import warnings
warnings.filterwarnings('ignore')

# Import from your original file
from customer_churn import (
    encode_new_customer,  # Your encoding function
    training_columns,      # Your training columns
    scaler,               # Your scaler
    model                 # Your model
)

# Page config
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2rem;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        color: white;
        margin-bottom: 2rem;
    }
    .prediction-card {
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
    }
    .churn-high {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
    }
    .churn-low {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="main-header">Customer Churn Prediction Pipeline</div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### Configuration")
    uploaded_file = st.file_uploader("Upload Customer Data (CSV)", type=['csv'])
    
    st.markdown("---")
    st.markdown("### Model Status")
    try:
        # Try to load existing model
        model = joblib.load("churn_model.pkl")
        scaler = joblib.load("scaler.pkl")
        training_cols = joblib.load("training_columns.pkl")
        st.success("Model loaded successfully")
        model_loaded = True
    except:
        st.warning("No model found. Train model first using your script.")
        model_loaded = False
    
    st.markdown("---")
    st.markdown("### Instructions")
    st.markdown("""
    1. First run: `python customer_churn.py`
    2. This trains and saves the model
    3. Then use this app for predictions
    """)

# Main content
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["Data Overview", "Make Predictions", "Batch Results"])
    
    with tab1:
        st.header("Dataset Overview")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Records", len(df))
        with col2:
            if 'churn' in df.columns:
                churn_count = (df['churn'] == 'Yes').sum() if df['churn'].dtype == 'object' else df['churn'].sum()
                st.metric("Churn Count", churn_count)
        with col3:
            st.metric("Features", len(df.columns))
        
        st.subheader("Data Preview")
        st.dataframe(df.head(10))
        
        st.subheader("Data Information")
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Basic Statistics**")
            st.dataframe(df.describe())
        with col2:
            st.write("**Missing Values**")
            missing_df = pd.DataFrame({
                'Column': df.columns,
                'Missing': df.isnull().sum(),
                'Percentage': (df.isnull().sum() / len(df) * 100).round(2)
            })
            st.dataframe(missing_df)
    
    with tab2:
        st.header("Make Single Prediction")
        
        if model_loaded:
            st.subheader("Enter Customer Details")
            
            col1, col2 = st.columns(2)
            
            with col1:
                tenure = st.number_input("Tenure (months)", 0, 100, 12)
                monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 200.0, 65.0)
                total_charges = st.number_input("Total Charges ($)", 0.0, 10000.0, 780.0)
                avg_gb = st.number_input("Avg Monthly GB Download", 0.0, 200.0, 25.0)
            
            with col2:
                avg_calls = st.number_input("Avg Calls per Month", 0, 200, 45)
                service_calls = st.number_input("Customer Service Calls", 0, 20, 2)
                contract_type = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
                paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
                payment_method = st.selectbox("Payment Method", ["Electronic check", "Credit card", "Bank transfer", "Mailed check"])
            
            if st.button("Predict Churn", type="primary"):
                # Create customer dataframe
                new_customer = pd.DataFrame([{
                    'tenure': tenure,
                    'monthly_charges': monthly_charges,
                    'total_charges': total_charges,
                    'avg_monthly_gb_download': avg_gb,
                    'avg_calls_per_month': avg_calls,
                    'customer_service_calls': service_calls,
                    'contract_type': contract_type,
                    'paperless_billing': paperless_billing,
                    'payment_method': payment_method
                }])
                
                try:
                    # Use the encode function from your original file
                    new_encoded = encode_new_customer(new_customer)
                    new_scaled = scaler.transform(new_encoded)
                    prediction = model.predict(new_scaled)[0]
                    probability = model.predict_proba(new_scaled)[0]
                    
                    # Display result
                    if prediction == 1:
                        st.markdown(f"""
                        <div class="prediction-card churn-high">
                            <h2>High Risk of Churn</h2>
                            <h3>Churn Probability: {probability[1]:.1%}</h3>
                            <p>The customer has a {probability[1]:.1%} probability of churning.</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="prediction-card churn-low">
                            <h2>Low Risk of Churn</h2>
                            <h3>Churn Probability: {probability[1]:.1%}</h3>
                            <p>The customer has a {probability[1]:.1%} probability of churning.</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Probability gauge
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=probability[1] * 100,
                        title={'text': "Churn Probability (%)"},
                        gauge={
                            'axis': {'range': [0, 100]},
                            'bar': {'color': "#f5576c"},
                            'steps': [
                                {'range': [0, 30], 'color': "lightgreen"},
                                {'range': [30, 70], 'color': "yellow"},
                                {'range': [70, 100], 'color': "salmon"}
                            ]
                        }
                    ))
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        else:
            st.warning("Model not loaded. Please run customer_churn.py first to train the model.")
    
    with tab3:
        st.header("Batch Predictions")
        
        if model_loaded:
            if st.button("Generate Batch Predictions", type="primary"):
                with st.spinner("Generating predictions for all customers..."):
                    try:
                        # Use your encoding function on the entire dataset
                        # First, check if all required columns exist
                        required_cols = ['contract_type', 'paperless_billing', 'payment_method']
                        missing_cols = [col for col in required_cols if col not in df.columns]
                        
                        if missing_cols:
                            st.error(f"Missing columns: {missing_cols}")
                        else:
                            # Encode all customers using your function
                            df_encoded = encode_new_customer(df)
                            df_scaled = scaler.transform(df_encoded)
                            
                            # Make predictions
                            predictions = model.predict(df_scaled)
                            probabilities = model.predict_proba(df_scaled)
                            
                            # Create results dataframe
                            results_df = df.copy()
                            results_df['Predicted_Churn'] = ['Yes' if p == 1 else 'No' for p in predictions]
                            results_df['Churn_Probability'] = [f"{prob[1]:.1%}" for prob in probabilities]
                            
                            # Show summary
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                pred_churn = (predictions == 1).mean() * 100
                                st.metric("Predicted Churn Rate", f"{pred_churn:.1f}%")
                            with col2:
                                if 'churn' in df.columns:
                                    actual_churn = (df['churn'] == 'Yes').mean() * 100 if df['churn'].dtype == 'object' else df['churn'].mean() * 100
                                    st.metric("Actual Churn Rate", f"{actual_churn:.1f}%")
                            with col3:
                                if 'churn' in df.columns:
                                    actual_labels = df['churn'].map({'Yes': 1, 'No': 0}) if df['churn'].dtype == 'object' else df['churn']
                                    accuracy = (predictions == actual_labels).mean() * 100
                                    st.metric("Model Accuracy", f"{accuracy:.1f}%")
                            
                            # Risk distribution
                            risk_levels = []
                            for prob in probabilities:
                                if prob[1] < 0.3:
                                    risk_levels.append('Low Risk')
                                elif prob[1] < 0.7:
                                    risk_levels.append('Medium Risk')
                                else:
                                    risk_levels.append('High Risk')
                            
                            results_df['Risk_Level'] = risk_levels
                            
                            st.subheader("Risk Distribution")
                            risk_counts = pd.Series(risk_levels).value_counts()
                            fig = px.pie(values=risk_counts.values, 
                                       names=risk_counts.index,
                                       title="Customer Risk Distribution",
                                       color_discrete_sequence=['#4facfe', '#ffd93d', '#f5576c'])
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Display results
                            st.subheader("Detailed Results")
                            display_cols = ['tenure', 'monthly_charges', 'contract_type', 'Predicted_Churn', 'Churn_Probability', 'Risk_Level']
                            available_cols = [col for col in display_cols if col in results_df.columns]
                            st.dataframe(results_df[available_cols])
                            
                            # Download button
                            csv = results_df.to_csv(index=False)
                            st.download_button(
                                label="Download Predictions (CSV)",
                                data=csv,
                                file_name="churn_predictions.csv",
                                mime="text/csv"
                            )
                            
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
        else:
            st.warning("Model not loaded. Please run customer_churn.py first to train the model.")
else:
    # Welcome screen
    st.info("Please upload a CSV file from the sidebar to begin")
    
    st.markdown("""
    ### How to use this application:
    
    1. **First, train the model:**
       ```bash
       python customer_churn.py
