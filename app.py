"""
Main Streamlit Application for Customer Churn Prediction
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from utils.data_preprocessing import preprocess_data, handle_missing_values
from utils.model_training import train_churn_model, save_model, load_model
from utils.prediction import predict_churn, predict_batch

import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        color: white;
        margin-bottom: 2rem;
    }
    .prediction-card {
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
    }
    .prediction-churn {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
    }
    .prediction-no-churn {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
    }
    .metric-card {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'model' not in st.session_state:
    st.session_state.model = None
if 'scaler' not in st.session_state:
    st.session_state.scaler = None
if 'training_columns' not in st.session_state:
    st.session_state.training_columns = None

# Title
st.markdown('<div class="main-header">Customer Churn Prediction Pipeline</div>', unsafe_allow_html=True)
st.markdown("### Predict customer churn with machine learning")

# Sidebar
with st.sidebar:
    st.markdown("### Configuration")
    st.markdown("---")
    
    # File upload
    uploaded_file = st.file_uploader("Upload Customer Data (CSV)", type=['csv'])
    
    st.markdown("---")
    
    # Model info
    st.markdown("### Model Status")
    if st.session_state.model_trained:
        st.success("Model is trained and ready")
    else:
        st.warning("Model not trained yet")
    
    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    This application predicts customer churn using:
    - Logistic Regression
    - Feature scaling
    - One-hot encoding
    """)

# Main content
if uploaded_file is not None:
    # Load data
    df = pd.read_csv(uploaded_file)
    df = handle_missing_values(df)
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Data Overview", "Model Training", "Make Predictions", "Results"])
    
    with tab1:
        st.markdown("### Dataset Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Customers", f"{len(df):,}")
        with col2:
            if 'churn' in df.columns:
                churn_rate = (df['churn'].map({'Yes': 1, 'No': 0}) if df['churn'].dtype == 'object' else df['churn']).mean() * 100
                st.metric("Churn Rate", f"{churn_rate:.1f}%")
        with col3:
            st.metric("Features", len(df.columns))
        with col4:
            st.metric("Missing Values", df.isnull().sum().sum())
        
        st.markdown("#### Data Preview")
        st.dataframe(df.head(10), use_container_width=True)
        
        st.markdown("#### Dataset Information")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Basic Statistics**")
            st.dataframe(df.describe(), use_container_width=True)
        with col2:
            st.markdown("**Data Types**")
            dtype_df = pd.DataFrame(df.dtypes, columns=['Data Type'])
            dtype_df['Count'] = df.count()
            dtype_df['Missing'] = df.isnull().sum()
            st.dataframe(dtype_df, use_container_width=True)
    
    with tab2:
        st.markdown("### Model Training")
        
        if st.button("Train Model", type="primary", use_container_width=True):
            with st.spinner("Training model... Please wait"):
                try:
                    # Preprocess data
                    X_scaled, y, training_columns, scaler = preprocess_data(
                        df, 
                        target_col='churn',
                        fit_scaler=True
                    )
                    
                    if y is not None:
                        # Train model
                        results = train_churn_model(X_scaled, y)
                        
                        # Save model artifacts
                        save_model(results['model'], scaler, training_columns)
                        
                        # Store in session state
                        st.session_state.model_trained = True
                        st.session_state.model = results['model']
                        st.session_state.scaler = scaler
                        st.session_state.training_columns = training_columns
                        
                        # Display results
                        st.success(f"Model trained successfully! Accuracy: {results['accuracy']:.2%}")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("#### Confusion Matrix")
                            fig = px.imshow(results['confusion_matrix'], 
                                          text_auto=True, 
                                          aspect="auto",
                                          labels=dict(x="Predicted", y="Actual"),
                                          x=["No Churn", "Churn"], 
                                          y=["No Churn", "Churn"])
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with col2:
                            st.markdown("#### Classification Report")
                            report_df = pd.DataFrame(results['classification_report']).transpose()
                            st.dataframe(report_df.round(3), use_container_width=True)
                        
                        # ROC Curve
                        from sklearn.metrics import roc_curve
                        fpr, tpr, _ = roc_curve(results['y_test'], results['y_prob'])
                        
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines',
                                                name=f'ROC Curve (AUC = {results["roc_auc"]:.3f})',
                                                line=dict(color='#667eea', width=2)))
                        fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines',
                                                name='Random Classifier',
                                                line=dict(dash='dash', color='gray')))
                        fig.update_layout(title='ROC Curve', 
                                        xaxis_title='False Positive Rate',
                                        yaxis_title='True Positive Rate', 
                                        height=400)
                        st.plotly_chart(fig, use_container_width=True)
                        
                    else:
                        st.error("Target column 'churn' not found in dataset!")
                        
                except Exception as e:
                    st.error(f"Error during training: {str(e)}")
    
    with tab3:
        st.markdown("### Make Predictions")
        
        if st.session_state.model_trained:
            st.markdown("#### Enter Customer Details")
            
            col1, col2 = st.columns(2)
            with col1:
                tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=12)
                monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=200.0, value=65.0)
                total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=780.0)
                avg_gb_download = st.number_input("Avg Monthly GB Download", min_value=0.0, max_value=200.0, value=25.0)
            
            with col2:
                avg_calls = st.number_input("Avg Calls per Month", min_value=0, max_value=200, value=45)
                service_calls = st.number_input("Customer Service Calls", min_value=0, max_value=20, value=2)
                contract_type = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
                paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
                payment_method = st.selectbox("Payment Method", ["Electronic check", "Credit card", "Bank transfer", "Mailed check"])
            
            if st.button("Predict Churn", type="primary", use_container_width=True):
                customer_data = {
                    'tenure': tenure,
                    'monthly_charges': monthly_charges,
                    'total_charges': total_charges,
                    'avg_monthly_gb_download': avg_gb_download,
                    'avg_calls_per_month': avg_calls,
                    'customer_service_calls': service_calls,
                    'contract_type': contract_type,
                    'paperless_billing': paperless_billing,
                    'payment_method': payment_method
                }
                
                try:
                    result = predict_churn(
                        customer_data,
                        st.session_state.model,
                        st.session_state.scaler,
                        st.session_state.training_columns
                    )
                    
                    # Display result
                    if result['churn_prediction'] == 'Yes':
                        st.markdown(f"""
                        <div class="prediction-card prediction-churn">
                            <h2>High Risk of Churn</h2>
                            <h3>Churn Probability: {result['churn_probability']:.1%}</h3>
                            <p>Risk Level: {result['risk_level']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="prediction-card prediction-no-churn">
                            <h2>Low Risk of Churn</h2>
                            <h3>Churn Probability: {result['churn_probability']:.1%}</h3>
                            <p>Risk Level: {result['risk_level']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Probability gauge
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=result['churn_probability'] * 100,
                        title={'text': "Churn Probability (%)"},
                        gauge={
                            'axis': {'range': [None, 100]},
                            'bar': {'color': "#f5576c"},
                            'steps': [
                                {'range': [0, 30], 'color': "lightgreen"},
                                {'range': [30, 70], 'color': "yellow"},
                                {'range': [70, 100], 'color': "salmon"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 70
                            }
                        }
                    ))
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Error making prediction: {str(e)}")
        else:
            st.warning("Please train the model first in the 'Model Training' tab!")
    
    with tab4:
        st.markdown("### Results & Insights")
        
        if st.session_state.model_trained and 'churn' in df.columns:
            if st.button("Generate Batch Predictions", type="primary", use_container_width=True):
                with st.spinner("Generating predictions for all customers..."):
                    try:
                        results_df = predict_batch(
                            df,
                            st.session_state.model,
                            st.session_state.scaler,
                            st.session_state.training_columns
                        )
                        
                        st.success("Predictions generated successfully!")
                        
                        # Summary metrics
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            predicted_churn_rate = (results_df['Predicted_Churn'] == 'Yes').mean() * 100
                            st.metric("Predicted Churn Rate", f"{predicted_churn_rate:.1f}%")
                        with col2:
                            actual_churn_rate = (df['churn'].map({'Yes': 1, 'No': 0}) if df['churn'].dtype == 'object' else df['churn']).mean() * 100
                            st.metric("Actual Churn Rate", f"{actual_churn_rate:.1f}%")
                        with col3:
                            accuracy = (results_df['Predicted_Churn'] == results_df['churn'].map({'Yes': 'Yes', 'No': 'No'})).mean() * 100
                            st.metric("Model Accuracy", f"{accuracy:.1f}%")
                        
                        # Risk distribution
                        st.markdown("#### Risk Distribution")
                        risk_counts = results_df['Risk_Level'].value_counts()
                        fig = px.pie(values=risk_counts.values, 
                                   names=risk_counts.index,
                                   title="Customer Risk Distribution",
                                   color_discrete_sequence=['#4facfe', '#ffd93d', '#f5576c'])
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Results table
                        st.markdown("#### Detailed Results")
                        display_cols = ['tenure', 'monthly_charges', 'contract_type', 
                                      'Predicted_Churn', 'Churn_Probability', 'Risk_Level']
                        if 'churn' in results_df.columns:
                            display_cols.insert(4, 'churn')
                        st.dataframe(results_df[display_cols], use_container_width=True)
                        
                        # Download button
                        csv = results_df.to_csv(index=False)
                        st.download_button(
                            label="Download Predictions (CSV)",
                            data=csv,
                            file_name="churn_predictions.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                        
                    except Exception as e:
                        st.error(f"Error generating predictions: {str(e)}")
        else:
            st.info("Train the model first to see batch predictions!")

else:
    # Welcome screen
    st.markdown("### Welcome to Customer Churn Predictor")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        #### Data Upload
        Upload your customer data in CSV format
        """)
    with col2:
        st.markdown("""
        #### Model Training
        Train the machine learning model
        """)
    with col3:
        st.markdown("""
        #### Get Predictions
        Make individual or batch predictions
        """)
    
    st.markdown("---")
    st.info("Upload a CSV file from the sidebar to begin!")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; padding: 1rem;'>
    <p>Customer Churn Prediction Pipeline | Built with Streamlit</p>
</div>
""", unsafe_allow_html=True)
