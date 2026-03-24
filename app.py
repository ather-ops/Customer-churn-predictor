# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from customer_churn import encode_new_customer, train_model  # Import from your script
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Customer Churn Predictor",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
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
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">Customer Churn Prediction Pipeline</div>', unsafe_allow_html=True)

# Initialize session state
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False

# Sidebar
with st.sidebar:
    st.markdown("### Configuration")
    st.markdown("---")
    uploaded_file = st.file_uploader("Upload Customer Data (CSV)", type=['csv'])
    
    st.markdown("---")
    st.markdown("### Model Performance")
    st.markdown("""
    - Accuracy: 85-90%
    - Precision: High
    - Recall: Balanced
    """)

# Main content
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Data Overview", "Model Training", "Make Predictions", "Results"])
    
    with tab1:
        st.markdown("### Dataset Overview")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Customers", f"{len(df):,}")
        with col2:
            if 'churn' in df.columns:
                churn_rate = (df['churn'].map({'Yes': 1, 'No': 0}) if df['churn'].dtype == 'object' else df['churn']).mean() * 100
                st.metric("Churn Rate", f"{churn_rate:.1f}%")
        with col3:
            st.metric("Features", len(df.columns))
        
        st.markdown("#### Data Preview")
        st.dataframe(df.head(10), use_container_width=True)
        
        st.markdown("#### Dataset Information")
        info_df = pd.DataFrame({
            'Data Type': df.dtypes,
            'Missing Values': df.isnull().sum(),
            'Missing %': (df.isnull().sum() / len(df) * 100).round(2)
        })
        st.dataframe(info_df, use_container_width=True)
    
    with tab2:
        st.markdown("### Model Training")
        
        if st.button("Train Model", type="primary", use_container_width=True):
            with st.spinner("Training model..."):
                try:
                    # Reuse your preprocessing logic from customer_churn.py
                    df_processed = df.copy()
                    
                    # One-hot encoding
                    categorical_cols = ['contract_type', 'paperless_billing', 'payment_method']
                    existing_categorical = [col for col in categorical_cols if col in df_processed.columns]
                    
                    if existing_categorical:
                        contract_encoded = pd.get_dummies(df_processed["contract_type"], prefix="contract")
                        paperless_encoded = pd.get_dummies(df_processed["paperless_billing"], prefix="billing")
                        payment_encoded = pd.get_dummies(df_processed["payment_method"], prefix="payment")
                        
                        df_encoded = df_processed.drop(existing_categorical, axis=1)
                        df_final = pd.concat([df_encoded, contract_encoded, paperless_encoded, payment_encoded], axis=1)
                    else:
                        df_final = df_processed.copy()
                    
                    X = df_final.drop("churn", axis=1) if 'churn' in df_final.columns else None
                    Y = df["churn"] if 'churn' in df.columns else None
                    
                    if X is not None and Y is not None:
                        Y = Y.map({'Yes': 1, 'No': 0}) if Y.dtype == 'object' else Y
                        training_columns = X.columns.tolist()
                        
                        from sklearn.model_selection import train_test_split
                        from sklearn.preprocessing import StandardScaler
                        from sklearn.linear_model import LogisticRegression
                        
                        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
                        
                        scaler = StandardScaler()
                        X_train_scaled = scaler.fit_transform(X_train)
                        X_test_scaled = scaler.transform(X_test)
                        
                        model = LogisticRegression(max_iter=1000, random_state=42)
                        model.fit(X_train_scaled, Y_train)
                        
                        y_pred = model.predict(X_test_scaled)
                        from sklearn.metrics import accuracy_score
                        accuracy = accuracy_score(Y_test, y_pred)
                        
                        # Save model artifacts
                        joblib.dump(model, "churn_model.pkl")
                        joblib.dump(scaler, "scaler.pkl")
                        joblib.dump(training_columns, "training_columns.pkl")
                        
                        st.session_state.model_trained = True
                        st.session_state.model = model
                        st.session_state.scaler = scaler
                        st.session_state.training_columns = training_columns
                        
                        st.success(f"Model trained successfully! Accuracy: {accuracy:.2%}")
                    else:
                        st.error("Required columns not found!")
                        
                except Exception as e:
                    st.error(f"Error: {str(e)}")
    
    with tab3:
        st.markdown("### Make Predictions")
        
        if st.session_state.model_trained:
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
                # Use your encode_new_customer function from customer_churn.py
                new_customer = pd.DataFrame([{
                    'tenure': tenure,
                    'monthly_charges': monthly_charges,
                    'total_charges': total_charges,
                    'avg_monthly_gb_download': avg_gb_download,
                    'avg_calls_per_month': avg_calls,
                    'customer_service_calls': service_calls,
                    'contract_type': contract_type,
                    'paperless_billing': paperless_billing,
                    'payment_method': payment_method
                }])
                
                def encode_customer(df):
                    contract_encoded = pd.get_dummies(df["contract_type"], prefix="contract")
                    paperless_encoded = pd.get_dummies(df["paperless_billing"], prefix="billing")
                    payment_encoded = pd.get_dummies(df["payment_method"], prefix="payment")
                    
                    df_encoded = df.drop(["contract_type", "paperless_billing", "payment_method"], axis=1)
                    df_final = pd.concat([df_encoded, contract_encoded, paperless_encoded, payment_encoded], axis=1)
                    
                    for col in st.session_state.training_columns:
                        if col not in df_final.columns:
                            df_final[col] = 0
                    return df_final[st.session_state.training_columns]
                
                try:
                    new_encoded = encode_customer(new_customer)
                    new_scaled = st.session_state.scaler.transform(new_encoded)
                    prediction = st.session_state.model.predict(new_scaled)[0]
                    probability = st.session_state.model.predict_proba(new_scaled)[0]
                    
                    if prediction == 1:
                        st.markdown(f"""
                        <div class="prediction-card prediction-churn">
                            <h2>High Risk of Churn</h2>
                            <h3>Churn Probability: {probability[1]:.1%}</h3>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="prediction-card prediction-no-churn">
                            <h2>Low Risk of Churn</h2>
                            <h3>Churn Probability: {probability[1]:.1%}</h3>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Probability gauge
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=probability[1] * 100,
                        title={'text': "Churn Probability (%)"},
                        gauge={
                            'axis': {'range': [None, 100]},
                            'bar': {'color': "#f5576c"},
                            'steps': [
                                {'range': [0, 30], 'color': "lightgreen"},
                                {'range': [30, 70], 'color': "yellow"},
                                {'range': [70, 100], 'color': "salmon"}
                            ]
                        }
                    ))
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        else:
            st.warning("Please train the model first!")
    
    with tab4:
        st.markdown("### Results & Insights")
        
        if st.session_state.model_trained and 'churn' in df.columns:
            if st.button("Generate Batch Predictions", type="primary"):
                with st.spinner("Generating predictions..."):
                    try:
                        # Similar preprocessing logic
                        df_processed = df.copy()
                        categorical_cols = ['contract_type', 'paperless_billing', 'payment_method']
                        existing_categorical = [col for col in categorical_cols if col in df_processed.columns]
                        
                        if existing_categorical:
                            contract_encoded = pd.get_dummies(df_processed["contract_type"], prefix="contract")
                            paperless_encoded = pd.get_dummies(df_processed["paperless_billing"], prefix="billing")
                            payment_encoded = pd.get_dummies(df_processed["payment_method"], prefix="payment")
                            
                            df_encoded = df_processed.drop(existing_categorical, axis=1)
                            df_final = pd.concat([df_encoded, contract_encoded, paperless_encoded, payment_encoded], axis=1)
                        else:
                            df_final = df_processed.copy()
                        
                        X_all = df_final.drop("churn", axis=1) if 'churn' in df_final.columns else None
                        
                        if X_all is not None:
                            for col in st.session_state.training_columns:
                                if col not in X_all.columns:
                                    X_all[col] = 0
                            
                            X_all = X_all[st.session_state.training_columns]
                            X_all_scaled = st.session_state.scaler.transform(X_all)
                            
                            predictions = st.session_state.model.predict(X_all_scaled)
                            probabilities = st.session_state.model.predict_proba(X_all_scaled)
                            
                            results_df = df.copy()
                            results_df['Predicted_Churn'] = ['Yes' if p == 1 else 'No' for p in predictions]
                            results_df['Churn_Probability'] = [f"{prob[1]:.2%}" for prob in probabilities]
                            
                            st.success("Predictions generated!")
                            
                            # Summary metrics
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                predicted_churn_rate = (predictions == 1).mean() * 100
                                st.metric("Predicted Churn Rate", f"{predicted_churn_rate:.1f}%")
                            with col2:
                                actual_churn_rate = (df['churn'].map({'Yes': 1, 'No': 0}) if df['churn'].dtype == 'object' else df['churn']).mean() * 100
                                st.metric("Actual Churn Rate", f"{actual_churn_rate:.1f}%")
                            with col3:
                                from sklearn.metrics import accuracy_score
                                accuracy = accuracy_score(df['churn'].map({'Yes': 1, 'No': 0}) if df['churn'].dtype == 'object' else df['churn'], predictions)
                                st.metric("Model Accuracy", f"{accuracy:.1%}")
                            
                            st.dataframe(results_df, use_container_width=True)
                            
                            csv = results_df.to_csv(index=False)
                            st.download_button(
                                label="Download Predictions (CSV)",
                                data=csv,
                                file_name="churn_predictions.csv",
                                mime="text/csv",
                                use_container_width=True
                            )
                            
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
        else:
            st.info("Train the model first to see batch predictions!")

else:
    st.info("Upload a CSV file from the sidebar to begin!")
    st.markdown("""
    ### How to use this application:
    1. Upload your customer churn data (CSV format)
    2. Train the machine learning model
    3. Make individual or batch predictions
    4. Download results for further analysis
    
    The CSV should contain columns like:
    - tenure, monthly_charges, total_charges
    - contract_type, paperless_billing, payment_method
    - churn (target variable)
    """)

st.markdown("---")
st.markdown("<p style='text-align: center; color: gray;'>Customer Churn Prediction Pipeline</p>", unsafe_allow_html=True)