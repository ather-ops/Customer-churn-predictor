"""
Customer Churn Prediction 
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

# Page config
st.set_page_config(
    page_title="Churn Prediction",
    page_icon=":chart:",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 20px;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 30px;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stButton>button {
        width: 100%;
        background: #667eea;
        color: white;
        font-weight: bold;
        padding: 10px;
        border-radius: 5px;
    }
    .stButton>button:hover {
        background: #764ba2;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='main-header'><h1>Customer Churn Prediction System</h1><p>Predict and prevent customer churn with machine learning</p></div>", unsafe_allow_html=True)

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'scaler' not in st.session_state:
    st.session_state.scaler = None
if 'training_cols' not in st.session_state:
    st.session_state.training_cols = None
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False

# Load model function
@st.cache_resource
def load_models():
    """Load pre-trained model, scaler, and training columns"""
    try:
        model = joblib.load("churn_model.pkl")
        scaler = joblib.load("scaler.pkl")
        training_cols = joblib.load("training_columns.pkl")
        return model, scaler, training_cols, True
    except Exception as e:
        st.warning("No pre-trained model found. Train a new model or upload model files.")
        return None, None, None, False

# Load models on startup
if not st.session_state.model_loaded:
    model, scaler, training_cols, loaded = load_models()
    if loaded:
        st.session_state.model = model
        st.session_state.scaler = scaler
        st.session_state.training_cols = training_cols
        st.session_state.model_loaded = True

# Sidebar
with st.sidebar:
    st.markdown("### Data Upload")
    uploaded_file = st.file_uploader("Upload CSV file", type=['csv'], help="Upload your customer data CSV file")
    
    st.markdown("---")
    st.markdown("### Model Status")
    
    if st.session_state.model_loaded:
        st.success("Model Ready")
        st.info(f"Features: {len(st.session_state.training_cols)}")
    else:
        st.warning("No Model Loaded")
        st.info("Upload data with 'churn' column to train a new model")
    
    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    This app predicts customer churn using:
    - Logistic Regression
    - Feature scaling
    - One-hot encoding
    
    Features used:
    - Tenure, Charges
    - Usage metrics
    - Contract details
    - Payment methods
    """)

# Main content
if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        
        # Remove customer_id if present
        if 'customer_id' in df.columns:
            customer_ids = df['customer_id'].copy()
            df_display = df.drop('customer_id', axis=1)
        else:
            df_display = df.copy()
        
        has_churn = 'churn' in df.columns
        
        # Metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Records", f"{len(df):,}")
        
        with col2:
            st.metric("Features", len(df.columns))
        
        if has_churn:
            churn_count = (df['churn'] == 'Yes').sum() if df['churn'].dtype == 'object' else df['churn'].sum()
            churn_rate = (churn_count/len(df)*100)
            
            with col3:
                st.metric("Churn Count", f"{churn_count:,}")
            
            with col4:
                st.metric("Churn Rate", f"{churn_rate:.1f}%",
                         delta="High" if churn_rate > 30 else "Normal" if churn_rate > 15 else "Low")
        
        # Tabs for organization
        tab1, tab2, tab3 = st.tabs(["Data Overview", "Model Training", "Predictions"])
        
        with tab1:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Data Preview")
                st.dataframe(df_display.head(10), use_container_width=True)
            
            with col2:
                st.subheader("Data Statistics")
                numeric_df = df.select_dtypes(include=[np.number])
                if not numeric_df.empty:
                    st.dataframe(numeric_df.describe(), use_container_width=True)
            
            st.subheader("Missing Values Analysis")
            missing_df = pd.DataFrame({
                'Column': df.columns,
                'Missing Count': df.isnull().sum(),
                'Missing %': (df.isnull().sum() / len(df) * 100).round(2)
            })
            missing_df = missing_df[missing_df['Missing Count'] > 0]
            if not missing_df.empty:
                st.dataframe(missing_df, use_container_width=True)
            else:
                st.success("No missing values found!")
        
        with tab2:
            if has_churn:
                st.subheader("Train New Model")
                
                col1, col2 = st.columns(2)
                with col1:
                    test_size = st.slider("Test Size", 0.1, 0.4, 0.2, 0.05)
                    random_state = st.number_input("Random State", 0, 100, 42)
                
                with col2:
                    max_iter = st.number_input("Max Iterations", 100, 5000, 1000, 100)
                    st.markdown("<br>", unsafe_allow_html=True)
                    train_button = st.button("Train Model", use_container_width=True)
                
                if train_button:
                    with st.spinner("Training model... This may take a moment."):
                        try:
                            # Data preprocessing
                            df_clean = df.copy()
                            
                            # Handle missing values
                            for col in df_clean.columns:
                                if col != 'churn':
                                    if df_clean[col].dtype == 'object':
                                        df_clean[col].fillna(df_clean[col].mode()[0] if not df_clean[col].mode().empty else 'Unknown', inplace=True)
                                    else:
                                        df_clean[col].fillna(df_clean[col].median(), inplace=True)
                            
                            # One-hot encoding
                            cat_cols = ['contract_type', 'paperless_billing', 'payment_method']
                            existing_cats = [c for c in cat_cols if c in df_clean.columns]
                            
                            if existing_cats:
                                df_encoded = pd.get_dummies(df_clean, columns=existing_cats)
                            else:
                                df_encoded = df_clean.copy()
                            
                            # Prepare features
                            X = df_encoded.drop('churn', axis=1)
                            y = df_encoded['churn']
                            
                            if y.dtype == 'object':
                                y = y.map({'Yes': 1, 'No': 0})
                            
                            # Train-test split
                            X_train, X_test, y_train, y_test = train_test_split(
                                X, y, test_size=test_size, random_state=random_state
                            )
                            
                            # Scale features
                            scaler_new = StandardScaler()
                            X_train_scaled = scaler_new.fit_transform(X_train)
                            X_test_scaled = scaler_new.transform(X_test)
                            
                            # Train model
                            model_new = LogisticRegression(max_iter=max_iter, random_state=random_state)
                            model_new.fit(X_train_scaled, y_train)
                            
                            # Predictions
                            y_pred = model_new.predict(X_test_scaled)
                            accuracy = accuracy_score(y_test, y_pred)
                            
                            # Save models
                            joblib.dump(model_new, "churn_model.pkl")
                            joblib.dump(scaler_new, "scaler.pkl")
                            joblib.dump(X.columns.tolist(), "training_columns.pkl")
                            
                            # Update session state
                            st.session_state.model = model_new
                            st.session_state.scaler = scaler_new
                            st.session_state.training_cols = X.columns.tolist()
                            st.session_state.model_loaded = True
                            
                            # Display results
                            st.success(f"Model trained successfully! Accuracy: {accuracy:.2%}")
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.subheader("Confusion Matrix")
                                cm = confusion_matrix(y_test, y_pred)
                                cm_df = pd.DataFrame(
                                    cm,
                                    columns=['Predicted No', 'Predicted Yes'],
                                    index=['Actual No', 'Actual Yes']
                                )
                                st.dataframe(cm_df, use_container_width=True)
                            
                            with col2:
                                st.subheader("Classification Report")
                                report = classification_report(y_test, y_pred, output_dict=True)
                                report_df = pd.DataFrame(report).transpose().round(3)
                                st.dataframe(report_df, use_container_width=True)
                            
                            # Feature importance
                            st.subheader("Feature Importance")
                            importance_df = pd.DataFrame({
                                'Feature': X.columns,
                                'Importance': model_new.coef_[0]
                            }).sort_values('Importance', key=abs, ascending=False).head(10)
                            
                            fig = go.Figure(go.Bar(
                                x=importance_df['Importance'],
                                y=importance_df['Feature'],
                                orientation='h',
                                marker_color='#667eea'
                            ))
                            fig.update_layout(
                                title="Top 10 Most Important Features",
                                xaxis_title="Coefficient Value",
                                yaxis_title="Feature",
                                height=400
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            
                        except Exception as e:
                            st.error(f"Training failed: {str(e)}")
            else:
                st.warning("Uploaded data doesn't contain 'churn' column. Cannot train model.")
        
        with tab3:
            if st.session_state.model_loaded:
                st.subheader("Make Predictions")
                
                prediction_type = st.radio(
                    "Select prediction type:",
                    ["Batch Prediction", "Single Customer"],
                    horizontal=True
                )
                
                if prediction_type == "Batch Prediction":
                    if st.button("Predict for All Customers", use_container_width=True):
                        with st.spinner("Making predictions..."):
                            try:
                                df_pred = df.copy()
                                if 'customer_id' in df_pred.columns:
                                    df_pred = df_pred.drop('customer_id', axis=1)
                                
                                # One-hot encoding
                                cat_cols = ['contract_type', 'paperless_billing', 'payment_method']
                                existing_cats = [c for c in cat_cols if c in df_pred.columns]
                                
                                if existing_cats:
                                    df_encoded = pd.get_dummies(df_pred, columns=existing_cats)
                                else:
                                    df_encoded = df_pred.copy()
                                
                                # Ensure all training columns exist
                                for col in st.session_state.training_cols:
                                    if col not in df_encoded.columns:
                                        df_encoded[col] = 0
                                
                                # Select and order columns
                                df_encoded = df_encoded[st.session_state.training_cols]
                                
                                # Scale and predict
                                df_scaled = st.session_state.scaler.transform(df_encoded)
                                preds = st.session_state.model.predict(df_scaled)
                                probs = st.session_state.model.predict_proba(df_scaled)
                                
                                # Create results dataframe
                                results = df.copy()
                                results['Predicted_Churn'] = ['Yes' if p == 1 else 'No' for p in preds]
                                results['Churn_Probability'] = probs[:, 1]
                                results['Risk_Level'] = pd.cut(
                                    probs[:, 1],
                                    bins=[0, 0.3, 0.7, 1],
                                    labels=['Low', 'Medium', 'High']
                                )
                                
                                # Summary metrics
                                col1, col2, col3, col4 = st.columns(4)
                                
                                risk_counts = results['Risk_Level'].value_counts()
                                
                                with col1:
                                    st.metric("High Risk", risk_counts.get('High', 0))
                                with col2:
                                    st.metric("Medium Risk", risk_counts.get('Medium', 0))
                                with col3:
                                    st.metric("Low Risk", risk_counts.get('Low', 0))
                                with col4:
                                    avg_prob = results['Churn_Probability'].mean()
                                    st.metric("Avg Probability", f"{avg_prob:.1%}")
                                
                                # Display results
                                st.subheader("Prediction Results")
                                display_cols = ['customer_id'] if 'customer_id' in results.columns else []
                                display_cols.extend(['tenure', 'monthly_charges', 'contract_type', 'Predicted_Churn', 'Churn_Probability', 'Risk_Level'])
                                available_cols = [c for c in display_cols if c in results.columns]
                                
                                st.dataframe(results[available_cols], use_container_width=True)
                                
                                # Risk distribution chart
                                fig = go.Figure(data=[
                                    go.Pie(
                                        labels=risk_counts.index,
                                        values=risk_counts.values,
                                        marker_colors=['#ff6b6b', '#ffd93d', '#6bcf7f'],
                                        hole=0.4
                                    )
                                ])
                                fig.update_layout(
                                    title="Risk Distribution",
                                    height=400
                                )
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Download button
                                csv = results.to_csv(index=False)
                                st.download_button(
                                    "Download Predictions",
                                    csv,
                                    f"churn_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                    "text/csv"
                                )
                                
                            except Exception as e:
                                st.error(f"Prediction failed: {str(e)}")
                
                else:  # Single Customer
                    st.subheader("Predict for Single Customer")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        tenure = st.number_input("Tenure (months)", 0, 100, 12)
                        monthly = st.number_input("Monthly Charges ($)", 0.0, 200.0, 65.0, 0.5)
                        total = st.number_input("Total Charges ($)", 0.0, 10000.0, 780.0, 10.0)
                        gb = st.number_input("Avg Monthly GB Download", 0.0, 200.0, 25.0, 0.5)
                        calls = st.number_input("Avg Calls per Month", 0, 200, 45)
                    
                    with col2:
                        service = st.number_input("Customer Service Calls", 0, 20, 2)
                        contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
                        paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
                        payment = st.selectbox("Payment Method", ["Electronic check", "Credit card", "Bank transfer", "Mailed check"])
                    
                    if st.button("Predict", use_container_width=True):
                        try:
                            customer = pd.DataFrame([{
                                'tenure': tenure,
                                'monthly_charges': monthly,
                                'total_charges': total,
                                'avg_monthly_gb_download': gb,
                                'avg_calls_per_month': calls,
                                'customer_service_calls': service,
                                'contract_type': contract,
                                'paperless_billing': paperless,
                                'payment_method': payment
                            }])
                            
                            # One-hot encoding
                            cat_cols = ['contract_type', 'paperless_billing', 'payment_method']
                            existing_cats = [c for c in cat_cols if c in customer.columns]
                            
                            if existing_cats:
                                customer_encoded = pd.get_dummies(customer, columns=existing_cats)
                            else:
                                customer_encoded = customer.copy()
                            
                            # Ensure all training columns exist
                            for col in st.session_state.training_cols:
                                if col not in customer_encoded.columns:
                                    customer_encoded[col] = 0
                            
                            # Select and order columns
                            customer_encoded = customer_encoded[st.session_state.training_cols]
                            
                            # Scale and predict
                            customer_scaled = st.session_state.scaler.transform(customer_encoded)
                            pred = st.session_state.model.predict(customer_scaled)[0]
                            prob = st.session_state.model.predict_proba(customer_scaled)[0][1]
                            
                            # Display result
                            st.markdown("---")
                            
                            col1, col2 = st.columns([1, 1])
                            
                            with col1:
                                if pred == 1:
                                    if prob >= 0.7:
                                        st.error("HIGH RISK - Customer likely to churn")
                                    else:
                                        st.warning("MEDIUM RISK - Customer may churn")
                                else:
                                    st.success("LOW RISK - Customer likely to stay")
                                
                                st.metric("Churn Probability", f"{prob:.1%}")
                            
                            with col2:
                                # Gauge chart
                                fig = go.Figure(go.Indicator(
                                    mode="gauge+number",
                                    value=prob * 100,
                                    title={"text": "Churn Risk"},
                                    number={'suffix': "%"},
                                    gauge={
                                        "axis": {"range": [0, 100]},
                                        "bar": {"color": "#667eea"},
                                        "steps": [
                                            {"range": [0, 30], "color": "#6bcf7f"},
                                            {"range": [30, 70], "color": "#ffd93d"},
                                            {"range": [70, 100], "color": "#ff6b6b"}
                                        ],
                                        "threshold": {
                                            "line": {"color": "red", "width": 4},
                                            "thickness": 0.75,
                                            "value": prob * 100
                                        }
                                    }
                                ))
                                fig.update_layout(height=300)
                                st.plotly_chart(fig, use_container_width=True)
                            
                        except Exception as e:
                            st.error(f"Prediction failed: {str(e)}")
            
            else:
                st.info("No model available. Please train a model first.")
    
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")

else:
    # Welcome screen
    st.markdown("### Welcome to the Churn Prediction System!")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        #### Instructions:
        
        1. Upload your data - Use the sidebar to upload a CSV file
        2. Train a model - If your data has a 'churn' column, you can train a new model
        3. Make predictions - Use the model to predict churn for new customers
        
        #### Required Data Format:
        
        Your CSV should contain these columns:
        - tenure, monthly_charges, total_charges
        - avg_monthly_gb_download, avg_calls_per_month
        - customer_service_calls
        - contract_type, paperless_billing, payment_method
        - churn (for training only)
        """)
    
    with col2:
        st.markdown("""
        #### Sample Data Preview:
        """)
        
        # Load and show sample data
        try:
            sample_df = pd.read_csv("Customer_churn1.csv")
            st.dataframe(sample_df.head(5), use_container_width=True)
            
            st.markdown("""
            #### Current Model Status:
            """)
            
            if st.session_state.model_loaded:
                st.success(f"Model loaded with {len(st.session_state.training_cols)} features")
            else:
                st.warning("No model found - train one with your data")
                
        except:
            st.info("Sample data file not found")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>"
    "Churn Prediction System | Built with Streamlit | "
    f"v1.0 | {datetime.now().year}"
    "</div>",
    unsafe_allow_html=True
)
