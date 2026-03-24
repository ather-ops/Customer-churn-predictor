"""
Prediction utilities
"""
import pandas as pd
import numpy as np
from .data_preprocessing import preprocess_data

def predict_churn(customer_data, model, scaler, training_columns):
    """
    Make single prediction for a customer
    
    Parameters:
    -----------
    customer_data : dict or pandas DataFrame
        Customer data
    model : sklearn model
        Trained model
    scaler : StandardScaler
        Fitted scaler
    training_columns : list
        List of columns used in training
    
    Returns:
    --------
    dict
        Prediction results
    """
    if isinstance(customer_data, dict):
        customer_df = pd.DataFrame([customer_data])
    else:
        customer_df = customer_data.copy()
    
    # Preprocess
    X_scaled, _, _, _ = preprocess_data(
        customer_df, 
        target_col=None,
        training_columns=training_columns,
        scaler=scaler,
        fit_scaler=False
    )
    
    # Predict
    prediction = model.predict(X_scaled)[0]
    probability = model.predict_proba(X_scaled)[0]
    
    return {
        'churn_prediction': 'Yes' if prediction == 1 else 'No',
        'churn_probability': probability[1],
        'stay_probability': probability[0],
        'risk_level': get_risk_level(probability[1])
    }

def predict_batch(dataframe, model, scaler, training_columns, target_col='churn'):
    """
    Make batch predictions for multiple customers
    
    Parameters:
    -----------
    dataframe : pandas DataFrame
        Customer data
    model : sklearn model
        Trained model
    scaler : StandardScaler
        Fitted scaler
    training_columns : list
        List of columns used in training
    target_col : str
        Name of target column (if exists)
    
    Returns:
    --------
    pandas DataFrame
        Original dataframe with predictions added
    """
    results_df = dataframe.copy()
    
    # Preprocess
    X_scaled, _, _, _ = preprocess_data(
        dataframe,
        target_col=target_col if target_col in dataframe.columns else None,
        training_columns=training_columns,
        scaler=scaler,
        fit_scaler=False
    )
    
    # Predict
    predictions = model.predict(X_scaled)
    probabilities = model.predict_proba(X_scaled)
    
    # Add results
    results_df['Predicted_Churn'] = ['Yes' if p == 1 else 'No' for p in predictions]
    results_df['Churn_Probability'] = [f"{prob[1]:.2%}" for prob in probabilities]
    results_df['Stay_Probability'] = [f"{prob[0]:.2%}" for prob in probabilities]
    results_df['Risk_Level'] = [get_risk_level(prob[1]) for prob in probabilities]
    
    return results_df

def get_risk_level(probability):
    """
    Categorize risk level based on churn probability
    
    Parameters:
    -----------
    probability : float
        Churn probability
    
    Returns:
    --------
    str
        Risk level category
    """
    if probability < 0.3:
        return 'Low Risk'
    elif probability < 0.7:
        return 'Medium Risk'
    else:
        return 'High Risk'
