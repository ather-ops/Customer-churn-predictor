"""
Data preprocessing utilities
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

def encode_categorical(df, categorical_cols=None):
    """
    Encode categorical variables using one-hot encoding
    
    Parameters:
    -----------
    df : pandas DataFrame
        Input dataframe
    categorical_cols : list
        List of categorical column names
    
    Returns:
    --------
    pandas DataFrame
        Encoded dataframe
    """
    if categorical_cols is None:
        categorical_cols = ['contract_type', 'paperless_billing', 'payment_method']
    
    # Check which categorical columns exist
    existing_categorical = [col for col in categorical_cols if col in df.columns]
    
    if not existing_categorical:
        return df
    
    # Create dummy variables
    encoded_dfs = []
    df_processed = df.copy()
    
    for col in existing_categorical:
        dummies = pd.get_dummies(df_processed[col], prefix=col.split('_')[0])
        encoded_dfs.append(dummies)
        df_processed = df_processed.drop(col, axis=1)
    
    # Concatenate all encoded columns
    if encoded_dfs:
        df_encoded = pd.concat([df_processed] + encoded_dfs, axis=1)
    else:
        df_encoded = df_processed
    
    return df_encoded

def preprocess_data(df, target_col='churn', training_columns=None, scaler=None, fit_scaler=False):
    """
    Complete data preprocessing pipeline
    
    Parameters:
    -----------
    df : pandas DataFrame
        Input dataframe
    target_col : str
        Name of target column
    training_columns : list
        List of columns from training (for prediction)
    scaler : StandardScaler
        Fitted scaler object
    fit_scaler : bool
        Whether to fit the scaler
    
    Returns:
    --------
    tuple
        (X, y, training_columns, scaler)
    """
    # Handle target variable
    if target_col in df.columns:
        y = df[target_col].copy()
        if y.dtype == 'object':
            y = y.map({'Yes': 1, 'No': 0})
    else:
        y = None
    
    # Remove target from features
    X = df.drop(target_col, axis=1) if target_col in df.columns else df.copy()
    
    # Encode categorical variables
    X_encoded = encode_categorical(X)
    
    # Ensure all training columns are present
    if training_columns is not None:
        for col in training_columns:
            if col not in X_encoded.columns:
                X_encoded[col] = 0
        X_encoded = X_encoded[training_columns]
    else:
        training_columns = X_encoded.columns.tolist()
    
    # Scale features
    if fit_scaler:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_encoded)
    else:
        X_scaled = scaler.transform(X_encoded) if scaler is not None else X_encoded.values
    
    return X_scaled, y, training_columns, scaler

def handle_missing_values(df):
    """
    Handle missing values in the dataset
    
    Parameters:
    -----------
    df : pandas DataFrame
        Input dataframe
    
    Returns:
    --------
    pandas DataFrame
        Dataframe with handled missing values
    """
    df_clean = df.copy()
    
    # Fill numerical missing values with median
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df_clean[col].isnull().any():
            df_clean[col].fillna(df_clean[col].median(), inplace=True)
    
    # Fill categorical missing values with mode
    categorical_cols = df_clean.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if df_clean[col].isnull().any():
            df_clean[col].fillna(df_clean[col].mode()[0] if not df_clean[col].mode().empty else 'Unknown', inplace=True)
    
    return df_clean
