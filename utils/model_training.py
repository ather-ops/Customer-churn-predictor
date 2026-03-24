"""
Model training utilities
"""
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import os

def train_churn_model(X, y, test_size=0.2, random_state=42):
    """
    Train logistic regression model for churn prediction
    
    Parameters:
    -----------
    X : array-like
        Feature matrix
    y : array-like
        Target vector
    test_size : float
        Proportion of data to use for testing
    random_state : int
        Random seed for reproducibility
    
    Returns:
    --------
    dict
        Dictionary containing model, predictions, and metrics
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Train model
    model = LogisticRegression(max_iter=1000, random_state=random_state)
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)
    
    return {
        'model': model,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'y_pred': y_pred,
        'y_prob': y_prob,
        'accuracy': accuracy,
        'roc_auc': roc_auc,
        'classification_report': report,
        'confusion_matrix': cm
    }

def save_model(model, scaler, training_columns, model_dir='models'):
    """
    Save model artifacts to disk
    
    Parameters:
    -----------
    model : sklearn model
        Trained model
    scaler : StandardScaler
        Fitted scaler
    training_columns : list
        List of column names used in training
    model_dir : str
        Directory to save models
    """
    os.makedirs(model_dir, exist_ok=True)
    
    joblib.dump(model, os.path.join(model_dir, 'churn_model.pkl'))
    joblib.dump(scaler, os.path.join(model_dir, 'scaler.pkl'))
    joblib.dump(training_columns, os.path.join(model_dir, 'training_columns.pkl'))
    
    print(f"Models saved to {model_dir}/")

def load_model(model_dir='models'):
    """
    Load model artifacts from disk
    
    Parameters:
    -----------
    model_dir : str
        Directory containing model files
    
    Returns:
    --------
    tuple
        (model, scaler, training_columns)
    """
    model = joblib.load(os.path.join(model_dir, 'churn_model.pkl'))
    scaler = joblib.load(os.path.join(model_dir, 'scaler.pkl'))
    training_columns = joblib.load(os.path.join(model_dir, 'training_columns.pkl'))
    
    return model, scaler, training_columns
